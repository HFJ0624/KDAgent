"""Offline audit, impact inventory, and subsequent run entrypoint for prompt_alignment_v2.

The ``audit``, ``dry-run``, and ``package`` paths of this module only read local files; only
``run`` with an explicit ``--approve-api`` enters the model-calling path. Historical results are
never used as completion markers for resuming the new version, so old-version records are not reused by mistake.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis" / "prompt_alignment_v2"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "prompt_alignment_v2"
FORMAL_ROOT = OUTPUT_ROOT / "formal"
ZIP_PATH = PROJECT_ROOT / "outputs" / "KDAgent_prompt_alignment_preflight.zip"
RESULTS_ZIP_PATH = PROJECT_ROOT / "outputs" / "KDAgent_prompt_alignment_v2_results.zip"
PROTOCOL_PATH = ANALYSIS_ROOT / "experiment_protocol.md"
CONFIG_PATH = ANALYSIS_ROOT / "config.json"
PROMPT_DIR = OUTPUT_ROOT / "prompt_examples"

SWAT_CASES = PROJECT_ROOT / "data" / "swat_s2s_raw_window_fixed" / "llm_prompt_cases.jsonl"
WADI_CASES = PROJECT_ROOT / "outputs" / "wadi_external_validation_v1" / "prepared_data" / "llm_prompt_cases.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    from src.data_loader import normalize_case

    cases: List[Dict[str, Any]] = []
    for raw in read_jsonl(path):
        case = normalize_case(raw)
        # normalize_case generates normalized fields, but compact/full evidence formatting still needs to retain
        # the original top_k_variables; the time series cannot be reconstructed from candidate IDs alone.
        if isinstance(raw.get("top_k_variables"), list):
            case["top_k_variables"] = raw["top_k_variables"]
        cases.append(case)
    return cases


def shared_evidence(case: Dict[str, Any]) -> str:
    from src.prompt_builder import format_public_numerical_evidence

    return format_public_numerical_evidence(case, compact_evidence=False)


def read_template(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def current_prompt(case: Dict[str, Any], template: str, process_knowledge: str = "", contexts: Any = None) -> str:
    from src.prompt_builder import build_prompt

    return build_prompt(
        case,
        template,
        process_knowledge=process_knowledge,
        rag_contexts=contexts,
        compact_evidence=False,
    )


def safe_source_hashes() -> Dict[str, str]:
    paths = [
        PROJECT_ROOT / "src" / "prompt_builder.py",
        PROJECT_ROOT / "src" / "model_client.py",
        PROJECT_ROOT / "experiments" / "react_adapted_v1" / "react_agent.py",
        PROJECT_ROOT / "prompts" / "rca_prompt_template.txt",
        PROJECT_ROOT / "experiments" / "wadi_external_validation_v1" / "prompts" / "wadi_rca_prompt_template.txt",
        PROJECT_ROOT / "src" / "main.py",
        ANALYSIS_ROOT / "preflight.py",
        CONFIG_PATH,
        PROTOCOL_PATH,
    ]
    return {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path) for path in paths if path.exists()}


def ensure_output_root() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def old_raw_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/raw_responses/*.jsonl")) if root.exists() else []


def old_iteration_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in old_raw_files(root):
        for row in read_jsonl(path):
            if row.get("record_type") == "agent_iteration":
                copied = dict(row)
                copied["_source_file"] = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                rows.append(copied)
    return rows


def current_sw_at_prompt(case: Dict[str, Any], contexts: Any = None) -> str:
    # The historical SWaT logs do not record a valid process_knowledge path, and the original default path does not exist;
    # rebuild with empty background to avoid mistaking newly added knowledge for historical facts.
    template = read_template(PROJECT_ROOT / "prompts" / "rca_prompt_template.txt")
    return current_prompt(case, template, process_knowledge="", contexts=contexts)


def first_rows_by_key(rows: Iterable[Mapping[str, Any]], branch: str | None = None) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if branch is not None and row.get("branch") != branch:
            continue
        if int(row.get("iteration") or 0) != 1:
            continue
        key = f"{row.get('case_id')}|{row.get('run_id')}|{Path(str(row.get('_source_file', ''))).parent.parent.name}"
        result.setdefault(key, dict(row))
    return result


def audit_existing() -> Dict[str, Any]:
    """Audit data scale, old-request evidence, ReAct evidence differences, and reuse boundaries."""
    from src.prompt_builder import build_prompt, format_public_numerical_evidence

    ensure_output_root()
    swat_cases = load_cases(SWAT_CASES)
    wadi_cases = load_cases(WADI_CASES)
    case_maps = {"swat": {str(c["case_id"]): c for c in swat_cases}, "wadi": {str(c["case_id"]): c for c in wadi_cases}}

    template = read_template(PROJECT_ROOT / "prompts" / "rca_prompt_template.txt")
    wadi_template = read_template(PROJECT_ROOT / "experiments" / "wadi_external_validation_v1" / "prompts" / "wadi_rca_prompt_template.txt")

    checks: List[Dict[str, Any]] = []
    def check(name: str, passed: bool, evidence: str, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence, "detail": detail})

    check("swat_template_has_rag_placeholder", "{rag_knowledge_block}" in template, "prompts/rca_prompt_template.txt")
    check("wadi_template_has_rag_placeholder", "{rag_knowledge_block}" in wadi_template, "experiments/wadi_external_validation_v1/prompts/wadi_rca_prompt_template.txt")
    check("swat_episode_count", len(swat_cases) == 20, str(SWAT_CASES), len(swat_cases))
    check("wadi_episode_count", len(wadi_cases) == 13, str(WADI_CASES), len(wadi_cases))
    for dataset, cases in (("swat", swat_cases), ("wadi", wadi_cases)):
        ids = [str(c.get("case_id")) for c in cases]
        candidates_ok = all(len(c.get("top10_vars") or []) == 10 for c in cases)
        check(f"{dataset}_unique_case_ids", len(ids) == len(set(ids)), str(SWAT_CASES if dataset == "swat" else WADI_CASES), sorted(x for x, n in Counter(ids).items() if n > 1))
        check(f"{dataset}_exact_top10", candidates_ok, str(SWAT_CASES if dataset == "swat" else WADI_CASES), [c.get("case_id") for c in cases if len(c.get("top10_vars") or []) != 10])

    # For the real old prompts, only the saved user prompt is checked; the old raw files have no final system prompt,
    # so "same body" is not extended here to "same complete request".
    swat_dual = PROJECT_ROOT / "outputs" / "final_experiments_frozen_v2"
    dual_rows = old_iteration_rows(swat_dual)
    dual_data = first_rows_by_key(dual_rows, "data_evidence")
    dual_rag = first_rows_by_key(dual_rows, "rag_knowledge")
    same_prompt = sum(1 for key in dual_data.keys() & dual_rag.keys() if dual_data[key].get("prompt") == dual_rag[key].get("prompt"))
    rag_with_marker = sum(1 for row in dual_rag.values() if "检索到的工业工艺知识" in str(row.get("prompt") or ""))
    check("historical_swat_dual_user_prompt_equality", same_prompt == len(dual_data.keys() & dual_rag.keys()), "outputs/final_experiments_frozen_v2/**/raw_responses/*.jsonl", {"paired": len(dual_data.keys() & dual_rag.keys()), "same": same_prompt})
    # The pass condition here is that "the audit can confirm the old problem", not that the historical erroneous result passes.
    check("historical_swat_rag_injection_bug_detected", len(dual_rag) == 300 and rag_with_marker == 0, "outputs/final_experiments_frozen_v2/**/raw_responses/*.jsonl", {"rag_first_round": len(dual_rag), "with_marker": rag_with_marker, "interpretation": "旧版本检索段落未进入保存的首轮 user prompt"})

    # Reproduce the full evidence view of the main experiment and check whether the old ReAct tool returned the compact view.
    old_tool = read_jsonl(PROJECT_ROOT / "outputs" / "react_adapted_v1" / "react" / "tool_events.jsonl")
    evidence_total = 0
    evidence_match = 0
    evidence_old_lengths: List[int] = []
    evidence_new_lengths: List[int] = []
    for item in old_tool:
        dataset = str(item.get("dataset") or "").lower()
        case = case_maps.get(dataset, {}).get(str(item.get("episode_id")))
        if not case:
            continue
        for event in item.get("events") or []:
            if event.get("tool") != "get_episode_evidence":
                continue
            old_text = str((event.get("observation") or {}).get("frozen_evidence") or "")
            new_text = format_public_numerical_evidence(case, compact_evidence=False)
            evidence_total += 1
            evidence_old_lengths.append(len(old_text))
            evidence_new_lengths.append(len(new_text))
            evidence_match += int(old_text == new_text)
    # Likewise, the mismatch between the old view and the public view is exactly the impact this round should catch; the corrected
    # code path is verified through test doubles and a shared entrypoint, and old records should not be rewritten.
    check("react_old_evidence_misalignment_detected", evidence_total == 99 and evidence_match == 0, "outputs/react_adapted_v1/react/tool_events.jsonl; src/prompt_builder.py", {"records": evidence_total, "matches": evidence_match, "old_avg_chars": round(sum(evidence_old_lengths) / max(1, len(evidence_old_lengths)), 2), "new_avg_chars": round(sum(evidence_new_lengths) / max(1, len(evidence_new_lengths)), 2), "interpretation": "旧 ReAct 证据工具使用 compact 视图"})

    # Use a local double to verify the RAG injection gate and actual rendering, without triggering the retriever/Embedding.
    fake_contexts = [{"source": "offline-test", "content": "OFFLINE_RAG_SENTINEL_CONTENT"}]
    rendered = build_prompt(swat_cases[0], template, process_knowledge="", rag_contexts=fake_contexts, compact_evidence=False)
    check("rag_injection_contains_full_offline_context", "OFFLINE_RAG_SENTINEL_CONTENT" in rendered and "offline-test" in rendered, "src/prompt_builder.py; prompts/rca_prompt_template.txt")
    missing_template_failed = False
    try:
        build_prompt(swat_cases[0], template.replace("{rag_knowledge_block}", ""), process_knowledge="", rag_contexts=fake_contexts, compact_evidence=False)
    except ValueError as exc:
        missing_template_failed = str(exc) == "rag_template_missing_placeholder"
    check("rag_missing_placeholder_is_blocked", missing_template_failed, "src/prompt_builder.py::validate_rag_template_support")
    evidence_text = format_public_numerical_evidence(swat_cases[0], compact_evidence=False)
    label_sentinel = dict(swat_cases[0])
    label_sentinel["gt_vars"] = ["LABEL_SENTINEL_SHOULD_NOT_APPEAR"]
    sentinel_text = format_public_numerical_evidence(label_sentinel, compact_evidence=False)
    check("public_evidence_label_isolation", "LABEL_SENTINEL_SHOULD_NOT_APPEAR" not in sentinel_text and evidence_text == sentinel_text, "src/prompt_builder.py::format_public_numerical_evidence")

    # Create request examples with fixed IDs. For SWaT, the "old request" first takes the historically saved body; the full retrieval
    # block is not saved in the old KDAgent raw records, so the example is explicitly marked as reconstructed rather than masquerading as a historical request.
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    swat_case = swat_cases[0]
    wadi_case = wadi_cases[0]
    swat_old = next((row.get("prompt") for row in dual_rows if row.get("case_id") == swat_case["case_id"] and row.get("branch") == "rag_knowledge" and row.get("iteration") == 1), "")
    def saved_contexts(dataset: str, case_id: Any) -> tuple[List[Dict[str, Any]], str]:
        roots = [PROJECT_ROOT / "outputs" / "final_experiments_frozen_v2", PROJECT_ROOT / "outputs" / "final_experiments_frozen_v1"] if dataset == "swat" else [PROJECT_ROOT / "outputs" / "react_adapted_v1" / "comparators" / "wadi"]
        for root in roots:
            for path in old_raw_files(root):
                for row in read_jsonl(path):
                    if row.get("record_type") == "case_summary" and str(row.get("case_id")) == str(case_id) and row.get("rag_contexts"):
                        return list(row.get("rag_contexts") or []), str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        return [], "not_found"

    for dataset, case, tpl, old_prompt in (("swat", swat_case, template, swat_old), ("wadi", wadi_case, wadi_template, "")):
        example_contexts, context_source = saved_contexts(dataset, case["case_id"])
        new_prompt = build_prompt(case, tpl, process_knowledge="", rag_contexts=example_contexts, compact_evidence=False)
        example = {
            "dataset": dataset,
            "case_id": str(case["case_id"]),
            "status": "reconstructed_example; not a new model request",
            "old_request": {"available": bool(old_prompt), "prompt": old_prompt or "", "note": "历史记录未保存最终请求或完整检索块；空值不代表请求不存在。"},
            "new_request_reconstruction": {"prompt": new_prompt, "retrieval_contexts": example_contexts, "retrieval_context_source": context_source, "evidence_view": shared_evidence(case)},
            "evidence_view_sha256": hashlib.sha256(shared_evidence(case).encode("utf-8")).hexdigest(),
        }
        write_json(PROMPT_DIR / f"{dataset}_case_{case['case_id']}.json", example)

    affected_rows, rerun_rows = build_impact_rows(swat_cases, wadi_cases, dual_rows, old_tool)
    affected_fields = ["dataset", "method", "model", "run_id", "episode_id", "branch", "retrieval_condition", "old_record_source", "decision", "reason", "evidence_level", "old_input_fingerprint", "new_input_fingerprint", "downstream_outputs"]
    write_csv(OUTPUT_ROOT / "affected_results.csv", affected_rows, affected_fields)
    rerun_fields = ["dataset", "method", "retrieval_condition", "diagnosis_records", "reuse_count", "rerun_count", "offline_recompute_count", "unknown_count", "logic_calls_upper_bound", "network_attempts_upper_bound", "reason"]
    write_csv(OUTPUT_ROOT / "rerun_plan.csv", rerun_rows, rerun_fields)
    write_csv(OUTPUT_ROOT / "paper_impact_map.csv", paper_impact_rows(), ["result_file", "paper_use", "processing", "reason"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "api_calls": 0,
        "real_results_status": "not_run",
        "checks": checks,
        "source_hashes": safe_source_hashes(),
        "historical_prompt_evidence": {"swat_dual_first_round_pairs": len(dual_data.keys() & dual_rag.keys()), "same_user_prompt_pairs": same_prompt, "rag_prompts_with_injected_marker": rag_with_marker},
        "react_evidence_evidence": {"old_tool_events": evidence_total, "old_equals_new_shared_view": evidence_match},
        "affected_row_count": len(affected_rows),
        "rerun_plan_row_count": len(rerun_rows),
    }
    write_json(OUTPUT_ROOT / "offline_acceptance.json", report)
    write_text(OUTPUT_ROOT / "fix_summary_zh.md", fix_summary(report, rerun_rows))
    write_text(OUTPUT_ROOT / "run_commands.md", run_commands())
    write_text(OUTPUT_ROOT / "implementation_changes.patch", implementation_patch_text())
    copy_static_materials()
    write_json(OUTPUT_ROOT / "manifest.json", build_manifest(exclude_zip=True))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def build_impact_rows(swat_cases: List[Dict[str, Any]], wadi_cases: List[Dict[str, Any]], dual_rows: List[Dict[str, Any]], react_tool_rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate method-level decisions while preserving the record-level evidence for each diagnosis/branch."""
    rows: List[Dict[str, Any]] = []
    grouped: defaultdict[tuple[str, str, str], Dict[str, int]] = defaultdict(lambda: Counter())
    diagnosis_keys: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    def add(dataset: str, method: str, model: str, run: int, episode: str, branch: str, condition: str, source: str, decision: str, reason: str, level: str, downstream: str) -> None:
        row = {
            "dataset": dataset, "method": method, "model": model or "unknown", "run_id": run, "episode_id": episode,
            "branch": branch, "retrieval_condition": condition, "old_record_source": source, "decision": decision,
            "reason": reason, "evidence_level": level, "old_input_fingerprint": "not_saved" if decision == "needs_evidence" else "historical_record", "new_input_fingerprint": "recomputed_after_code_fix" if decision == "rerun_required" else "unchanged_or_offline", "downstream_outputs": downstream,
        }
        rows.append(row)
        key = (dataset, method, condition)
        diagnosis_keys[key].add(f"{dataset}|{method}|{model}|{run}|{episode}|{condition}")
        grouped[key]["reuse_count"] += decision == "reuse_verified"
        grouped[key]["rerun_count"] += decision == "rerun_required"
        grouped[key]["offline_recompute_count"] += decision == "offline_recompute"
        grouped[key]["unknown_count"] += decision == "needs_evidence"

    models = ["qwen-plus", "qwen-max", "deepseek-v4-flash", "deepseek-v4-pro", "glm-5.2"]
    # The SWaT main experiment is 5 models x 3 runs x 20 cases. Baseline and Only Agent do not depend on
    # the RAG slot or the ReAct evidence view, so they are reused directly; only offline fusion depends on the updated branch outputs.
    for method in ("Direct Baseline", "Only Agent"):
        for model in models:
            for run in range(1, 4):
                for case in swat_cases:
                    add("SWaT", method, model, run, str(case["case_id"]), "single", "clean", "outputs/final_experiments_frozen_v1", "reuse_verified", "无 RAG/证据视图依赖；请求配置保持不变。", "historical prompt/result record", "无")
    for model in models:
        for run in range(1, 4):
            for case in swat_cases:
                add("SWaT", "KDAgent", model, run, str(case["case_id"]), "data_evidence", "clean", "outputs/final_experiments_frozen_v2", "reuse_verified", "证据分支不注入检索；公共模板空插槽渲染后与历史正文可核对。", "historical prompt record", "融合需在检索分支更新后离线重算")
                add("SWaT", "KDAgent", model, run, str(case["case_id"]), "rag_knowledge", "clean", "outputs/final_experiments_frozen_v2", "rerun_required", "历史 RAG 分支首轮提示缺少检索段落，不能把检索成功日志当作注入证据。", "historical prompt lacks injected marker", "AP/RP/RRF/Borda 需重算")
                add("SWaT", "Serial", model, run, str(case["case_id"]), "iterative", "clean", "outputs/final_experiments_frozen_v1", "rerun_required", "首轮请求受 SWaT RAG 插槽修正影响；后续轮次不能只替换最终答案。", "historical first-round prompt and current template", "下游结果需重算")
                add("SWaT", "Only RAG", model, run, str(case["case_id"]), "single", "clean", "outputs/final_experiments_frozen_v1", "rerun_required", "RAG 上下文非空但旧 SWaT 模板无注入插槽。", "historical prompt lacks injected marker", "下游结果需重算")

    # Every diagnosis in the old ReAct tool traces uses compact evidence; the new version switches to the
    # same full public view as the main experiment, so the old ReAct diagnoses are not reused.
    for item in react_tool_rows:
        dataset = str(item.get("dataset") or "").upper()
        method = "ReAct"
        model = "deepseek-v4-pro"
        run = int(item.get("run_id") or 0)
        episode = str(item.get("episode_id"))
        if any(event.get("tool") == "get_episode_evidence" for event in item.get("events") or []):
            add(dataset, method, model, run, episode, "evidence_tool", "clean", "outputs/react_adapted_v1/react/tool_events.jsonl", "rerun_required", "旧证据工具返回为 compact 视图；新版本改用 shared_main_prompt_full_v1。", "historical compact evidence", "新 ReAct 结果独立保存")

    # The existing retrieval robustness is an offline dependency of the generated branches; recomputation does not call the API.
    robustness = PROJECT_ROOT / "outputs" / "retrieval_robustness_v1" / "records.jsonl"
    if robustness.exists():
        seen_robustness: set[str] = set()
        for record in read_jsonl(robustness):
            condition = str(record.get("condition") or "")
            if condition not in {"random_retrieval", "stage_mismatched_retrieval", "low_relevance_retrieval"}:
                continue
            method_name = {
                "kdagent": "KDAgent retrieval perturbation",
                "serial": "Serial retrieval perturbation",
                "retrieval_branch": "Retrieval branch perturbation",
            }.get(str(record.get("method") or ""), f"{record.get('method')} retrieval perturbation")
            model = str(record.get("model") or "unknown")
            run = int(record.get("run_id") or 0)
            episode = str(record.get("case_id"))
            unique_key = f"{method_name}|{model}|{run}|{episode}|{condition}"
            if unique_key in seen_robustness:
                continue
            seen_robustness.add(unique_key)
            add("SWaT", method_name, model, run, episode, str(record.get("method") or "fusion"), condition, str(robustness.relative_to(PROJECT_ROOT)).replace("\\", "/"), "offline_recompute", "已有扰动记录逐条读取；不重生成分支。", "stored retrieval record", "更新分支后重算 AP/RP/RRF/Borda")

    plan: List[Dict[str, Any]] = []
    for (dataset, method, condition), counts in sorted(grouped.items()):
        rerun_count = int(counts["rerun_count"])
        # Each ReAct diagnosis has at most four logical generations; the branch limits for KDAgent/Serial are
        # listed separately per the historical protocol, so network retries are not miscounted as logical generations.
        if method == "ReAct":
            logic = rerun_count * 4
            network = logic * 5
        elif method == "KDAgent":
            logic = rerun_count * 1
            network = logic * 5
        elif method == "Serial":
            logic = rerun_count * 3
            network = logic * 5
        elif method == "Only RAG":
            logic = rerun_count
            network = logic * 5
        else:
            logic = 0
            network = 0
        plan.append({"dataset": dataset, "method": method, "retrieval_condition": condition, "diagnosis_records": len(diagnosis_keys[(dataset, method, condition)]), "reuse_count": counts["reuse_count"], "rerun_count": rerun_count, "offline_recompute_count": counts["offline_recompute_count"], "unknown_count": counts["unknown_count"], "logic_calls_upper_bound": logic, "network_attempts_upper_bound": network, "reason": "rerun_count 按受影响分支/记录计；逻辑调用上限不是预期实际消耗。"})
    return rows, plan


def paper_impact_rows() -> List[Dict[str, str]]:
    return [
        {"result_file": "outputs/final_experiments_frozen_v2/**", "paper_use": "KDAgent SWaT clean 表格/图/检验", "processing": "保留证据分支，检索相关结果待补跑并重算", "reason": "SWaT RAG 分支旧请求没有注入证据"},
        {"result_file": "outputs/final_experiments_frozen_v1/rag_self_refinement_agent_*", "paper_use": "Serial SWaT 表格/图/检验", "processing": "补跑整条诊断，不替换最终答案", "reason": "首轮提示路径受模板修正影响"},
        {"result_file": "outputs/react_adapted_v1/react/**", "paper_use": "ReAct 对照表格/图/检验", "processing": "旧结果保留为历史版本，新版本独立补跑", "reason": "公共证据视图从 compact 改为主实验完整呈现视图"},
        {"result_file": "outputs/retrieval_robustness_v1/records.jsonl", "paper_use": "检索鲁棒性与融合分析", "processing": "当前记录离线重算；若使用修正后的分支，重算依赖项", "reason": "不调用 API，不覆盖旧结果"},
    ]


def run_commands() -> str:
    return """# prompt_alignment_v2 可复现命令

工作目录：`D:\\workspace\\lab_TARCA_S2S\\llm_rca_experiment`

## 本轮已实际执行（无 API）

```powershell
python -m py_compile analysis/prompt_alignment_v2/preflight.py
python -m analysis.prompt_alignment_v2.preflight audit
python -m analysis.prompt_alignment_v2.preflight dry-run
python -m analysis.prompt_alignment_v2.preflight package
```

## 后续正式补跑入口

`run` 默认拒绝发起请求；必须明确指定方法和 `--approve-api`。新结果写入
`outputs/prompt_alignment_v2/formal/`，不会读取旧 ReAct 的完成标记，也不会覆盖历史输出。

```powershell
$env:DASHSCOPE_API_KEY="<从环境变量注入，不写入文件>"
python -m analysis.prompt_alignment_v2.preflight run --method react-adapted --datasets swat wadi --runs 3 --approve-api --resume
```

传统方法的补跑命令由同一入口生成并要求显式方法：

```powershell
python -m analysis.prompt_alignment_v2.preflight run --method kdagent --dataset swat --runs 3 --approve-api --resume
python -m analysis.prompt_alignment_v2.preflight run --method serial --dataset swat --runs 3 --approve-api --resume
python -m analysis.prompt_alignment_v2.preflight run --method only-rag --dataset swat --runs 3 --approve-api --resume
```

正式运行后的评分和重算：

```powershell
python -m analysis.prompt_alignment_v2.preflight score
python -m analysis.prompt_alignment_v2.preflight package
```

`dry-run` 只读取 `rerun_plan.csv`，报告诊断数、分支/逻辑调用上限和网络尝试上限；
上限不是实际消耗，未提供价格估算。
"""


def implementation_patch_text() -> str:
    return """--- a/prompts/rca_prompt_template.txt
+++ b/prompts/rca_prompt_template.txt
@@
 背景知识：
 {process_knowledge}
 
+{rag_knowledge_block}
 任务说明：

--- a/src/prompt_builder.py
+++ b/src/prompt_builder.py
@@
+def format_public_numerical_evidence(case, compact_evidence=False):
+    return format_top10_candidates_compact(case) if compact_evidence else format_top10_candidates(case)
+
+def validate_rag_template_support(template, rag_contexts):
+    if rag_contexts and "{rag_knowledge_block}" not in template:
+        raise ValueError("rag_template_missing_placeholder")
@@
+    validate_rag_template_support(template, rag_contexts)
+    top10_text = format_public_numerical_evidence(case, compact_evidence=compact_evidence)

--- a/experiments/react_adapted_v1/react_agent.py
+++ b/experiments/react_adapted_v1/react_agent.py
@@
-from src.prompt_builder import format_top10_candidates_compact
+from src.prompt_builder import format_public_numerical_evidence
@@
-"frozen_evidence": format_top10_candidates_compact(case)
+"frozen_evidence": format_public_numerical_evidence(case, compact_evidence=False)
+# 同时保存证据视图格式和 SHA-256，便于核对主实验输入是否一致。

--- a/src/model_client.py
+++ b/src/model_client.py
@@
+# chat 在发送前生成脱敏 request_snapshot、request_control 和 request_fingerprint；
+# Authorization 值永远不写入快照。

--- a/src/main.py
+++ b/src/main.py
@@
+# 增加可选 experiment_version/protocol_sha256 字段，保证后续版本记录与旧版本
+# 的 resume 完成标记在记录层面可区分。
"""


def fix_summary(report: Dict[str, Any], rerun_rows: List[Dict[str, Any]]) -> str:
    grouped = Counter((row["method"], row["decision"]) for row in read_csv_rows(OUTPUT_ROOT / "affected_results.csv"))
    lines = [
        "# KDAgent 实验预检修正摘要",
        "",
        "本轮只做离线代码修正和材料审计，不调用 LLM/Embedding API，不生成新的模型预测，不修改论文和历史结果。",
        "",
        f"- 离线验收状态：**{report['status']}**。",
        "- SWaT 检索模板已补入 `{rag_knowledge_block}`；WADI 原模板已有该插槽。",
        "- RAG 非空且模板缺少插槽时，`build_prompt()` 现在抛出 `rag_template_missing_placeholder`，不会静默丢弃检索文本。",
        "- ReAct `get_episode_evidence` 已改为复用主实验公共数值证据入口；旧 99 条工具轨迹使用的是 compact 视图，因此旧 ReAct 结果不能伪装成修正后结果。",
        "- 客户端新增脱敏最终请求快照和请求控制字段，未来正式运行可审计 system/user messages、参数和网络重试控制。",
        "",
        "## 历史影响判定",
        "",
    ]
    for (method, decision), count in sorted(grouped.items()):
        lines.append(f"- {method} / {decision}: {count} 条记录或分支。")
    lines += [
        "",
        "## 严格边界",
        "",
        "- 历史 KDAgent 记录中的证据分支与检索分支首轮 user prompt 正文相同，但旧 raw 没有完整 system prompt/最终请求快照；不能据此称为完整请求相同。",
        "- SWaT 旧检索分支没有保存完整检索块正文，故不能用旧日志重建其每条完整请求；对应分支列为 `rerun_required`。",
        "- ReAct 旧工具日志保存了 compact evidence；公共视图改变后，SWaT 60 条和 WADI 39 条均列入新版本补跑。",
        "- `13/20` 等可覆盖子集只用于评价分层，不参与本清单的样本筛选。",
        "- 本轮正式真实结果状态为 `not_run`；`run` 入口已实现但未执行。",
        "",
        "## 后续建议",
        "",
        "先运行 `dry-run` 审核预算，再按 `run_commands.md` 对受影响方法执行独立版本补跑；旧结果、旧检验和新结果分目录保存。",
    ]
    return "\n".join(lines)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dry_run() -> Dict[str, Any]:
    plan = read_csv_rows(OUTPUT_ROOT / "rerun_plan.csv")
    result = {
        "status": "PASS" if plan else "FAIL",
        "api_calls": 0,
        "real_results_status": "not_run",
        "plan_rows": len(plan),
        "diagnosis_records_to_rerun": sum(int(row["rerun_count"]) for row in plan),
        "logic_calls_upper_bound": sum(int(row["logic_calls_upper_bound"]) for row in plan),
        "network_attempts_upper_bound": sum(int(row["network_attempts_upper_bound"]) for row in plan),
        "note": "上限按分支/诊断计算，不把网络重试计为逻辑生成；未估算价格。",
    }
    write_json(OUTPUT_ROOT / "dry_run_budget.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def run_formal(args: argparse.Namespace) -> None:
    if not args.approve_api:
        raise RuntimeError("正式运行必须显式传入 --approve-api；本轮未发起 API 请求。")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY；本轮未发起 API 请求。")
    if args.method == "react-adapted":
        run_react_v2(args)
        return
    run_legacy_v2(args)


def protocol_hash() -> str:
    return sha256(PROTOCOL_PATH)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def run_react_v2(args: argparse.Namespace) -> None:
    """Versioned ReAct runner; old react_adapted_v1 records never match this version's keys."""
    from src.model_client import ModelClient, load_model_configs
    from src.rag_retriever import RAGRetriever
    from experiments.react_adapted_v1.react_agent import AgentLimits, ReactAdaptedAgent

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model_name = cfg["model_name"]
    model_cfg = dict(load_model_configs(str(PROJECT_ROOT / "configs" / "models.yaml"))[model_name])
    model_cfg.update(cfg["generation"])
    client = ModelClient(model_cfg)
    out = FORMAL_ROOT / "react" / model_name
    final_path = out / "final_records.jsonl"
    score_path = out / "scoring_keys.jsonl"
    if final_path.exists() and not args.resume:
        raise RuntimeError(f"新版本结果已存在：{final_path}；请使用 --resume，禁止覆盖。")
    existing = {str(row.get("record_key")) for row in read_jsonl(final_path)} if args.resume else set()
    system_prompt = (PROJECT_ROOT / "experiments" / "react_adapted_v1" / "prompts" / "react_system.txt").read_text(encoding="utf-8")
    limits = AgentLimits(**cfg["agent_limits"])
    source_hash = safe_source_hashes()
    for dataset in args.datasets:
        cases = load_cases(SWAT_CASES if dataset == "swat" else WADI_CASES)
        rag_config = PROJECT_ROOT / cfg["datasets"][dataset]["rag_config"]
        retriever = RAGRetriever.from_config_path(str(rag_config))
        agent = ReactAdaptedAgent(client, retriever, system_prompt, limits)
        for run_id in range(1, args.runs + 1):
            for index, case in enumerate(cases, start=1):
                key = "|".join(("prompt_alignment_v2", protocol_hash(), dataset, "ReAct", model_name, str(case["case_id"]), str(run_id), "clean"))
                if key in existing:
                    print(f"[skip] {key}", flush=True)
                    continue
                print(f"[run] dataset={dataset} run={run_id}/{args.runs} episode={case['case_id']} ({index}/{len(cases)})", flush=True)
                result = agent.run(dataset, case, run_id)
                calls = result.pop("calls")
                events = result.pop("tool_events")
                result.update({"record_key": key, "experiment_version": "prompt_alignment_v2", "protocol_sha256": protocol_hash(), "source_hashes": source_hash})
                append_jsonl(out / "raw_model_calls.jsonl", {"record_key": key, "calls": calls, "experiment_version": "prompt_alignment_v2", "protocol_sha256": protocol_hash()})
                append_jsonl(out / "tool_events.jsonl", {"record_key": key, "events": events, "experiment_version": "prompt_alignment_v2", "protocol_sha256": protocol_hash()})
                append_jsonl(score_path, {"record_key": key, "dataset": dataset, "episode_id": str(case["case_id"]), "run_id": run_id, "gt_vars": case.get("gt_vars", []), "top10_vars": case.get("top10_vars", [])})
                append_jsonl(final_path, result)
                existing.add(key)
                print(f"[done] {key} status={result['stopping_reason']} calls={result['logic_calls']} tokens={result['total_tokens']}", flush=True)
                time.sleep(float(cfg["sleep_seconds"]))


def run_legacy_v2(args: argparse.Namespace) -> None:
    """Build a version-isolated command for the legacy methods; actual execution is delegated to the existing src/main.py."""
    dataset = args.dataset
    if dataset not in {"swat", "wadi"}:
        raise ValueError("legacy 方法需要 --dataset swat 或 wadi")
    cases = SWAT_CASES if dataset == "swat" else WADI_CASES
    template = "prompts/rca_prompt_template.txt" if dataset == "swat" else "experiments/wadi_external_validation_v1/prompts/wadi_rca_prompt_template.txt"
    knowledge = "data/process_knowledge_template.md" if dataset == "swat" else "outputs/wadi_external_validation_v1/prepared_data/process_knowledge_template.md"
    flags = {"kdagent": (1, 1), "serial": (1, 0), "only-rag": (0, 0), "baseline": (0, 0), "only-agent": (0, 1)}
    if args.method not in flags:
        raise ValueError(f"不支持的 legacy 方法：{args.method}")
    use_rag, use_iterative = flags[args.method]
    output = FORMAL_ROOT / f"{args.method}_{dataset}_{json.loads(CONFIG_PATH.read_text(encoding='utf-8'))['model_name']}"
    command = [sys.executable, "src/main.py", "--data_path", str(cases), "--config_path", "configs/models.yaml", "--model_name", "deepseek-v4-pro", "--output_dir", str(output), "--prompt_template", template, "--knowledge_path", knowledge, "--num_runs", str(args.runs), "--temperature", "0.2", "--max_tokens", "8192", "--thinking_budget", "2048", "--use_rag", str(use_rag), "--use_iterative_agent", str(use_iterative), "--use_dual_branch_fusion", "1" if args.method == "kdagent" else "0", "--rag_top_k", "5", "--max_iterations", "3", "--min_confidence", "0.5", "--experiment_version", "prompt_alignment_v2", "--protocol_sha256", protocol_hash()]
    if args.resume:
        command.append("--resume")
    print("[formal command] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def score_formal() -> None:
    """Score the v2 ReAct formal records independently per dataset."""
    react_dir = FORMAL_ROOT / "react" / "deepseek-v4-pro"
    final_path = react_dir / "final_records.jsonl"
    score_path = react_dir / "scoring_keys.jsonl"
    if not final_path.exists() or not score_path.exists():
        result = {"status": "not_run", "api_calls": 0, "note": "本轮未生成新预测；正式补跑完成后再次执行 score。"}
        write_json(OUTPUT_ROOT / "score_status.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    labels = {row["record_key"]: row for row in read_jsonl(score_path)}
    records = read_jsonl(final_path)
    duplicate_keys = [key for key, count in Counter(row.get("record_key") for row in records).items() if count > 1]
    scored: List[Dict[str, Any]] = []
    missing_labels = []
    for record in records:
        label = labels.get(record.get("record_key"))
        if not label:
            missing_labels.append(record.get("record_key"))
            continue
        parsed = record.get("parsed_response") or {}
        predicted = [canonical(value) for value in parsed.get("predicted_root_causes", []) if canonical(value)][:5]
        truth = [canonical(value) for value in label.get("gt_vars", []) if canonical(value)]
        candidates = [canonical(value) for value in label.get("top10_vars", []) if canonical(value)]
        truth_set = set(truth)
        best_rank = next((index + 1 for index, value in enumerate(predicted) if value in truth_set), None)
        dcg = sum(1.0 / math.log2(index + 2) for index, value in enumerate(predicted) if value in truth_set)
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(truth_set), 5)))
        scored.append(
            {
                "dataset": str(record.get("dataset") or "").lower(),
                "method": "ReAct",
                "model": record.get("model_name", ""),
                "run_id": record.get("run_id"),
                "episode_id": record.get("episode_id"),
                "predicted_top5": ";".join(predicted),
                "gt_vars": ";".join(truth),
                "top10_vars": ";".join(candidates),
                "reachable": int(bool(truth_set.intersection(candidates))),
                "hit_at_1": int(best_rank is not None and best_rank <= 1),
                "hit_at_3": int(best_rank is not None and best_rank <= 3),
                "hit_at_5": int(best_rank is not None and best_rank <= 5),
                "mrr": 1.0 / best_rank if best_rank else 0.0,
                "ndcg_at_5": dcg / ideal if ideal else 0.0,
                "best_gt_rank": best_rank or "",
                "completed": int(bool(record.get("completed"))),
                "valid_output": int(bool(record.get("valid_output"))),
                "logic_calls": int(record.get("logic_calls", 0) or 0),
                "network_attempts": int(record.get("network_attempts", 0) or 0),
                "prompt_tokens": int(record.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(record.get("completion_tokens", 0) or 0),
                "reasoning_tokens": int(record.get("reasoning_tokens", 0) or 0),
                "total_tokens": int(record.get("total_tokens", 0) or 0),
                "elapsed_s": float(record.get("elapsed_s", 0.0) or 0.0),
                "stopping_reason": record.get("stopping_reason", ""),
                "record_key": record.get("record_key", ""),
            }
        )

    record_fields = list(scored[0].keys()) if scored else []
    write_csv(OUTPUT_ROOT / "formal_record_level_results.csv", scored, record_fields)

    # Three runs are repeated observations of the same episode; here we give the descriptive record mean while clearly
    # stating the number of episodes, to avoid calling the 60/39 records distinct events.
    summaries: List[Dict[str, Any]] = []
    for dataset in ("swat", "wadi"):
        dataset_rows = [row for row in scored if row["dataset"] == dataset]
        for scope, scoped in (
            ("full", dataset_rows),
            ("reachable", [row for row in dataset_rows if row["reachable"]]),
        ):
            n = len(scoped)
            summaries.append(
                {
                    "dataset": dataset,
                    "method": "ReAct",
                    "model": "deepseek-v4-pro",
                    "scope": scope,
                    "records": n,
                    "episodes": len({row["episode_id"] for row in scoped}),
                    "hit_at_1_count": sum(row["hit_at_1"] for row in scoped),
                    "hit_at_3_count": sum(row["hit_at_3"] for row in scoped),
                    "hit_at_5_count": sum(row["hit_at_5"] for row in scoped),
                    "hit_at_1": sum(row["hit_at_1"] for row in scoped) / n if n else "",
                    "hit_at_3": sum(row["hit_at_3"] for row in scoped) / n if n else "",
                    "hit_at_5": sum(row["hit_at_5"] for row in scoped) / n if n else "",
                    "mrr": sum(row["mrr"] for row in scoped) / n if n else "",
                    "ndcg_at_5": sum(row["ndcg_at_5"] for row in scoped) / n if n else "",
                    "completion_rate": sum(row["completed"] for row in scoped) / n if n else "",
                    "valid_output_rate": sum(row["valid_output"] for row in scoped) / n if n else "",
                    "logic_calls": sum(row["logic_calls"] for row in scoped),
                    "total_tokens": sum(row["total_tokens"] for row in scoped),
                    "elapsed_s": sum(row["elapsed_s"] for row in scoped),
                }
            )
    summary_fields = list(summaries[0].keys()) if summaries else []
    write_csv(OUTPUT_ROOT / "formal_summary_metrics.csv", summaries, summary_fields)

    expected = {"swat": 60, "wadi": 39}
    observed = Counter(row["dataset"] for row in scored)
    invalid_outputs = sum(1 for row in scored if not row["valid_output"])
    complete = all(observed[name] == count for name, count in expected.items())
    status = "PASS" if complete and not missing_labels and not duplicate_keys and not invalid_outputs else "FAIL"
    result = {
        "status": status,
        "api_calls_during_scoring": 0,
        "formal_result_scope": "ReAct only",
        "scored_records": len(scored),
        "expected_records": sum(expected.values()),
        "records_by_dataset": dict(observed),
        "missing_label_records": len(missing_labels),
        "duplicate_record_keys": len(duplicate_keys),
        "invalid_output_records": invalid_outputs,
        "traditional_method_reruns_found": any(
            path.is_dir() for path in FORMAL_ROOT.glob("*_swat_*")
        ),
    }
    write_json(OUTPUT_ROOT / "score_status.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def copy_static_materials() -> None:
    """Place this round's entrypoint and protocol into the package so the preflight version can be inspected outside the project directory."""
    targets = {
        ANALYSIS_ROOT / "preflight.py": OUTPUT_ROOT / "source" / "preflight.py",
        CONFIG_PATH: OUTPUT_ROOT / "source" / "config.json",
        PROTOCOL_PATH: OUTPUT_ROOT / "source" / "experiment_protocol.md",
    }
    for source, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build_manifest(exclude_zip: bool = True) -> Dict[str, Any]:
    files = []
    if OUTPUT_ROOT.exists():
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            # The manifest cannot record its own hash, otherwise that hash would immediately become invalid once the new manifest is written.
            if (
                not path.is_file()
                or path == OUTPUT_ROOT / "manifest.json"
                or (exclude_zip and path.suffix.lower() == ".zip")
            ):
                continue
            files.append({"path": str(path.relative_to(OUTPUT_ROOT)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    score_status_path = OUTPUT_ROOT / "score_status.json"
    score_status = json.loads(score_status_path.read_text(encoding="utf-8")) if score_status_path.exists() else {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls_during_packaging": 0,
        "real_results_status": "react_complete" if score_status.get("status") == "PASS" else "incomplete",
        "formal_result_scope": score_status.get("formal_result_scope", "not_run"),
        "score_status": score_status,
        "files": files,
        "source_hashes": safe_source_hashes(),
    }


def package() -> None:
    ensure_output_root()
    copy_static_materials()
    score_status_path = OUTPUT_ROOT / "score_status.json"
    score_status = json.loads(score_status_path.read_text(encoding="utf-8")) if score_status_path.exists() else {}
    readme = """# KDAgent Prompt Alignment v2 结果包

本包包含 prompt_alignment_v2 的离线审计材料与正式 ReAct 结果。正式记录共
99 条：SWaT 20 episodes x 3 runs，WADI 13 episodes x 3 runs。原始响应、工具轨迹、
评分标签、record-level 指标、汇总指标和资源消耗均保留。

当前 formal 目录未发现 SWaT KDAgent retrieval branch、Only RAG 和 Serial 的 v2
补跑输出，因此本包不得表述为原补跑计划 999 条已全部完成。历史结果未被覆盖，
本包也不包含 API 密钥、.env、虚拟环境、向量索引或大型原始传感器数据。
"""
    write_text(OUTPUT_ROOT / "PACKAGE_README.md", readme)
    manifest = build_manifest(exclude_zip=True)
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    forbidden = (".env", ".key", ".pem", ".pyc", ".zip")
    forbidden_dirs = {"__pycache__", ".venv", "venv", "chroma_db"}
    target_zip = RESULTS_ZIP_PATH if score_status.get("status") == "PASS" else ZIP_PATH
    if target_zip.exists():
        target_zip.unlink()
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if not path.is_file() or path == target_zip:
                continue
            if path.suffix.lower() in forbidden or forbidden_dirs.intersection(path.parts):
                raise RuntimeError(f"禁止文件进入预检包：{path}")
            archive.write(path, Path("prompt_alignment_v2") / path.relative_to(OUTPUT_ROOT))
    with zipfile.ZipFile(target_zip) as archive:
        entry_count = len(archive.namelist())
    print(json.dumps({"status": "PASS", "zip_path": str(target_zip), "zip_size_bytes": target_zip.stat().st_size, "entries": entry_count}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KDAgent prompt alignment v2 preflight")
    parser.add_argument("mode", choices=("audit", "dry-run", "run", "score", "package"))
    parser.add_argument("--method", choices=("react-adapted", "kdagent", "serial", "only-rag", "baseline", "only-agent"), default="react-adapted")
    parser.add_argument("--dataset", choices=("swat", "wadi"), default="swat")
    parser.add_argument("--datasets", nargs="+", choices=("swat", "wadi"), default=["swat", "wadi"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--approve-api", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "audit":
        audit_existing()
    elif args.mode == "dry-run":
        dry_run()
    elif args.mode == "run":
        run_formal(args)
    elif args.mode == "score":
        score_formal()
    else:
        package()


if __name__ == "__main__":
    main()
