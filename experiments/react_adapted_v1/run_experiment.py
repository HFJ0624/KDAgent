"""Unified entry point for ReAct-adapted auditing, running, scoring, and packaging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "react_adapted_v1"
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cases, normalize_case  # noqa: E402
from src.model_client import ModelClient, load_model_configs  # noqa: E402
from src.rag_retriever import RAGRetriever  # noqa: E402
from src.utils import append_jsonl, read_yaml  # noqa: E402
from experiments.react_adapted_v1.react_agent import AgentLimits, ReactAdaptedAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ReAct-adapted 工业 RCA 对照实验。")
    parser.add_argument(
        "--mode",
        choices=("audit", "synth-test", "dry-run", "run-react", "run-wadi-comparators", "score", "export-txt", "package", "all-offline", "formal"),
        default="all-offline",
    )
    parser.add_argument("--datasets", nargs="+", choices=("swat", "wadi"), default=["swat", "wadi"])
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--approve-api", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config() -> Dict[str, Any]:
    return read_yaml(str(CONFIG_PATH))


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(materialized[0].keys()) if materialized else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_normalized_cases(path: Path) -> List[Dict[str, Any]]:
    """Normalize cases while keeping the raw candidate structure needed for compact evidence formatting."""
    cases: List[Dict[str, Any]] = []
    for raw in load_cases(str(path)):
        case = normalize_case(raw)
        if isinstance(raw.get("top_k_variables"), list):
            case["top_k_variables"] = raw["top_k_variables"]
        cases.append(case)
    return cases


def record_key(dataset: str, method: str, model: str, episode: Any, run_id: int) -> str:
    return "|".join((dataset, method, model, str(episode), str(run_id)))


def data_audit(output: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"generated_at": datetime.now().isoformat(), "datasets": {}, "existing_results": {}}
    for dataset, spec in cfg["datasets"].items():
        cases_path = resolve(spec["cases"])
        cases = load_normalized_cases(cases_path)
        ids = [str(case["case_id"]) for case in cases]
        duplicate_ids = sorted({value for value in ids if ids.count(value) > 1})
        invalid_candidate_counts = [case["case_id"] for case in cases if len(case.get("top10_vars") or []) != 10]
        retriever = RAGRetriever.from_config_path(str(resolve(spec["rag_config"])))
        report["datasets"][dataset] = {
            "cases_path": str(cases_path),
            "cases_sha256": sha256(cases_path),
            "expected_episodes": int(spec["expected_episodes"]),
            "actual_episodes": len(cases),
            "duplicate_episode_ids": duplicate_ids,
            "episodes_without_exactly_10_candidates": invalid_candidate_counts,
            "reachable_episodes": sum(bool(set(case.get("gt_vars") or []) & set(case.get("top10_vars") or [])) for case in cases),
            "rag_config": str(resolve(spec["rag_config"])),
            "knowledge_chunk_count": retriever.store.count(),
            "audit_manifest": str(resolve(spec["audit_manifest"])),
            "audit_manifest_sha256": sha256(resolve(spec["audit_manifest"])),
        }

    for name, path in cfg.get("existing_results", {}).items():
        result_path = resolve(path)
        raw_dir = result_path / "raw_responses"
        files = sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
        summaries = 0
        for file in files:
            summaries += sum(1 for row in load_jsonl(file) if row.get("record_type") == "case_summary")
        report["existing_results"][name] = {"path": str(result_path), "run_files": len(files), "case_summary_records": summaries}

    failures = []
    for dataset, item in report["datasets"].items():
        if item["actual_episodes"] != item["expected_episodes"]:
            failures.append(f"{dataset}: episode count {item['actual_episodes']} != {item['expected_episodes']}")
        if item["duplicate_episode_ids"] or item["episodes_without_exactly_10_candidates"]:
            failures.append(f"{dataset}: duplicate IDs or non-Top-10 candidate panels")
        if item["knowledge_chunk_count"] <= 0:
            failures.append(f"{dataset}: knowledge collection unavailable")
    report["status"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    write_json(output / "data_availability_report.json", report)

    kb_roots = {
        "swat": PROJECT_ROOT / "data" / "rag_kb",
        "wadi": PROJECT_ROOT / "outputs" / "wadi_external_validation_v1" / "prepared_data" / "rag_kb",
    }
    kb_audit = []
    forbidden_fields = ("gt_vars", "root_tags", "episode_id", "case_id", "official_attack_id")
    for dataset, root in kb_roots.items():
        extra_sources = [
            PROJECT_ROOT / "data" / "swat_s2s_raw_window_fixed" / "process_knowledge_template.md"
        ] if dataset == "swat" else [
            PROJECT_ROOT / "outputs" / "wadi_external_validation_v1" / "prepared_data" / "process_knowledge_template.md"
        ]
        for path in sorted(root.glob("*.md")) + extra_sources:
            text = path.read_text(encoding="utf-8").lower()
            matched = [field for field in forbidden_fields if field in text]
            kb_audit.append(
                {
                    "dataset": dataset,
                    "source_file": str(path),
                    "sha256": sha256(path),
                    "structured_label_fields_found": ";".join(matched),
                    "contains_ground_truth_phrase": int("ground truth" in text or "ground-truth" in text),
                    "audit_result": "FAIL" if matched else "PASS",
                    "notes": "The phrase 'ground truth' may occur only in a generic non-case disclaimer; structured case-label fields are forbidden.",
                }
            )
    write_csv(output / "kb_label_isolation_audit.csv", kb_audit)

    lines = ["# 数据可用性报告", "", f"状态：**{report['status']}**", ""]
    for dataset, item in report["datasets"].items():
        lines.extend(
            [
                f"## {dataset.upper()}",
                f"- 冻结 episode：{item['actual_episodes']}（预期 {item['expected_episodes']}）",
                f"- Top-10 可覆盖 episode：{item['reachable_episodes']}",
                f"- 知识块：{item['knowledge_chunk_count']}",
                f"- 输入：`{item['cases_path']}`",
                "",
            ]
        )
    lines.append("## 已保存对照")
    for name, item in report["existing_results"].items():
        lines.append(f"- {name}: {item['case_summary_records']} 条 case summary，{item['run_files']} 个 run 文件")
    if failures:
        lines.extend(["", "## 阻塞", *[f"- {failure}" for failure in failures]])
    (output / "data_availability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def synthetic_test() -> None:
    subprocess.run([sys.executable, "-m", "unittest", "experiments.react_adapted_v1.test_react_adapted", "-v"], cwd=PROJECT_ROOT, check=True)


def dry_run(output: Path, cfg: Dict[str, Any], runs: int) -> Dict[str, Any]:
    diagnosis_count = sum(int(cfg["datasets"][name]["expected_episodes"]) for name in ("swat", "wadi")) * runs
    # WADI is still missing KDAgent runs 2-3 and Serial runs 1-3; the SWaT comparisons can be reused once verified.
    comparator_diagnoses = 13 * 2 + 13 * 3
    budget = cfg["budget"]
    result = {
        "model_name": cfg["model_name"],
        "react_diagnoses": diagnosis_count,
        "react_expected_by_dataset": {name: int(cfg["datasets"][name]["expected_episodes"]) * runs for name in ("swat", "wadi")},
        "react_max_logic_calls": diagnosis_count * int(budget["max_logic_calls_per_diagnosis"]),
        "react_max_completion_tokens": diagnosis_count * int(budget["max_completion_tokens_per_diagnosis"]),
        "wadi_comparator_diagnoses_to_add": comparator_diagnoses,
        "wadi_comparator_max_logic_calls": 13 * 2 * 4 + 13 * 3 * 3,
        "api_key_configured_in_current_process": bool(os.environ.get("DASHSCOPE_API_KEY")),
        "formal_ready_except_credentials": diagnosis_count == 99,
        "note": "调用上限不是预期实际消耗；通过即停止会减少逻辑调用。未估算价格。",
    }
    write_json(output / "dry_run_budget.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def configured_model(cfg: Dict[str, Any]) -> ModelClient:
    models = load_model_configs(str(PROJECT_ROOT / "configs" / "models.yaml"))
    model_cfg = dict(models[cfg["model_name"]])
    budget = cfg["budget"]
    model_cfg.update(
        {
            "temperature": 0.2,
            "max_tokens": int(budget["max_completion_tokens_per_call"]),
            "thinking_budget": int(budget["thinking_budget"]),
            "timeout": int(budget["timeout_seconds"]),
            "max_retries": int(budget["max_network_attempts_per_logic_call"]),
            "retry_base_sleep": int(budget["retry_base_sleep_seconds"]),
        }
    )
    return ModelClient(model_cfg)


def require_api(client: ModelClient, approved: bool) -> None:
    if not approved:
        raise RuntimeError("真实运行需要显式增加 --approve-api。")
    if not client.is_configured():
        raise RuntimeError("当前进程未读取到 DASHSCOPE_API_KEY；未发出任何模型或 embedding 请求。")


def run_react(output: Path, cfg: Dict[str, Any], datasets: Sequence[str], runs: int, approved: bool, resume: bool) -> None:
    client = configured_model(cfg)
    require_api(client, approved)
    protocol = EXPERIMENT_DIR / "experiment_protocol.md"
    if not protocol.exists():
        raise RuntimeError("正式运行前必须先保存 experiment_protocol.md。")
    system_prompt = (EXPERIMENT_DIR / "prompts" / "react_system.txt").read_text(encoding="utf-8")
    budget = cfg["budget"]
    agent_cfg = cfg["agent"]
    limits = AgentLimits(
        max_logic_calls=int(budget["max_logic_calls_per_diagnosis"]),
        max_completion_tokens=int(budget["max_completion_tokens_per_diagnosis"]),
        max_ranking_length=int(agent_cfg["max_ranking_length"]),
        retrieval_top_k=int(agent_cfg["retrieval_top_k"]),
        max_query_chars=int(agent_cfg["max_query_chars"]),
    )
    final_path = output / "react" / "final_records.jsonl"
    score_path = output / "react" / "scoring_keys.jsonl"
    completed = {row["record_key"] for row in load_jsonl(final_path)} if resume else set()
    if final_path.exists() and not resume:
        raise RuntimeError(f"正式结果已存在：{final_path}。请使用 --resume，禁止覆盖正式记录。")

    for dataset in datasets:
        spec = cfg["datasets"][dataset]
        cases = load_normalized_cases(resolve(spec["cases"]))
        if len(cases) != int(spec["expected_episodes"]):
            raise RuntimeError(f"{dataset} episode 数不匹配，拒绝凑数运行。")
        retriever = RAGRetriever.from_config_path(str(resolve(spec["rag_config"])))
        agent = ReactAdaptedAgent(client, retriever, system_prompt, limits)
        for run_id in range(1, runs + 1):
            for index, case in enumerate(cases, start=1):
                key = record_key(dataset, "ReAct-adapted", client.model, case["case_id"], run_id)
                if key in completed:
                    print(f"[跳过] {key}", flush=True)
                    continue
                print(f"[运行] dataset={dataset} run={run_id}/{runs} episode={case['case_id']} ({index}/{len(cases)})", flush=True)
                result = agent.run(dataset, case, run_id)
                calls = result.pop("calls")
                tools = result.pop("tool_events")
                result["record_key"] = key
                result["protocol_sha256"] = sha256(protocol)
                append_jsonl(str(output / "react" / "raw_model_calls.jsonl"), {"record_key": key, "dataset": dataset, "episode_id": str(case["case_id"]), "run_id": run_id, "calls": calls})
                append_jsonl(str(output / "react" / "tool_events.jsonl"), {"record_key": key, "dataset": dataset, "episode_id": str(case["case_id"]), "run_id": run_id, "events": tools})
                append_jsonl(str(score_path), {"record_key": key, "dataset": dataset, "episode_id": str(case["case_id"]), "run_id": run_id, "gt_vars": case.get("gt_vars") or [], "top10_vars": case.get("top10_vars") or []})
                append_jsonl(str(final_path), result)
                completed.add(key)
                print(f"[完成] {key} status={result['stopping_reason']} calls={result['logic_calls']} tokens={result['total_tokens']}", flush=True)
                time.sleep(float(budget["sleep_ms_between_diagnoses"]) / 1000.0)


def main_command(cfg: Dict[str, Any], output_dir: Path, method: str, runs: int) -> List[str]:
    spec = cfg["datasets"]["wadi"]
    is_kdagent = method == "KDAgent"
    return [
        sys.executable,
        "src/main.py",
        "--data_path", str(resolve(spec["cases"])),
        "--config_path", str(PROJECT_ROOT / "configs" / "models.yaml"),
        "--model_name", cfg["model_name"],
        "--output_dir", str(output_dir),
        "--prompt_template", str(PROJECT_ROOT / "experiments" / "wadi_external_validation_v1" / "prompts" / "wadi_rca_prompt_template.txt"),
        "--knowledge_path", str(PROJECT_ROOT / "outputs" / "wadi_external_validation_v1" / "prepared_data" / "process_knowledge_template.md"),
        "--num_runs", str(runs),
        "--temperature", "0.2",
        "--max_tokens", "8192",
        "--thinking_budget", "2048",
        "--use_rag", "1",
        "--rag_config", str(resolve(spec["rag_config"])),
        "--rag_top_k", "5",
        "--use_iterative_agent", "1",
        "--use_dual_branch_fusion", "1" if is_kdagent else "0",
        "--max_iterations", "3",
        "--min_confidence", "0.5",
        "--max_retries", "5",
        "--request_timeout", "120",
        "--sleep_ms", "300",
        "--resume",
    ]


def run_wadi_comparators(output: Path, cfg: Dict[str, Any], runs: int, approved: bool) -> None:
    client = configured_model(cfg)
    require_api(client, approved)
    root = output / "comparators" / "wadi"
    kdagent_dir = root / f"kdagent_{cfg['model_name']}"
    serial_dir = root / f"serial_{cfg['model_name']}"
    copied = kdagent_dir / "raw_responses" / f"{cfg['model_name']}_run_1.jsonl"
    old = resolve(cfg["existing_results"]["wadi_kdagent_run1"]) / "raw_responses" / f"{cfg['model_name']}_run_1.jsonl"
    if not copied.exists():
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, copied)
        print(f"[复用] WADI KDAgent run 1: {old}", flush=True)
    for method, target in (("KDAgent", kdagent_dir), ("Serial", serial_dir)):
        command = main_command(cfg, target, method, runs)
        print(f"[执行] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def normalize_ids(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip().upper().replace(" ", "") for value in values if str(value).strip()]


def metrics_for(ranking: Sequence[str], truth: Sequence[str]) -> Dict[str, Any]:
    truth_set = set(normalize_ids(list(truth)))
    rank = next((index + 1 for index, value in enumerate(normalize_ids(list(ranking))[:5]) if value in truth_set), None)
    dcg = sum(1.0 / math.log2(index + 2) for index, value in enumerate(normalize_ids(list(ranking))[:5]) if value in truth_set)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(truth_set), 5)))
    return {
        "hit_at_1": int(bool(rank and rank <= 1)),
        "hit_at_3": int(bool(rank and rank <= 3)),
        "hit_at_5": int(bool(rank and rank <= 5)),
        "mrr": 1.0 / rank if rank else 0.0,
        "ndcg_at_5": dcg / ideal if ideal else 0.0,
        "best_gt_rank": rank or "",
    }


def load_case_summaries(path: Path, model: str) -> List[Dict[str, Any]]:
    raw = path / "raw_responses"
    rows = []
    for file in sorted(raw.glob(f"{model}_run_*.jsonl")) if raw.exists() else []:
        rows.extend(row for row in load_jsonl(file) if row.get("record_type") == "case_summary")
    return rows


def old_method_rows(dataset: str, method: str, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        parsed = record.get("parsed_response") or {}
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "model": str(record.get("model_name") or "deepseek-v4-pro"),
                "run_id": int(record.get("run_id") or 0),
                "episode_id": str(record.get("case_id")),
                "predicted_top5": normalize_ids(parsed.get("predicted_root_causes") or [])[:5],
                "gt_vars": normalize_ids(record.get("gt_vars") or []),
                "top10_vars": normalize_ids(record.get("top10_vars") or []),
                "completed": bool(record.get("api_ok")),
                "valid_output": bool(parsed.get("predicted_root_causes")),
                "logic_calls": int(record.get("iteration_count", 0) or 0) + (1 if method == "KDAgent" else 0),
                "prompt_tokens": int(record.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(record.get("completion_tokens", 0) or 0),
                "reasoning_tokens": int(record.get("reasoning_tokens", 0) or 0),
                "total_tokens": int(record.get("total_tokens", 0) or 0),
                "elapsed_s": float(record.get("elapsed_s", 0.0) or 0.0),
                "stopping_reason": str(record.get("agent_status") or "saved_result"),
            }
        )
    return rows


def collect_rows(output: Path, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    model = cfg["model_name"]
    rows: List[Dict[str, Any]] = []
    rows += old_method_rows("swat", "KDAgent", load_case_summaries(resolve(cfg["existing_results"]["swat_kdagent"]), model))
    rows += old_method_rows("swat", "Serial", load_case_summaries(resolve(cfg["existing_results"]["swat_serial"]), model))

    wadi_kdagent = load_case_summaries(output / "comparators" / "wadi" / f"kdagent_{model}", model)
    if not wadi_kdagent:
        wadi_kdagent = load_case_summaries(resolve(cfg["existing_results"]["wadi_kdagent_run1"]), model)
    rows += old_method_rows("wadi", "KDAgent", wadi_kdagent)
    rows += old_method_rows("wadi", "Serial", load_case_summaries(output / "comparators" / "wadi" / f"serial_{model}", model))

    scoring = {row["record_key"]: row for row in load_jsonl(output / "react" / "scoring_keys.jsonl")}
    for record in load_jsonl(output / "react" / "final_records.jsonl"):
        label = scoring.get(record["record_key"])
        if not label:
            continue
        parsed = record.get("parsed_response") or {}
        rows.append(
            {
                "dataset": record["dataset"],
                "method": "ReAct-adapted",
                "model": record["model_name"],
                "run_id": int(record["run_id"]),
                "episode_id": str(record["episode_id"]),
                "predicted_top5": normalize_ids(parsed.get("predicted_root_causes") or [])[:5],
                "gt_vars": normalize_ids(label.get("gt_vars") or []),
                "top10_vars": normalize_ids(label.get("top10_vars") or []),
                "completed": bool(record.get("completed")),
                "valid_output": bool(record.get("valid_output")),
                "logic_calls": int(record.get("logic_calls", 0) or 0),
                "prompt_tokens": int(record.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(record.get("completion_tokens", 0) or 0),
                "reasoning_tokens": int(record.get("reasoning_tokens", 0) or 0),
                "total_tokens": int(record.get("total_tokens", 0) or 0),
                "elapsed_s": float(record.get("elapsed_s", 0.0) or 0.0),
                "stopping_reason": str(record.get("stopping_reason") or ""),
            }
        )

    # Keep only one record per key; raise immediately on duplicates to forbid cherry-picking by score.
    unique: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset"], row["method"], row["episode_id"], row["run_id"])
        if key in unique:
            raise RuntimeError(f"重复 method-run-episode 记录：{key}")
        unique[key] = row
    return list(unique.values())


def exact_sign_flip(values: Sequence[float]) -> float:
    nonzero = [float(value) for value in values if abs(float(value)) > 1e-12]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero) / len(nonzero))
    extreme = 0
    total = 1 << len(nonzero)
    for mask in range(total):
        mean = sum((-value if (mask >> index) & 1 else value) for index, value in enumerate(nonzero)) / len(nonzero)
        extreme += abs(mean) >= observed - 1e-12
    return extreme / total


def bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    return tuple(float(value) for value in np.quantile(array[indices].mean(axis=1), [0.025, 0.975]))


def holm(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def interval_groups(dataset: str, cfg: Dict[str, Any]) -> List[List[str]]:
    manifest = resolve(cfg["datasets"][dataset]["audit_manifest"])
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    intervals: List[Tuple[float, float, str]] = []
    for row in source:
        if dataset == "swat":
            start = datetime.strptime(row["episode_window_start"].strip(), "%d/%m/%Y %I:%M:%S %p").timestamp()
            end = datetime.strptime(row["episode_window_end"].strip(), "%d/%m/%Y %I:%M:%S %p").timestamp()
            episode = str(row["episode_id"])
        else:
            if str(row.get("included", "")).lower() not in ("true", "1", "yes"):
                continue
            start, end = float(row["attack_row_start"]), float(row["attack_row_end"])
            episode = str(row["episode_id"])
        intervals.append((start, end, episode))
    groups: List[List[str]] = []
    current: List[str] = []
    current_end = float("-inf")
    for start, end, episode in sorted(intervals):
        if current and start > current_end:
            groups.append(current)
            current = []
        current.append(episode)
        current_end = max(current_end, end)
    if current:
        groups.append(current)
    return groups


def score(output: Path, cfg: Dict[str, Any]) -> None:
    base_rows = collect_rows(output, cfg)
    records: List[Dict[str, Any]] = []
    for row in base_rows:
        metric = metrics_for(row["predicted_top5"], row["gt_vars"])
        records.append(
            {
                **row,
                "predicted_top5": ";".join(row["predicted_top5"]),
                "gt_vars": ";".join(row["gt_vars"]),
                "top10_vars": ";".join(row["top10_vars"]),
                "reachable": int(bool(set(row["gt_vars"]) & set(row["top10_vars"]))),
                **metric,
            }
        )
    write_csv(output / "record_level_results.csv", records)

    summaries = []
    for (dataset, method), group_iter in itertools.groupby(sorted(records, key=lambda row: (row["dataset"], row["method"])), key=lambda row: (row["dataset"], row["method"])):
        group = list(group_iter)
        for scope, scoped in (("full", group), ("reachable", [row for row in group if row["reachable"]])):
            expected_episodes = int(cfg["datasets"][dataset]["expected_episodes"])
            expected_records = expected_episodes * int(cfg["num_runs"]) if scope == "full" else ""
            summaries.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "scope": scope,
                    "records": len(scoped),
                    "episodes": len({row["episode_id"] for row in scoped}),
                    "expected_full_records": expected_records,
                    "hit_at_1": np.mean([row["hit_at_1"] for row in scoped]) if scoped else "",
                    "hit_at_3": np.mean([row["hit_at_3"] for row in scoped]) if scoped else "",
                    "hit_at_5": np.mean([row["hit_at_5"] for row in scoped]) if scoped else "",
                    "mrr": np.mean([row["mrr"] for row in scoped]) if scoped else "",
                    "ndcg_at_5": np.mean([row["ndcg_at_5"] for row in scoped]) if scoped else "",
                    "completion_rate": np.mean([int(row["completed"]) for row in scoped]) if scoped else "",
                    "valid_output_rate": np.mean([int(row["valid_output"]) for row in scoped]) if scoped else "",
                }
            )
    write_csv(output / "summary_metrics.csv", summaries)
    write_csv(output / "paper_ready_table.csv", [{key: row[key] for key in ("dataset", "method", "scope", "records", "episodes", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5", "completion_rate", "valid_output_rate")} for row in summaries])

    comparisons = []
    labels = [("swat", "KDAgent"), ("swat", "Serial"), ("wadi", "KDAgent"), ("wadi", "Serial")]
    by_key = {(row["dataset"], row["method"], row["episode_id"], row["run_id"]): row for row in records}
    for position, (dataset, comparator) in enumerate(labels):
        episodes = sorted({row["episode_id"] for row in records if row["dataset"] == dataset})
        differences = []
        matched_records = 0
        for episode in episodes:
            per_run = []
            for run_id in range(1, int(cfg["num_runs"]) + 1):
                left = by_key.get((dataset, comparator, episode, run_id))
                right = by_key.get((dataset, "ReAct-adapted", episode, run_id))
                if left and right:
                    per_run.append(float(left["hit_at_5"]) - float(right["hit_at_5"]))
                    matched_records += 1
            if len(per_run) == int(cfg["num_runs"]):
                differences.append(sum(per_run) / len(per_run))
        complete = len(differences) == int(cfg["datasets"][dataset]["expected_episodes"])
        low, high = bootstrap_ci(differences, int(cfg["bootstrap_samples"]), int(cfg["seed"]) + position) if differences else (float("nan"), float("nan"))
        comparisons.append(
            {
                "dataset": dataset,
                "comparison": f"{comparator} - ReAct-adapted",
                "metric": "Hit@5",
                "matched_records": matched_records,
                "matched_episodes": len(differences),
                "complete": complete,
                "mean_difference": np.mean(differences) if differences else "",
                "ci95_low": low if differences else "",
                "ci95_high": high if differences else "",
                "raw_p": exact_sign_flip(differences) if differences else "",
                "holm_p": "",
                "family_complete": False,
            }
        )
    family_complete = all(row["complete"] for row in comparisons)
    if family_complete:
        adjusted = holm([float(row["raw_p"]) for row in comparisons])
        for row, value in zip(comparisons, adjusted):
            row["holm_p"] = value
            row["family_complete"] = True
    write_csv(output / "paired_comparisons.csv", comparisons)

    paired_episode_rows = []
    metric_names = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5")
    for dataset, comparator in labels:
        episodes = sorted({row["episode_id"] for row in records if row["dataset"] == dataset})
        for episode in episodes:
            left = [row for row in records if row["dataset"] == dataset and row["method"] == comparator and row["episode_id"] == episode]
            right = [row for row in records if row["dataset"] == dataset and row["method"] == "ReAct-adapted" and row["episode_id"] == episode]
            left_by_run = {int(row["run_id"]): row for row in left}
            right_by_run = {int(row["run_id"]): row for row in right}
            matched_runs = sorted(set(left_by_run) & set(right_by_run))
            row: Dict[str, Any] = {
                "dataset": dataset,
                "comparison": f"{comparator} - ReAct-adapted",
                "episode_id": episode,
                "matched_runs": len(matched_runs),
                "complete_three_runs": len(matched_runs) == int(cfg["num_runs"]),
            }
            for metric in metric_names:
                left_mean = np.mean([float(left_by_run[run][metric]) for run in matched_runs]) if matched_runs else ""
                right_mean = np.mean([float(right_by_run[run][metric]) for run in matched_runs]) if matched_runs else ""
                row[f"comparator_{metric}"] = left_mean
                row[f"react_{metric}"] = right_mean
                row[f"difference_{metric}"] = float(left_mean) - float(right_mean) if matched_runs else ""
            paired_episode_rows.append(row)
    write_csv(output / "paired_episode_differences.csv", paired_episode_rows)

    budget_rows = []
    for row in records:
        budget_rows.append({key: row[key] for key in ("dataset", "method", "model", "episode_id", "run_id", "completed", "valid_output", "logic_calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "elapsed_s", "stopping_reason")})
    write_csv(output / "budget_and_failure_audit.csv", budget_rows)

    overlap_rows = []
    for dataset in ("swat", "wadi"):
        groups = interval_groups(dataset, cfg)
        for method in ("KDAgent", "Serial", "ReAct-adapted"):
            selected = [row for row in records if row["dataset"] == dataset and row["method"] == method]
            episode_means = defaultdict(list)
            for row in selected:
                episode_means[row["episode_id"]].append(float(row["hit_at_5"]))
            component_means = []
            for component in groups:
                values = [np.mean(episode_means[episode]) for episode in component if episode in episode_means]
                if values:
                    component_means.append(float(np.mean(values)))
            overlap_rows.append({"dataset": dataset, "method": method, "overlap_components": len(groups), "observed_components": len(component_means), "component_balanced_hit_at_5": np.mean(component_means) if component_means else "", "groups": json.dumps(groups, ensure_ascii=False)})
    write_csv(output / "overlap_group_sensitivity.csv", overlap_rows)
    write_result_summary(output, cfg, summaries, comparisons, records)


def write_result_summary(output: Path, cfg: Dict[str, Any], summaries: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> None:
    expected = 99
    react = [row for row in records if row["method"] == "ReAct-adapted"]
    complete = len(react) == expected
    summary_lookup = {(row["dataset"], row["method"], row["scope"]): row for row in summaries}
    lines = [
        "# ReAct-adapted 补充实验结果说明",
        "",
        "## 完成状态",
        f"- ReAct-adapted 正式记录：{len(react)}/{expected}；{'已完成全部目标' if complete else '尚未完成全部目标'}。",
        f"- 当前进程 API Key：{'可用' if os.environ.get('DASHSCOPE_API_KEY') else '不可用'}。模拟测试不计入论文结果。",
        "",
        "## 方法边界",
        "- ReAct-adapted 是单 Agent 的自主工具循环：模型逐步决定读取证据、检索知识或提交答案。",
        "- Serial 在生成前固定完成检索并把知识放入同一迭代上下文；Only Agent 只有证据推理和验证修复，二者都没有自主工具选择。",
        "- KDAgent 独立生成证据与检索两个分支，并用来源权限规则融合；ReAct-adapted 不分支，也不固定证据首位。",
        "",
        "## 实测结论",
    ]
    if complete:
        for dataset in ("swat", "wadi"):
            for method in ("KDAgent", "Serial", "ReAct-adapted"):
                row = summary_lookup.get((dataset, method, "full"))
                if row:
                    lines.append(f"- {dataset.upper()} {method}: Hit@1={float(row['hit_at_1']):.3f}, Hit@3={float(row['hit_at_3']):.3f}, Hit@5={float(row['hit_at_5']):.3f}, MRR={float(row['mrr']):.3f}, NDCG@5={float(row['ndcg_at_5']):.3f}。")
    else:
        lines.append("- 正式 ReAct 记录不完整，因此不能声称 KDAgent 优于或不优于该新对照，也不能解释统计显著性。")
    lines.extend(
        [
            "",
            "## 统计边界",
            f"- 四项预定 Hit@5 比较族：{'完整，已统一 Holm 校正' if all(row['family_complete'] for row in comparisons) else '不完整，Holm 校正值留空'}。",
            "- 该实验回答的是相同候选、证据和知识条件下，双分支来源控制与单 Agent 自主工具组织的差异；它不等于官方外部 RCA 系统复现，也不能证明对所有数据集、模型或故障类型全面领先。",
            "",
            "## 8 页论文最小加入建议",
            "在实验设置中用 2-3 句定义 ReAct-adapted 和共同预算，在主结果表增加一行，并用 2-3 句报告两数据集 Hit@5 配对差值与四项 Holm 结果。工具轨迹、完整指标和失败审计放补充材料。",
        ]
    )
    (output / "results_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    english = [
        "# Paper-ready paragraph draft",
        "",
        "Suitable placement: Experimental Evaluation / Strong Baselines.",
        "",
        "We added ReAct-adapted, a task-specific single-agent baseline that alternates between model-selected evidence inspection, domain-knowledge retrieval, and ranking submission. Unlike KDAgent, it neither generates independent evidence and retrieval branches nor reserves Rank 1 for an admissible evidence primary. All methods used the same frozen candidate panels and dataset-specific knowledge sources under a common per-call and per-diagnosis generation envelope. See paper_ready_table.csv and paired_comparisons.csv for the verified numerical results; no superiority or significance claim should be inserted until all 99 ReAct-adapted records and the four pre-specified paired comparisons are complete.",
    ]
    (output / "paper_ready_paragraph_en.md").write_text("\n".join(english) + "\n", encoding="utf-8")


def export_results_txt(output: Path) -> Path:
    """Consolidate this round's results and audit information into a single UTF-8 text file for external plotting/writing tools."""
    result_file = output / "paper_ready_results.txt"

    def read_csv(name: str) -> List[Dict[str, str]]:
        path = output / name
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    summary = read_csv("summary_metrics.csv")
    comparison = read_csv("paired_comparisons.csv")
    episode_diffs = read_csv("paired_episode_differences.csv")
    records = read_csv("record_level_results.csv")
    budget = read_csv("budget_and_failure_audit.csv")
    availability = json.loads((output / "data_availability_report.json").read_text(encoding="utf-8"))
    formal_status = json.loads((output / "formal_run_status.json").read_text(encoding="utf-8")) if (output / "formal_run_status.json").exists() else {}

    lines = [
        "KDAgent ReAct-adapted 补充实验结果汇总",
        "=" * 80,
        "用途：将本轮正式实验的统计结果、逐记录结果和资源审计集中保存，供论文写作或绘图使用。",
        "本文件不包含模型原始长响应；原始响应和工具轨迹保存在 outputs/react_adapted_v1/react/。",
        "",
        "一、正式状态",
        "- formal_status: " + json.dumps(formal_status, ensure_ascii=False),
        "- ReAct-adapted 目标记录：99（SWaT 20 episodes x 3 runs；WADI 13 episodes x 3 runs）",
        f"- ReAct-adapted 实际记录：{sum(1 for row in records if row.get('method') == 'ReAct-adapted')}",
        f"- 全部 record-level 行数（KDAgent/Serial/ReAct-adapted）：{len(records)}",
        f"- SWaT 数据核查：{availability['datasets']['swat']['actual_episodes']} episodes，{availability['datasets']['swat']['reachable_episodes']} reachable，KB {availability['datasets']['swat']['knowledge_chunk_count']} chunks",
        f"- WADI 数据核查：{availability['datasets']['wadi']['actual_episodes']} episodes，{availability['datasets']['wadi']['reachable_episodes']} reachable，KB {availability['datasets']['wadi']['knowledge_chunk_count']} chunks",
        "- 指标分母：同一 episode 的 3 runs 是重复观测，不是 3 个独立事件；配对统计先在 episode 内聚合。",
        "",
        "二、汇总指标",
        "字段：dataset | method | scope | records | episodes | expected_full_records | Hit@1 | Hit@3 | Hit@5 | MRR | NDCG@5 | completion_rate | valid_output_rate",
    ]
    for row in summary:
        lines.append(" | ".join(row.get(field, "") for field in ("dataset", "method", "scope", "records", "episodes", "expected_full_records", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5", "completion_rate", "valid_output_rate")))

    lines.extend(["", "三、四项预设 Hit@5 配对比较（comparator - ReAct-adapted）", "字段：dataset | comparison | matched_records | matched_episodes | complete | mean_difference | CI95_low | CI95_high | raw_p | Holm_p | family_complete"])
    for row in comparison:
        lines.append(" | ".join(row.get(field, "") for field in ("dataset", "comparison", "matched_records", "matched_episodes", "complete", "mean_difference", "ci95_low", "ci95_high", "raw_p", "holm_p", "family_complete")))

    lines.extend(["", "四、逐 episode 配对差异", "字段：dataset | comparison | episode_id | matched_runs | complete_three_runs | comparator/react/difference for Hit@1, Hit@3, Hit@5, MRR, NDCG@5"])
    episode_fields = ["dataset", "comparison", "episode_id", "matched_runs", "complete_three_runs"]
    for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"):
        episode_fields.extend([f"comparator_{metric}", f"react_{metric}", f"difference_{metric}"])
    for row in episode_diffs:
        lines.append(" | ".join(row.get(field, "") for field in episode_fields))

    lines.extend(["", "五、完整 record-level 结果", "字段：dataset | method | model | run_id | episode_id | predicted_top5 | gt_vars | top10_vars | reachable | hit_at_1 | hit_at_3 | hit_at_5 | mrr | ndcg_at_5 | best_gt_rank | completed | valid_output | logic_calls | prompt_tokens | completion_tokens | reasoning_tokens | total_tokens | elapsed_s | stopping_reason"])
    record_fields = ["dataset", "method", "model", "run_id", "episode_id", "predicted_top5", "gt_vars", "top10_vars", "reachable", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5", "best_gt_rank", "completed", "valid_output", "logic_calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "elapsed_s", "stopping_reason"]
    for row in records:
        lines.append(" | ".join(row.get(field, "") for field in record_fields))

    lines.extend(["", "六、预算与失败审计", "字段：dataset | method | model | episode_id | run_id | completed | valid_output | logic_calls | prompt_tokens | completion_tokens | reasoning_tokens | total_tokens | elapsed_s | stopping_reason"])
    budget_fields = ["dataset", "method", "model", "episode_id", "run_id", "completed", "valid_output", "logic_calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "elapsed_s", "stopping_reason"]
    for row in budget:
        lines.append(" | ".join(row.get(field, "") for field in budget_fields))

    lines.extend([
        "",
        "七、文件索引",
        "- record_level_results.csv：完整逐记录评分结果。",
        "- summary_metrics.csv：full/reachable 汇总指标。",
        "- paired_comparisons.csv：四项预设 Hit@5 比较、95% bootstrap CI、exact sign-flip 和 Holm。",
        "- paired_episode_differences.csv：以 episode 为单位的配对差异。",
        "- budget_and_failure_audit.csv：调用次数、tokens、耗时、失败和停止原因。",
        "- react/raw_model_calls.jsonl：模型原始响应与 provider token 记录。",
        "- react/tool_events.jsonl：工具调用、参数、返回 observation 和工具顺序。",
        "",
        "八、口径说明",
        "- Hit@k：前 k 个预测中任意一个 ground-truth variable 命中即为 1。",
        "- MRR：最早 ground-truth rank 的倒数；未命中为 0。",
        "- NDCG@5：仅使用前 5 位，多标签 ground truth 按理想排序归一化。",
        "- ReAct-adapted 是任务适配版工具型 Agent，不是 RCAgent 官方完整复现。",
    ])
    result_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[导出] {result_file}（{result_file.stat().st_size} bytes）", flush=True)
    return result_file


def write_manifest(output: Path, cfg: Dict[str, Any]) -> None:
    tracked = [CONFIG_PATH, EXPERIMENT_DIR / "experiment_protocol.md", EXPERIMENT_DIR / "react_agent.py", EXPERIMENT_DIR / "run_experiment.py", EXPERIMENT_DIR / "prompts" / "react_system.txt"]
    write_json(
        output / "reproducibility_manifest.json",
        {
            "generated_at": datetime.now().isoformat(),
            "git_repository": False,
            "model_config": {
                "config_entry": cfg["model_name"],
                "provider": "openai_compatible",
                "api_model_identifier": "deepseek-v4-pro",
                "backend_snapshot_or_revision": "not exposed by the saved provider responses",
            },
            "reference": {"repository": "https://github.com/ysymyth/ReAct", "branch": "master", "observed_commit": "6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9", "claim": "task adaptation, not official industrial RCA reproduction"},
            "file_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in tracked},
        },
    )


def package(output: Path) -> Path:
    archive = output / "react_adapted_v1_delivery.zip"
    include_roots = [EXPERIMENT_DIR, output]
    excluded_parts = {"__pycache__", "chroma_db", ".venv", "venv"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for root in include_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path == archive or excluded_parts.intersection(path.parts):
                    continue
                if path.suffix.lower() in {".key", ".pem", ".env"}:
                    continue
                arc_root = "implementation" if root == EXPERIMENT_DIR else "results"
                zipped.write(path, Path(arc_root) / path.relative_to(root))
    print(f"[打包] {archive}", flush=True)
    return archive


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    cfg = load_config()
    if args.runs is not None:
        cfg["num_runs"] = args.runs
    runs = int(cfg["num_runs"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, cfg)
    status_path = output / "formal_run_status.json"
    if args.mode == "formal":
        write_json(status_path, {"status": "started", "started_at": datetime.now().isoformat(), "api_key_configured": bool(os.environ.get("DASHSCOPE_API_KEY"))})

    if args.mode in ("audit", "all-offline", "formal"):
        audit = data_audit(output, cfg)
        if audit["status"] != "PASS":
            raise RuntimeError("数据可用性核查失败，正式运行已阻止。")
    if args.mode in ("synth-test", "all-offline", "formal"):
        synthetic_test()
    if args.mode in ("dry-run", "all-offline", "formal"):
        dry_run(output, cfg, runs)
    if args.mode in ("run-react", "formal"):
        if args.mode == "formal" and not os.environ.get("DASHSCOPE_API_KEY"):
            write_json(status_path, {"status": "blocked_before_api_call", "time": datetime.now().isoformat(), "reason": "DASHSCOPE_API_KEY is not visible to the current process", "api_requests_sent": 0})
        run_react(output, cfg, args.datasets, runs, args.approve_api, args.resume)
    if args.mode in ("run-wadi-comparators", "formal"):
        run_wadi_comparators(output, cfg, runs, args.approve_api)
    if args.mode in ("score", "all-offline", "formal"):
        score(output, cfg)
    if args.mode in ("export-txt", "all-offline", "formal"):
        export_results_txt(output)
    if args.mode in ("package", "all-offline", "formal"):
        package(output)
    if args.mode == "formal":
        write_json(status_path, {"status": "completed", "completed_at": datetime.now().isoformat(), "api_key_configured": True})


if __name__ == "__main__":
    main()
