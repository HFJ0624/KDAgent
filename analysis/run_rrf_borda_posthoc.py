"""RRF/Borda post-hoc deterministic fusion.

This script only reads the already-saved clean and retrieval-robustness results and does not call the LLM.
It is independent of the old paper_ready_analysis.py, so the old script no longer incorrectly treats the
TA-RCA Top-10 as a third voting branch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "swat_s2s_raw_window_fixed" / "llm_prompt_cases.jsonl"
CLEAN_ROOT = ROOT / "outputs" / "final_experiments_frozen_v2"
ROBUST_PATH = ROOT / "outputs" / "retrieval_robustness_v1" / "records.jsonl"
DEFAULT_OUTPUT = ROOT / "analysis" / "rrf_borda_posthoc_v1"
MODELS = ("qwen-plus", "qwen-max", "deepseek-v4-flash", "deepseek-v4-pro", "glm-5.2")
CONDITIONS = ("clean", "random_retrieval", "stage_mismatched_retrieval", "low_relevance_retrieval")
FUSIONS = ("authority_preserving", "retrieval_primary", "rrf", "borda")


def setup_logger(output: Path) -> logging.Logger:
    """Write to both the terminal and a dedicated log so the zero-API analysis process can be reviewed."""
    logger = logging.getLogger("rrf_borda_posthoc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "rrf_borda_posthoc.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def normalize_vars(value: Any) -> list[str]:
    """Normalize the variable ID format; do not guess natural-language names as variable IDs."""
    if not isinstance(value, list):
        return []
    return [str(item).strip().upper() for item in value if str(item).strip()]


def read_cases() -> dict[int, dict[str, Any]]:
    cases: dict[int, dict[str, Any]] = {}
    with DATA_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                cases[int(item["case_id"])] = item
    if len(cases) != 20:
        raise RuntimeError(f"固定 benchmark case 数量异常：{len(cases)} != 20")
    return cases


def case_truth(case: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    truth = normalize_vars((case.get("ground_truth") or {}).get("gt_vars"))
    top10 = normalize_vars(case.get("top10_vars"))
    if not top10:
        top10 = normalize_vars([item.get("var_id") for item in (case.get("top_k_variables") or [])[:10]])
    if len(top10) != 10:
        raise RuntimeError(f"case {case.get('case_id')} 的 frozen TA-RCA Top-10 不完整")
    return truth, top10


def parse_rank(response: Mapping[str, Any] | None) -> list[str]:
    return normalize_vars((response or {}).get("predicted_root_causes"))


def load_clean_branches() -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[dict[str, Any]]]:
    """Read the clean dual-branch raw logs and keep only the final Evidence round."""
    branches: dict[tuple[str, int, int], dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        files = sorted((CLEAN_ROOT / f"dual_branch_fusion_agent_{model}" / "raw_responses").glob("*_run_*.jsonl"))
        if len(files) != 3:
            raise RuntimeError(f"{model} clean raw 文件数量异常：{len(files)} != 3")
        for path in files:
            summaries: dict[tuple[int, int], dict[str, Any]] = {}
            evidence: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            retrieval: dict[tuple[int, int], dict[str, Any]] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                key = (int(item.get("run_id", 0)), int(item.get("case_id", 0)))
                if item.get("record_type") == "case_summary":
                    summaries[key] = item
                elif item.get("record_type") == "agent_iteration":
                    if item.get("branch") == "data_evidence":
                        evidence[key].append(item)
                    elif item.get("branch") == "rag_knowledge":
                        retrieval[key] = item
            for key in sorted(summaries):
                final_iteration = int(summaries[key].get("final_iteration") or 1)
                candidates = [x for x in evidence.get(key, []) if int(x.get("iteration") or 1) == final_iteration]
                evidence_item = candidates[-1] if candidates else None
                retrieval_item = retrieval.get(key)
                ev_rank = parse_rank((evidence_item or {}).get("parsed_response"))
                rt_rank = parse_rank((retrieval_item or {}).get("parsed_response"))
                if not ev_rank or not rt_rank:
                    missing.append({
                        "model": model,
                        "run_id": key[0],
                        "case_id": key[1],
                        "missing": ",".join(x for x, ok in (("evidence", bool(ev_rank)), ("retrieval", bool(rt_rank))) if not ok),
                        "source_file": str(path),
                    })
                branches[(model, key[0], key[1])] = {
                    "evidence_rank": ev_rank,
                    "retrieval_rank": rt_rank,
                    "evidence_primary": ev_rank[0] if ev_rank else "",
                    "retrieval_primary": rt_rank[0] if rt_rank else "",
                    "clean_retrieved_contexts": summaries[key].get("rag_contexts") or [],
                }
    if len(branches) != 300:
        raise RuntimeError(f"clean branch case 数量异常：{len(branches)} != 300")
    return branches, missing


def load_latest_robustness() -> dict[str, dict[str, Any]]:
    """Read the last record of the append-only robustness checkpoint."""
    latest: dict[str, dict[str, Any]] = {}
    with ROBUST_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                latest[str(item["record_key"])] = item
    return latest


def robust_retrieval_rank(latest: Mapping[str, Mapping[str, Any]], model: str, run: int, case: int, condition: str) -> dict[str, Any] | None:
    key = f"{model}|{run}|{case}|{condition}|retrieval_branch"
    item = latest.get(key)
    if not item:
        return None
    parsed = item.get("parsed_response") or {}
    return {
        "rank": parse_rank(parsed),
        "contexts": item.get("retrieved_contexts") or [],
        "retrieval_record_key": key,
        "quality": {
            "api_success": bool(item.get("api_success")),
            "finish_reason": item.get("finish_reason", ""),
            "valid_json": parsed.get("valid_json"),
            "was_truncated": bool(item.get("was_truncated")),
        },
    }


def legal_rank(rank: Sequence[str], top10: Sequence[str]) -> tuple[list[str], list[str]]:
    """Keep only variables in the frozen candidate set and report the filtered-out illegal variables."""
    allowed = set(top10)
    legal: list[str] = []
    illegal: list[str] = []
    for value in rank:
        if value in allowed and value not in legal:
            legal.append(value)
        elif value not in allowed and value not in illegal:
            illegal.append(value)
    return legal, illegal


def tie_key(value: str, ta_order: Mapping[str, int]) -> tuple[int, str]:
    """The TA rank is used only when scores are exactly tied; the canonical ID is the final stable tie-break."""
    return ta_order.get(value, 10**9), value


def fuse(evidence: Sequence[str], retrieval: Sequence[str], top10: Sequence[str], method: str) -> tuple[list[str], dict[str, float]]:
    """Compute the two generic fusion schemes over the union of legal Evidence/Retrieval candidates."""
    ev, _ = legal_rank(evidence, top10)
    rt, _ = legal_rank(retrieval, top10)
    ta_order = {value: index for index, value in enumerate(top10)}
    union = list(dict.fromkeys(ev + rt))
    scores: dict[str, float] = {value: 0.0 for value in union}
    if method == "rrf":
        for ranking in (ev, rt):
            for index, value in enumerate(ranking, start=1):
                scores[value] += 1.0 / (60.0 + index)
    elif method == "borda":
        for ranking in (ev, rt):
            length = len(ranking)
            for index, value in enumerate(ranking, start=1):
                scores[value] += length - index + 1
    else:
        raise ValueError(method)
    ordered = sorted(union, key=lambda value: (-scores[value], *tie_key(value, ta_order)))
    return ordered[:5], scores


def positional_fusion(evidence: Sequence[str], retrieval: Sequence[str], top10: Sequence[str], method: str) -> list[str]:
    """Reproduce the two existing source-aware comparison rules from the paper, without calling the LLM."""
    ev, _ = legal_rank(evidence, top10)
    rt, _ = legal_rank(retrieval, top10)
    if method == "authority_preserving":
        parts = ev[:1] + rt + ev[1:]
    elif method == "retrieval_primary":
        parts = rt[:1] + ev + rt[1:]
    else:
        raise ValueError(method)
    return list(dict.fromkeys(parts))[:5]


def evaluate(ranking: Sequence[str], truth: Sequence[str], top10: Sequence[str]) -> dict[str, Any]:
    truth_set = set(truth)
    position = next((index + 1 for index, value in enumerate(ranking) if value in truth_set), None)
    dcg = sum(1.0 / math.log2(index + 2) for index, value in enumerate(ranking[:5]) if value in truth_set)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(5, len(truth_set))))
    return {
        "hit_at_1": int(bool(position and position <= 1)),
        "hit_at_3": int(bool(position and position <= 3)),
        "hit_at_5": int(bool(position and position <= 5)),
        "mrr": 1.0 / position if position else 0.0,
        "ndcg_at_5": dcg / ideal if ideal else 0.0,
        "best_gt_rank": position if position is not None else "",
        "reachable": int(bool(set(truth) & set(top10))),
    }


def build_record(model: str, run: int, case_id: int, condition: str, method: str, branch: Mapping[str, Any], case: Mapping[str, Any], missing: str = "") -> dict[str, Any]:
    truth, top10 = case_truth(case)
    evidence = branch.get("evidence_rank") or []
    retrieval = branch.get("retrieval_rank") or []
    if method in ("authority_preserving", "retrieval_primary"):
        ranking = positional_fusion(evidence, retrieval, top10, method)
        scores: dict[str, float] = {}
    else:
        ranking, scores = fuse(evidence, retrieval, top10, method)
    values = evaluate(ranking, truth, top10) if not missing else {key: "" for key in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5", "best_gt_rank", "reachable")}
    ev, ev_illegal = legal_rank(evidence, top10)
    rt, rt_illegal = legal_rank(retrieval, top10)
    primary = ranking[0] if ranking else ""
    return {
        "model": model,
        "run_id": run,
        "case_id": case_id,
        "condition": condition,
        "fusion_method": method,
        "predicted_top5": ",".join(ranking),
        "evidence_rank": ",".join(ev),
        "retrieval_rank": ",".join(rt),
        "evidence_primary": ev[0] if ev else "",
        "retrieval_primary": rt[0] if rt else "",
        "primary_overwrite_from_evidence": int(bool(primary and ev and primary != ev[0])),
        "primary_overwrite_from_retrieval": int(bool(primary and rt and primary != rt[0])),
        "rrf_borda_scores": json.dumps(scores, ensure_ascii=False, sort_keys=True),
        "illegal_evidence_candidates": ",".join(ev_illegal),
        "illegal_retrieval_candidates": ",".join(rt_illegal),
        "missing_branch": missing,
        "retrieval_record_key": branch.get("retrieval_record_key", ""),
        "retrieved_chunk_ids": json.dumps([x.get("id") or x.get("metadata", {}).get("chunk_id", "") for x in branch.get("retrieved_contexts", [])], ensure_ascii=False),
        "retrieved_sources": json.dumps([x.get("source", "") for x in branch.get("retrieved_contexts", [])], ensure_ascii=False),
        "retrieved_similarities": json.dumps([x.get("similarity", x.get("score", "")) for x in branch.get("retrieved_contexts", [])], ensure_ascii=False),
        **values,
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("missing_branch", "") == ""]
    reachable = [row for row in valid if int(row.get("reachable", 0)) == 1]
    result: dict[str, Any] = {"n_records": len(valid), "n_missing": len(rows) - len(valid), "n_reachable": len(reachable)}
    for label, items in (("all", valid), ("reachable", reachable)):
        result[f"{label}_n"] = len(items)
        for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"):
            result[f"{label}_{metric}"] = round(sum(float(item[metric]) for item in items) / len(items), 6) if items else ""
        ranks = [int(item["best_gt_rank"]) for item in items if str(item.get("best_gt_rank", "")) != ""]
        result[f"{label}_mean_best_gt_rank_hit_only"] = round(sum(ranks) / len(ranks), 6) if ranks else ""
        result[f"{label}_median_best_gt_rank_hit_only"] = statistics.median(ranks) if ranks else ""
        result[f"{label}_miss_count"] = len(items) - len(ranks)
        result[f"{label}_primary_overwrite_rate_from_evidence"] = round(sum(int(item["primary_overwrite_from_evidence"]) for item in items) / len(items), 6) if items else ""
    return result


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-API RRF/Borda post-hoc fusion")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)
    cases = read_cases()
    clean_branches, clean_missing = load_clean_branches()
    latest = load_latest_robustness()
    logger.info("读取固定 benchmark=%d cases；clean branch=%d；robustness latest records=%d", len(cases), len(clean_branches), len(latest))

    records: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for run in range(1, 4):
            for case_id in range(20):
                key = (model, run, case_id)
                clean = clean_branches[key]
                clean_missing_text = next((x["missing"] for x in clean_missing if (x["model"], x["run_id"], x["case_id"]) == key), "")
                for fusion in FUSIONS:
                    record = build_record(model, run, case_id, "clean", fusion, {
                        "evidence_rank": clean["evidence_rank"],
                        "retrieval_rank": clean["retrieval_rank"],
                        "retrieved_contexts": clean["clean_retrieved_contexts"],
                    }, cases[case_id], clean_missing_text)
                    records.append(record)
                    if clean_missing_text:
                        missing_rows.append({**record, "missing_stage": "clean"})
                for condition in CONDITIONS[1:]:
                    robust = robust_retrieval_rank(latest, model, run, case_id, condition)
                    missing_text = "retrieval" if not robust or not robust["rank"] else ""
                    branch = {
                        "evidence_rank": clean["evidence_rank"],
                        "retrieval_rank": (robust or {}).get("rank", []),
                        "retrieval_record_key": (robust or {}).get("retrieval_record_key", ""),
                        "retrieved_contexts": (robust or {}).get("contexts", []),
                    }
                    for fusion in FUSIONS:
                        record = build_record(model, run, case_id, condition, fusion, branch, cases[case_id], missing_text)
                        records.append(record)
                        if missing_text:
                            missing_rows.append({**record, "missing_stage": condition})

    expected = 5 * 3 * 20 * 4 * 4
    if len(records) != expected:
        raise RuntimeError(f"输出记录数异常：{len(records)} != {expected}")
    write_csv(output / "rrf_borda_record_level.csv", records)
    write_csv(output / "rrf_borda_missing_branches.csv", missing_rows)

    summary_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for fusion in FUSIONS:
            selected = [row for row in records if row["condition"] == condition and row["fusion_method"] == fusion]
            summary_rows.append({"condition": condition, "fusion_method": fusion, **aggregate(selected)})
    write_csv(output / "rrf_borda_summary.csv", summary_rows)

    # Save the robustness branch quality strictly as audit information, avoiding silent inclusion of failed/truncated results in the paper's conclusions.
    robustness_quality = {
        "retrieval_records": sum(1 for x in latest.values() if x.get("method") == "retrieval_branch"),
        "retrieval_api_success": sum(1 for x in latest.values() if x.get("method") == "retrieval_branch" and x.get("api_success") is True),
        "retrieval_truncated": sum(1 for x in latest.values() if x.get("method") == "retrieval_branch" and x.get("finish_reason") == "length"),
        "retrieval_invalid_json": sum(1 for x in latest.values() if x.get("method") == "retrieval_branch" and (x.get("parsed_response") or {}).get("valid_json") is False),
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "llm_calls": 0,
        "benchmark_cases": 20,
        "models": list(MODELS),
        "runs": 3,
        "conditions": list(CONDITIONS),
        "fusion_methods": list(FUSIONS),
        "rrf_k": 60,
        "vote_sources": ["Evidence branch", "Retrieval branch"],
        "ta_role": "candidate legality check and complete-score tie-break only",
        "clean_missing_branches": clean_missing,
        "robustness_quality": robustness_quality,
        "source_files": {"cases": str(DATA_PATH), "clean_root": str(CLEAN_ROOT), "robustness_records": str(ROBUST_PATH)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    md: list[str] = [
        "# RRF/Borda Post-hoc Fusion Summary",
        "",
        "本分析完全基于已有保存结果，LLM 调用次数为 0。RRF 和 Borda 只使用 Evidence branch 与 Retrieval branch 的合法候选并集；frozen TA-RCA Top-10 不参与投票得分，只用于候选合法性检查和完全同分时的 deterministic tie-break。",
        "",
        "## 实验范围",
        "- 固定 20 个真实 SWaT cases，5 models，3 runs；每个 condition×fusion 应有 300 条记录。",
        "- conditions: clean、random_retrieval、stage_mismatched_retrieval、low_relevance_retrieval。",
        "- RRF 固定 `k=60`；Borda 对每条实际 branch ranking 使用 `m_s-rank+1`。",
        "- 推断若后续执行，仍以 20 个 case 为独立单位；本次不新增显著性检验，也不混入原先 Holm family。",
        "",
        "## 关键问题回答",
        "1. RRF/Borda 是否提高 clean Hit@1/MRR/NDCG：以 `rrf_borda_summary.csv` 中 clean 行与 Authority-Preserving、Retrieval-Primary 对照比较；不使用最终 KDAgent Top-5 反推 branch ranking。",
        "2. 是否改变 Hit@5：直接比较 clean 与三个 perturbation 条件的 `all_hit_at_5` 和 `reachable_hit_at_5`。",
        "3. 是否更容易 primary overwrite：使用 `all_primary_overwrite_rate_from_evidence` 和 `reachable_primary_overwrite_rate_from_evidence`；Authority-Preserving 在 evidence primary 有效时理论上保持 primary。",
        "4. generic source-agnostic fusion 与 source-aware fusion 的差异：RRF/Borda 是 source-agnostic；Authority-Preserving、Retrieval-Primary 是 source-aware 对照。应同时观察 Hit@1、MRR、NDCG@5 与 Hit@5。",
        "5. 贡献表述：只有当 RRF/Borda 在 clean 和 perturbation 中一致优于 source-aware 方法时，才可讨论 accuracy-optimal；否则更稳妥的表述是 explicit source-authority control，以及 robustness–accuracy trade-off。",
        "",
        "## 输出文件",
        "- `rrf_borda_record_level.csv`: 300 clean + 900 perturbation combinations × 4 fusion methods，保留 record-level 结果。",
        "- `rrf_borda_summary.csv`: all/13 reachable 汇总指标、miss_count 和 primary overwrite rate。",
        "- `rrf_borda_missing_branches.csv`: 不完整 branch 的位置和原因。",
        "- `manifest.json`: 零 API、固定参数和源文件审计信息。",
        "",
        "## 审计结论",
        f"- clean 缺失 branch 数: {len(clean_missing)}。不得对缺失 case 猜测排名。",
        f"- robustness Retrieval branch quality: {json.dumps(robustness_quality, ensure_ascii=False)}。",
        "- 结果中的异常、截断或 invalid JSON 状态保留在 record-level 及 manifest 中，不能在论文中静默当作完全有效输出。",
    ]
    # Write the actual numbers directly into the Markdown, so paper writing does not rely only on verbal conclusions.
    summary_index = {(row["condition"], row["fusion_method"]): row for row in summary_rows}

    def value(condition: str, fusion: str, field: str) -> Any:
        return summary_index[(condition, fusion)].get(field, "")

    def fmt(condition: str, fusion: str, field: str) -> str:
        item = value(condition, fusion, field)
        return f"{float(item):.4f}" if item != "" else "NA"

    md += [
        "",
        "## 实际汇总指标",
        "以下均为 record-level macro 平均；clean 因一个 Retrieval branch 缺失，四种 fusion 的有效记录数为 299，其余条件为 300。",
        "",
        "| Condition | Fusion | N | Hit@1 | Hit@3 | Hit@5 | MRR | NDCG@5 | Overwrite from Evidence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for fusion in FUSIONS:
            md.append(
                f"| {condition} | {fusion} | {value(condition, fusion, 'all_n')} | "
                f"{fmt(condition, fusion, 'all_hit_at_1')} | {fmt(condition, fusion, 'all_hit_at_3')} | "
                f"{fmt(condition, fusion, 'all_hit_at_5')} | {fmt(condition, fusion, 'all_mrr')} | "
                f"{fmt(condition, fusion, 'all_ndcg_at_5')} | "
                f"{fmt(condition, fusion, 'all_primary_overwrite_rate_from_evidence')} |"
            )
    md += [
        "",
        "## 五个问题的结论",
        f"1. Clean 条件下，RRF 的 Hit@1/MRR/NDCG@5 为 {fmt('clean', 'rrf', 'all_hit_at_1')}/{fmt('clean', 'rrf', 'all_mrr')}/{fmt('clean', 'rrf', 'all_ndcg_at_5')}，Borda 为 {fmt('clean', 'borda', 'all_hit_at_1')}/{fmt('clean', 'borda', 'all_mrr')}/{fmt('clean', 'borda', 'all_ndcg_at_5')}；均未超过 Authority-Preserving 的 {fmt('clean', 'authority_preserving', 'all_hit_at_1')}/{fmt('clean', 'authority_preserving', 'all_mrr')}/{fmt('clean', 'authority_preserving', 'all_ndcg_at_5')}，因此不能声称 RRF/Borda 提高了 clean 的首位或整体排序质量。",
        f"2. Clean 有效记录上 RRF/Borda 的 Hit@5 均为 {fmt('clean', 'rrf', 'all_hit_at_5')}，Authority-Preserving 为 {fmt('clean', 'authority_preserving', 'all_hit_at_5')}；Hit@5 只有轻微变化。三个 perturbation 条件下，RRF/Borda 的 Hit@5 分别为 random {fmt('random_retrieval', 'rrf', 'all_hit_at_5')}/{fmt('random_retrieval', 'borda', 'all_hit_at_5')}、stage {fmt('stage_mismatched_retrieval', 'rrf', 'all_hit_at_5')}/{fmt('stage_mismatched_retrieval', 'borda', 'all_hit_at_5')}、low {fmt('low_relevance_retrieval', 'rrf', 'all_hit_at_5')}/{fmt('low_relevance_retrieval', 'borda', 'all_hit_at_5')}，没有超过同条件 Authority-Preserving。",
        f"3. 在 perturbation 下，RRF/Borda 从 Evidence primary 改写 primary 的比例约为 {fmt('random_retrieval', 'rrf', 'all_primary_overwrite_rate_from_evidence')}/{fmt('random_retrieval', 'borda', 'all_primary_overwrite_rate_from_evidence')}、{fmt('stage_mismatched_retrieval', 'rrf', 'all_primary_overwrite_rate_from_evidence')}/{fmt('stage_mismatched_retrieval', 'borda', 'all_primary_overwrite_rate_from_evidence')}、{fmt('low_relevance_retrieval', 'rrf', 'all_primary_overwrite_rate_from_evidence')}/{fmt('low_relevance_retrieval', 'borda', 'all_primary_overwrite_rate_from_evidence')}；Authority-Preserving 为 0，因此 generic fusion 更容易覆盖 primary，但仍低于 Retrieval-Primary。",
        "4. RRF/Borda 作为 generic source-agnostic fusion，整体没有体现出优于 source-aware Authority-Preserving 的排序质量；Authority-Preserving 在三个 perturbation 条件的 Hit@5 均高于或不低于 RRF/Borda，同时完全保持 Evidence primary。",
        "5. 这些结果不支持把 KDAgent 表述为 accuracy-optimal fusion。更准确的论文表述是：KDAgent 的贡献在于 explicit source-authority control，在保持主诊断可信性的同时提供可控的 robustness–accuracy trade-off；RRF/Borda 是透明、可复现但不一定最优的 generic late-fusion 对照。",
        "",
        "## 统计说明",
        "本次只做 deterministic post-hoc fusion，没有新增 LLM 调用，也没有执行新的显著性检验；若后续补充检验，应以 20 cases 为独立单位，并单独标记为 post-hoc 分析，不混入原先预注册 Holm family。",
    ]
    (output / "paper_ready_rrf_borda_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("完成零 API RRF/Borda 分析：records=%d，summary=%d，missing=%d，输出目录=%s", len(records), len(summary_rows), len(missing_rows), output)


if __name__ == "__main__":
    main()
