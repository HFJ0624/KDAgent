"""Unified entry point for KDAgent mechanism validation.

`audit` and `offline` only read existing results and do not call the LLM; `feedback`
runs in dry-run mode by default. Paid calls are only made when both `--execute`
and `--allow-llm` are passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_client import ModelClient, load_model_configs  # noqa: E402
from src.iterative_agent import IterativeSelfRefinementAgent  # noqa: E402
from src.response_parser import parse_response  # noqa: E402
from src.response_validator import validate_response  # noqa: E402
from analysis.rrf_borda_posthoc_v1.run_rrf_borda_posthoc import fuse, positional_fusion  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
DEFAULT_OUTPUT = ROOT / "outputs" / "mechanism_validation_v1"
CASES_PATH = ROOT / "data" / "swat_s2s_raw_window_fixed" / "llm_prompt_cases.jsonl"
AUDIT_PATH = ROOT / "analysis" / "episode_audit_v1" / "swat_episode_audit.csv"
CLEAN_ROOT = ROOT / "outputs" / "final_experiments_frozen_v2"
ROBUST_PATH = ROOT / "outputs" / "retrieval_robustness_v1" / "records.jsonl"
FUSION_PATH = ROOT / "analysis" / "rrf_borda_posthoc_v1" / "rrf_borda_record_level.csv"
MODEL_CONFIG = ROOT / "configs" / "models.yaml"

METRICS = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr")


def setup_logger(output: Path) -> logging.Logger:
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    logger = logging.getLogger("mechanism_validation_v1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "logs" / "mechanism_validation.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML，无法读取机制验证配置。") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                item = json.loads(line)
                item["_source_file"] = str(path)
                item["_source_line"] = line_number
                rows.append(item)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_vars(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        item = str(value).strip().upper().replace(" ", "")
        if item and item not in result:
            result.append(item)
    return result


def parse_csv_rank(value: Any) -> list[str]:
    return normalize_vars(str(value or "").split(","))


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def load_cases() -> dict[int, dict[str, Any]]:
    cases: dict[int, dict[str, Any]] = {}
    for item in read_jsonl(CASES_PATH):
        case_id = int(item["case_id"])
        truth = normalize_vars((item.get("ground_truth") or {}).get("gt_vars"))
        top10 = normalize_vars(item.get("top10_vars"))
        if not top10:
            top10 = normalize_vars([x.get("var_id") for x in (item.get("top_k_variables") or [])[:10]])
        cases[case_id] = {**item, "truth": truth, "top10": top10}
    return cases


def load_episode_map() -> dict[int, str]:
    return {int(row["episode_id"]): str(row["episode_id"]) for row in read_csv(AUDIT_PATH)}


def clean_source_files(config: Mapping[str, Any]) -> list[Path]:
    files: list[Path] = []
    for model in config["models"]:
        files.extend(sorted((CLEAN_ROOT / f"dual_branch_fusion_agent_{model}" / "raw_responses").glob("*_run_*.jsonl")))
    return files


def model_from_path(path: Path) -> str:
    return path.parents[1].name.removeprefix("dual_branch_fusion_agent_")


def load_clean(config: Mapping[str, Any]) -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[dict[str, Any]]]:
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    duplicate_rows: list[dict[str, Any]] = []
    for path in clean_source_files(config):
        model = model_from_path(path)
        grouped: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {"evidence": [], "retrieval": [], "summary": []})
        for item in read_jsonl(path):
            key = (int(item.get("run_id", 0)), int(item.get("case_id", -1)))
            if item.get("record_type") == "case_summary":
                grouped[key]["summary"].append(item)
            elif item.get("record_type") == "agent_iteration" and item.get("branch") == "data_evidence":
                grouped[key]["evidence"].append(item)
            elif item.get("record_type") == "agent_iteration" and item.get("branch") == "rag_knowledge":
                grouped[key]["retrieval"].append(item)
        for (run_id, case_id), parts in grouped.items():
            key = (model, run_id, case_id)
            if key in records:
                duplicate_rows.append({"source": "clean", "model": model, "run_id": run_id, "episode_id": case_id, "reason": "duplicate_pair_key"})
            summary = parts["summary"][-1] if parts["summary"] else {}
            evidence = sorted(parts["evidence"], key=lambda x: int(x.get("iteration", 0)))
            retrieval = parts["retrieval"][-1] if parts["retrieval"] else None
            records[key] = {"summary": summary, "evidence": evidence, "retrieval": retrieval, "source_file": str(path)}
    return records, duplicate_rows


def load_robust_latest() -> tuple[dict[str, dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    raw_count = 0
    for item in read_jsonl(ROBUST_PATH):
        raw_count += 1
        key = str(item.get("record_key") or "")
        if key:
            latest[key] = item
    return latest, raw_count - len(latest)


def load_fusions() -> dict[tuple[str, int, int, str, str], dict[str, str]]:
    result: dict[tuple[str, int, int, str, str], dict[str, str]] = {}
    for row in read_csv(FUSION_PATH):
        key = (row["model"], int(row["run_id"]), int(row["case_id"]), row["condition"], row["fusion_method"])
        if key in result:
            raise RuntimeError(f"融合 CSV 出现重复配对键：{key}")
        result[key] = row
    return result


def expected_keys(config: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    return list(itertools.product(config["models"], config["runs"], range(int(config["episodes"]))))


def validate_saved(parsed: Mapping[str, Any], top10: Sequence[str], api_success: bool, finish_reason: str, threshold: float) -> dict[str, Any]:
    # Uniformly replay the project's current Validator instead of relying only on the historical boolean fields.
    return validate_response(dict(parsed), list(top10), threshold, api_success=api_success, finish_reason=finish_reason)


def acceptable_primary(parsed: Mapping[str, Any], top10: Sequence[str]) -> str:
    primary = str(parsed.get("primary_root_cause") or "").strip().upper().replace(" ", "")
    return primary if primary in set(top10) else ""


def metric_values(ranking: Sequence[str], truth: Sequence[str]) -> dict[str, float | int | str]:
    truth_set = set(truth)
    rank = next((index for index, value in enumerate(ranking[:5], start=1) if value in truth_set), None)
    return {
        "hit_at_1": int(rank == 1),
        "hit_at_3": int(rank is not None and rank <= 3),
        "hit_at_5": int(rank is not None and rank <= 5),
        "mrr": 1.0 / rank if rank else 0.0,
        "best_gt_rank": rank if rank else "",
    }


def bootstrap_episode_ci(values: Mapping[int, Sequence[float]], samples: int, seed: int) -> tuple[float, float, float]:
    import random
    episode_ids = sorted(values)
    episode_means = [statistics.fmean(values[x]) for x in episode_ids]
    if not episode_means:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    draws = [statistics.fmean(rng.choice(episode_means) for _ in episode_means) for _ in range(samples)]
    draws.sort()
    low = draws[int(0.025 * samples)]
    high = draws[min(samples - 1, int(0.975 * samples))]
    return statistics.fmean(episode_means), low, high


def audit(config: Mapping[str, Any], output: Path, logger: logging.Logger) -> dict[str, Any]:
    cases = load_cases()
    episode_map = load_episode_map()
    clean, clean_duplicates = load_clean(config)
    robust, robust_duplicates = load_robust_latest()
    fusions = load_fusions()
    expected = expected_keys(config)
    exclusions: list[dict[str, Any]] = list(clean_duplicates)
    clean_missing = 0
    for model, run_id, case_id in expected:
        record = clean.get((model, run_id, case_id))
        if not record:
            clean_missing += 1
            exclusions.append({"source": "clean", "model": model, "run_id": run_id, "episode_id": case_id, "reason": "missing_clean_pair"})
            continue
        if not record["evidence"]:
            exclusions.append({"source": "clean", "model": model, "run_id": run_id, "episode_id": case_id, "reason": "missing_evidence_branch"})
        if not record["retrieval"]:
            exclusions.append({"source": "clean", "model": model, "run_id": run_id, "episode_id": case_id, "reason": "missing_retrieval_branch"})
    threshold = float(config["min_confidence"])
    for key in expected:
        model, run_id, case_id = key
        record = clean.get(key, {})
        if not record:
            continue
        evidence_valid, _, _ = evidence_state(record, cases[case_id]["top10"], threshold)
        for condition in config["conditions"]:
            retrieval_valid, _, _, _ = retrieval_state(condition, key, record, robust, cases[case_id]["top10"])
            if evidence_valid is False:
                exclusions.append({"source": "mechanism_comparison", "model": model, "run_id": run_id, "episode_id": case_id, "condition": condition, "reason": "evidence_contract_failed"})
            elif retrieval_valid is None:
                exclusions.append({"source": "mechanism_comparison", "model": model, "run_id": run_id, "episode_id": case_id, "condition": condition, "reason": "missing_retrieval_branch"})
            elif retrieval_valid is False:
                exclusions.append({"source": "mechanism_comparison", "model": model, "run_id": run_id, "episode_id": case_id, "condition": condition, "reason": "unacceptable_retrieval_primary"})
    robust_expected = len(expected) * 3
    robust_branch = [x for x in robust.values() if x.get("method") == "retrieval_branch"]
    fusion_expected = len(expected) * len(config["conditions"]) * len(config["fusion_methods"])
    model_cfgs = load_model_configs(str(MODEL_CONFIG))
    model_details = []
    for model in config["models"]:
        cfg = model_cfgs[model]
        model_details.append({key: cfg.get(key) for key in ("name", "model", "provider", "base_url", "temperature", "max_tokens", "thinking_budget", "timeout", "max_retries", "retry_base_sleep")})
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "llm_calls": 0,
        "cases": len(cases),
        "episode_map_rows": len(episode_map),
        "expected_clean_pairs": len(expected),
        "loaded_clean_pairs": len(clean),
        "missing_clean_pairs": clean_missing,
        "robustness_raw_rows": len(read_jsonl(ROBUST_PATH)),
        "robustness_unique_record_keys": len(robust),
        "robustness_duplicate_checkpoint_rows": robust_duplicates,
        "robustness_retrieval_records": len(robust_branch),
        "expected_robustness_retrieval_records": robust_expected,
        "fusion_rows": len(fusions),
        "expected_fusion_rows": fusion_expected,
        "exclusion_count": len(exclusions),
        "model_configs": model_details,
        "model_revision_limitation": "Saved responses expose provider model aliases and request IDs, but no immutable provider revision/hash.",
        "source_files": [str(CASES_PATH), str(AUDIT_PATH), str(CLEAN_ROOT), str(ROBUST_PATH), str(FUSION_PATH), str(MODEL_CONFIG)],
    }
    write_json(output / "data_availability.json", report)
    write_csv(output / "data_exclusions.csv", exclusions)
    lines = [
        "# 数据可用性与排除报告", "",
        f"- 固定案例：{len(cases)}；clean 配对：{len(clean)}/{len(expected)}。",
        f"- Robustness Retrieval 唯一记录：{len(robust_branch)}/{robust_expected}；append-only 重复 checkpoint 行：{robust_duplicates}，分析按 record_key 取最后一条。",
        f"- AP/RP/RRF/Borda 配对行：{len(fusions)}/{fusion_expected}。",
        f"- 条件级排除状态：{len(exclusions)} 条；详见 `data_exclusions.csv`。同一模型-run-episode 在不同 retrieval 条件下分别计数。",
        "- 模型日志保留 provider model alias 与 request ID，但不包含不可变供应商 revision/hash。",
        "- 本审计不调用 LLM，且不修改原始结果。",
    ]
    (output / "data_availability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("数据审计完成：clean=%d，robust retrieval=%d，fusion=%d，exclusions=%d", len(clean), len(robust_branch), len(fusions), len(exclusions))
    return report


def evidence_state(record: Mapping[str, Any], top10: Sequence[str], threshold: float) -> tuple[bool | None, list[str], str]:
    rounds = record.get("evidence") or []
    if not rounds:
        return None, [], ""
    item = rounds[-1]
    parsed = item.get("parsed_response") or {}
    validation = validate_saved(parsed, top10, bool(item.get("api_success")), str(item.get("finish_reason") or ""), threshold)
    return bool(validation["is_valid"]), normalize_vars(parsed.get("predicted_root_causes")), acceptable_primary(parsed, top10)


def retrieval_state(condition: str, key: tuple[str, int, int], clean_record: Mapping[str, Any], robust: Mapping[str, Mapping[str, Any]], top10: Sequence[str]) -> tuple[bool | None, list[str], str, dict[str, Any] | None]:
    model, run_id, case_id = key
    if condition == "clean":
        item = clean_record.get("retrieval")
    else:
        item = robust.get(f"{model}|{run_id}|{case_id}|{condition}|retrieval_branch")
    if not item:
        return None, [], "", None
    parsed = item.get("parsed_response") or {}
    rank = [x for x in normalize_vars(parsed.get("predicted_root_causes")) if x in set(top10)]
    primary = acceptable_primary(parsed, top10)
    return bool(primary), rank, primary, item


def experiment_conflict(config: Mapping[str, Any], output: Path, logger: logging.Logger) -> list[dict[str, Any]]:
    cases = load_cases()
    clean, _ = load_clean(config)
    robust, _ = load_robust_latest()
    fusions = load_fusions()
    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    replay_mismatches: list[dict[str, Any]] = []
    threshold = float(config["min_confidence"])
    for key in expected_keys(config):
        model, run_id, case_id = key
        case = cases[case_id]
        base = clean.get(key, {})
        ev_valid, ev_rank, ev_primary = evidence_state(base, case["top10"], threshold)
        for condition in config["conditions"]:
            rt_acceptable, rt_rank, rt_primary, _ = retrieval_state(condition, key, base, robust, case["top10"])
            if not base:
                status = "missing_clean_record"
            elif ev_valid is None:
                status = "unknown_evidence_validation"
            elif not ev_valid:
                status = "evidence_contract_failed"
            elif rt_acceptable is None:
                status = "missing_retrieval_branch"
            elif not rt_acceptable:
                status = "unacceptable_retrieval_primary"
            else:
                status = "agreement" if ev_primary == rt_primary else "disagreement"
            stored_ap = fusions.get((model, run_id, case_id, condition, "authority_preserving"), {})
            clean_summary = base.get("summary") or {}
            status_rows.append({"model": model, "run_id": run_id, "episode_id": case_id, "condition": condition, "status": status, "evidence_primary": ev_primary, "retrieval_primary": rt_primary, "evidence_rank": ",".join(ev_rank), "retrieval_rank": ",".join(rt_rank), "saved_ap_top5": stored_ap.get("predicted_top5", ""), "saved_ap_missing_branch": stored_ap.get("missing_branch", ""), "clean_agent_status": clean_summary.get("agent_status", "") if condition == "clean" else "", "clean_evidence_branch_success": clean_summary.get("evidence_branch_success", "") if condition == "clean" else "", "clean_retrieval_branch_success": clean_summary.get("rag_branch_success", "") if condition == "clean" else ""})
            for method in config["fusion_methods"]:
                stored = fusions.get((model, run_id, case_id, condition, method))
                if status not in {"agreement", "disagreement"} or not stored or stored.get("missing_branch"):
                    continue
                ranking = parse_csv_rank(stored.get("predicted_top5"))
                replayed = positional_fusion(ev_rank, rt_rank, case["top10"], method) if method in {"authority_preserving", "retrieval_primary"} else fuse(ev_rank, rt_rank, case["top10"], method)[0]
                if ranking != replayed:
                    replay_mismatches.append({"model": model, "run_id": run_id, "episode_id": case_id, "condition": condition, "fusion_method": method, "saved_top5": ",".join(ranking), "replayed_top5": ",".join(replayed)})
                metrics = metric_values(ranking, case["truth"])
                rows.append({
                    "model": model, "run_id": run_id, "episode_id": case_id, "condition": condition,
                    "group": status, "fusion_method": method, "ground_truth": ",".join(case["truth"]),
                    "evidence_primary": ev_primary, "retrieval_primary": rt_primary,
                    "evidence_rank": ",".join(ev_rank), "retrieval_rank": ",".join(rt_rank),
                    "predicted_top5": ",".join(ranking), **metrics,
                })
    write_csv(output / "exp1_conflict_record_level.csv", rows)
    write_csv(output / "exp1_status_and_exclusions.csv", status_rows)
    write_csv(output / "exp1_fusion_replay_mismatches.csv", replay_mismatches)
    write_json(output / "exp1_fusion_replay_check.json", {"checked_records": len(rows), "mismatch_count": len(replay_mismatches), "fusion_implementation": str(ROOT / "analysis" / "rrf_borda_posthoc_v1" / "run_rrf_borda_posthoc.py")})
    if replay_mismatches:
        raise RuntimeError(f"现有融合函数重放结果与保存结果不一致：{len(replay_mismatches)} 条，详见 exp1_fusion_replay_mismatches.csv")

    ap_index = {(r["model"], r["run_id"], r["episode_id"], r["condition"], r["group"]): r for r in rows if r["fusion_method"] == "authority_preserving"}
    transitions: list[dict[str, Any]] = []
    for row in rows:
        if row["fusion_method"] == "authority_preserving":
            continue
        ap = ap_index[(row["model"], row["run_id"], row["episode_id"], row["condition"], row["group"])]
        a, b = int(ap["hit_at_1"]), int(row["hit_at_1"])
        transition = "correct_to_wrong" if (a, b) == (1, 0) else "wrong_to_correct" if (a, b) == (0, 1) else "correctness_unchanged"
        transitions.append({**{k: row[k] for k in ("model", "run_id", "episode_id", "condition", "group", "fusion_method")}, "ap_hit_at_1": a, "comparison_hit_at_1": b, "transition": transition, "delta_hit_at_1": b - a, "delta_mrr": float(row["mrr"]) - float(ap["mrr"])})
    write_csv(output / "exp1_paired_transitions_vs_ap.csv", transitions)

    transition_summary: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        for method in ("retrieval_primary", "rrf", "borda"):
            selected = [r for r in transitions if r["condition"] == condition and r["group"] == "disagreement" and r["fusion_method"] == method]
            counts = Counter(str(r["transition"]) for r in selected)
            transition_summary.append({"condition": condition, "fusion_method": method, "n_paired_records": len(selected), "n_episodes": len({r["episode_id"] for r in selected}), "correct_to_wrong": counts["correct_to_wrong"], "wrong_to_correct": counts["wrong_to_correct"], "correctness_unchanged": counts["correctness_unchanged"], "hit_at_1_net_gain": counts["wrong_to_correct"] - counts["correct_to_wrong"]})
    write_csv(output / "exp1_transition_summary.csv", transition_summary)

    summaries: list[dict[str, Any]] = []

    def episode_balanced_metrics(selected: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        # Average over models and repeated runs within the same episode first, then weigh episodes equally.
        by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for item in selected:
            by_episode[int(item["episode_id"])].append(item)
        return {
            metric: round(statistics.fmean(statistics.fmean(float(x[metric]) for x in items) for items in by_episode.values()), 6)
            for metric in METRICS
        }

    for condition in config["conditions"]:
        for group in ("agreement", "disagreement"):
            for method in config["fusion_methods"]:
                selected = [r for r in rows if r["condition"] == condition and r["group"] == group and r["fusion_method"] == method]
                if not selected:
                    continue
                summaries.append({"scope": "episode_balanced_all_models", "model": "ALL", "weighting": "mean over model-runs within episode, then equal-weight mean over episodes", "condition": condition, "group": group, "fusion_method": method, "n_records": len(selected), "n_episodes": len({r["episode_id"] for r in selected}), **episode_balanced_metrics(selected)})
                for model in config["models"]:
                    part = [r for r in selected if r["model"] == model]
                    if part:
                        summaries.append({"scope": "episode_balanced_per_model", "model": model, "weighting": "mean over runs within episode, then equal-weight mean over episodes", "condition": condition, "group": group, "fusion_method": method, "n_records": len(part), "n_episodes": len({r["episode_id"] for r in part}), **episode_balanced_metrics(part)})
    write_csv(output / "exp1_conflict_summary.csv", summaries)

    episode_diffs: list[dict[str, Any]] = []
    inference: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        for method in ("retrieval_primary", "rrf", "borda"):
            selected = [r for r in transitions if r["condition"] == condition and r["group"] == "disagreement" and r["fusion_method"] == method]
            for metric in ("delta_hit_at_1", "delta_mrr"):
                by_episode: dict[int, list[float]] = defaultdict(list)
                for row in selected:
                    by_episode[int(row["episode_id"])].append(float(row[metric]))
                for episode_id, values in sorted(by_episode.items()):
                    episode_diffs.append({"condition": condition, "fusion_method": method, "metric": metric, "episode_id": episode_id, "n_model_runs": len(values), "mean_difference": round(statistics.fmean(values), 6)})
                mean, low, high = bootstrap_episode_ci(by_episode, int(config["bootstrap_samples"]), int(config["seed"]))
                inference.append({"condition": condition, "fusion_method": method, "metric": metric, "n_episodes": len(by_episode), "episode_balanced_mean": round(mean, 6), "bootstrap_ci95_low": round(low, 6), "bootstrap_ci95_high": round(high, 6), "bootstrap_samples": config["bootstrap_samples"], "seed": config["seed"], "post_hoc_family": "mechanism_validation_separate_from_original_holm"})
    write_csv(output / "exp1_episode_level_paired_differences.csv", episode_diffs)
    write_csv(output / "exp1_episode_cluster_bootstrap.csv", inference)
    logger.info("实验1完成：可比较策略记录=%d，状态记录=%d", len(rows), len(status_rows))
    return rows


def rank_from_iteration(item: Mapping[str, Any], top10: Sequence[str]) -> list[str]:
    parsed = item.get("parsed_response") or {}
    return [x for x in normalize_vars(parsed.get("predicted_root_causes")) if x in set(top10)][:5]


def experiment_repair(config: Mapping[str, Any], output: Path, logger: logging.Logger) -> list[dict[str, Any]]:
    cases = load_cases()
    clean, _ = load_clean(config)
    rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    errors_recovered: Counter[str] = Counter()
    threshold = float(config["min_confidence"])
    for model, run_id, case_id in expected_keys(config):
        record = clean[(model, run_id, case_id)]
        case = cases[case_id]
        rounds = record["evidence"]
        if not rounds:
            continue
        validations = [validate_saved(x.get("parsed_response") or {}, case["top10"], bool(x.get("api_success")), str(x.get("finish_reason") or ""), threshold) for x in rounds]
        for index, (iteration, validation) in enumerate(zip(rounds, validations), start=1):
            iteration_rows.append({
                "model": model, "run_id": run_id, "episode_id": case_id,
                "iteration": int(iteration.get("iteration") or index),
                "prompt_type": iteration.get("prompt_type", ""),
                "retry_reasons_from_previous_round": ",".join(iteration.get("retry_reasons_from_previous_round") or []),
                "response_content": iteration.get("content", ""),
                "response_sha256": hashlib.sha256(str(iteration.get("content") or "").encode("utf-8")).hexdigest(),
                "parsed_response_json": json.dumps(iteration.get("parsed_response") or {}, ensure_ascii=False),
                "validation_passed": int(bool(validation["is_valid"])),
                "validation_error_codes": ",".join(validation["retry_reasons"]),
                "api_success": int(bool(iteration.get("api_success"))),
                "finish_reason": iteration.get("finish_reason", ""),
                "request_id": extract_request_id(iteration.get("raw_response")),
                "total_tokens": int(iteration.get("total_tokens") or 0),
                "elapsed_s": float(iteration.get("elapsed_s") or 0.0),
            })
        first, final = rounds[0], rounds[-1]
        first_rank, final_rank = rank_from_iteration(first, case["top10"]), rank_from_iteration(final, case["top10"])
        first_metrics, final_metrics = metric_values(first_rank, case["truth"]), metric_values(final_rank, case["truth"])
        first_reasons = list(validations[0]["retry_reasons"])
        for reason in first_reasons:
            errors[reason] += 1
            if validations[-1]["is_valid"]:
                errors_recovered[reason] += 1
        summary = record.get("summary") or {}
        final_fused = normalize_vars((summary.get("parsed_response") or {}).get("predicted_root_causes"))[:5]
        fused_metrics = metric_values(final_fused, case["truth"])
        rows.append({
            "model": model, "run_id": run_id, "episode_id": case_id, "ground_truth": ",".join(case["truth"]),
            "round_count": len(rounds), "first_validation_passed": int(validations[0]["is_valid"]),
            "final_validation_passed": int(validations[-1]["is_valid"]),
            "recovery_round": next((index + 1 for index, val in enumerate(validations) if val["is_valid"]), ""),
            "stop_reason": "validator_passed" if validations[-1]["is_valid"] else "max_rounds_exhausted",
            "first_error_codes": ",".join(first_reasons),
            "first_primary": acceptable_primary(first.get("parsed_response") or {}, case["top10"]),
            "final_primary": acceptable_primary(final.get("parsed_response") or {}, case["top10"]),
            "first_top5": ",".join(first_rank), "final_evidence_top5": ",".join(final_rank), "final_fused_top5": ",".join(final_fused),
            "primary_changed": int((first_rank[:1] or [""])[0] != (final_rank[:1] or [""])[0]),
            "top5_changed": int(first_rank != final_rank),
            "first_evidence_hit_at_1": first_metrics["hit_at_1"], "first_evidence_hit_at_5": first_metrics["hit_at_5"],
            "final_evidence_hit_at_1": final_metrics["hit_at_1"], "final_evidence_hit_at_5": final_metrics["hit_at_5"],
            "final_fusion_hit_at_1": fused_metrics["hit_at_1"], "final_fusion_hit_at_5": fused_metrics["hit_at_5"],
            "logical_calls": len(rounds), "extra_logical_calls": max(0, len(rounds) - 1),
            "first_tokens": int(first.get("total_tokens") or 0),
            "extra_tokens": sum(int(x.get("total_tokens") or 0) for x in rounds[1:]),
            "first_elapsed_s": float(first.get("elapsed_s") or 0.0),
            "extra_elapsed_s": sum(float(x.get("elapsed_s") or 0.0) for x in rounds[1:]),
            "request_ids": ",".join(extract_request_id(x.get("raw_response")) for x in rounds if extract_request_id(x.get("raw_response"))),
        })
    write_csv(output / "exp2_repair_record_level.csv", rows)
    write_csv(output / "exp2_repair_iteration_trace.csv", iteration_rows)
    error_rows = [{"error_code": code, "first_round_occurrences": count, "records_eventually_recovered": errors_recovered[code], "recovery_rate": round(errors_recovered[code] / count, 6)} for code, count in errors.most_common()]
    write_csv(output / "exp2_error_recovery.csv", error_rows)

    summary_rows: list[dict[str, Any]] = []
    for model in ["ALL", *config["models"]]:
        selected = rows if model == "ALL" else [r for r in rows if r["model"] == model]
        failed = [r for r in selected if not int(r["first_validation_passed"])]
        hit1_gain = sum(int(r["first_evidence_hit_at_1"]) == 0 and int(r["final_evidence_hit_at_1"]) == 1 for r in failed)
        hit1_loss = sum(int(r["first_evidence_hit_at_1"]) == 1 and int(r["final_evidence_hit_at_1"]) == 0 for r in failed)
        hit5_gain = sum(int(r["first_evidence_hit_at_5"]) == 0 and int(r["final_evidence_hit_at_5"]) == 1 for r in failed)
        hit5_loss = sum(int(r["first_evidence_hit_at_5"]) == 1 and int(r["final_evidence_hit_at_5"]) == 0 for r in failed)
        summary_rows.append({
            "model": model, "weighting": "balanced record macro" if model == "ALL" else "model-run-episode records",
            "n_records": len(selected), "first_round_pass_rate": round(statistics.fmean(int(r["first_validation_passed"]) for r in selected), 6),
            "final_pass_rate": round(statistics.fmean(int(r["final_validation_passed"]) for r in selected), 6),
            "round2_recovered": sum(str(r["recovery_round"]) == "2" for r in selected),
            "round3_recovered": sum(str(r["recovery_round"]) == "3" for r in selected),
            "unrecovered": sum(not int(r["final_validation_passed"]) for r in selected),
            "repair_needed": len(failed), "primary_changed_among_repair": sum(int(r["primary_changed"]) for r in failed),
            "top5_changed_among_repair": sum(int(r["top5_changed"]) for r in failed),
            "repair_subset_first_hit_at_1": round(statistics.fmean(float(r["first_evidence_hit_at_1"]) for r in failed), 6) if failed else "",
            "repair_subset_final_hit_at_1": round(statistics.fmean(float(r["final_evidence_hit_at_1"]) for r in failed), 6) if failed else "",
            "repair_subset_hit_at_1_gain_count": hit1_gain,
            "repair_subset_hit_at_1_loss_count": hit1_loss,
            "repair_subset_first_hit_at_5": round(statistics.fmean(float(r["first_evidence_hit_at_5"]) for r in failed), 6) if failed else "",
            "repair_subset_final_hit_at_5": round(statistics.fmean(float(r["final_evidence_hit_at_5"]) for r in failed), 6) if failed else "",
            "repair_subset_hit_at_5_gain_count": hit5_gain,
            "repair_subset_hit_at_5_loss_count": hit5_loss,
            "extra_logical_calls": sum(int(r["extra_logical_calls"]) for r in selected),
            "extra_tokens": sum(int(r["extra_tokens"]) for r in selected),
            "extra_elapsed_s": round(sum(float(r["extra_elapsed_s"]) for r in selected), 3),
            "first_evidence_hit_at_1": round(statistics.fmean(float(r["first_evidence_hit_at_1"]) for r in selected), 6),
            "final_evidence_hit_at_1": round(statistics.fmean(float(r["final_evidence_hit_at_1"]) for r in selected), 6),
            "first_evidence_hit_at_5": round(statistics.fmean(float(r["first_evidence_hit_at_5"]) for r in selected), 6),
            "final_evidence_hit_at_5": round(statistics.fmean(float(r["final_evidence_hit_at_5"]) for r in selected), 6),
        })
    write_csv(output / "exp2_repair_summary.csv", summary_rows)
    example_candidates = [r for r in rows if not int(r["first_validation_passed"]) and int(r["final_validation_passed"]) and int(r["top5_changed"])]
    example = sorted(example_candidates, key=lambda r: (-len(r["first_error_codes"].split(",")), r["model"], r["episode_id"]))[0] if example_candidates else None
    if example:
        source = clean[(example["model"], int(example["run_id"]), int(example["episode_id"]))]
        trace = [{"iteration": int(x.get("iteration", 0)), "prompt_type": x.get("prompt_type"), "validation_result": validate_saved(x.get("parsed_response") or {}, cases[int(example["episode_id"])]["top10"], bool(x.get("api_success")), str(x.get("finish_reason") or ""), threshold), "parsed_response": x.get("parsed_response"), "finish_reason": x.get("finish_reason"), "tokens": x.get("total_tokens"), "elapsed_s": x.get("elapsed_s")} for x in source["evidence"]]
        write_json(output / "exp2_repair_case_example.json", {"selection_rule": "recovered record with changed Top-5; ties by most initial errors then model/episode", "record": example, "trace": trace})
    logger.info("实验2完成：records=%d，首轮失败=%d，最终未恢复=%d", len(rows), sum(not int(r["first_validation_passed"]) for r in rows), sum(not int(r["final_validation_passed"]) for r in rows))
    return rows


def extract_request_id(raw: Any) -> str:
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return str(parsed.get("id") or "") if isinstance(parsed, dict) else ""
        except json.JSONDecodeError:
            return ""
    if isinstance(raw, dict):
        return str(raw.get("id") or "")
    return ""


def feedback_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = load_cases()
    clean, _ = load_clean(config)
    selected: list[dict[str, Any]] = []
    threshold = float(config["min_confidence"])
    for key in expected_keys(config):
        record = clean[key]
        if not record["evidence"]:
            continue
        first = record["evidence"][0]
        validation = validate_saved(first.get("parsed_response") or {}, cases[key[2]]["top10"], bool(first.get("api_success")), str(first.get("finish_reason") or ""), threshold)
        if not validation["is_valid"]:
            selected.append({"model": key[0], "run_id": key[1], "episode_id": key[2], "case": cases[key[2]], "first": first, "validation": validation})
    return selected


def feedback_prompt(
    snapshot: Mapping[str, Any],
    treatment: str,
    previous_content: str | None = None,
    previous_validation: Mapping[str, Any] | None = None,
    useful_information: Mapping[str, Any] | None = None,
    repair_round: int = 1,
) -> str:
    """Build paired retry prompts from the original Agent template; the only treatment difference is the granularity of the error feedback."""
    first = snapshot["first"]
    case = snapshot["case"]
    top10 = case["top10"]
    validation = dict(previous_validation or snapshot["validation"])
    if treatment == "generic":
        validation["retry_reasons"] = ["上一轮输出未满足要求，请重新检查并修正。"]
    elif treatment != "targeted":
        raise ValueError(f"未知反馈组：{treatment}")

    agent = IterativeSelfRefinementAgent(model_client=None, min_confidence=0.5)
    content = previous_content if previous_content is not None else str(first.get("content") or "")
    useful = dict(useful_information or agent.extract_useful_information(first.get("parsed_response") or {}, top10))
    if repair_round == 1:
        return agent.build_refinement_prompt(case, top10, content, validation, useful, rag_contexts=None)
    return agent.build_forced_choice_prompt(case, top10, content, validation, useful, rag_contexts=None)


def feedback_dry_run(config: Mapping[str, Any], output: Path, logger: logging.Logger) -> dict[str, Any]:
    selected = feedback_cases(config)
    model_configs = load_model_configs(str(MODEL_CONFIG))
    by_model = Counter(x["model"] for x in selected)
    first_tokens = [int(x["first"].get("total_tokens") or 0) for x in selected]
    n = len(selected)
    max_calls = n * 2 * int(config["max_repair_rounds"])
    empirical_token_reference = round(statistics.fmean(first_tokens) * max_calls) if first_tokens else 0
    completion_token_ceiling = sum(
        count * 2 * int(config["max_repair_rounds"]) * int(model_configs[model]["max_tokens"])
        for model, count in by_model.items()
    )
    budget = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "execution": "dry_run", "llm_calls": 0,
        "eligible_first_round_failures": n, "by_model": dict(sorted(by_model.items())),
        "treatments": ["targeted", "generic"], "max_additional_rounds_per_treatment": int(config["max_repair_rounds"]),
        "max_logical_generation_calls": max_calls,
        "network_retry_policy": "ModelClient retries are transport attempts and are recorded separately; they are not independent experimental observations.",
        "observed_first_round_total_tokens_sum": sum(first_tokens),
        "observed_first_round_total_tokens_mean": round(statistics.fmean(first_tokens), 2) if first_tokens else 0,
        "empirical_total_token_planning_reference_at_max_calls": empirical_token_reference,
        "configured_completion_token_ceiling": completion_token_ceiling,
        "token_budget_note": "Future prompt and completion lengths are stochastic. The empirical reference multiplies the saved first-round mean by the maximum call count; the configured ceiling covers completion tokens only and excludes prompt tokens.",
        "money_budget": "not estimated: no reliable immutable per-model price table is stored in the repository",
        "model_config_path": str(MODEL_CONFIG), "min_confidence": config["min_confidence"], "seed": config["seed"],
    }
    write_json(output / "exp3_dry_run_budget.json", budget)
    rows = []
    for item in selected:
        first = item["first"]
        model_cfg = model_configs[item["model"]]
        rows.append({"model": item["model"], "run_id": item["run_id"], "episode_id": item["episode_id"], "first_request_id": extract_request_id(first.get("raw_response")), "first_error_codes": ",".join(item["validation"]["retry_reasons"]), "first_prompt_sha256": hashlib.sha256(str(first.get("prompt") or "").encode("utf-8")).hexdigest(), "first_response_sha256": hashlib.sha256(str(first.get("content") or "").encode("utf-8")).hexdigest(), "targeted_prompt_sha256": hashlib.sha256(feedback_prompt(item, "targeted").encode("utf-8")).hexdigest(), "generic_prompt_sha256": hashlib.sha256(feedback_prompt(item, "generic").encode("utf-8")).hexdigest(), "temperature": model_cfg["temperature"], "max_tokens": model_cfg["max_tokens"], "thinking_budget": model_cfg.get("thinking_budget", "")})
    write_csv(output / "exp3_eligible_snapshots.csv", rows)
    logger.info("实验3 dry-run：N=%d，最大新增逻辑调用=%d，实际 API 调用=0", n, budget["max_logical_generation_calls"])
    return budget


def feedback_execute(config: Mapping[str, Any], output: Path, logger: logging.Logger, resume: bool, allow_llm: bool) -> None:
    if not allow_llm:
        raise RuntimeError("正式调用被安全门禁阻止：必须同时指定 --execute --allow-llm。")
    selected = feedback_cases(config)
    checkpoint = output / "exp3_feedback_records.jsonl"
    if checkpoint.exists() and not resume:
        raise RuntimeError("实验3 checkpoint 已存在。为避免重复付费调用，请使用 --resume 继续。")
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if checkpoint.exists():
        for row in read_jsonl(checkpoint):
            history[str(row["pair_key"])].append(row)
    cfgs = load_model_configs(str(MODEL_CONFIG))
    for snapshot in selected:
        case = snapshot["case"]
        for treatment in ("targeted", "generic"):
            pair_key = f"{snapshot['model']}|{snapshot['run_id']}|{snapshot['episode_id']}|{treatment}"
            previous_rows = sorted(history.get(pair_key, []), key=lambda x: int(x["repair_round"]))
            if previous_rows and previous_rows[-1].get("terminal"):
                logger.info("[skip] %s", pair_key)
                continue
            client = ModelClient(cfgs[snapshot["model"]], logger=logger)
            agent = IterativeSelfRefinementAgent(client, float(config["min_confidence"]), logger)
            useful_information = agent.extract_useful_information(snapshot["first"].get("parsed_response") or {}, case["top10"])
            previous_content = str(snapshot["first"].get("content") or "")
            previous_validation = snapshot["validation"]
            for old in previous_rows:
                previous_content = str(old.get("content") or "")
                previous_validation = old.get("validation_result") or {}
                useful_information = agent.merge_useful_information(
                    useful_information,
                    agent.extract_useful_information(old.get("parsed_response") or {}, case["top10"]),
                )
            start_round = int(previous_rows[-1]["repair_round"]) + 1 if previous_rows else 1
            for repair_round in range(start_round, int(config["max_repair_rounds"]) + 1):
                prompt = feedback_prompt(
                    snapshot, treatment, previous_content, previous_validation,
                    useful_information, repair_round,
                )
                result = client.chat(prompt=prompt, system_prompt="You are a careful industrial root cause analysis engineer. Output strictly valid JSON only.", max_retries=3, timeout=120, retry_base_sleep=5)
                parsed = parse_response(result.get("content", "") if result.get("success") else "")
                validation = validate_saved(parsed, case["top10"], bool(result.get("success")), str(result.get("finish_reason") or ""), float(config["min_confidence"]))
                terminal = bool(validation["is_valid"]) or repair_round == int(config["max_repair_rounds"])
                network_attempts = int(result.get("network_attempts") or 0)
                record = {"pair_key": pair_key, "model": snapshot["model"], "run_id": snapshot["run_id"], "episode_id": snapshot["episode_id"], "treatment": treatment, "repair_round": repair_round, "prompt": prompt, "previous_content": previous_content, "content": result.get("content", ""), "raw_response": result.get("raw_response", ""), "request_id": extract_request_id(result.get("raw_response")), "parsed_response": parsed, "validation_result": validation, "api_success": result.get("success"), "finish_reason": result.get("finish_reason"), "error_type": result.get("error_type"), "error_message": result.get("error_message"), "prompt_tokens": result.get("prompt_tokens", 0), "completion_tokens": result.get("completion_tokens", 0), "reasoning_tokens": result.get("reasoning_tokens", 0), "total_tokens": result.get("total_tokens", 0), "elapsed_s": result.get("elapsed", 0.0), "logical_calls": 1, "network_attempts": network_attempts, "network_retries": max(0, network_attempts - 1), "terminal": terminal, "config": {k: cfgs[snapshot["model"]].get(k) for k in ("model", "temperature", "max_tokens", "thinking_budget")}}
                with checkpoint.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                if terminal:
                    break
                previous_content = str(result.get("content") or "")
                previous_validation = validation
                useful_information = agent.merge_useful_information(
                    useful_information,
                    agent.extract_useful_information(parsed, case["top10"]),
                )
    summarize_feedback(config, output, logger)


def summarize_feedback(config: Mapping[str, Any], output: Path, logger: logging.Logger) -> None:
    """Aggregate the append-only checkpoint into paired, directly analyzable Experiment 3 results."""
    checkpoint = output / "exp3_feedback_records.jsonl"
    if not checkpoint.exists():
        return
    cases = load_cases()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in read_jsonl(checkpoint):
        grouped[str(item["pair_key"])].append(item)

    record_rows: list[dict[str, Any]] = []
    for pair_key, attempts in sorted(grouped.items()):
        attempts.sort(key=lambda x: int(x["repair_round"]))
        final = attempts[-1]
        case = cases[int(final["episode_id"])]
        ranking = rank_from_iteration(final, case["top10"])
        metrics = metric_values(ranking, case["truth"])
        record_rows.append({
            "pair_key": pair_key, "model": final["model"], "run_id": final["run_id"],
            "episode_id": final["episode_id"], "treatment": final["treatment"],
            "terminal": int(bool(final.get("terminal"))),
            "final_validation_passed": int(bool((final.get("validation_result") or {}).get("is_valid"))),
            "final_top5": ",".join(ranking), "hit_at_1": metrics["hit_at_1"],
            "hit_at_5": metrics["hit_at_5"], "mrr": metrics["mrr"],
            "logical_calls": len(attempts),
            "network_attempts": sum(int(x.get("network_attempts") or 0) for x in attempts),
            "network_retries": sum(int(x.get("network_retries") or 0) for x in attempts),
            "total_tokens": sum(int(x.get("total_tokens") or 0) for x in attempts),
            "elapsed_s": round(sum(float(x.get("elapsed_s") or 0.0) for x in attempts), 3),
            "final_error_codes": ",".join((final.get("validation_result") or {}).get("retry_reasons") or []),
        })
    write_csv(output / "exp3_feedback_record_level.csv", record_rows)

    summary_rows: list[dict[str, Any]] = []
    for treatment in ("targeted", "generic"):
        part = [x for x in record_rows if x["treatment"] == treatment and x["terminal"]]
        recovered = sum(int(x["final_validation_passed"]) for x in part)
        by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in part:
            by_episode[int(item["episode_id"])].append(item)

        def episode_rate(field: str) -> float | str:
            return round(statistics.fmean(statistics.fmean(float(x[field]) for x in values) for values in by_episode.values()), 6) if by_episode else ""

        summary_rows.append({
            "treatment": treatment, "weighting": "mean over eligible model-runs within episode, then equal-weight mean over episodes", "n_terminal_pairs": len(part), "n_episodes": len(by_episode), "n_recovered": recovered,
            "final_validation_pass_rate": episode_rate("final_validation_passed"),
            "evidence_hit_at_1": episode_rate("hit_at_1"),
            "evidence_hit_at_5": episode_rate("hit_at_5"),
            "logical_calls": sum(int(x["logical_calls"]) for x in part),
            "network_attempts": sum(int(x["network_attempts"]) for x in part),
            "network_retries": sum(int(x["network_retries"]) for x in part),
            "total_tokens": sum(int(x["total_tokens"]) for x in part),
            "elapsed_s": round(sum(float(x["elapsed_s"]) for x in part), 3),
            "tokens_per_recovered_output": round(sum(int(x["total_tokens"]) for x in part) / recovered, 2) if recovered else "",
            "logical_calls_per_recovered_output": round(sum(int(x["logical_calls"]) for x in part) / recovered, 3) if recovered else "",
        })
    write_csv(output / "exp3_feedback_summary.csv", summary_rows)

    index = {(x["model"], int(x["run_id"]), int(x["episode_id"]), x["treatment"]): x for x in record_rows if x["terminal"]}
    paired_rows: list[dict[str, Any]] = []
    for model, run_id, episode_id in expected_keys(config):
        targeted = index.get((model, run_id, episode_id, "targeted"))
        generic = index.get((model, run_id, episode_id, "generic"))
        if not targeted or not generic:
            continue
        paired_rows.append({
            "model": model, "run_id": run_id, "episode_id": episode_id,
            "delta_validation_pass_targeted_minus_generic": int(targeted["final_validation_passed"]) - int(generic["final_validation_passed"]),
            "delta_hit_at_1_targeted_minus_generic": int(targeted["hit_at_1"]) - int(generic["hit_at_1"]),
            "delta_hit_at_5_targeted_minus_generic": int(targeted["hit_at_5"]) - int(generic["hit_at_5"]),
            "delta_tokens_targeted_minus_generic": int(targeted["total_tokens"]) - int(generic["total_tokens"]),
            "delta_elapsed_s_targeted_minus_generic": round(float(targeted["elapsed_s"]) - float(generic["elapsed_s"]), 3),
        })
    write_csv(output / "exp3_feedback_paired_differences.csv", paired_rows)
    episode_rows: list[dict[str, Any]] = []
    for episode_id in sorted({int(x["episode_id"]) for x in paired_rows}):
        part = [x for x in paired_rows if int(x["episode_id"]) == episode_id]
        episode_rows.append({"episode_id": episode_id, "n_model_runs": len(part), **{field: round(statistics.fmean(float(x[field]) for x in part), 6) for field in ("delta_validation_pass_targeted_minus_generic", "delta_hit_at_1_targeted_minus_generic", "delta_hit_at_5_targeted_minus_generic", "delta_tokens_targeted_minus_generic", "delta_elapsed_s_targeted_minus_generic")}})
    write_csv(output / "exp3_feedback_episode_level_paired_differences.csv", episode_rows)
    logger.info("实验3汇总完成：terminal pairs=%d，paired records=%d", sum(int(x["terminal"]) for x in record_rows), len(paired_rows))


def write_results_summary(config: Mapping[str, Any], output: Path, exp1: Sequence[Mapping[str, Any]], exp2: Sequence[Mapping[str, Any]], budget: Mapping[str, Any]) -> None:
    disagreement = [r for r in exp1 if r["group"] == "disagreement"]
    repair_needed = [r for r in exp2 if not int(r["first_validation_passed"])]

    def episode_mean(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
        by_episode: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            by_episode[int(row["episode_id"])].append(float(row[metric]))
        return statistics.fmean(statistics.fmean(values) for values in by_episode.values()) if by_episode else math.nan

    lines = [
        "# KDAgent 三项机制验证结果说明", "",
        "## 执行边界", "- 实验1和实验2已经基于保存日志离线运行，LLM 调用为0。", "- 实验3只完成程序、配对快照和dry-run，未调用LLM。", "",
        "## 实验1 分支冲突",
        "下表先在每个 episode 内平均模型与 run，再对 episode 等权平均；model-run 记录不被当作独立推断样本。", "",
        "| Condition | Method | Records | Episodes | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in config["conditions"]:
        for method in config["fusion_methods"]:
            selected = [r for r in disagreement if r["condition"] == condition and r["fusion_method"] == method]
            lines.append(f"| {condition} | {method} | {len(selected)} | {len({r['episode_id'] for r in selected})} | {episode_mean(selected, 'hit_at_1'):.4f} | {episode_mean(selected, 'hit_at_3'):.4f} | {episode_mean(selected, 'hit_at_5'):.4f} | {episode_mean(selected, 'mrr'):.4f} |")

    recovered = sum(int(r["final_validation_passed"]) for r in repair_needed)
    lines += [
        "", "AP首位不变是融合规则保证的设计性质；策略间Hit@1/MRR差异才是来源选择的诊断结果。条件级排除及原因见 `exp1_status_and_exclusions.csv`。", "",
        "## 实验2 验证修复",
        f"- 300条Evidence记录中，首轮失败{len(repair_needed)}条；最终恢复{recovered}条，未恢复{len(repair_needed) - recovered}条。",
        f"- 修复新增逻辑调用{sum(int(r['extra_logical_calls']) for r in exp2)}次，新增token {sum(int(r['extra_tokens']) for r in exp2)}，累计额外耗时{sum(float(r['extra_elapsed_s']) for r in exp2):.2f}秒。",
        f"- 在修复子集中，Evidence Hit@1从{statistics.fmean(float(r['first_evidence_hit_at_1']) for r in repair_needed):.4f}变为{statistics.fmean(float(r['final_evidence_hit_at_1']) for r in repair_needed):.4f}，Hit@5从{statistics.fmean(float(r['first_evidence_hit_at_5']) for r in repair_needed):.4f}变为{statistics.fmean(float(r['final_evidence_hit_at_5']) for r in repair_needed):.4f}。",
        "- 合规恢复与诊断正确性分别报告；不能把Validator通过直接解释为根因命中。", "",
        "## 实验3 定向反馈对照",
        f"- 首轮失败配对样本N={budget['eligible_first_round_failures']}，两组各最多追加2轮，理论上限{budget['max_logical_generation_calls']}次逻辑生成调用。",
        f"- 历史首轮均值推算的规划参考为{budget['empirical_total_token_planning_reference_at_max_calls']} tokens；配置的completion-only上限为{budget['configured_completion_token_ceiling']} tokens，两者都不是实际账单承诺。",
        "- 未估算金额，因为仓库没有可靠且不可变的逐模型价格表。", "",
        "## 可放入论文的英文段落草稿", "### Placement: Evaluation, Mechanism Analysis",
        "On the saved SWaT branch outputs, we separated source agreement from source conflict and replayed authority-preserving, retrieval-primary, RRF, and Borda fusion under identical generated hypotheses. The authority-preserving rule deterministically retained an evidence-contract-valid primary; diagnostic differences were assessed through paired Hit@1 and MRR changes after averaging model-run observations within each episode. Primary invariance is treated as a design property rather than an empirical accuracy gain, and incomplete or inadmissible branches are reported outside the conflict comparison.", "",
        "### Placement: Evaluation, Validation-Guided Repair",
        f"Across 300 evidence-branch records, {len(repair_needed)} first-round responses failed the evidence contract and {recovered} were recovered within the two permitted repair calls. We report contract recovery separately from ground-truth localization and account for the additional logical calls, tokens, and elapsed time. A paired targeted-feedback versus generic-retry experiment has been prepared from the same saved first-round snapshots but was not executed without explicit API authorization.",
    ]
    (output / "results_summary_zh_and_paper_draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KDAgent unified entry for the three mechanism validations")
    parser.add_argument("command", choices=("audit", "offline", "feedback", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="For the feedback subcommand, actually call the API; default is dry-run")
    parser.add_argument("--allow-llm", action="store_true", help="Paid-call second confirmation gate")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config.resolve())
    output = args.output_dir.resolve()
    logger = setup_logger(output)
    if args.command in {"audit", "all"}:
        audit(config, output, logger)
    exp1: list[dict[str, Any]] = []
    exp2: list[dict[str, Any]] = []
    if args.command in {"offline", "all"}:
        exp1 = experiment_conflict(config, output, logger)
        exp2 = experiment_repair(config, output, logger)
    if args.command in {"feedback", "all"}:
        if args.execute:
            feedback_execute(config, output, logger, args.resume, args.allow_llm)
        else:
            budget = feedback_dry_run(config, output, logger)
            if exp1 and exp2:
                write_results_summary(config, output, exp1, exp2, budget)
    write_json(output / "run_manifest.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "command": args.command, "config": config, "llm_execution_requested": bool(args.execute), "llm_allowed": bool(args.allow_llm), "source_files": [str(CASES_PATH), str(AUDIT_PATH), str(CLEAN_ROOT), str(ROBUST_PATH), str(FUSION_PATH), str(MODEL_CONFIG)]})


if __name__ == "__main__":
    main()
