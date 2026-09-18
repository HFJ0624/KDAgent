"""KDAgent strong-baseline re-evaluation (zero API calls).

This script only reads the already-saved experiment results, recomputes the metrics of all methods in a
unified way, and generates a traceable paired-comparison table. Note in particular: RRF/Borda use only the
Evidence and Retrieval branches; the frozen TA-RCA Top-10 is used only for candidate legality checks and for
ordering on complete ties.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "swat_s2s_raw_window_fixed" / "llm_prompt_cases.jsonl"
V1 = ROOT / "outputs" / "final_experiments_frozen_v1"
V2 = ROOT / "outputs" / "final_experiments_frozen_v2"
LEGACY = ROOT / "outputs" / "final_experiments"
ROBUSTNESS = ROOT / "outputs" / "retrieval_robustness_v1" / "records.jsonl"
OUT = ROOT / "analysis" / "strong_baseline_reevaluation_v1"
MODELS = ("qwen-plus", "qwen-max", "deepseek-v4-flash", "deepseek-v4-pro", "glm-5.2")
RUNS = (1, 2, 3)
CASES = tuple(range(20))
METRICS = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5")

METHOD_LABELS = {
    "direct_llm": "Direct LLM Baseline",
    "only_rag": "Only RAG",
    "only_agent": "Only Agent",
    "serial": "Serial RAG+Agent",
    "kdagent": "KDAgent authority-preserving",
    "retrieval_primary": "Retrieval-primary fusion",
    "rrf": "Standard RRF",
    "borda": "Standard Borda",
    "authority_constrained_rrf": "Authority-constrained RRF",
    "authority_constrained_borda": "Authority-constrained Borda",
}


def normalize(value: Any) -> list[str]:
    """Be compatible with CSV strings, JSON lists, and Python lists; do not guess variable aliases."""
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                values = json.loads(text)
            except json.JSONDecodeError:
                values = text.replace(";", ",").split(",")
        else:
            values = text.replace(";", ",").split(",")
    else:
        values = []
    result: list[str] = []
    for item in values:
        item = str(item).strip().upper().replace(" ", "")
        if item and item not in result:
            result.append(item)
    return result


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("strong_baseline_reevaluation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    output.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "strong_baseline_reevaluation.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_cases() -> dict[int, dict[str, Any]]:
    cases: dict[int, dict[str, Any]] = {}
    with DATA.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                cases[int(item["case_id"])] = item
    if tuple(sorted(cases)) != CASES:
        raise RuntimeError(f"固定 benchmark case 键异常：{sorted(cases)}")
    for case in cases.values():
        gt = normalize((case.get("ground_truth") or {}).get("gt_vars"))
        top10 = normalize(case.get("top10_vars"))
        if not top10:
            top10 = normalize([x.get("var_id") for x in case.get("top_k_variables", [])[:10]])
        if len(top10) != 10 or not gt:
            raise RuntimeError(f"case {case['case_id']} 的 ground truth 或 frozen Top-10 不完整")
        case["_gt"] = gt
        case["_top10"] = top10
    return cases


def method_dir(condition: str, model: str) -> Path:
    """Read according to the existing frozen/legacy directories; copy or overwrite nothing."""
    if condition == "kdagent":
        return V2 / f"dual_branch_fusion_agent_{model}"
    frozen = V1 / f"{condition}_{model}"
    return frozen if frozen.exists() else LEGACY / f"{condition}_{model}"


def load_method_rows(condition: str, model: str) -> dict[tuple[int, int], dict[str, Any]]:
    path = method_dir(condition, model)
    files = list((path / "parsed_results").glob("*_parsed.csv"))
    if len(files) != 1:
        raise RuntimeError(f"{path} parsed CSV 数量异常：{len(files)}")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    with files[0].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["run_id"]), int(row["case_id"]))
            if key in rows:
                raise RuntimeError(f"{path.name} 存在重复键 {key}")
            rows[key] = row
    return rows


def parse_rank(value: Any) -> list[str]:
    return normalize(value)


def load_all_methods(logger: logging.Logger) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    """Load the five top-level methods and check that every method keeps 300 identical paired keys."""
    mapping = {
        "direct_llm": "baseline",
        "only_rag": "only_rag",
        "only_agent": "only_self_refinement_agent",
        "serial": "rag_self_refinement_agent",
        "kdagent": "kdagent",
    }
    data: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    reference: set[tuple[int, int]] | None = None
    for method, condition in mapping.items():
        for model in MODELS:
            rows = load_method_rows(condition, model)
            keys = set(rows)
            if keys != {(run, case) for run in RUNS for case in CASES}:
                raise RuntimeError(f"{method}/{model} 配对键不完整：{len(keys)}")
            reference = reference or keys
            if keys != reference:
                raise RuntimeError(f"{method}/{model} 与其他方法键不一致")
            for (run, case), row in rows.items():
                data[(method, model, run, case)] = row
    logger.info("五类直接方法加载完成：%d records，所有方法键一致", len(data))
    return data


def rank_metrics(ranking: Sequence[str], case: Mapping[str, Any]) -> dict[str, Any]:
    gt = set(case["_gt"])
    rank = next((i + 1 for i, value in enumerate(ranking) if value in gt), None)
    dcg = sum(1 / math.log2(i + 2) for i, value in enumerate(ranking[:5]) if value in gt)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(gt))))
    return {
        "hit_at_1": int(rank is not None and rank <= 1),
        "hit_at_3": int(rank is not None and rank <= 3),
        "hit_at_5": int(rank is not None and rank <= 5),
        "mrr": 1 / rank if rank else 0.0,
        "ndcg_at_5": dcg / ideal if ideal else 0.0,
        "best_gt_rank": rank if rank is not None else "",
        "reachable": int(bool(gt & set(case["_top10"]))),
    }


def legal(rank: Sequence[str], top10: Sequence[str]) -> list[str]:
    allowed = set(top10)
    return list(dict.fromkeys(x for x in rank if x in allowed))


def fuse(evidence: Sequence[str], retrieval: Sequence[str], top10: Sequence[str], method: str) -> list[str]:
    """Implement five offline fusion schemes; TA-RCA only participates in legality checks and complete-score tie-breaks."""
    ev, ret = legal(evidence, top10), legal(retrieval, top10)
    if method == "retrieval_primary":
        return list(dict.fromkeys(ret[:1] + ev + ret[1:]))[:5]
    if method == "authority_preserving":
        return list(dict.fromkeys(ev[:1] + ret + ev[1:]))[:5]
    primary = ev[0] if ev else ""
    remaining = list(dict.fromkeys(ev + ret))
    # Both branches of the standard RRF/Borda participate fully in voting; only the constrained versions
    # pin the admissible Evidence primary to Rank 1 and rank the remaining candidates.
    vote_candidates = [x for x in remaining if not (method.startswith("authority_constrained") and x == primary)]
    scores: dict[str, float] = {x: 0.0 for x in vote_candidates}
    rankings = (ev, ret)
    if method in ("rrf", "authority_constrained_rrf"):
        for ranking in rankings:
            for i, value in enumerate(ranking, 1):
                if value in scores:
                    scores[value] += 1 / (60 + i)
    elif method in ("borda", "authority_constrained_borda"):
        for ranking in rankings:
            for i, value in enumerate(ranking, 1):
                if value in scores:
                    scores[value] += len(ranking) - i + 1
    else:
        raise ValueError(method)
    ta_order = {value: i for i, value in enumerate(top10)}
    ordered = sorted(vote_candidates, key=lambda x: (-scores[x], ta_order.get(x, 10**9), x))
    if method.startswith("authority_constrained") and primary:
        return [primary] + ordered[:4]
    return ordered[:5]


def load_branches(cases: Mapping[int, Mapping[str, Any]], logger: logging.Logger) -> tuple[dict[tuple[str, int, int, str], dict[str, Any]], list[dict[str, Any]]]:
    """Read the final clean Evidence round and the final robustness Retrieval records."""
    branches: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for model in MODELS:
        for path in sorted((V2 / f"dual_branch_fusion_agent_{model}" / "raw_responses").glob("*_run_*.jsonl")):
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
                elif item.get("record_type") == "agent_iteration" and item.get("branch") == "data_evidence":
                    evidence[key].append(item)
                elif item.get("record_type") == "agent_iteration" and item.get("branch") == "rag_knowledge":
                    retrieval[key] = item
            run = int(path.stem.rsplit("_run_", 1)[1])
            for case in CASES:
                key = (run, case)
                summary = summaries.get(key, {})
                final_iteration = int(summary.get("final_iteration") or 1)
                finals = [x for x in evidence.get(key, []) if int(x.get("iteration") or 1) == final_iteration]
                ev = parse_rank((finals[-1] if finals else {}).get("parsed_response", {}).get("predicted_root_causes"))
                rt = parse_rank(retrieval.get(key, {}).get("parsed_response", {}).get("predicted_root_causes"))
                missing = ",".join(x for x, present in (("evidence", bool(ev)), ("retrieval", bool(rt))) if not present)
                branches[(model, run, case, "clean")] = {"evidence": ev, "retrieval": rt, "missing": missing, "source": str(path)}
                if missing:
                    audit.append({"model": model, "run_id": run, "case_id": case, "condition": "clean", "missing": missing, "source": str(path)})
    latest: dict[str, dict[str, Any]] = {}
    with ROBUSTNESS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                latest[item["record_key"]] = item
    for model in MODELS:
        for run in RUNS:
            for case in CASES:
                clean = branches[(model, run, case, "clean")]
                for condition in ("random_retrieval", "stage_mismatched_retrieval", "low_relevance_retrieval"):
                    key = f"{model}|{run}|{case}|{condition}|retrieval_branch"
                    item = latest.get(key)
                    parsed = (item or {}).get("parsed_response") or {}
                    rt = parse_rank(parsed.get("predicted_root_causes"))
                    missing = "retrieval" if not item or not rt else ""
                    branches[(model, run, case, condition)] = {"evidence": clean["evidence"], "retrieval": rt, "missing": missing, "source": key,
                        "retrieved_chunk_ids": json.dumps([x.get("id") or x.get("metadata", {}).get("chunk_id", "") for x in (item or {}).get("retrieved_contexts", [])], ensure_ascii=False),
                        "retrieved_sources": json.dumps([x.get("source", "") for x in (item or {}).get("retrieved_contexts", [])], ensure_ascii=False),
                        "retrieved_similarities": json.dumps([x.get("similarity", x.get("score", "")) for x in (item or {}).get("retrieved_contexts", [])], ensure_ascii=False),
                        "retrieval_quality": json.dumps({x: (item or {}).get(x) for x in ("api_success", "finish_reason", "fallback")}, ensure_ascii=False)}
                    if missing:
                        audit.append({"model": model, "run_id": run, "case_id": case, "condition": condition, "missing": missing, "source": key})
    logger.info("分支加载完成：%d branch records；缺失位置=%d；robustness latest=%d", len(branches), len(audit), len(latest))
    return branches, audit


def metric_summary(rows: Sequence[Mapping[str, Any]], scope: str, group: str) -> list[dict[str, Any]]:
    selected = [x for x in rows if x["scope"] == scope]
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        if group == "model_method":
            key = (row["model"], row["method"])
        elif group == "condition_method":
            key = (row["condition"], row["method"])
        else:
            key = row[group]
        groups[key].append(row)
    result = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        valid = [x for x in items if not x["missing"]]
        row = {"group": key, "scope": scope, "n_records": len(valid), "miss_count": len(items) - len(valid),
            **{m: round(sum(float(x[m]) for x in valid) / len(valid), 6) if valid else "" for m in METRICS},
            "top1_hits": sum(int(x["hit_at_1"]) for x in valid), "top3_hits": sum(int(x["hit_at_3"]) for x in valid), "top5_hits": sum(int(x["hit_at_5"]) for x in valid),
            "mean_best_gt_rank_hit_only": round(statistics.mean([int(x["best_gt_rank"]) for x in valid if x["best_gt_rank"] != ""]), 6) if any(x["best_gt_rank"] != "" for x in valid) else "",
            "median_best_gt_rank_hit_only": statistics.median([int(x["best_gt_rank"]) for x in valid if x["best_gt_rank"] != ""]) if any(x["best_gt_rank"] != "" for x in valid) else "",
            "evidence_primary_overwrite_count": sum(int(x["evidence_primary_overwritten"]) for x in valid),
            "evidence_primary_overwrite_rate": round(sum(int(x["evidence_primary_overwritten"]) for x in valid) / len(valid), 6) if valid else ""}
        if isinstance(key, tuple):
            row["model"] = key[0]
            row["method"] = key[1]
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-API strong-baseline reevaluation")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    logger = setup_logger(args.output_dir)
    cases = load_cases()
    direct = load_all_methods(logger)
    branches, audit = load_branches(cases, logger)
    rows: list[dict[str, Any]] = []
    all_methods = tuple(METHOD_LABELS)
    for model in MODELS:
        for run in RUNS:
            for case_id in CASES:
                case = cases[case_id]
                for condition in ("clean", "random_retrieval", "stage_mismatched_retrieval", "low_relevance_retrieval"):
                    branch = branches[(model, run, case_id, condition)]
                    for method in all_methods:
                        missing = branch["missing"] if method not in ("direct_llm", "only_rag", "only_agent", "serial", "kdagent") else ""
                        if condition != "clean" and method in ("direct_llm", "only_rag", "only_agent", "serial"):
                            continue
                        if method in ("direct_llm", "only_rag", "only_agent", "serial", "kdagent"):
                            if condition == "clean":
                                row = direct[(method, model, run, case_id)]
                                ranking = parse_rank(row.get("predicted_root_causes"))
                            else:
                                ranking = fuse(branch["evidence"], branch["retrieval"], case["_top10"], "authority_preserving")
                        else:
                            ranking = fuse(branch["evidence"], branch["retrieval"], case["_top10"], method)
                        values = rank_metrics(ranking, case) if not missing else {m: "" for m in METRICS} | {"best_gt_rank": "", "reachable": int(bool(set(case["_gt"]) & set(case["_top10"]))) }
                        rows.append({"model": model, "run_id": run, "case_id": case_id, "condition": condition, "method": method,
                            "method_label": METHOD_LABELS[method], "predicted_top5": ",".join(ranking), "missing": missing,
                            "evidence_primary": branch["evidence"][0] if branch["evidence"] else "", "retrieval_primary": branch["retrieval"][0] if branch["retrieval"] else "",
                            "evidence_primary_overwritten": int(bool(branch["evidence"] and ranking and ranking[0] != branch["evidence"][0])),
                            "retrieved_chunk_ids": branch.get("retrieved_chunk_ids", ""), "retrieved_sources": branch.get("retrieved_sources", ""),
                            "retrieved_similarities": branch.get("retrieved_similarities", ""), "retrieval_quality": branch.get("retrieval_quality", ""), **values})
    write_csv(args.output_dir / "record_level_results.csv", rows)
    write_csv(args.output_dir / "data_completeness_audit.csv", audit)

    # Only the five direct methods from the clean runs and the five branch-derived fusion methods enter the main comparison table.
    clean = [x for x in rows if x["condition"] == "clean"]
    # The full panel contains all records; reachable is a post-hoc subset of the same records. The two must not
    # be made mutually exclusive, otherwise the full panel would incorrectly drop the 13 reachable cases.
    scoped_clean = [{**row, "scope": "full"} for row in clean]
    scoped_clean += [{**row, "scope": "reachable"} for row in clean if row["reachable"]]
    summary = []
    for scope in ("full", "reachable"):
        for row in metric_summary(scoped_clean, scope, "method"):
            row["scope_type"] = "full_20" if scope == "full" else "reachable_13"
            summary.append(row)
    write_csv(args.output_dir / "main_comparison_table.csv", summary)
    per_backbone = []
    for scope in ("full", "reachable"):
        for row in metric_summary(scoped_clean, scope, "model_method"):
            row["scope_type"] = "full_20" if scope == "full" else "reachable_13"
            per_backbone.append(row)
    write_csv(args.output_dir / "per_backbone_comparison.csv", per_backbone)

    # Five-model macro-average: first obtain each backbone's results, then average the backbones with equal weight.
    macro_average = []
    for scope in ("full", "reachable"):
        for method in all_methods:
            items = [x for x in per_backbone if x.get("scope") == scope and x.get("method") == method]
            if not items:
                continue
            macro_average.append({"method": method, "method_label": METHOD_LABELS[method], "scope": scope,
                "n_backbones": len(items), "mean_n_records": round(statistics.mean(float(x["n_records"]) for x in items), 6),
                "mean_miss_count": round(statistics.mean(float(x["miss_count"]) for x in items), 6),
                **{metric: round(statistics.mean(float(x[metric]) for x in items), 6) for metric in METRICS},
                **{f"mean_{field}": round(statistics.mean(float(x[field]) for x in items), 6) for field in ("top1_hits", "top3_hits", "top5_hits")}})
    write_csv(args.output_dir / "macro_average_comparison.csv", macro_average)

    # Export KDAgent's best backbone separately, to avoid mistaking the macro-average for the best model.
    best_backbone = []
    for scope in ("full", "reachable"):
        candidates = [x for x in per_backbone if x.get("scope") == scope and x.get("method") == "kdagent"]
        for metric in METRICS:
            best_value = max(float(x[metric]) for x in candidates)
            for winner in [x for x in candidates if float(x[metric]) == best_value]:
                best_backbone.append({"method": "kdagent", "scope": scope, "selection_metric": metric,
                    "best_model": winner["model"], "metric_value": winner[metric],
                    **{field: winner[field] for field in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5", "top1_hits", "top3_hits", "top5_hits")}})
    write_csv(args.output_dir / "kdagent_best_backbone.csv", best_backbone)
    # deepseek-v4-pro is overall best on KDAgent's main ranking metrics; export the ten-method
    # comparison for this backbone separately, so the paper's main table does not treat the macro-average as the best result.
    best_model = "deepseek-v4-pro"
    best_model_rows = [x for x in per_backbone if x.get("model") == best_model]
    write_csv(args.output_dir / "best_backbone_main_comparison.csv", best_model_rows)

    # At the case level, first average model/run, then compute KDAgent's paired difference relative to each method.
    case_rows = defaultdict(lambda: defaultdict(list))
    for row in clean:
        if not row["missing"]:
            case_rows[(row["case_id"], row["method"])]["metrics"].append(row)
    paired = []
    for case_id in CASES:
        for method in all_methods:
            a = case_rows[(case_id, "kdagent")]["metrics"]
            b = case_rows[(case_id, method)]["metrics"]
            if method == "kdagent" or not a or not b:
                continue
            am = {m: sum(float(x[m]) for x in a) / len(a) for m in METRICS}
            bm = {m: sum(float(x[m]) for x in b) / len(b) for m in METRICS}
            paired.append({"case_id": case_id, "baseline": method, "baseline_label": METHOD_LABELS[method],
                **{f"kdagent_minus_{m}": round(am[m] - bm[m], 6) for m in METRICS},
                **{f"kdagent_{m}": round(am[m], 6) for m in METRICS}, **{f"baseline_{m}": round(bm[m], 6) for m in METRICS},
                "hit5_relation": "win" if am["hit_at_5"] > bm["hit_at_5"] else "loss" if am["hit_at_5"] < bm["hit_at_5"] else "tie"})
    write_csv(args.output_dir / "paired_case_differences.csv", paired)

    constrained_methods = ("kdagent", "rrf", "borda", "authority_constrained_rrf", "authority_constrained_borda")
    constrained = [x for x in scoped_clean if x["method"] in constrained_methods]
    write_csv(args.output_dir / "authority_constrained_comparison.csv", metric_summary(constrained, "full", "method"))
    scoped_robust = [{**x, "scope": "full"} for x in rows if x["condition"] != "clean"]
    scoped_robust += [{**x, "scope": "reachable"} for x in rows if x["condition"] != "clean" and x["reachable"]]
    write_csv(args.output_dir / "robustness_fusion_summary.csv", [r for scope in ("full", "reachable") for r in metric_summary(scoped_robust, scope, "condition_method")])

    # Summarize each baseline's case-level difference, relative change, and W/T/L versus KDAgent.
    paired_summary = []
    for scope in ("full", "reachable"):
        scope_cases = set(CASES) if scope == "full" else {case_id for case_id, item in cases.items() if set(item["_gt"]) & set(item["_top10"])}
        for method in (m for m in all_methods if m != "kdagent"):
            items = [x for x in paired if x["baseline"] == method and int(x["case_id"]) in scope_cases]
            result = {"scope": scope, "baseline": method, "baseline_label": METHOD_LABELS[method], "n_cases": len(items)}
            for metric in METRICS:
                diffs = [float(x[f"kdagent_minus_{metric}"]) for x in items]
                nonzero = [x for x in items if float(x[f"baseline_{metric}"]) != 0]
                result[f"mean_kdagent_minus_{metric}"] = round(statistics.mean(diffs), 6) if diffs else ""
                result[f"relative_improvement_{metric}"] = round(statistics.mean([(float(x[f"kdagent_{metric}"]) - float(x[f"baseline_{metric}"])) / abs(float(x[f"baseline_{metric}"])) for x in nonzero]), 6) if nonzero else ""
                result[f"{metric}_wins"] = sum(float(x[f"kdagent_{metric}"]) > float(x[f"baseline_{metric}"]) for x in items)
                result[f"{metric}_ties"] = sum(float(x[f"kdagent_{metric}"]) == float(x[f"baseline_{metric}"]) for x in items)
                result[f"{metric}_losses"] = sum(float(x[f"kdagent_{metric}"]) < float(x[f"baseline_{metric}"]) for x in items)
            paired_summary.append(result)
    write_csv(args.output_dir / "paired_summary.csv", paired_summary)

    # The Markdown only references tables actually produced by this script, so paper authors do not misread old results.
    md = ["# Strong-Baseline Re-evaluation", "", "本分析只读取已有结果，LLM API calls = 0。所有模型、run、episode 均保留；缺失分支进入审计表，不用最终 Top-5 反推。", "",
          "## 方法与口径", "- 固定 20 SWaT cases、5 models、3 runs；直接方法使用 clean 原始结果。", "- RRF 固定 k=60，Borda 使用每个 branch 的 `m_s-rank+1`。TA-RCA Top-10 只用于合法性和完全同分 tie-break。", "- Best-GT Rank 只对命中记录统计，并报告 miss_count。", "- paired analysis 先按 case 对 model/run 求平均，case 是独立单位。", "",
          "## 输出文件", "- `main_comparison_table.csv`: clean 全 20 cases、13 reachable cases 的主比较。", "- `per_backbone_comparison.csv`: 各 backbone 的 clean 汇总。", "- `macro_average_comparison.csv`: 五个 backbone 的等权 macro-average。", "- `kdagent_best_backbone.csv`: KDAgent 按每个指标筛选的最佳 backbone。", "- `paired_case_differences.csv`: KDAgent 与各方法的 case-level 配对差值。", "- `paired_summary.csv`: 各指标的 case-level mean difference、relative improvement 和 W/T/L。", "- `authority_constrained_comparison.csv`: clean constrained RRF/Borda 对照。", "- `robustness_fusion_summary.csv`: 三种 retrieval 条件按 condition×method 的融合结果。", "- `data_completeness_audit.csv`: 缺失 branch 和来源审计。", "", "## 严格可支持的结论", "1. 只有在主表和配对表同时支持时，才能声称 KDAgent 在对应指标上优于 baseline。", "2. Authority-constrained 方法把 Evidence primary 固定在第 1 位，因此其首位变化应与 KDAgent 的 admissible Evidence primary 一致。", "3. generic RRF/Borda 的结果不能被表述为自动优于 source-aware authority-preserving fusion。", "4. 任意缺失、截断或 invalid JSON 必须结合审计表说明，不能删除或静默修复。", ""]
    clean_index = {(x["group"], x["scope"]): x for x in summary}
    def clean_value(method: str, field: str) -> str:
        value = clean_index[(method, "full")].get(field, "")
        return f"{float(value):.4f}" if value != "" else "NA"
    md += [
        "## A. Main comparison table",
        f"Clean full-panel key values: KDAgent Hit@1/3/5={clean_value('kdagent', 'hit_at_1')}/{clean_value('kdagent', 'hit_at_3')}/{clean_value('kdagent', 'hit_at_5')}, MRR={clean_value('kdagent', 'mrr')}, NDCG@5={clean_value('kdagent', 'ndcg_at_5')}; Direct LLM Hit@1/5={clean_value('direct_llm', 'hit_at_1')}/{clean_value('direct_llm', 'hit_at_5')}; RRF Hit@1/5={clean_value('rrf', 'hit_at_1')}/{clean_value('rrf', 'hit_at_5')}; Borda Hit@1/5={clean_value('borda', 'hit_at_1')}/{clean_value('borda', 'hit_at_5')}. Full values and hit counts are in `main_comparison_table.csv`.",
        "",
        "## B. Per-backbone comparison table",
        "`per_backbone_comparison.csv` reports every backbone × method × scope, including valid record counts and misses. `macro_average_comparison.csv` reports equal-backbone macro averages, while `kdagent_best_backbone.csv` reports the best KDAgent backbone per metric.",
        "",
        "## C. KDAgent vs each baseline paired differences",
        "`paired_case_differences.csv` contains one row per episode after averaging model/run records within the episode. `paired_summary.csv` contains absolute differences, relative changes and metric-wise win/tie/loss counts.",
        "",
        "## D. Authority-constrained RRF/Borda comparison",
        f"On clean valid records, constrained RRF and constrained Borda preserve the Evidence primary by construction and have Hit@1={clean_value('authority_constrained_rrf', 'hit_at_1')} and {clean_value('authority_constrained_borda', 'hit_at_1')}. See `authority_constrained_comparison.csv` and `robustness_fusion_summary.csv`.",
        "",
        "## E. Whether any previous paper numbers need correction",
        "RRF/Borda values produced by the old `scripts/paper_ready_analysis.py` require correction because that implementation included TA-RCA Top-10 as a third voting branch. The present results use only Evidence and Retrieval. The one missing clean Retrieval branch must also be reported rather than imputed.",
        "",
        "## F. Strongest claims strictly supported by the rerun",
        "The rerun supports a transparent comparison of source-aware and generic late fusion under identical saved model/run/episode keys. It does not support a blanket claim that KDAgent is accuracy-optimal; the paired tables must determine which metrics and episodes improve. The robustness results support discussing explicit source-authority control and an accuracy-robustness trade-off.",
        "",
    ]
    (args.output_dir / "paper_ready_strong_baseline_summary.md").write_text("\n".join(md), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps({"llm_calls": 0, "models": MODELS, "runs": RUNS, "cases": CASES, "methods": METHOD_LABELS, "rrf_k": 60, "audit_rows": len(audit)}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("完成 strong-baseline 离线复评：records=%d，audit=%d，输出=%s", len(rows), len(audit), args.output_dir)


if __name__ == "__main__":
    main()
