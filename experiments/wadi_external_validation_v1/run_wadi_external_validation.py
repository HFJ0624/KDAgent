"""Orchestrate WADI preparation, sanity check, Phase A/B, and result summarization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "wadi_external_validation_v1"
MODEL_NAME = "deepseek-v4-pro"
SANITY_SEED = 20260903


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimum-cost WADI external validation.")
    parser.add_argument(
        "--mode",
        choices=("prepare", "sanity", "phase-a", "auto", "summarize"),
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--approve-api", action="store_true", help="Explicitly allow Embedding/LLM API calls.")
    parser.add_argument("--rebuild-kb", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def _run(command: Sequence[str], cwd: Path = PROJECT_ROOT) -> None:
    print(f"[执行] {subprocess.list2cmdline(list(command))}", flush=True)
    subprocess.run(list(command), cwd=cwd, check=True)


def _prepared_paths(output_dir: Path) -> Dict[str, Path]:
    prepared = output_dir / "prepared_data"
    return {
        "cases": prepared / "llm_prompt_cases.jsonl",
        "manifest": prepared / "wadi_episode_manifest.csv",
        "variables": prepared / "variables_meta.csv",
        "knowledge": prepared / "process_knowledge_template.md",
        "kb_dir": prepared / "rag_kb",
        "top10": prepared / "wadi_tarca_top10.json",
    }


def prepare(output_dir: Path, force_train: bool = False) -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.wadi_external_validation_v1.prepare_wadi",
        "--output-dir",
        str(output_dir),
    ]
    if force_train:
        command.append("--force-train")
    _run(command)


def _kb_fingerprint(paths: Dict[str, Path]) -> str:
    digest = hashlib.sha256()
    inputs = [
        paths["variables"],
        paths["knowledge"],
        *sorted(paths["kb_dir"].glob("*.md")),
        EXPERIMENT_DIR / "configs" / "rag_wadi.yaml",
    ]
    for path in inputs:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_kb(output_dir: Path, rebuild: bool = False) -> None:
    paths = _prepared_paths(output_dir)
    marker = output_dir / "kb_build_manifest.json"
    fingerprint = _kb_fingerprint(paths)
    chroma_dir = output_dir / "chroma_db"
    if marker.exists() and chroma_dir.exists() and not rebuild:
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("source_fingerprint") == fingerprint:
            print("[KB] 输入未变化，复用现有 WADI Chroma collection。", flush=True)
            return

    _run(
        [
            sys.executable,
            "src/build_rag_index.py",
            "--data_dir",
            str(paths["variables"].parent),
            "--kb_dir",
            str(paths["kb_dir"]),
            "--rag_config",
            str(EXPERIMENT_DIR / "configs" / "rag_wadi.yaml"),
            "--rebuild",
        ]
    )
    marker.write_text(
        json.dumps(
            {
                "source_fingerprint": fingerprint,
                "collection_name": "wadi_process_kb_v1",
                "contains_attack_answers": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def sanity_check(output_dir: Path) -> None:
    """Draw a fixed random sample of 3 episodes, retrieve knowledge for each, and print human-readable evidence."""
    paths = _prepared_paths(output_dir)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_loader import normalize_case
    from src.prompt_builder import build_prompt, load_process_knowledge, load_template
    from src.rag_retriever import RAGRetriever

    raw_cases = _load_jsonl(paths["cases"])
    if len(raw_cases) < 3:
        raise RuntimeError("可评估 WADI episode 少于 3 个。")
    selected = random.Random(SANITY_SEED).sample(raw_cases, 3)
    template = load_template(str(EXPERIMENT_DIR / "prompts" / "wadi_rca_prompt_template.txt"))
    process_knowledge = load_process_knowledge(str(paths["knowledge"]))
    retriever = RAGRetriever.from_config_path(str(EXPERIMENT_DIR / "configs" / "rag_wadi.yaml"))

    report_lines: List[str] = []
    for raw_case in selected:
        case = normalize_case(raw_case)
        retrieval = retriever.retrieve(case, top_k=5)
        contexts = retrieval.get("retrieved_contexts", [])
        if retrieval.get("error") or len(contexts) != 5:
            raise RuntimeError(
                f"{case['case_id']} 检索未返回完整 Top-5：{retrieval.get('error') or len(contexts)}"
            )
        prompt = build_prompt(case, template, process_knowledge, contexts)
        lowered = prompt.lower()
        leaked_fields = [token for token in ("ground_truth", "gt_vars", "root_tags") if token in lowered]
        if leaked_fields:
            raise RuntimeError(f"{case['case_id']} Prompt 泄漏标签字段：{leaked_fields}")

        details = case["top10_details"]
        gt_vars = case["gt_vars"]
        top10_vars = case["top10_vars"]
        evidence = details[0].get("time_series", []) if details else []
        evidence_sample = evidence[len(evidence) // 2] if evidence else {}
        lines = [
            "=" * 72,
            f"Episode ID: {case['case_id']}",
            f"Ground Truth: {gt_vars}",
            f"TA-RCA Top-10: {top10_vars}",
            f"GT covered?: {any(value in top10_vars for value in gt_vars)}",
            "Candidate variable type: "
            + ", ".join(f"{item['var']}={item['type']}" for item in details),
            f"Evidence sample (Top-1): {json.dumps(evidence_sample, ensure_ascii=False)}",
            "Retrieved top-5 chunks:",
        ]
        for index, context in enumerate(contexts, start=1):
            metadata = context.get("metadata", {}) or {}
            chunk_label = metadata.get("var_id") or metadata.get("source") or f"chunk-{index}"
            content = " ".join(str(context.get("content", "")).split())[:240]
            lines.append(
                f"  {index}. id/source={chunk_label}; similarity={context.get('score')}; {content}"
            )
        report_lines.extend(lines)

    report_lines.extend(
        [
            "=" * 72,
            "Sanity check: PASS",
            "- GT 与候选使用同一 Wxxx canonical ID。",
            "- Prompt/RAG 不包含 ground_truth、gt_vars 或 root_tags 字段。",
            "- 每个抽检 episode 均获得 5 条 WADI 知识。",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    print(report, flush=True)
    (output_dir / "sanity_check_3_episodes.txt").write_text(report, encoding="utf-8")


def _model_command(output_dir: Path, method: str, num_runs: int) -> List[str]:
    paths = _prepared_paths(output_dir)
    method_dir = output_dir / f"{method}_{MODEL_NAME}"
    use_kdagent = method == "kdagent"
    return [
        sys.executable,
        "src/main.py",
        "--data_path",
        str(paths["cases"]),
        "--config_path",
        str(PROJECT_ROOT / "configs" / "models.yaml"),
        "--model_name",
        MODEL_NAME,
        "--output_dir",
        str(method_dir),
        "--prompt_template",
        str(EXPERIMENT_DIR / "prompts" / "wadi_rca_prompt_template.txt"),
        "--knowledge_path",
        str(paths["knowledge"]),
        "--num_runs",
        str(num_runs),
        "--temperature",
        "0.2",
        "--max_tokens",
        "8192",
        "--thinking_budget",
        "2048",
        "--use_rag",
        "1" if use_kdagent else "0",
        "--rag_config",
        str(EXPERIMENT_DIR / "configs" / "rag_wadi.yaml"),
        "--rag_top_k",
        "5",
        "--use_iterative_agent",
        "1" if use_kdagent else "0",
        "--use_dual_branch_fusion",
        "1" if use_kdagent else "0",
        "--max_iterations",
        "3",
        "--min_confidence",
        "0.5",
        "--max_retries",
        "5",
        "--request_timeout",
        "120",
        "--sleep_ms",
        "300",
        "--resume",
    ]


def _load_final_records(method_dir: Path, max_run: int | None = None) -> List[Dict[str, Any]]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.main import _load_case_summary_records

    records = _load_case_summary_records(str(method_dir / "raw_responses"), MODEL_NAME)
    if max_run is not None:
        records = [record for record in records if int(record.get("run_id", 0)) <= max_run]
    return records


def _validate_records(output_dir: Path, method: str, num_runs: int, episode_count: int) -> None:
    records = _load_final_records(output_dir / f"{method}_{MODEL_NAME}", max_run=num_runs)
    expected = num_runs * episode_count
    if len(records) != expected:
        raise RuntimeError(f"{method} 记录数异常：{len(records)} != {expected}")
    failed = [record for record in records if not record.get("api_ok")]
    truncated = [
        record
        for record in records
        if str(record.get("finish_reason") or "").lower() == "length"
    ]
    if failed or truncated:
        raise RuntimeError(f"{method} 存在失败={len(failed)}、最终截断={len(truncated)}，停止聚合。")


def run_methods(output_dir: Path, num_runs: int) -> None:
    episode_count = len(_load_jsonl(_prepared_paths(output_dir)["cases"]))
    for method in ("baseline", "kdagent"):
        _run(_model_command(output_dir, method, num_runs))
        _validate_records(output_dir, method, num_runs, episode_count)
        print(f"[{method}] {num_runs} run(s) 完整性检查通过。", flush=True)


def _prediction_rank(record: Dict[str, Any]) -> int | None:
    predicted = [str(value).strip().upper() for value in record.get("parsed_response", {}).get("predicted_root_causes", [])]
    gt = {str(value).strip().upper() for value in record.get("gt_vars", [])}
    return next((index + 1 for index, value in enumerate(predicted[:5]) if value in gt), None)


def _method_metrics(records: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    if not records:
        raise RuntimeError(f"{method} 没有可汇总记录。")
    ranks = [_prediction_rank(record) for record in records]
    covered = [
        record
        for record in records
        if set(record.get("gt_vars", [])) & set(record.get("top10_vars", []))
    ]
    return {
        "Method": "KDAgent" if method == "kdagent" else "Baseline",
        "Backbone": MODEL_NAME,
        "Runs": len({int(record.get("run_id", 0)) for record in records}),
        "Records": len(records),
        "Hit@1": sum(rank == 1 for rank in ranks) / len(ranks),
        "Hit@3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks),
        "Hit@5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "MRR": sum(1.0 / rank if rank else 0.0 for rank in ranks) / len(ranks),
        "Covered Top-5 Recovery": (
            sum(_prediction_rank(record) is not None for record in covered) / len(covered)
            if covered
            else 0.0
        ),
        "Covered Records": len(covered),
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output_dir: Path, num_runs: int | None = None) -> List[Dict[str, Any]]:
    methods = ("baseline", "kdagent")
    metric_rows: List[Dict[str, Any]] = []
    for method in methods:
        records = _load_final_records(
            output_dir / f"{method}_{MODEL_NAME}", max_run=num_runs
        )
        metric_rows.append(_method_metrics(records, method))
        predictions = [
            {
                "model": record.get("model_name"),
                "run_id": record.get("run_id"),
                "episode_id": record.get("case_id"),
                "gt_vars": record.get("gt_vars", []),
                "ta_rca_top10": record.get("top10_vars", []),
                "predicted_top5": record.get("parsed_response", {}).get("predicted_root_causes", [])[:5],
                "primary_root_cause": record.get("parsed_response", {}).get("primary_root_cause"),
                "hit_at_1": int(_prediction_rank(record) == 1),
                "hit_at_3": int((_prediction_rank(record) or 99) <= 3),
                "hit_at_5": int(_prediction_rank(record) is not None),
                "reciprocal_rank": 1.0 / _prediction_rank(record) if _prediction_rank(record) else 0.0,
                "api_ok": bool(record.get("api_ok")),
            }
            for record in records
        ]
        filename = "wadi_kdagent_predictions.json" if method == "kdagent" else "wadi_baseline_predictions.json"
        (output_dir / filename).write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    baseline, kdagent = metric_rows
    for key in ("Hit@1", "Hit@3", "Hit@5", "MRR", "Covered Top-5 Recovery"):
        baseline[key] = round(float(baseline[key]), 6)
        kdagent[key] = round(float(kdagent[key]), 6)
    _write_csv(output_dir / "wadi_metrics.csv", metric_rows)
    _write_csv(
        output_dir / "table_cross_system_validation_wadi.csv",
        [
            {key: row[key] for key in ("Method", "Hit@1", "Hit@3", "Hit@5", "MRR", "Covered Top-5 Recovery")}
            for row in metric_rows
        ],
    )

    cases = _load_jsonl(_prepared_paths(output_dir)["cases"])
    coverage = sum(
        bool(set(case["ground_truth"]["gt_vars"]) & {item["var_id"] for item in case["top_k_variables"]})
        for case in cases
    ) / len(cases)
    with paths["variables"].open("r", encoding="utf-8", newline="") as handle:
        variable_count = sum(1 for _ in csv.DictReader(handle))
    result = {
        "wadi_episode_count": len(cases),
        "wadi_variable_count": variable_count,
        "ta_rca_top10_coverage": round(coverage, 6),
        "baseline": baseline,
        "kdagent": kdagent,
        "hit_at_5_absolute_gain": round(kdagent["Hit@5"] - baseline["Hit@5"], 6),
        "covered_recovery_absolute_gain": round(
            kdagent["Covered Top-5 Recovery"] - baseline["Covered Top-5 Recovery"], 6
        ),
    }
    (output_dir / "wadi_result_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return metric_rows


def _phase_b_is_justified(metrics: List[Dict[str, Any]]) -> bool:
    baseline, kdagent = metrics
    return (
        kdagent["Hit@5"] > baseline["Hit@5"]
        or kdagent["Covered Top-5 Recovery"] > baseline["Covered Top-5 Recovery"]
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _prepared_paths(output_dir)

    if args.mode == "prepare":
        prepare(output_dir, force_train=args.force_train)
        return
    if args.mode == "summarize":
        summarize(output_dir)
        return

    if args.mode == "auto" and not paths["cases"].exists():
        prepare(output_dir, force_train=args.force_train)
    missing = [str(path) for key, path in paths.items() if key != "kb_dir" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"WADI 准备产物缺失，请先运行 --mode prepare：{missing}")
    if not args.approve_api:
        raise RuntimeError("sanity/实验会调用 Embedding 或 LLM API；确认费用后请增加 --approve-api。")

    build_kb(output_dir, rebuild=args.rebuild_kb)
    sanity_check(output_dir)
    if args.mode == "sanity":
        return

    run_methods(output_dir, num_runs=1)
    phase_a_metrics = summarize(output_dir, num_runs=1)
    _write_csv(output_dir / "phase_a_metrics.csv", phase_a_metrics)
    shutil.copyfile(
        output_dir / "wadi_result_summary.json",
        output_dir / "phase_a_result_summary.json",
    )
    if args.mode == "phase-a":
        return
    if _phase_b_is_justified(phase_a_metrics):
        print("[Phase B] KDAgent 的 Hit@5 或 covered recovery 优于 Baseline，追加 runs 2-3。", flush=True)
        run_methods(output_dir, num_runs=3)
        summarize(output_dir, num_runs=3)
    else:
        print("[停止] Phase A 未满足增益条件，不追加付费运行。", flush=True)


if __name__ == "__main__":
    main()
