"""Run the official RCA ablation experiments for five models and the explicitly selected v2 fusion experiments.

By default it runs 5 models x 4 conditions, repeating each case 3 times. Each result group is
written independently to ``outputs/final_experiments_frozen_v1/{condition}_{model}/`` without
overwriting the older debug results from earlier runs. Use ``--dry-run`` to only inspect and
print the full commands without calling the API.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_experiments_frozen_v1"

MODELS = (
    "qwen-plus",
    "qwen-max",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5.2",
)

# The four conditions form the paper's complete ablation matrix. Apart from the RAG and Agent
# switches, all models share the exact same data, prompt, temperature, repeats, and retrieval count.
CONDITIONS = (
    ("baseline", 0, 0, 0),
    ("only_rag", 1, 0, 0),
    ("only_self_refinement_agent", 0, 1, 0),
    ("rag_self_refinement_agent", 1, 1, 0),
    ("dual_branch_fusion_agent", 1, 1, 1),
)
DEFAULT_CONDITIONS = tuple(condition[0] for condition in CONDITIONS[:4])

# Tuned experiment parameters shared by every ablation group with the Agent/RAG conditions.
# Kept as named constants (instead of inline magic numbers) so the runtime parameters
# stay identical across conditions and are self-documenting.
DEFAULT_NUM_RUNS = 3          # repeats per case
DEFAULT_TEMPERATURE = 0.2     # sampling temperature
DEFAULT_MAX_TOKENS = 8192     # max generated tokens per request
DEFAULT_RAG_TOP_K = 5         # retrieval count for the RAG knowledge branch
DEFAULT_MAX_ITERATIONS = 3    # self-refinement agent rounds
DEFAULT_MIN_CONFIDENCE = 0.5  # agent validation confidence threshold
# The official SWaT dataset always contains exactly this many cases.
OFFICIAL_CASE_COUNT = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official RCA ablation matrix.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Run one or more models serially, in the given order.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=[condition[0] for condition in CONDITIONS],
        default=list(DEFAULT_CONDITIONS),
        help="Select one or more experiment conditions to run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root output directory for experiment groups; use a separate dir for preflight to avoid polluting official results.",
    )
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS, help="Number of repeated runs per case.")
    parser.add_argument("--max-cases", type=int, default=-1, help="Maximum number of cases to run; -1 means all.")
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Run only the given case IDs, in the given order; useful for worst-case preflight.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Override sampling temperature for the selected models.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Override max output tokens for the selected models.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing final-aggregate records.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands for the selected combos; do not call the model API.")
    args = parser.parse_args(argv)
    if len(args.models) != len(set(args.models)):
        parser.error("--models cannot contain the same model more than once.")
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("--conditions cannot contain the same condition more than once.")
    if args.case_ids and len(args.case_ids) != len(set(args.case_ids)):
        parser.error("--case-ids cannot contain the same case more than once.")
    return args


def build_command(
    model: str,
    output_dir: Path,
    use_rag: int,
    use_agent: int,
    args: argparse.Namespace,
    use_dual_branch_fusion: int = 0,
) -> list[str]:
    """Build the command for a single experiment group, ensuring all ablation groups share the same set of common parameters."""
    command = [
        sys.executable,
        "src/main.py",
        "--data_path",
        str(PROJECT_ROOT / "data" / "swat_s2s_raw_window_fixed" / "llm_prompt_cases.jsonl"),
        "--config_path",
        str(PROJECT_ROOT / "configs" / "models.yaml"),
        "--model_name",
        model,
        "--output_dir",
        str(output_dir),
        "--num_runs",
        str(args.num_runs),
        "--max_cases",
        str(args.max_cases),
        "--temperature",
        str(args.temperature),
        "--max_tokens",
        str(args.max_tokens),
        "--use_rag",
        str(use_rag),
        "--rag_config",
        str(PROJECT_ROOT / "configs" / "rag.yaml"),
        "--rag_top_k",
        str(DEFAULT_RAG_TOP_K),
        "--use_iterative_agent",
        str(use_agent),
        "--use_dual_branch_fusion",
        str(use_dual_branch_fusion),
        "--max_iterations",
        str(DEFAULT_MAX_ITERATIONS),
        "--min_confidence",
        str(DEFAULT_MIN_CONFIDENCE),
    ]
    if args.resume:
        command.append("--resume")
    if args.case_ids:
        command.extend(["--case_ids", *args.case_ids])
    return command


def validate_environment(dry_run: bool) -> None:
    """Check data, configuration, and API key before any billed call to avoid failing halfway through the run."""
    required_paths = (
        PROJECT_ROOT / "data" / "swat_s2s_raw_window_fixed" / "llm_prompt_cases.jsonl",
        PROJECT_ROOT / "configs" / "models.yaml",
        PROJECT_ROOT / "configs" / "rag.yaml",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少实验文件：" + ", ".join(missing))
    if not dry_run and not os.environ.get("DASHSCOPE_API_KEY", "").strip():
        raise RuntimeError("未设置 DASHSCOPE_API_KEY，禁止启动正式实验。")


def validate_experiment_result(
    output_dir: Path,
    model: str,
    expected_records: int,
) -> None:
    """Validate the infrastructure completeness of a single experiment group, stopping subsequent models on truncation or API failure."""
    metrics_path = output_dir / "metrics" / f"{model}_metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"实验结束但未生成 metrics：{metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    if metrics.get("num_records") != expected_records:
        raise RuntimeError(
            f"{output_dir.name} 记录数异常："
            f"{metrics.get('num_records')} != {expected_records}"
        )
    if metrics.get("api_failure_count", 0):
        raise RuntimeError(
            f"{output_dir.name} 存在 {metrics['api_failure_count']} 条 API 失败记录。"
        )
    if metrics.get("truncated_response_count", 0):
        raise RuntimeError(
            f"{output_dir.name} 存在 {metrics['truncated_response_count']} 条输出截断，"
            "已停止后续模型，请先调整生成或推理预算。"
        )


def main() -> None:
    args = parse_args()
    validate_environment(args.dry_run)
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    condition_by_name = {item[0]: item for item in CONDITIONS}
    selected_conditions = [condition_by_name[name] for name in args.conditions]
    total_groups = len(selected_conditions) * len(args.models)
    completed_groups = 0

    for condition_name, use_rag, use_agent, use_dual_branch_fusion in selected_conditions:
        for model in args.models:
            experiment_name = f"{condition_name}_{model}"
            output_dir = output_root / experiment_name
            command = build_command(
                model,
                output_dir,
                use_rag,
                use_agent,
                args,
                use_dual_branch_fusion=use_dual_branch_fusion,
            )
            print(f"[{experiment_name}] {subprocess.list2cmdline(command)}", flush=True)

            if args.dry_run:
                continue

            # main.py creates the condition directory and the raw/parsed/metrics/logs subdirectories.
            # check=True stops immediately when any group fails, avoiding billing in an incomplete environment.
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            if args.case_ids:
                expected_records = args.num_runs * len(args.case_ids)
            elif args.max_cases < 0:
                # No --max-cases limit given: the official SWaT data always contains
                # exactly OFFICIAL_CASE_COUNT cases, so all of them are expected.
                expected_records = args.num_runs * OFFICIAL_CASE_COUNT
            else:
                # An explicit --max-cases was provided, so only that many cases are expected.
                expected_records = args.num_runs * args.max_cases
            validate_experiment_result(output_dir, model, expected_records)
            completed_groups += 1
            print(
                f"[{completed_groups}/{total_groups}] {experiment_name} 完成并通过完整性检查。",
                flush=True,
            )

    print(f"全部实验任务已处理，结果目录：{output_root}")


if __name__ == "__main__":
    main()
