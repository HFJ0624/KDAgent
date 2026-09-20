"""Batch runner.

This module is the batch entry point for the LLM RCA experiment. It sequentially
runs the same RCA (root cause analysis) experiment for multiple models listed in
the config file, aggregates each model's metrics, and finally produces a
cross-model comparison summary CSV / JSON file as well as a summary table printed
in the terminal.

Typical workflow:
    1. Parse command-line arguments (data path, config path, output directory,
       RAG options, etc.).
    2. Load the model config and determine the target model list for this run.
    3. Iterate over each target model, build a separate ``argparse.Namespace``
       and call :func:`src.main.run_single_model` to run the experiment.
    4. Capture each model's returned metrics or exceptions and aggregate them
       into the ``summaries`` list.
    5. Write the aggregated results to ``all_models_summary.csv`` and
       ``all_models_summary.json``.
    6. Print a concise cross-model comparison table in the terminal for quick review.
"""

import argparse
import concurrent.futures
import json
import os
import sys
from typing import Any, Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.main import run_single_model
from src.model_client import list_available_models
from src.utils import ensure_dir, sanitize_filename, write_df_csv, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return a Namespace object.

    See the ``help`` text in each ``add_argument`` below for parameter meanings.
    """
    p = argparse.ArgumentParser(
        description="Run the LLM RCA experiment for all models in the config YAML."
    )
    # data file path: JSONL file containing the RCA cases to evaluate
    p.add_argument("--data_path", default="data/llm_prompt_cases.jsonl",
                   help="Path to the RCA case data (JSONL).")
    # model config file path: YAML config listing all models to run
    p.add_argument("--config_path", default="configs/models.yaml",
                   help="Path to the model config file (YAML), defining all available models.")
    # output directory: both intermediate artifacts and final aggregate results
    p.add_argument("--output_dir", default="outputs",
                   help="Output directory for per-model artifacts and the final aggregate results.")
    # prompt template path: template used to build the LLM input
    p.add_argument("--prompt_template", default="prompts/rca_prompt_template.txt",
                   help="Path to the RCA prompt template file.")
    # knowledge base template path: process knowledge template used in RAG
    p.add_argument("--knowledge_path", default="data/process_knowledge_template.md",
                   help="Path to the process-knowledge template used for RAG augmentation.")
    # number of repeated runs per case, used to compute average performance
    p.add_argument("--num_runs", type=int, default=3,
                   help="Number of repeated inferences per case (for stability statistics).")
    # max number of cases to process; -1 means all
    p.add_argument("--max_cases", type=int, default=-1,
                   help="Maximum number of cases to process; -1 means all.")
    # LLM temperature controlling output randomness; uses model default when None
    p.add_argument("--temperature", type=float, default=None,
                   help="LLM sampling temperature controlling output randomness; None uses the model default.")
    # whether to enable resume: skip already completed cases
    p.add_argument("--resume", action="store_true",
                   help="Enable resume: skip already-completed cases.")
    # during resume, whether to skip historically failed cases (0=no, 1=yes)
    p.add_argument("--resume_skip_failed", type=int, default=0, choices=[0, 1],
                   help="Whether to skip historically failed cases on resume (0=no, 1=yes).")
    # sleep milliseconds between requests, for rate limiting
    p.add_argument("--sleep_ms", type=int, default=300,
                   help="Sleep between requests in milliseconds (rate limiting).")
    # max retry count after a request failure
    p.add_argument("--max_retries", type=int, default=5,
                   help="Maximum retry count after a request failure.")
    # timeout for a single HTTP request (seconds)
    p.add_argument("--request_timeout", type=int, default=120,
                   help="Timeout for a single HTTP request (seconds).")
    # base sleep seconds for exponential backoff retries
    p.add_argument("--retry_base_sleep", type=int, default=5,
                   help="Base sleep seconds for exponential-backoff retries.")
    # whether to enable RAG retrieval enhancement (0=no, 1=yes)
    p.add_argument("--use_rag", type=int, default=0, choices=[0, 1],
                   help="Whether to enable RAG retrieval augmentation (0=no, 1=yes).")
    # RAG config file path
    p.add_argument("--rag_config", default="configs/rag.yaml",
                   help="Path to the RAG config file (YAML).")
    # RAG retrieval Top-K; None means use the default from the config file
    p.add_argument("--rag_top_k", type=int, default=None,
                   help="RAG retrieval Top-K; None uses the default from the RAG config.")
    # Optional model subset: only run the specified subset of models
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model subset (space-separated). Omit to run every model in the config.",
    )
    # reproducibility seed shared with src/main.py
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible shared-state sampling (e.g. RAG jitter).",
    )
    # number of models to run concurrently; 1 keeps the historical serial order
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of models to run concurrently; 1 keeps serial execution. Mind API rate limits.",
    )
    return p.parse_args()


def _resolve_path(p: str) -> str:
    """Resolve a relative path to an absolute path.

    Resolution precedence:
        1. If already an absolute path, return it directly.
        2. If it exists relative to the current working directory, return it directly.
        3. Otherwise resolve it relative to the project root.
    """
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    return os.path.join(_PROJECT_ROOT, p)


def main() -> None:
    """Batch run entry point.

    Processing flow:
        1. Parse command-line arguments; resolve paths such as ``config_path`` /
           ``output_dir`` and create the ``metrics`` subdirectory for summary files.
        2. Call :func:`list_available_models` to read all available models from the
           config and determine the target model subset based on ``--models``.
        3. Iterate over the target model list, build an independent
           ``argparse.Namespace`` for each model, and call :func:`run_single_model`
           to run the single-model RCA experiment.
        4. Capture exceptions from ``run_single_model`` with a fallback (fill with
           zero metrics) and aggregate each model's results into the ``summaries`` list.
        5. Use :func:`_write_summary_csv` to write the aggregated results to
           ``metrics/all_models_summary.csv``.
        6. Use :func:`_print_summary_table` to print the cross-model comparison table
           in the terminal.
        7. Additionally call :func:`write_json` to write ``all_models_summary.json``
           for programmatic reading later.
    """
    args = parse_args()
    config_path = _resolve_path(args.config_path)
    output_dir = _resolve_path(args.output_dir)
    metrics_dir = os.path.join(output_dir, "metrics")
    ensure_dir(metrics_dir)

    available = list_available_models(config_path)
    targets = args.models or available

    print(f"Available models: {available}")
    print(f"This run: {targets}")

    def _run_one(model_name: str) -> Optional[Dict[str, Any]]:
        """Run a single target model and return its metrics (or None if unknown)."""
        if model_name not in available:
            print(f"[skip] unknown model: {model_name}", file=sys.stderr)
            return None

        # Build an argparse.Namespace for each model
        ns = argparse.Namespace(
            data_path=args.data_path,
            config_path=args.config_path,
            model_name=model_name,
            output_dir=args.output_dir,
            prompt_template=args.prompt_template,
            knowledge_path=args.knowledge_path,
            num_runs=args.num_runs,
            max_cases=args.max_cases,
            temperature=args.temperature,
            resume=args.resume,
            resume_skip_failed=args.resume_skip_failed,
            sleep_ms=args.sleep_ms,
            max_retries=args.max_retries,
            request_timeout=args.request_timeout,
            retry_base_sleep=args.retry_base_sleep,
            use_rag=args.use_rag,
            rag_config=args.rag_config,
            rag_top_k=args.rag_top_k,
            seed=args.seed,
        )

        print(f"\n========== Running model: {model_name} ==========")
        try:
            return run_single_model(ns)
        except Exception as e:
            print(f"[error] model {model_name} raised: {e}", file=sys.stderr)
            return {
                "model_name": model_name,
                "num_cases": 0,
                "num_runs": args.num_runs,
                "accuracy_at_1": 0.0,
                "accuracy_at_3": 0.0,
                "accuracy_at_5": 0.0,
                "valid_json_rate": 0.0,
                "hallucination_rate": 0.0,
                "average_confidence": 0.0,
                "error": str(e),
            }

    if args.parallel > 1:
        # Run several models concurrently; each writes to its own per-model
        # directory, so the writes do not collide. Keep the model order stable.
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.parallel)) as ex:
            summaries: List[Dict[str, Any]] = [
                m for m in ex.map(_run_one, targets) if m is not None
            ]
    else:
        summaries = []
        for model_name in targets:
            metrics = _run_one(model_name)
            if metrics is not None:
                summaries.append(metrics)

    # ---- Write the summary CSV ----
    if summaries:
        summary_path = os.path.join(metrics_dir, "all_models_summary.csv")
        _write_summary_csv(summary_path, summaries)
        print(f"\nSummary written: {summary_path}")

        # Print a table
        cols = [
            "model_name",
            "num_cases",
            "num_runs",
            "accuracy_at_1",
            "accuracy_at_3",
            "accuracy_at_5",
            "valid_json_rate",
            "hallucination_rate",
            "average_confidence",
        ]
        _print_summary_table(summaries, cols)

        # Also write a JSON summary
        write_json(
            os.path.join(metrics_dir, "all_models_summary.json"),
            {
                "num_models": len(summaries),
                "results": summaries,
            },
        )


def _write_summary_csv(path: str, rows: list) -> None:
    """Write the aggregated results to a CSV file.

    Prefer ``pandas.DataFrame.to_csv`` for better formatting and encoding support;
    if pandas is unavailable, fall back to the project's custom :func:`write_df_csv`.

    :param path: Output CSV file path.
    :param rows: List of rows to write, each a ``dict``.
    """
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception:
        write_df_csv(path, rows)


def _print_summary_table(rows: list, cols: list) -> None:
    """Print the cross-model comparison summary table in the terminal.

    If ``pandas`` is available in the environment, prefer ``DataFrame.to_string``
    for a well-aligned table; otherwise manually construct a simple ASCII table
    as a fallback.

    :param rows: A list of metrics ``dict`` for each model.
    :param cols: The list of column names to display.
    """
    filtered_cols = [c for c in cols if any(c in r for r in rows)]
    if not rows:
        return
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        cols_present = [c for c in filtered_cols if c in df.columns]
        print(df[cols_present].to_string(index=False))
    except Exception:
        # Fallback: simple ASCII table
        col_widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in filtered_cols}
        sep = "+" + "+".join("-" * (col_widths[c] + 2) for c in filtered_cols) + "+"
        header = "|" + "|".join(" " + c.ljust(col_widths[c]) + " " for c in filtered_cols) + "|"
        print(sep)
        print(header)
        print(sep)
        for r in rows:
            line = "|" + "|".join(" " + str(r.get(c, "")).ljust(col_widths[c]) + " " for c in filtered_cols) + "|"
            print(line)
        print(sep)


if __name__ == "__main__":
    main()
