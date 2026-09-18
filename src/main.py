"""
main.py: Main entry point for the industrial Root Cause Analysis (RCA) experiment.

This module is the core of the entire experiment framework and is responsible
for coordinating the following tasks:
1. Parse command-line arguments (Argument Parser)
2. Load configuration files (YAML) and experiment data (JSONL)
3. Initialize the large language model client (ModelClient)
4. Optional: initialize the RAG (Retrieval-Augmented Generation) system
5. Run the core experiment loop: call the LLM API for each case
6. Parse model responses and compute evaluation metrics
7. Save results to JSONL (raw responses), CSV (parsed data), and JSON (metrics)
8. Support resume functionality

The overall experiment flow is as follows:
    main() -> run_single_model() -> iterate over run_id and case_id
                                    |
                                    +-> build the Prompt (RAG enhancement optional)
                                    +-> call ModelClient.chat()
                                    +-> parse the response (ResponseParser)
                                    +-> evaluate (Evaluator)
                                    +-> write to the JSONL file
                                    +-> after the loop, aggregate into CSV and Metrics
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Set

# -----------------------------------------------------------------------------
# Path config: ensure the project root is on sys.path so modules can be imported.
# -----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_loader import load_cases, normalize_case
from src import evaluator as evaluator
from src.evaluator import build_parsed_rows, compute_metrics, evaluate_case
from src.model_client import ModelClient, get_model_config, list_available_models
from src.prompt_builder import build_prompt, load_process_knowledge, load_template
from src.response_parser import clamp_to_candidates, parse_response
from src.iterative_agent import IterativeSelfRefinementAgent
from src.dual_branch_fusion_agent import DualBranchFusionAgent
from src.utils import (
    ensure_dir,
    sanitize_filename,
    setup_logger,
    write_df_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Defines all configuration items needed to run the experiment, including data
    paths, model selection, and experiment parameters. Returns an
    argparse.Namespace object containing the values of all arguments.
    """
    p = argparse.ArgumentParser(
        description="Run single-model industrial root cause analysis experiment."
    )
    # --- Data and path parameters ---
    p.add_argument(
        "--data_path",
        default="data/llm_prompt_cases.jsonl",
        help="JSONL file path containing all cases.",
    )
    p.add_argument(
        "--config_path",
        default="configs/models.yaml",
        help="Model YAML config file path.",
    )
    p.add_argument(
        "--model_name",
        required=True,
        help="Model name to run (key in models.yaml).",
    )
    p.add_argument(
        "--output_dir",
        default="outputs",
        help="Experiment output root directory.",
    )
    p.add_argument(
        "--experiment_version",
        default="",
        help="Optional version namespace written into new records for resume isolation.",
    )
    p.add_argument(
        "--protocol_sha256",
        default="",
        help="Optional frozen protocol hash written into new records.",
    )
    p.add_argument(
        "--prompt_template",
        default="prompts/rca_prompt_template.txt",
        help="Optional: custom prompt template path.",
    )
    p.add_argument(
        "--knowledge_path",
        default="data/process_knowledge_template.md",
        help="Optional: industrial process knowledge markdown path.",
    )
    
    # --- Experiment control parameters ---
    p.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of times to repeat inference per case.",
    )
    p.add_argument(
        "--max_cases",
        type=int,
        default=-1,
        help="Max number of cases to run (-1 = all).",
    )
    p.add_argument(
        "--case_ids",
        nargs="+",
        default=None,
        help="Run only the specified case IDs, preserving the given order.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override temperature in YAML config.",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Override maximum output tokens in YAML config.",
    )
    p.add_argument(
        "--thinking_budget",
        type=int,
        default=None,
        help="Override reasoning token budget (0 disables the request field).",
    )
    
    # --- Resume parameters ---
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip completed (model, run_id, case_id) combos based on raw_responses.",
    )
    p.add_argument(
        "--resume_skip_failed",
        type=int,
        default=0,
        choices=[0, 1],
        help="When resuming, also skip cases that previously failed (1=skip, 0=rerun failed).",
    )
    
    # --- API call control parameters ---
    p.add_argument(
        "--sleep_ms",
        type=int,
        default=300,
        help="Sleep milliseconds between API calls (to avoid rate limiting).",
    )
    p.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max retries per API call.",
    )
    p.add_argument(
        "--request_timeout",
        type=int,
        default=120,
        help="Request timeout in seconds.",
    )
    p.add_argument(
        "--retry_base_sleep",
        type=int,
        default=5,
        help="Base sleep seconds for exponential backoff.",
    )
    
    # --- RAG (Retrieval-Augmented Generation) parameters ---
    p.add_argument(
        "--use_rag",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable RAG retrieval (1=enable, 0=disable).",
    )
    p.add_argument(
        "--rag_config",
        default="configs/rag.yaml",
        help="RAG config YAML path (only effective with --use_rag 1).",
    )
    p.add_argument(
        "--rag_top_k",
        type=int,
        default=None,
        help="Override top_k in rag.yaml (only effective with --use_rag 1).",
    )
    
    # --- Iterative self-refinement Agent parameters ---
    p.add_argument(
        "--use_iterative_agent",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable iterative self-correction agent (1=enable, 0=disable).",
    )
    p.add_argument(
        "--use_dual_branch_fusion",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable v2 evidence-Agent and RAG dual-branch deterministic fusion.",
    )
    p.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        choices=[3],
        help="Max iterations for iterative agent (default=3).",
    )
    p.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for iterative agent validation (default=0.5).",
    )
    
    # --- Compact evidence mode parameters ---
    p.add_argument(
        "--compact_evidence",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use compact evidence format to reduce prompt length (1=enable, 0=disable).",
    )
    return p.parse_args()


def _resolve_path(p: str) -> str:
    """
    Resolve a relative path to an absolute path.

    Prefer the current working directory if the path exists; otherwise resolve
    it relative to the project root.
    """
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    return os.path.join(_PROJECT_ROOT, p)


def _write_rows_csv(path: str, rows: list) -> None:
    """
    Write a list of dict rows to a CSV file.

    Prefer pandas (when available) for better compatibility; otherwise fall back
    to utils.write_df_csv.
    """
    if not rows:
        return
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        ensure_dir(os.path.dirname(path))
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception:
        write_df_csv(path, rows)


def _append_jsonl_with_flush(path: str, record: Dict[str, Any]) -> None:
    """
    Append one record to a JSONL (JSON Lines) file and flush it to disk immediately.

    This is critical for experiment stability: after each case is processed, the
    record is written and synced to prevent data loss if the program is interrupted.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_case_summary_records(
    raw_dir: str,
    model_id: str,
    allowed_keys: Set[str] | None = None,
) -> List[Dict[str, Any]]:
    """Load the last final summary record for each ``(run_id, case_id)`` from JSONL.

    Agent intermediate rounds do not participate in the metrics. After a failed
    record is rerun, the file may contain both the old failure and the new success,
    so the last record per key must be kept to avoid double counting. allowed_keys
    filters out historical records outside the current data range and repetition count.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    pending_truncated_iterations: Dict[str, bool] = {}
    fname_prefix = f"{model_id}_run_"
    if not os.path.isdir(raw_dir):
        return []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".jsonl"):
            continue
        if not fname.startswith(fname_prefix):
            continue
        run_part = fname[len(fname_prefix):-len(".jsonl")]
        try:
            run_id = int(run_part)
        except ValueError:
            continue
        full = os.path.join(raw_dir, fname)
        try:
            with open(full, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        cid = rec.get("case_id")
                        if cid is not None:
                            rec["run_id"] = rec.get("run_id", run_id)
                            key = f"{run_id}|{cid}"
                            if allowed_keys is not None and key not in allowed_keys:
                                continue
                            if rec.get("record_type") == "agent_iteration":
                                was_truncated = bool(rec.get("was_truncated")) or (
                                    str(rec.get("finish_reason") or "").lower() == "length"
                                )
                                pending_truncated_iterations[key] = (
                                    pending_truncated_iterations.get(key, False)
                                    or was_truncated
                                )
                                continue

                            # Older Agent summaries have no had_truncated_iteration.
                            # Intermediate rounds and case_summary are written to disk in
                            # order, so the flag can be rebuilt losslessly here; once a
                            # summary is encountered, the pending state is cleared so a
                            # later rerun does not inherit the historical flag from a
                            # previous call.
                            rec.setdefault(
                                "had_truncated_iteration",
                                pending_truncated_iterations.pop(key, False),
                            )
                            latest[key] = rec
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    def sort_key(item: Dict[str, Any]) -> tuple[int, str]:
        return int(item.get("run_id", 0)), str(item.get("case_id", ""))

    return sorted(latest.values(), key=sort_key)


def _load_completed_keys(
    raw_dir: str, model_id: str, skip_failed: bool = False
) -> Set[str]:
    """Load the (run_id, case_id) combos that resume should skip; only fully
    successful final records are skipped by default."""
    done: Set[str] = set()
    for rec in _load_case_summary_records(raw_dir, model_id):
        success = bool(rec.get("api_ok", rec.get("success", True)))
        truncated = evaluator.is_final_response_truncated(rec)
        # API success only means the request completed; a length truncated output
        # is still an incomplete result and must be rerun on resume by default.
        # resume_skip_failed=1 is an explicit switch that keeps all old records,
        # so truncated items are allowed to be skipped.
        if (success and not truncated) or skip_failed:
            done.add(f"{rec.get('run_id')}|{rec.get('case_id')}")
    return done


def _summarize_agent_api_status(
    all_iterations: List[Dict[str, Any]],
) -> tuple[bool, Dict[str, Any]]:
    """Distinguish Agent content-validation failure from three-round API total
    failure, and return the last API error."""
    api_ok = any(bool(item.get("api_success")) for item in all_iterations)
    if api_ok:
        return True, {}
    last_error = next(
        (
            item
            for item in reversed(all_iterations)
            if not item.get("api_success")
        ),
        {},
    )
    return False, last_error


def _get_log_filename(model_name: str, use_rag: bool) -> str:
    """
    Generate a log filename based on the model name and RAG flag.

    Format: rca_experiment_{model_name}_{rag_flag}_{timestamp}.log
    """
    safe_name = sanitize_filename(model_name)
    rag_flag = "with-rag" if use_rag else "no-rag"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"rca_experiment_{safe_name}_{rag_flag}_{timestamp}.log"


def _build_record_failure(
    case_id: Any,
    model_name: str,
    run_id: int,
    error_type: str,
    error_message: str,
    gt_vars: list,
    top10_vars: list,
    top10_details: list,
    raw_response: str = "",
) -> Dict[str, Any]:
    """
    Build a failed experiment record.

    When the API call ultimately fails (or parsing fails), this function creates
    a complete failure record so that the CSV and Metrics include that case.
    """
    return {
        "record_type": "case_summary",
        "case_id": case_id,
        "model_name": model_name,
        "run_id": run_id,
        "prompt": "",
        "raw_response": raw_response,
        "parsed_response": {
            "predicted_root_causes": [],
            "primary_root_cause": None,
            "root_cause_name": "",
            "reasoning": "",
            "inference_process": [],
            "numerical_evidence": "",
            "temporal_evidence": "",
            "type_aware_reasoning": "",
            "process_relation_reasoning": "",
            "why_not_other_candidates": "",
            "evidence_variables": [],
            "uncertainty_analysis": "",
            "is_valid_prediction": False,
            "confidence": 0.0,
            "valid_json": False,
            "parse_method": "error",
        },
        "gt_vars": gt_vars,
        "top10_vars": top10_vars,
        "top10_details": top10_details,
        "api_ok": False,
        "api_error": error_message,
        "error_type": error_type,
        "elapsed_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "was_truncated": False,
        "use_rag": False,
        "rag_query": "",
        "rag_contexts": [],
        "is_hit_at_1": 0,
        "is_hit_at_3": 0,
        "is_hit_at_5": 0,
        "is_hallucination": 0,
        "is_api_failure": 1,
    }


def run_single_model(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the complete experiment for a single model.

    This is the core function of the experiment, executing the following steps:
    1. Resolve paths and create output directories
    2. Load the model config and apply command-line overrides
    3. Set up the logging system
    4. Initialize the ModelClient
    5. Load the experiment data and Prompt template
    6. Initialize the RAG retriever (if enabled)
    7. Load resume records
    8. Main loop: iterate over num_runs and cases
        a. Skip completed cases (resume mode)
        b. Perform RAG retrieval
        c. Build the Prompt
        d. Call the API
        e. Parse the response
        f. Evaluate the result
        g. Write to JSONL
    9. Aggregate and generate CSV and Metrics

    Returns the final metrics dictionary.
    """
    # --- Step 1: Path resolution and directory creation ---
    data_path = _resolve_path(args.data_path)
    config_path = _resolve_path(args.config_path)
    output_dir = _resolve_path(args.output_dir)
    template_path = _resolve_path(args.prompt_template)
    knowledge_path = _resolve_path(args.knowledge_path)

    raw_dir = os.path.join(output_dir, "raw_responses")
    parsed_dir = os.path.join(output_dir, "parsed_results")
    metrics_dir = os.path.join(output_dir, "metrics")
    log_dir = os.path.join(output_dir, "logs")
    # All artifacts of a formal experiment live inside the same condition/model
    # directory, avoiding logs scattered into the project root again and making
    # it easy to archive and reproduce the experiment configuration later.
    for d in (raw_dir, parsed_dir, metrics_dir, log_dir):
        ensure_dir(d)

    # --- Step 2: Load model config and allow command-line overrides ---
    model_cfg = get_model_config(config_path, args.model_name)
    if args.temperature is not None:
        model_cfg["temperature"] = args.temperature
    if args.max_tokens is not None:
        if args.max_tokens <= 0:
            raise ValueError("--max_tokens 必须是正整数。")
        model_cfg["max_tokens"] = args.max_tokens
    if args.thinking_budget is not None:
        if args.thinking_budget < 0:
            raise ValueError("--thinking_budget 不能为负数。")
        # 0 explicitly means not to send thinking_budget, which helps with
        # compatibility troubleshooting.
        model_cfg["thinking_budget"] = args.thinking_budget or None
    if args.request_timeout is not None:
        model_cfg["timeout"] = args.request_timeout
    if args.max_retries is not None:
        model_cfg["max_retries"] = args.max_retries
    if args.retry_base_sleep is not None:
        model_cfg["retry_base_sleep"] = args.retry_base_sleep

    use_rag = bool(args.use_rag)
    use_dual_branch_fusion = bool(args.use_dual_branch_fusion)
    if use_dual_branch_fusion and not use_rag:
        raise ValueError("双分支融合必须同时启用 --use_rag 1。")
    model_id = sanitize_filename(model_cfg.get("name", args.model_name))

    # --- Step 3: Set up the logging system ---
    log_filename = _get_log_filename(model_id, use_rag)
    log_filepath = os.path.join(log_dir, log_filename)
    logger = setup_logger(
        log_dir=log_dir,
        name=f"rca_{model_id}",
        log_filename=log_filename,
        model_name=model_id,
        use_rag=use_rag,
    )

    # Print the full experiment configuration to the log
    logger.info("=" * 60)
    logger.info("EXPERIMENT CONFIG:")
    logger.info("  model_name: %s", model_id)
    logger.info("  model_backend: %s", model_cfg.get("provider", "unknown"))
    logger.info("  use_rag: %s", use_rag)
    logger.info("  rag_top_k: %s", args.rag_top_k or 0)
    logger.info("  use_iterative_agent: %s", bool(args.use_iterative_agent))
    logger.info("  use_dual_branch_fusion: %s", use_dual_branch_fusion)
    logger.info("  max_iterations: %d", args.max_iterations)
    logger.info("  num_runs: %d", args.num_runs)
    logger.info("  max_cases: %d", args.max_cases)
    logger.info("  temperature: %s", model_cfg.get("temperature"))
    logger.info("  max_tokens: %d", model_cfg.get("max_tokens", 2048))
    logger.info("  thinking_budget: %s", model_cfg.get("thinking_budget"))
    logger.info("  output_dir: %s", output_dir)
    logger.info("  data_path: %s", data_path)
    logger.info("  request_timeout: %ds", args.request_timeout)
    logger.info("  max_retries: %d", args.max_retries)
    logger.info("  retry_base_sleep: %ds", args.retry_base_sleep)
    logger.info("  resume: %s", args.resume)
    logger.info("  log_file: %s", log_filepath)
    logger.info("=" * 60)

    # --- Step 4: Initialize the model client ---
    client = ModelClient(model_cfg, logger=logger)

    if not client.is_configured():
        error_msg = f"Model '{client.name}' not configured (missing API key env '{client.api_key_env}')."
        logger.error("%s", error_msg)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return {
            "model_name": client.name,
            "num_cases": 0,
            "num_records": 0,
            "num_runs": args.num_runs,
            "error": error_msg,
        }

    logger.info("Model: %s (provider=%s, endpoint=%s)", client.name, client.provider, client.base_url)

    # --- Step 5: Load experiment data and the Prompt template ---
    # When case_ids are specified, the full data must be loaded first and then
    # selected precisely by ID; otherwise max_cases would truncate the data early,
    # preventing later cases (e.g. 12, 13, 16) from being used for isolated preflight.
    cases = load_cases(data_path, max_cases=-1 if args.case_ids else args.max_cases)
    cases = [normalize_case(c) for c in cases]
    if args.case_ids:
        case_by_id = {str(case.get("case_id")): case for case in cases}
        missing_case_ids = [
            case_id for case_id in args.case_ids if case_id not in case_by_id
        ]
        if missing_case_ids:
            raise ValueError(f"未找到 case_id：{', '.join(missing_case_ids)}")
        cases = [case_by_id[case_id] for case_id in args.case_ids]
    logger.info("Loaded %d cases.", len(cases))

    template = load_template(template_path)
    process_knowledge = load_process_knowledge(knowledge_path)

    # --- Step 6: Initialize the RAG retriever (optional) ---
    rag_retriever = None
    if use_rag:
        rag_config_path = _resolve_path(args.rag_config)
        logger.info("RAG enabled. Initializing retriever with config: %s", rag_config_path)
        try:
            from src.rag_retriever import RAGRetriever
            rag_retriever = RAGRetriever.from_config_path(rag_config_path, logger=logger)
            logger.info("RAG retriever ready (top_k=%s, store=%s).", args.rag_top_k or rag_retriever.top_k, rag_retriever.store.collection_name)
        except Exception as exc:
            logger.warning("RAG initialization failed; continuing without RAG. Error: %s", exc)
            rag_retriever = None
            use_rag = False

    # --- Step 7: Load resume records ---
    completed: Set[str] = set()
    if args.resume:
        completed = _load_completed_keys(raw_dir, model_id, skip_failed=bool(args.resume_skip_failed))
        logger.info("Resuming: found %d completed entries (skip_failed=%s).", len(completed), bool(args.resume_skip_failed))

    # --- Variables used for statistics ---
    all_records: List[Dict[str, Any]] = []
    total_cases = len(cases) * args.num_runs
    processed_count = 0
    success_count = 0
    failure_count = 0

    # --- Step 8: Main loop ---
    for run_id in range(1, args.num_runs + 1):
        raw_fname = os.path.join(raw_dir, f"{model_id}_run_{run_id}.jsonl")
        logger.info("===== Run %d/%d (output: %s) =====", run_id, args.num_runs, raw_fname)

        for case in cases:
            case_id = case.get("case_id")
            processed_count += 1

            # Resume check
            if args.resume and f"{run_id}|{case_id}" in completed:
                logger.info("[skip][run=%s][case=%s] already completed.", run_id, case_id)
                continue

            logger.info("[run=%s][case=%s] calling API...", run_id, case_id)

            # --- RAG retrieval ---
            rag_query = ""
            rag_contexts: List[Dict[str, Any]] = []
            if use_rag and rag_retriever is not None:
                try:
                    rag_res = rag_retriever.retrieve(case, top_k=args.rag_top_k)
                    rag_query = rag_res.get("rag_query", "") or ""
                    rag_contexts = rag_res.get("retrieved_contexts", []) or []
                except Exception as exc:
                    logger.warning("[run=%s][case=%s] RAG retrieval failed: %s", run_id, case_id, exc)
                    rag_query = ""
                    rag_contexts = []

            # --- Build the Prompt ---
            prompt = build_prompt(
                case,
                template,
                process_knowledge=process_knowledge,
                rag_contexts=rag_contexts if (use_rag and rag_retriever is not None) else None,
                compact_evidence=bool(args.compact_evidence),
            )
            evidence_prompt = prompt
            if use_dual_branch_fusion:
                # v2 must keep the data branch from seeing the retrieved text at
                # all, avoiding RAG polluting the first root cause before fusion;
                # the knowledge branch continues to use the RAG-bearing Prompt built above.
                evidence_prompt = build_prompt(
                    case,
                    template,
                    process_knowledge=process_knowledge,
                    rag_contexts=None,
                    compact_evidence=bool(args.compact_evidence),
                )

            # --- Call the API (supporting iterative self-refinement) ---
            use_iterative = bool(args.use_iterative_agent) or use_dual_branch_fusion
            candidate_vars = case.get("top10_vars", [])
            rag_contexts_for_agent = rag_contexts if (use_rag and rag_retriever is not None) else None

            if use_iterative:
                if use_dual_branch_fusion:
                    agent = DualBranchFusionAgent(
                        model_client=client,
                        min_confidence=args.min_confidence,
                        max_retries=args.max_retries,
                        timeout=args.request_timeout,
                        retry_base_sleep=args.retry_base_sleep,
                        logger=logger,
                    )
                    agent_result = agent.run_case(
                        case=case,
                        evidence_prompt=evidence_prompt,
                        rag_prompt=prompt,
                        top10_vars=candidate_vars,
                        max_iterations=args.max_iterations,
                        run_id=run_id,
                    )
                else:
                    agent = IterativeSelfRefinementAgent(
                        model_client=client,
                        min_confidence=args.min_confidence,
                        max_retries=args.max_retries,
                        timeout=args.request_timeout,
                        retry_base_sleep=args.retry_base_sleep,
                        logger=logger,
                    )
                    agent_result = agent.run_case(
                        case=case,
                        initial_prompt=prompt,
                        top10_vars=candidate_vars,
                        rag_contexts=rag_contexts_for_agent,
                        max_iterations=args.max_iterations,
                        run_id=run_id,
                    )

                # Save all iterations to raw_responses
                for iter_record in agent_result.get("all_iterations", []):
                    _append_jsonl_with_flush(raw_fname, iter_record)

                parsed_response = agent_result.get("parsed_response", {})
                # The raw view is used to count out-of-candidate variables; the final
                # view is used to record a validated answer or UNKNOWN.
                raw_parsed_response = agent_result.get("raw_parsed_response", parsed_response)
                raw_pred = list(raw_parsed_response.get("predicted_root_causes", []) or [])
                raw_primary = raw_parsed_response.get("primary_root_cause")
                record_is_halluc = int(
                    evaluator.is_hallucination(
                        raw_pred + ([raw_primary] if raw_primary else []),
                        candidate_vars,
                    )
                )
                # Agent validation already guarantees that successful answers are in the
                # candidate set; failed answers must remain UNKNOWN, so the clamping
                # function that would clear UNKNOWN cannot be invoked here.

                all_iterations = agent_result.get("all_iterations", [])
                api_ok, last_api_error = _summarize_agent_api_status(all_iterations)

                record = {
                    "record_type": "case_summary",
                    "case_id": case_id,
                    "model_name": client.name,
                    "run_id": run_id,
                    "prompt": prompt,
                    "content": agent_result.get("content", ""),
                    "reasoning_content": agent_result.get("reasoning_content", ""),
                    "finish_reason": agent_result.get("finish_reason", ""),
                    "raw_response": agent_result.get("content", ""),
                    "parsed_response": parsed_response,
                    "gt_vars": case.get("gt_vars", []),
                    "top10_vars": case.get("top10_vars", []),
                    "top10_details": case.get("top10_details", []),
                    # If at least one round succeeded, this is an Agent validation
                    # failure rather than an API total failure; if all three rounds
                    # failed, it must enter the API failure statistics.
                    "api_ok": api_ok,
                    "api_error": None if api_ok else last_api_error.get("error_message"),
                    "error_type": None if api_ok else last_api_error.get("error_type"),
                    "error_message": None if api_ok else last_api_error.get("error_message"),
                    "elapsed_s": sum(iter.get("elapsed_s", 0.0) for iter in all_iterations),
                    "prompt_tokens": sum(iter.get("prompt_tokens", 0) for iter in all_iterations),
                    "completion_tokens": sum(iter.get("completion_tokens", 0) for iter in all_iterations),
                    "reasoning_tokens": sum(iter.get("reasoning_tokens", 0) for iter in all_iterations),
                    "total_tokens": sum(iter.get("total_tokens", 0) for iter in all_iterations),
                    # Final truncation decides whether the case is complete; intermediate
                    # round truncation is kept separately for auditing, so that batch
                    # gates do not misjudge a case as incomplete after the Agent has
                    # already advanced to later rounds.
                    "was_truncated": agent_result.get("finish_reason") == "length",
                    "had_truncated_iteration": any(
                        iter.get("was_truncated", False) for iter in all_iterations
                    ),
                    "use_rag": bool(use_rag and rag_retriever is not None),
                    "rag_query": rag_query,
                    "rag_contexts": rag_contexts,
                    "is_api_failure": 0 if api_ok else 1,
                    "iteration_count": agent_result.get("iteration_count", 1),
                    "final_iteration": agent_result.get("final_iteration", 1),
                    "retry_count": agent_result.get("retry_count", 0),
                    "validation_passed": agent_result.get("validation_passed", False),
                    "retry_reasons": agent_result.get("retry_reasons", []),
                    "agent_status": agent_result.get("agent_status", "success"),
                    "agent_failed": agent_result.get("agent_failed", False),
                    "final_prompt_type": agent_result.get("final_prompt_type", "initial"),
                    "use_dual_branch_fusion": bool(
                        agent_result.get("use_dual_branch_fusion", False)
                    ),
                    "fusion_strategy": agent_result.get("fusion_strategy", ""),
                    "evidence_branch_primary": agent_result.get(
                        "evidence_branch_primary", ""
                    ),
                    "rag_branch_primary": agent_result.get("rag_branch_primary", ""),
                    "branch_primary_agreement": bool(
                        agent_result.get("branch_primary_agreement", False)
                    ),
                    "rag_supplemented_candidates": agent_result.get(
                        "rag_supplemented_candidates", []
                    ),
                    "evidence_branch_success": bool(
                        agent_result.get("evidence_branch_success", False)
                    ),
                    "rag_branch_success": bool(
                        agent_result.get("rag_branch_success", False)
                    ),
                }

            else:
                try:
                    api_result = client.chat(
                        prompt=prompt,
                        system_prompt="You are a careful industrial root cause analysis engineer. Output strictly valid JSON only.",
                        max_retries=args.max_retries,
                        timeout=args.request_timeout,
                        retry_base_sleep=args.retry_base_sleep,
                    )
                except Exception as exc:
                    error_type = type(exc).__name__
                    error_message = str(exc)
                    logger.error("[run=%s][case=%s] Unexpected exception: %s - %s", run_id, case_id, error_type, error_message)

                    failure_record = _build_record_failure(
                        case_id=case_id,
                        model_name=client.name,
                        run_id=run_id,
                        error_type=error_type,
                        error_message=error_message,
                        gt_vars=case.get("gt_vars", []),
                        top10_vars=case.get("top10_vars", []),
                        top10_details=case.get("top10_details", []),
                    )
                    if args.experiment_version:
                        failure_record["experiment_version"] = args.experiment_version
                    if args.protocol_sha256:
                        failure_record["protocol_sha256"] = args.protocol_sha256
                    _append_jsonl_with_flush(raw_fname, failure_record)
                    all_records.append(failure_record)
                    failure_count += 1
                    logger.info("[run=%s][case=%s] Failure written. Continuing.", run_id, case_id)
                    continue

                if not api_result.get("success"):
                    error_type = api_result.get("error_type", "UnknownError")
                    error_message = api_result.get("error_message", "Unknown error")
                    logger.error("[run=%s][case=%s] API failed after %d retries: %s - %s",
                               run_id, case_id, args.max_retries, error_type, error_message)

                    failure_record = _build_record_failure(
                        case_id=case_id,
                        model_name=client.name,
                        run_id=run_id,
                        error_type=error_type,
                        error_message=error_message,
                        gt_vars=case.get("gt_vars", []),
                        top10_vars=case.get("top10_vars", []),
                        top10_details=case.get("top10_details", []),
                        raw_response=api_result.get("raw_response", ""),
                    )
                    if args.experiment_version:
                        failure_record["experiment_version"] = args.experiment_version
                    if args.protocol_sha256:
                        failure_record["protocol_sha256"] = args.protocol_sha256
                    _append_jsonl_with_flush(raw_fname, failure_record)
                    all_records.append(failure_record)
                    failure_count += 1
                    logger.info("[run=%s][case=%s] Failure record written and continue.", run_id, case_id)
                    continue

                parsed_response = parse_response(api_result.get("content", ""))
                raw_pred = list(parsed_response.get("predicted_root_causes", []) or [])
                raw_primary = parsed_response.get("primary_root_cause")

                record_is_halluc = int(
                    evaluator.is_hallucination(
                        raw_pred + ([raw_primary] if raw_primary else []),
                        case.get("top10_vars", []),
                    )
                )
                parsed_response = clamp_to_candidates(parsed_response, case.get("top10_vars", []))

                record = {
                    "record_type": "case_summary",
                    "case_id": case_id,
                    "model_name": client.name,
                    "run_id": run_id,
                    "prompt": prompt,
                    "content": api_result.get("content", ""),
                    "reasoning_content": api_result.get("reasoning_content", ""),
                    "finish_reason": api_result.get("finish_reason", ""),
                    "raw_response": api_result.get("raw_response", ""),
                    "parsed_response": parsed_response,
                    "gt_vars": case.get("gt_vars", []),
                    "top10_vars": case.get("top10_vars", []),
                    "top10_details": case.get("top10_details", []),
                    "api_ok": True,
                    "api_error": None,
                    "error_type": api_result.get("error_type"),
                    "error_message": api_result.get("error_message"),
                    "elapsed_s": api_result.get("elapsed", 0.0),
                    "prompt_tokens": api_result.get("prompt_tokens", 0),
                    "completion_tokens": api_result.get("completion_tokens", 0),
                    "reasoning_tokens": api_result.get("reasoning_tokens", 0),
                    "total_tokens": api_result.get("total_tokens", 0),
                    "was_truncated": api_result.get("finish_reason") == "length",
                    "use_rag": bool(use_rag and rag_retriever is not None),
                    "rag_query": rag_query,
                    "rag_contexts": rag_contexts,
                    "is_api_failure": 0,
                    "iteration_count": 1,
                    "final_iteration": 1,
                    "retry_count": 0,
                    "validation_passed": None,
                    "retry_reasons": [],
                    # Non-Agent branches also write the full summary state, so that
                    # CSV and resume use the same record-decision logic.
                    "agent_status": "disabled",
                    "agent_failed": False,
                    "final_prompt_type": "initial",
                }

            # Evaluate this record
            # Version fields are written only when explicitly passed, so the record
            # structure of historical default runs is not rewritten; prompt_alignment_v2
            # uses them to isolate old-version resume markers.
            if args.experiment_version:
                record["experiment_version"] = args.experiment_version
            if args.protocol_sha256:
                record["protocol_sha256"] = args.protocol_sha256
            record = evaluate_case(record)
            record["is_hallucination"] = record_is_halluc
            record["raw_predicted_root_causes"] = raw_pred
            record["raw_primary_root_cause"] = raw_primary

            # Write to file
            _append_jsonl_with_flush(raw_fname, record)
            all_records.append(record)
            if record.get("api_ok"):
                success_count += 1
            else:
                failure_count += 1

            logger.info(
                "[run=%s][case=%s] Case complete. api_ok=%s, elapsed=%.2fs",
                run_id,
                case_id,
                bool(record.get("api_ok")),
                record.get("elapsed_s", 0.0),
            )

            # Prevent API rate limiting
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    # --- Step 9: Aggregate and generate reports ---
    # The report is always rebuilt from the JSONL on disk, not only from records
    # added by the current process. This way, interrupted reruns, rerun-after-failure,
    # and "all skipped this time" all produce complete and deduplicated results.
    allowed_keys = {
        f"{run_id}|{case.get('case_id')}"
        for run_id in range(1, args.num_runs + 1)
        for case in cases
    }
    all_records = _load_case_summary_records(raw_dir, model_id, allowed_keys)
    processed_count = len(all_records)
    success_count = sum(bool(record.get("api_ok")) for record in all_records)
    failure_count = processed_count - success_count

    if all_records:
        # Generate CSV
        rows = build_parsed_rows(all_records)
        parsed_csv = os.path.join(parsed_dir, f"{model_id}_parsed.csv")
        _write_rows_csv(parsed_csv, rows)
        logger.info("Wrote parsed results: %s (%d rows)", parsed_csv, len(rows))

        # Compute metrics
        metrics = compute_metrics(all_records, client.name, num_runs=args.num_runs)
        rag_enabled = bool(use_rag and rag_retriever is not None)
        metrics["use_rag"] = rag_enabled
        # rag_top_k indicates the actual number of retrieved items that participated
        # in Prompt construction in this experiment. Even if baseline received the
        # unified batch parameter --rag_top_k=5, it never performed retrieval, so it
        # must be recorded as 0.
        metrics["rag_top_k"] = int(
            args.rag_top_k or rag_retriever.top_k
        ) if rag_enabled else 0
        metrics["use_iterative_agent"] = bool(args.use_iterative_agent)
        metrics["use_dual_branch_fusion"] = use_dual_branch_fusion
        iterative_enabled = bool(args.use_iterative_agent) or use_dual_branch_fusion
        metrics["max_iterations"] = int(args.max_iterations) if iterative_enabled else 0
        metrics["min_confidence"] = float(args.min_confidence) if iterative_enabled else None
        fusion_records = [
            record for record in all_records if record.get("use_dual_branch_fusion")
        ]
        metrics["fusion_strategy"] = (
            "evidence_primary_rag_supplement" if fusion_records else None
        )
        metrics["branch_primary_agreement_rate"] = round(
            sum(bool(record.get("branch_primary_agreement")) for record in fusion_records)
            / len(fusion_records),
            4,
        ) if fusion_records else 0.0
        metrics["average_rag_supplemented_candidates"] = round(
            sum(
                len(record.get("rag_supplemented_candidates", []) or [])
                for record in fusion_records
            ) / len(fusion_records),
            4,
        ) if fusion_records else 0.0
        metrics["evidence_branch_failure_count"] = sum(
            not bool(record.get("evidence_branch_success")) for record in fusion_records
        )
        metrics["rag_branch_failure_count"] = sum(
            not bool(record.get("rag_branch_success")) for record in fusion_records
        )
        metrics["max_tokens"] = int(client.max_tokens)
        metrics["thinking_budget"] = client.thinking_budget
        metrics["temperature"] = float(client.temperature)
        metrics["api_success_rate"] = round(success_count / processed_count, 4) if processed_count > 0 else 0.0
        metrics["api_failure_count"] = failure_count
        metrics["api_failure_rate"] = round(failure_count / processed_count, 4) if processed_count > 0 else 0.0
        metrics["total_cases_processed"] = processed_count
        metrics["successful_cases"] = success_count
        metrics["failed_cases"] = failure_count

        # Save metrics
        metrics_path = os.path.join(metrics_dir, f"{model_id}_metrics.json")
        write_json(metrics_path, metrics)
        logger.info("Metrics: %s", json.dumps(metrics, ensure_ascii=False))
        logger.info("Success: %d/%d (%.1f%%), Failed: %d",
                   success_count, processed_count,
                   (success_count / processed_count * 100) if processed_count > 0 else 0,
                   failure_count)
        return metrics

    # If there are no records, return default zero metrics
    logger.warning("No records produced for model %s.", client.name)
    return {
        "model_name": client.name,
        "num_cases": 0,
        "num_records": 0,
        "num_runs": args.num_runs,
        "accuracy_at_1": 0.0,
        "accuracy_at_3": 0.0,
        "accuracy_at_5": 0.0,
        "valid_json_rate": 0.0,
        "hallucination_rate": 0.0,
        "average_confidence": 0.0,
        "valid_inference_process_rate": 0.0,
        "average_inference_process_steps": 0.0,
        "truncated_response_count": 0,
        "truncated_response_rate": 0.0,
        "average_prompt_tokens": 0.0,
        "average_completion_tokens": 0.0,
        "average_reasoning_tokens": 0.0,
        "average_total_tokens": 0.0,
        "use_rag": bool(use_rag and rag_retriever is not None),
        "rag_top_k": int(args.rag_top_k or 0) if use_rag and rag_retriever is not None else 0,
        "api_success_rate": 0.0,
        "api_failure_count": 0,
        "api_failure_rate": 0.0,
    }


def main() -> None:
    """
    Program main entry point.

    1. Parse command-line arguments
    2. Verify that the model name exists in the config file
    3. Call run_single_model to execute the experiment
    4. Print the final metrics results
    """
    args = parse_args()

    config_path = _resolve_path(args.config_path)
    available = list_available_models(config_path)
    if args.model_name not in available:
        print(
            f"ERROR: model '{args.model_name}' not found in {config_path}. "
            f"Available models: {available}",
            file=sys.stderr,
        )
        sys.exit(2)

    metrics = run_single_model(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if metrics.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
