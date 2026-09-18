"""
evaluator.py: evaluation module for RCA experiments.

This module provides the core functionality for quantitatively evaluating the
large-model root cause analysis (RCA) results, mainly including:
- hit@k metric computation (whether the prediction hits the true root cause)
- hallucination rate detection (whether the prediction exceeds the candidate set)
- evaluation and normalization of a single record
- summary statistics over multiple records (computing average metrics)
- flattening records into CSV-format rows for export

Throughout the experiment pipeline, it plays the role of "referee", turning the
model's text output into quantifiable performance metrics.
"""

import json
from typing import Any, Dict, Iterable, List, Optional

from .utils import normalize_var_name


def is_final_response_truncated(record: Dict[str, Any]) -> bool:
    """Determine whether the final response submitted to the evaluator is truncated, and tolerate legacy Agent summary records."""
    finish_reason = str(record.get("finish_reason") or "").strip().lower()
    if finish_reason == "length":
        return True

    # Legacy Agent summaries write any intermediate-round truncation into was_truncated.
    # When the Agent has moved to later rounds and finally stops normally, this field
    # only represents a historical event and must not block the final result from evaluation.
    agent_status = str(record.get("agent_status") or "").strip().lower()
    if agent_status and agent_status != "disabled":
        return False
    return bool(record.get("was_truncated"))


def _normalize_var_list(vals: Any) -> List[str]:
    """
    Normalize a variable list of any format into a string list.
    
    Normalization rules:
    - None -> empty list
    - string -> one-element list
    - list/tuple -> process each element
    - all strings are uppercased and whitespace-stripped, so comparisons have no format differences
    """
    if vals is None:
        return []
    if isinstance(vals, str):
        vals = [vals]
    if isinstance(vals, (list, tuple)):
        out = []
        for v in vals:
            if v is None:
                continue
            out.append(normalize_var_name(v))
        return [v for v in out if v]
    return []


def _normalize_single(v: Any) -> Optional[str]:
    """
    Normalize a single variable name.
    
    Rule: uppercase, strip whitespace, drop None/empty strings.
    """
    if v is None:
        return None
    s = normalize_var_name(v)
    return s or None


def is_hit_at_k(predicted: List[str], gt_vars: List[str], k: int) -> int:
    """
    Compute the hit@k metric.
    
    Criterion: if any true root-cause variable (gt_vars) is among the first k
    variables of the predicted list, it is a hit (returns 1); otherwise a miss (returns 0).
    """
    top = predicted[:k]
    gt_set = set(gt_vars)
    return 1 if any(v in gt_set for v in top) else 0


def is_hallucination(predicted: List[str], candidate_vars: List[str]) -> int:
    """
    Detect hallucination.
    
    Criterion: if any variable in the predicted list is not in the given candidate
    set (candidate_vars), it is considered a hallucination (returns 1); otherwise normal (returns 0).
    """
    canon_candidates = {normalize_var_name(c) for c in candidate_vars}
    for v in predicted:
        vv = normalize_var_name(v or "")
        # UNKNOWN is the abstention marker after the Agent exhausts retries; it is counted as a miss, not as an out-of-candidate hallucination.
        if vv == "UNKNOWN":
            continue
        if vv and vv not in canon_candidates:
            return 1
    return 0


def _is_valid_inference_process(
    process: Any, candidate_vars: List[str]
) -> bool:
    """Determine whether the four-step reasoning process is complete; used for unified statistics across the baseline and Agent groups."""
    if not isinstance(process, list) or len(process) != 4:
        return False
    candidates = set(_normalize_var_list(candidate_vars))
    for expected_step, step in enumerate(process, start=1):
        if not isinstance(step, dict) or step.get("step") != expected_step:
            return False
        if not str(step.get("stage") or "").strip():
            return False
        if len(str(step.get("analysis") or "").strip()) < 10:
            return False
        evidence = _normalize_var_list(step.get("evidence_variables"))
        if not evidence or any(value not in candidates for value in evidence):
            return False
    return True


def evaluate_case(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single experiment record.

    This function processes one complete experiment record, mainly doing the following:
    1. Parses and normalizes the predicted results, ground-truth labels, and candidate set from the record.
    2. Unifies the structure of primary and predicted:
       - If there is a primary but no predicted, use primary as predicted
       - If there is a predicted but no primary, use predicted[0] as primary
       - If both are empty, keep the empty prediction and count it as a miss; never forge a model answer with the candidate Top-1
    3. Again ensures primary is at the head of the predicted list.
    4. Computes hit@1, hit@3, hit@5, and hallucination rate.
    5. Backfills auxiliary fields (such as root_cause_name, evidence_variables).
    
    Returns the record dict after evaluation and field backfilling.
    """
    parsed = record.get("parsed_response", {}) or {}
    predicted = _normalize_var_list(parsed.get("predicted_root_causes", []))
    gt_vars = _normalize_var_list(record.get("gt_vars", []))
    top10_vars = _normalize_var_list(record.get("top10_vars", []))

    primary = _normalize_single(parsed.get("primary_root_cause"))

    # UNKNOWN is the formal failure semantics and must bypass the regular Top-1 fallback to avoid polluting accuracy.
    is_unknown = primary == "UNKNOWN"

    # This only fixes the inconsistency of the same model answer across the two fields. If both are
    # empty, they must stay empty and be counted as a miss; backfilling with the TA-RCA Top-1 would
    # wrongly credit the candidate model's prior performance to the LLM's reasoning, polluting the baseline.
    if not predicted and primary and not is_unknown:
        predicted = [primary]
    elif predicted and not primary:
        primary = predicted[0]

    # Defense again: ensure predicted and primary are consistent (primary must be in predicted)
    if primary and primary not in predicted and not is_unknown:
        predicted = [primary] + [v for v in predicted if v != primary]

    # Even if parsed_response is already clipped to the candidate list, keep a defensive layer here
    hit_at_1 = is_hit_at_k(predicted, gt_vars, 1)
    hit_at_3 = is_hit_at_k(predicted, gt_vars, 3)
    hit_at_5 = is_hit_at_k(predicted, gt_vars, 5)
    halluc = is_hallucination(predicted, top10_vars)

    parsed_out = dict(parsed)
    parsed_out["predicted_root_causes"] = predicted[:5]
    parsed_out["primary_root_cause"] = primary

    # If root_cause_name is missing, try to look it up from top10_details
    if not parsed_out.get("root_cause_name") and primary:
        top10_details = record.get("top10_details", []) or []
        found_name = _lookup_variable_name(primary, top10_details)
        if found_name:
            parsed_out["root_cause_name"] = found_name

    # Ensure the evidence_variables field exists
    evidence = _normalize_var_list(parsed_out.get("evidence_variables"))
    if not evidence:
        parsed_out["evidence_variables"] = list(parsed_out["predicted_root_causes"])

    result = dict(record)
    result["parsed_response"] = parsed_out
    result["is_hit_at_1"] = hit_at_1
    result["is_hit_at_3"] = hit_at_3
    result["is_hit_at_5"] = hit_at_5
    result["is_hallucination"] = halluc
    return result


def _lookup_variable_name(var_id: str, top10_details: List[Dict[str, Any]]) -> str:
    """
    Look up the real name of a variable by its ID from the top10_details list.
    
    Mainly used to map an ID such as V27 to the specific sensor/actuator name.
    """
    var_id_upper = var_id.upper().replace(" ", "")
    for detail in top10_details:
        if not isinstance(detail, dict):
            continue
        detail_var = str(detail.get("var") or detail.get("var_id") or detail.get("variable") or "").upper().replace(" ", "")
        if detail_var == var_id_upper:
            name = str(detail.get("name") or "")
            if name:
                return name
    return ""


def compute_metrics(records: List[Dict[str, Any]], model_name: str, num_runs: int = 1) -> Dict[str, Any]:
    """
    Summarize multiple experiment records and compute overall evaluation metrics.
    
    Computed metrics include:
    - accuracy_at_k (k=1,3,5): average hit@k over all records
    - hallucination_rate: average hallucination rate over all records
    - valid_json_rate: proportion of records successfully parsed as valid JSON
    - average_confidence: average model confidence over all valid records
    - covered-case metrics: computed only on cases where the true root cause is within the Top-10 candidates
    """
    n = len(records)
    if n == 0:
        return {
            "model_name": model_name,
            "num_cases": 0,
            "num_records": 0,
            "num_runs": num_runs,
            "accuracy_at_1": 0.0,
            "accuracy_at_3": 0.0,
            "accuracy_at_5": 0.0,
            "valid_json_rate": 0.0,
            "hallucination_rate": 0.0,
            "average_confidence": 0.0,
            "parse_failure_count": 0,
            "empty_response_count": 0,
            "valid_inference_process_rate": 0.0,
            "average_inference_process_steps": 0.0,
            "truncated_response_count": 0,
            "truncated_response_rate": 0.0,
            "average_prompt_tokens": 0.0,
            "average_completion_tokens": 0.0,
            "average_reasoning_tokens": 0.0,
            "average_total_tokens": 0.0,
            "covered_case_count": 0,
            "covered_record_count": 0,
            "covered_case_ratio": 0.0,
            "covered_accuracy_at_1": 0.0,
            "covered_accuracy_at_3": 0.0,
            "covered_accuracy_at_5": 0.0,
        }

    # Compute the sum of each metric
    acc1 = sum(r.get("is_hit_at_1", 0) for r in records) / n
    acc3 = sum(r.get("is_hit_at_3", 0) for r in records) / n
    acc5 = sum(r.get("is_hit_at_5", 0) for r in records) / n
    halluc = sum(r.get("is_hallucination", 0) for r in records) / n

    # Compute the valid-JSON rate, parse-failure count, and average confidence
    valid_json_count = 0
    parse_failure_count = 0
    empty_response_count = 0
    confidences: List[float] = []
    valid_inference_process_count = 0
    inference_process_step_count = 0
    truncated_response_count = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    reasoning_tokens_total = 0
    total_tokens_total = 0
    for r in records:
        pr = r.get("parsed_response", {}) or {}
        if pr.get("valid_json"):
            valid_json_count += 1
        else:
            parse_failure_count += 1
        
        raw_resp = r.get("raw_response", "")
        if not raw_resp or not raw_resp.strip():
            empty_response_count += 1
        
        c = pr.get("confidence")
        if isinstance(c, (int, float)) and c is not None:
            confidences.append(float(c))

        process = pr.get("inference_process", []) or []
        if isinstance(process, list):
            inference_process_step_count += len(process)
        if _is_valid_inference_process(process, r.get("top10_vars", [])):
            valid_inference_process_count += 1

        if is_final_response_truncated(r):
            truncated_response_count += 1
        prompt_tokens_total += int(r.get("prompt_tokens", 0) or 0)
        completion_tokens_total += int(r.get("completion_tokens", 0) or 0)
        reasoning_tokens_total += int(r.get("reasoning_tokens", 0) or 0)
        total_tokens_total += int(r.get("total_tokens", 0) or 0)

    valid_json_rate = valid_json_count / n if n else 0.0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # Compute the covered-case metrics
    covered_records = []
    for r in records:
        gt_vars = _normalize_var_list(r.get("gt_vars", []))
        top10_vars = _normalize_var_list(r.get("top10_vars", []))
        # covered = any gt_var appears in top10_vars
        is_covered = any(gt in top10_vars for gt in gt_vars) if gt_vars else False
        if is_covered:
            covered_records.append(r)
    
    # num_cases is the deduplicated data-case count; num_records is the number of
    # final records produced by repeated experiments. For example, 20 cases repeated
    # 3 times yield 20 and 60 respectively.
    unique_case_ids = {
        str(r.get("case_id")) for r in records if r.get("case_id") is not None
    }
    covered_case_ids = {
        str(r.get("case_id"))
        for r in covered_records
        if r.get("case_id") is not None
    }
    covered_record_count = len(covered_records)
    covered_case_count = len(covered_case_ids)
    covered_ratio = (
        covered_case_count / len(unique_case_ids) if unique_case_ids else 0.0
    )
    
    covered_acc1 = 0.0
    covered_acc3 = 0.0
    covered_acc5 = 0.0
    if covered_record_count > 0:
        covered_acc1 = sum(r.get("is_hit_at_1", 0) for r in covered_records) / covered_record_count
        covered_acc3 = sum(r.get("is_hit_at_3", 0) for r in covered_records) / covered_record_count
        covered_acc5 = sum(r.get("is_hit_at_5", 0) for r in covered_records) / covered_record_count

    # Compute the Iterative Self-refinement Agent metrics
    total_iterations = 0
    first_round_success = 0
    second_round_attempts = 0
    second_round_recoveries = 0
    third_round_attempts = 0
    third_round_recoveries = 0
    agent_failed = 0
    validation_passed_count = 0
    
    for r in records:
        iteration_count = r.get("iteration_count", 1)
        final_iteration = r.get("final_iteration", 1)
        validation_passed = r.get("validation_passed")
        agent_status = r.get("agent_status", "")
        agent_failed_flag = r.get("agent_failed", False)
        
        total_iterations += iteration_count
        
        if final_iteration == 1 and validation_passed is True:
            first_round_success += 1
        
        if iteration_count >= 2:
            second_round_attempts += 1
            if final_iteration == 2 and validation_passed is True:
                second_round_recoveries += 1
        
        if iteration_count >= 3:
            third_round_attempts += 1
            if final_iteration == 3 and validation_passed is True:
                third_round_recoveries += 1
        
        if agent_failed_flag:
            agent_failed += 1
        
        if validation_passed is True:
            validation_passed_count += 1

    average_iteration_count = total_iterations / n if n else 0.0
    first_round_success_rate = first_round_success / n if n else 0.0
    second_round_recovery_rate = second_round_recoveries / second_round_attempts if second_round_attempts else 0.0
    third_round_recovery_rate = third_round_recoveries / third_round_attempts if third_round_attempts else 0.0
    agent_failed_rate = agent_failed / n if n else 0.0
    validation_pass_rate = validation_passed_count / n if n else 0.0

    return {
        "model_name": model_name,
        "num_cases": len(unique_case_ids),
        "num_records": n,
        "num_runs": num_runs,
        "accuracy_at_1": round(acc1, 4),
        "accuracy_at_3": round(acc3, 4),
        "accuracy_at_5": round(acc5, 4),
        "valid_json_rate": round(valid_json_rate, 4),
        "hallucination_rate": round(halluc, 4),
        "average_confidence": round(avg_conf, 4),
        "parse_failure_count": parse_failure_count,
        "empty_response_count": empty_response_count,
        "valid_inference_process_rate": round(
            valid_inference_process_count / n, 4
        ),
        "average_inference_process_steps": round(
            inference_process_step_count / n, 4
        ),
        "truncated_response_count": truncated_response_count,
        "truncated_response_rate": round(truncated_response_count / n, 4),
        "average_prompt_tokens": round(prompt_tokens_total / n, 2),
        "average_completion_tokens": round(completion_tokens_total / n, 2),
        "average_reasoning_tokens": round(reasoning_tokens_total / n, 2),
        "average_total_tokens": round(total_tokens_total / n, 2),
        "covered_case_count": covered_case_count,
        "covered_record_count": covered_record_count,
        "covered_case_ratio": round(covered_ratio, 4),
        "covered_accuracy_at_1": round(covered_acc1, 4),
        "covered_accuracy_at_3": round(covered_acc3, 4),
        "covered_accuracy_at_5": round(covered_acc5, 4),
        "average_iteration_count": round(average_iteration_count, 4),
        "first_round_success_rate": round(first_round_success_rate, 4),
        "second_round_attempt_count": second_round_attempts,
        "second_round_recovery_rate": round(second_round_recovery_rate, 4),
        "third_round_attempt_count": third_round_attempts,
        "third_round_recovery_rate": round(third_round_recovery_rate, 4),
        "agent_failed_count": agent_failed,
        "agent_failed_rate": round(agent_failed_rate, 4),
        "validation_pass_rate": round(validation_pass_rate, 4),
    }


def build_parsed_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten a complex nested experiment record into simple dict rows for export to a CSV file.
    
    Main processing:
    - Extracts each field from parsed_response
    - Truncates and cleans long text fields (such as reasoning) to avoid breaking the CSV format
    - Formats boolean fields (such as is_valid_prediction)
    - Assembles them into a flat row structure
    """
    rows: List[Dict[str, Any]] = []
    for r in records:
        pr = r.get("parsed_response", {}) or {}
        # Clean fields that may contain special characters such as newlines
        reasoning = _sanitize_text(pr.get("reasoning", ""), max_len=500)
        # The reasoning process is written as complete JSON into a single CSV cell without
        # the 500-char truncation; the CSV writer handles quote escaping, and the
        # four-step structure can later be restored directly via json.loads.
        inference_process = json.dumps(
            pr.get("inference_process", []) or [],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        content_excerpt = _sanitize_text(r.get("content", ""), max_len=500)
        reasoning_content_excerpt = _sanitize_text(r.get("reasoning_content", ""), max_len=500)
        raw_response_excerpt = _sanitize_text(r.get("raw_response", ""), max_len=500)

        root_cause_name = _sanitize_text(pr.get("root_cause_name", ""), max_len=100)
        numerical_evidence = _sanitize_text(pr.get("numerical_evidence", ""), max_len=500)
        temporal_evidence = _sanitize_text(pr.get("temporal_evidence", ""), max_len=500)
        type_aware_reasoning = _sanitize_text(pr.get("type_aware_reasoning", ""), max_len=500)
        process_relation_reasoning = _sanitize_text(pr.get("process_relation_reasoning", ""), max_len=500)
        why_not_other_candidates = _sanitize_text(pr.get("why_not_other_candidates", ""), max_len=500)
        uncertainty_analysis = _sanitize_text(pr.get("uncertainty_analysis", ""), max_len=500)
        
        # Handle boolean fields
        is_valid_prediction = pr.get("is_valid_prediction")
        if is_valid_prediction is not None:
            is_valid_prediction = str(bool(is_valid_prediction))
        else:
            is_valid_prediction = ""

        # Handle API status and error info
        api_ok = r.get("api_ok", False)
        api_success = 1 if api_ok else 0

        error_type = r.get("error_type", "") or ""
        error_message = _sanitize_text(r.get("error_message", "") or "", max_len=500)

        rows.append(
            {
                "case_id": r.get("case_id"),
                "model_name": r.get("model_name", ""),
                "run_id": r.get("run_id", 1),
                "gt_vars": ",".join(_normalize_var_list(r.get("gt_vars", []))),
                "top10_vars": ",".join(_normalize_var_list(r.get("top10_vars", []))),
                "primary_root_cause": pr.get("primary_root_cause", "") or "",
                "predicted_root_causes": ",".join(pr.get("predicted_root_causes", []) or []),
                "is_hit_at_1": r.get("is_hit_at_1", 0),
                "is_hit_at_3": r.get("is_hit_at_3", 0),
                "is_hit_at_5": r.get("is_hit_at_5", 0),
                "reasoning": reasoning,
                "inference_process": inference_process,
                "inference_process_step_count": len(
                    pr.get("inference_process", []) or []
                ),
                "confidence": pr.get("confidence"),
                "content_excerpt": content_excerpt,
                "reasoning_content_excerpt": reasoning_content_excerpt,
                "raw_response_excerpt": raw_response_excerpt,
                "finish_reason": r.get("finish_reason", ""),
                "was_truncated": 1 if is_final_response_truncated(r) else 0,
                "had_truncated_iteration": 1 if r.get("had_truncated_iteration") else 0,
                "prompt_tokens": r.get("prompt_tokens", 0),
                "completion_tokens": r.get("completion_tokens", 0),
                "reasoning_tokens": r.get("reasoning_tokens", 0),
                "total_tokens": r.get("total_tokens", 0),
                "root_cause_name": root_cause_name,
                "numerical_evidence": numerical_evidence,
                "temporal_evidence": temporal_evidence,
                "type_aware_reasoning": type_aware_reasoning,
                "process_relation_reasoning": process_relation_reasoning,
                "why_not_other_candidates": why_not_other_candidates,
                "uncertainty_analysis": uncertainty_analysis,
                "is_valid_prediction": is_valid_prediction,
                "api_success": api_success,
                "error_type": error_type,
                "error_message": error_message,
                "iteration_count": r.get("iteration_count", 1),
                "final_iteration": r.get("final_iteration", 1),
                "retry_count": r.get("retry_count", 0),
                "validation_passed": str(r.get("validation_passed")) if r.get("validation_passed") is not None else "",
                "retry_reasons": ",".join(r.get("retry_reasons", [])),
                "agent_status": r.get("agent_status", ""),
                "agent_failed": 1 if r.get("agent_failed", False) else 0,
                "final_prompt_type": r.get("final_prompt_type", "initial"),
                "use_dual_branch_fusion": 1 if r.get("use_dual_branch_fusion") else 0,
                "fusion_strategy": r.get("fusion_strategy", ""),
                "evidence_branch_primary": r.get("evidence_branch_primary", ""),
                "rag_branch_primary": r.get("rag_branch_primary", ""),
                "branch_primary_agreement": 1 if r.get("branch_primary_agreement") else 0,
                "rag_supplemented_candidates": ",".join(
                    r.get("rag_supplemented_candidates", []) or []
                ),
                "evidence_branch_success": 1 if r.get("evidence_branch_success") else 0,
                "rag_branch_success": 1 if r.get("rag_branch_success") else 0,
            }
        )
    return rows


def _sanitize_text(text: Any, max_len: int = 500) -> str:
    """
    Clean text for safe writing to a CSV file.
    
    Main operations:
    - Replaces newlines and carriage returns with spaces to prevent CSV row misalignment
    - Truncates overly long text to the specified maximum length
    - Handles None values
    """
    if text is None:
        return ""
    s = str(text)[:max_len]
    return s.replace("\r", " ").replace("\n", " ")
