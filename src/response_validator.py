"""Response Validator: Validates LLM RCA outputs against structural and content rules.

Ground-truth labels are only used for offline evaluation and are never used by the Iterative Self-refinement Agent.

This module implements the validation logic for Iterative Self-refinement Agent, ensuring:
1. Output is valid JSON
2. Variables are from Top-10 candidates only (no hallucination)
3. Confidence is within valid range
4. All required explanation fields are present and non-empty
5. No access to ground_truth, gt_vars, or gt_names during validation
"""

from typing import Any, Dict, List

from .utils import normalize_var_name


def validate_response(
    parsed: Dict[str, Any],
    top10_vars: List[str],
    min_confidence: float = 0.5,
    *,
    api_success: bool = True,
    finish_reason: str = "",
) -> Dict[str, Any]:
    """
    Validate an LLM response against structural and content rules.
    
    Ground-truth labels are only used for offline evaluation and are never used by the Iterative Self-refinement Agent.
    
    Args:
        parsed: Parsed response dictionary from response_parser.parse_response()
        top10_vars: List of valid candidate variable IDs
        min_confidence: Minimum acceptable confidence score (default: 0.5)
        api_success: Whether the model API call completed successfully
        finish_reason: Provider finish reason; ``length`` means output was truncated
    
    Returns:
        Dict with "is_valid" (bool) and "retry_reasons" (list of strings)
    """
    retry_reasons: List[str] = []
    canon_candidates = {normalize_var_name(v) for v in top10_vars if v}

    # When the API call fails or the output is truncated by the length limit,
    # the body is not a complete model conclusion even if it can still be parsed
    # as a fallback; it must enter the next self-refinement round instead of
    # being wrongly judged as successful by structural validation.
    if not api_success:
        retry_reasons.append("api_call_failed")
    if str(finish_reason or "").strip().lower() == "length":
        retry_reasons.append("output_truncated")

    # 1. Check valid_json flag
    if not parsed.get("valid_json"):
        retry_reasons.append("invalid_json")

    # 2. Check predicted_root_causes is non-empty list
    predicted = parsed.get("predicted_root_causes") or []
    if not isinstance(predicted, list) or len(predicted) == 0:
        retry_reasons.append("empty_predicted_root_causes")

    # 3. Check primary_root_cause is non-empty
    primary = parsed.get("primary_root_cause")
    if not primary:
        retry_reasons.append("empty_primary_root_cause")

    # 4. Check primary_root_cause is in top10_vars
    if primary:
        primary_norm = normalize_var_name(primary)
        if primary_norm and primary_norm not in canon_candidates:
            retry_reasons.append("primary_not_in_top10")

    # 5. Check all predicted_root_causes are in top10_vars
    if isinstance(predicted, list):
        for v in predicted:
            vv = normalize_var_name(v)
            if vv and vv not in canon_candidates:
                retry_reasons.append("predicted_contains_outside_var")
                break

    # 6. Check primary_root_cause == predicted_root_causes[0]
    if isinstance(predicted, list) and len(predicted) > 0 and primary:
        primary_norm = normalize_var_name(primary)
        first_pred_norm = normalize_var_name(predicted[0])
        if primary_norm != first_pred_norm:
            retry_reasons.append("primary_not_first_in_predicted")

    # 7. Check confidence is valid number in [0, 1] and >= min_confidence
    confidence = parsed.get("confidence")
    if confidence is None:
        retry_reasons.append("missing_confidence")
    else:
        try:
            c = float(confidence)
            if not (0.0 <= c <= 1.0):
                retry_reasons.append("confidence_out_of_range")
            elif c < min_confidence:
                retry_reasons.append("low_confidence")
        except (TypeError, ValueError):
            retry_reasons.append("confidence_not_number")

    # 8. Check reasoning is non-empty and not too short
    reasoning = parsed.get("reasoning") or ""
    if not reasoning or len(str(reasoning).strip()) < 10:
        retry_reasons.append("reasoning_too_short")

    # 9. Check evidence_variables is list and all from Top-10
    evidence_vars = parsed.get("evidence_variables") or []
    if not isinstance(evidence_vars, list):
        retry_reasons.append("evidence_variables_not_list")
    else:
        for v in evidence_vars:
            vv = normalize_var_name(v)
            if vv and vv not in canon_candidates:
                retry_reasons.append("evidence_variable_not_in_top10")
                break

    # 10. Check the structured inference process. The four steps correspond to
    # candidate comparison, temporal judgment, process propagation, and the final
    # decision; every step's evidence variables must come from the Top-10.
    inference_process = parsed.get("inference_process") or []
    if not isinstance(inference_process, list) or len(inference_process) != 4:
        retry_reasons.append("invalid_inference_process_step_count")
    else:
        for expected_step, step in enumerate(inference_process, start=1):
            if not isinstance(step, dict):
                retry_reasons.append("invalid_inference_process_step")
                break
            if step.get("step") != expected_step:
                retry_reasons.append("invalid_inference_process_step_order")
                break
            if not str(step.get("stage") or "").strip():
                retry_reasons.append("missing_inference_process_stage")
                break
            if len(str(step.get("analysis") or "").strip()) < 10:
                retry_reasons.append("inference_process_analysis_too_short")
                break
            step_evidence = step.get("evidence_variables")
            if not isinstance(step_evidence, list) or not step_evidence:
                retry_reasons.append("invalid_inference_process_evidence")
                break
            if any(
                normalize_var_name(value) not in canon_candidates
                for value in step_evidence
            ):
                retry_reasons.append("inference_process_evidence_not_in_top10")
                break

    # 11. Check required explanation fields are non-empty
    required_fields = [
        ("root_cause_name", "missing_root_cause_name"),
        ("numerical_evidence", "missing_numerical_evidence"),
        ("temporal_evidence", "missing_temporal_evidence"),
        ("type_aware_reasoning", "missing_type_aware_reasoning"),
        ("process_relation_reasoning", "missing_process_relation_reasoning"),
        ("why_not_other_candidates", "missing_why_not_other_candidates"),
    ]
    for field_name, error_code in required_fields:
        value = parsed.get(field_name)
        if not value or len(str(value).strip()) == 0:
            retry_reasons.append(error_code)

    return {
        "is_valid": len(retry_reasons) == 0,
        "retry_reasons": retry_reasons,
        "confidence": confidence,
        "primary_root_cause": primary,
        "predicted_root_causes": predicted,
        "reasoning_length": len(str(reasoning).strip()) if reasoning else 0,
    }
