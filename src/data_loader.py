"""Data loader: reads case data for the LLM RCA experiment."""

import os
from typing import Any, Dict, List

from .utils import load_jsonl


def load_cases(data_path: str, max_cases: int = -1) -> List[Dict[str, Any]]:
    """
    Load a list of cases from a JSONL file. Each case should contain at least:
    case_id, prompt (or prompt-related fields), gt_vars, top10_vars.

    When `max_cases` is positive, only the first `max_cases` entries are returned.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在：{data_path}")

    cases = load_jsonl(data_path)

    if max_cases > 0:
        cases = cases[:max_cases]

    return cases


def _normalize_str_list(v: Any) -> List[str]:
    """Normalize a value into a list of strings. Supports list / tuple / comma-separated string / None."""
    if v is None:
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    return []


def _top10_details_from_top_k(top_k: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert the top_k_variables structure into the unified top10_details format."""
    details: List[Dict[str, Any]] = []
    for item in top_k[:10]:
        if not isinstance(item, dict):
            continue
        var_id = str(item.get("var_id") or item.get("var") or item.get("variable") or "").strip()
        name = str(item.get("name") or "").strip()
        vtype = str(item.get("type") or "").strip()
        score = item.get("case_level_score")
        if score is None:
            score = item.get("score")
        description = str(item.get("description") or "").strip()
        ts = item.get("time_series") or []

        details.append(
            {
                "var": var_id,
                "name": name,
                "type": vtype,
                "score": score,
                "description": description,
                "time_series": ts,
                "raw_item": item,
            }
        )
    return details


def normalize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single case into the standard structure used internally by the framework.

    Normalized structure:
        {
            "case_id": int/str,
            "prompt": str,
            "gt_vars": List[str],
            "top10_vars": List[str],
            "top10_details": List[Dict],
            "top10_names": List[str],  # readable name of each top10 variable
            "ground_truth": dict,     # original ground_truth preserved
        }
    """
    normalized: Dict[str, Any] = {}

    normalized["case_id"] = case.get("case_id", case.get("id", 0))
    # A cross-dataset adapter may explicitly declare system and task; when these
    # fields are missing in old SWaT data, downstream still uses the original
    # defaults, so historical experiment behavior is unchanged.
    normalized["system_name"] = str(case.get("system_name") or "").strip()
    normalized["task"] = str(case.get("task") or "").strip()

    # Prompt: prefer the complete prompt that comes with the case
    prompt = case.get("prompt")
    if not prompt:
        prompt = case.get("llm_prompt", "")
    normalized["prompt"] = prompt or ""

    # Ground-truth root cause variables: read from ground_truth.gt_vars or gt_vars
    gt_vars = None
    ground_truth = case.get("ground_truth")
    if isinstance(ground_truth, dict):
        gt_vars = ground_truth.get("gt_vars")
        gt_names = ground_truth.get("gt_names")
        normalized["ground_truth"] = ground_truth
    else:
        gt_names = None

    if gt_vars is None:
        gt_vars = case.get("gt_vars")
    normalized["gt_vars"] = _normalize_str_list(gt_vars)

    # Ground-truth root cause names (optional, for debugging/display)
    normalized["gt_names"] = _normalize_str_list(gt_names)

    # Top-10 candidate variables: parsed from top_k_variables
    top_k = case.get("top_k_variables")
    top10_vars: List[str] = []
    top10_details: List[Dict[str, Any]] = []
    top10_names: List[str] = []

    if isinstance(top_k, list) and top_k:
        top10_details = _top10_details_from_top_k(top_k)
        top10_vars = [d["var"] for d in top10_details if d["var"]]
        top10_names = [d["name"] for d in top10_details if d["name"]]

    # Compatible with the old format: direct top10_vars / top10_details
    if not top10_vars:
        top10_vars = _normalize_str_list(case.get("top10_vars"))
    if not top10_details:
        raw_details = case.get("top10_details") or case.get("candidate_details") or []
        if isinstance(raw_details, list):
            normalized_details: List[Dict[str, Any]] = []
            for item in raw_details[:10]:
                if isinstance(item, dict):
                    normalized_details.append(
                        {
                            "var": str(item.get("var") or item.get("variable") or item.get("name") or "").strip(),
                            "name": str(item.get("name") or "").strip(),
                            "type": str(item.get("type") or "").strip(),
                            "score": item.get("score"),
                            "description": str(item.get("description") or item.get("semantic") or "").strip(),
                            "time_series": item.get("time_series") or [],
                            "raw_item": item,
                        }
                    )
                else:
                    normalized_details.append(
                        {
                            "var": str(item).strip(),
                            "name": "",
                            "type": "",
                            "score": None,
                            "description": "",
                            "time_series": [],
                            "raw_item": item,
                        }
                    )
            top10_details = normalized_details

    # Fallback: if Top-10 is still empty, find candidates from top_k_variables or prompt as best as possible
    if not top10_vars:
        fallback = case.get("candidate_vars") or []
        top10_vars = _normalize_str_list(fallback)

    # Require top10_vars to be non-empty: if it is still empty, log a warning but
    # do not throw, so the downstream evaluator can continue (this case should
    # normally not occur).
    normalized["top10_vars"] = top10_vars or []
    normalized["top10_details"] = top10_details or []
    normalized["top10_names"] = top10_names or []

    # Extra fields: preserve the original structure
    extra: Dict[str, Any] = {}
    for k, v in case.items():
        if k not in normalized:
            extra[k] = v
    normalized["extra"] = extra

    return normalized
