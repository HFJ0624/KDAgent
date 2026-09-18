"""Response parser module: extract structured root-cause fields from raw LLM text.

This module is responsible for turning the model's free-text output into a structured
RCA result, and is the bridge connecting `ModelClient` (which returns raw text) and
`Evaluator` (which computes metrics).

## Three-level fault-tolerant parsing strategy

Because different models produce unstable output formats, this parser tries the
following priorities in order and returns immediately once any level succeeds:

1. **XML tag parsing** (``_parse_tagged``): prefers to match
   ``<answer>`` / ``<reasoning>`` / ``<semantic_observations>`` and other
   XML-style tags; this is the preferred path for the XML format explicitly
   specified in the Prompt.
2. **JSON object parsing** (``_extract_json_object`` + ``json.loads``):
   locates and parses the first complete JSON object in the text; supports
   ```json``` fences, regex matching, and balanced-bracket scanning.
3. **Regex fallback parsing** (``_VAR_TOKEN_RE``): when both of the above fail,
   degrades to grabbing all ``V\\d+`` identifiers from the whole text as the
   predicted result, so even a pure-text description can still yield
   evaluable structured fields.

## Main public functions

* :func:`parse_response`: parses a single raw response and returns a dict
  containing ``predicted_root_causes``, ``primary_root_cause``,
  ``reasoning``, ``confidence`` and other fields.
* :func:`clamp_to_candidates`: intersects the parsed variable set with the
  candidate set, filtering out "hallucinated" variables for downstream evaluation.
"""

import json
import re
from typing import Any, Dict, List, Optional

from .utils import normalize_var_name


_VAR_TOKEN_RE = re.compile(r"\bV\s*\d+\b", re.IGNORECASE)
# Locate a JSON object in the response text (possibly nested)
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")
# Look for XML-style <answer>V27</answer> output
_ANSWER_TAG_RE = re.compile(
    r"<\s*(?:answer|final_answer|root_cause|primary_root_cause|predicted_root_causes)\b[^>]*>"
    r"([\s\S]*?)"
    r"</\s*(?:answer|final_answer|root_cause|primary_root_cause|predicted_root_causes)\s*>",
    re.IGNORECASE,
)
# Look for the <semantic_observations> / <reasoning> auxiliary fields
_SEMANTIC_TAG_RE = re.compile(
    r"<\s*semantic_observations\b[^>]*>([\s\S]*?)</\s*semantic_observations\s*>",
    re.IGNORECASE,
)
_REASONING_TAG_RE = re.compile(
    r"<\s*reasoning\b[^>]*>([\s\S]*?)</\s*reasoning\s*>",
    re.IGNORECASE,
)

# Look for the XML tags of the explanatory fields
_ROOT_CAUSE_NAME_TAG_RE = re.compile(
    r"<\s*root_cause_name\b[^>]*>([\s\S]*?)</\s*root_cause_name\s*>",
    re.IGNORECASE,
)
_NUMERICAL_EVIDENCE_TAG_RE = re.compile(
    r"<\s*numerical_evidence\b[^>]*>([\s\S]*?)</\s*numerical_evidence\s*>",
    re.IGNORECASE,
)
_TEMPORAL_EVIDENCE_TAG_RE = re.compile(
    r"<\s*temporal_evidence\b[^>]*>([\s\S]*?)</\s*temporal_evidence\s*>",
    re.IGNORECASE,
)
_TYPE_AWARE_REASONING_TAG_RE = re.compile(
    r"<\s*type_aware_reasoning\b[^>]*>([\s\S]*?)</\s*type_aware_reasoning\s*>",
    re.IGNORECASE,
)
_PROCESS_RELATION_REASONING_TAG_RE = re.compile(
    r"<\s*process_relation_reasoning\b[^>]*>([\s\S]*?)</\s*process_relation_reasoning\s*>",
    re.IGNORECASE,
)
_WHY_NOT_OTHERS_TAG_RE = re.compile(
    r"<\s*why_not_other_candidates\b[^>]*>([\s\S]*?)</\s*why_not_other_candidates\s*>",
    re.IGNORECASE,
)
_UNCERTAINTY_TAG_RE = re.compile(
    r"<\s*uncertainty_analysis\b[^>]*>([\s\S]*?)</\s*uncertainty_analysis\s*>",
    re.IGNORECASE,
)
_IS_VALID_PREDICTION_TAG_RE = re.compile(
    r"<\s*is_valid_prediction\b[^>]*>([\s\S]*?)</\s*is_valid_prediction\s*>",
    re.IGNORECASE,
)


def _extract_json_object(text: str) -> Optional[str]:
    """
    Try to extract a JSON object from the text, trying in the following order:
      1. The first ```json ... ``` fenced code block
      2. The first {...} fragment matched by regex
      3. A balanced-bracket scan starting from the first '{'
    Returns the candidate string or None.
    """
    if not text:
        return None

    # 1) Fenced code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) Greedy regex match for the first {...}
    m = _JSON_OBJ_RE.search(text)
    if m:
        candidate = m.group(0)
        # Basic validation
        if candidate.count("{") == candidate.count("}"):
            return candidate

    # 3) Balanced-bracket scan
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    """Safely parse a JSON string, returning a dict or None."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def _extract_vars_with_regex(text: str) -> List[str]:
    """Fallback: extract all V<n> identifiers from the text (case-insensitive, uppercased)."""
    return sorted({normalize_var_name(m.group(0)) for m in _VAR_TOKEN_RE.finditer(text)})


def _normalize_var_list(vals: Any) -> List[str]:
    """Normalize any form of variable set into a deduplicated, uppercased, whitespace-stripped string list.

    Rules:
      - ``None`` / empty -> return an empty list;
      - a single string -> treated as a one-element list;
      - list/tuple -> each element is processed and cleaned into ``V001`` form.
    """
    if vals is None:
        return []
    if isinstance(vals, str):
        vals = [vals]
    if isinstance(vals, (list, tuple)):
        out = []
        for v in vals:
            if isinstance(v, str):
                out.append(normalize_var_name(v))
            elif v is None:
                continue
            else:
                out.append(normalize_var_name(v))
        return [v for v in out if v]
    return []


def _normalize_confidence(val: Any) -> Optional[float]:
    """Normalize the various confidence forms a model may output into a float in [0, 1].

    Supports both 0.85 and 85 (percent); invalid values return None.
    """
    if val is None:
        return None
    try:
        c = float(val)
        if 0.0 <= c <= 1.0:
            return c
        # Some models output a percentage, e.g. 85
        if 1.0 < c <= 100.0:
            return c / 100.0
        return None
    except (TypeError, ValueError):
        return None


def _normalize_inference_process(value: Any) -> List[Dict[str, Any]]:
    """Normalize the structured reasoning process output by the model, shared by JSONL, CSV, and Agent consumption.

    Each step keeps only its sequence number, stage, analysis summary, and evidence
    variables. Missing steps are not manufactured here; if the model does not follow
    the protocol, the field stays empty and the Agent validator decides whether to
    enter the next correction round.
    """
    if not isinstance(value, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(value[:6], start=1):
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "").strip()
        analysis = str(item.get("analysis") or "").strip()
        evidence_variables = _normalize_var_list(item.get("evidence_variables"))
        try:
            step = int(item.get("step", index))
        except (TypeError, ValueError):
            step = index
        normalized.append(
            {
                "step": step,
                "stage": stage,
                "analysis": analysis,
                "evidence_variables": evidence_variables,
            }
        )
    return normalized


def _parse_tagged(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse <answer> / <reasoning> / <semantic_observations>-style output.

    Parsing strategy:
      1. Use ``_ANSWER_TAG_RE`` to locate the answer tag block;
      2. Further grab all variable identifiers within the answer text via the
         ``V\\d+`` regex, as ``predicted_root_causes`` and ``primary_root_cause``;
      3. Similarly parse reasoning / semantic_observations as explanatory fields;
      4. Parse the newly added explainability fields (root_cause_name, numerical_evidence, etc.)
         as supplementary notes;
      5. If the answer tag is missing or empty, return None directly and let the
         caller fall back to the next level.
    """
    parsed: Dict[str, Any] = {}
    answer_match = _ANSWER_TAG_RE.search(text)
    if not answer_match:
        return None
    answer_text = (answer_match.group(1) or "").strip()
    if not answer_text:
        return None

    # Extract all V<n> from answer_text
    vars_in_answer = _extract_vars_with_regex(answer_text)

    # If answer_text is multiple variables separated by comma/semicolon/space, split it directly
    tokens = [t.strip().upper() for t in re.split(r"[,\s;，；]+", answer_text) if t.strip()]
    var_tokens = [t for t in tokens if re.match(r"^V\s*\d+$", t, re.IGNORECASE)]

    predicted = var_tokens if var_tokens else vars_in_answer
    if not predicted and vars_in_answer:
        predicted = vars_in_answer

    primary = predicted[0] if predicted else None

    parsed["predicted_root_causes"] = predicted[:5]
    parsed["primary_root_cause"] = primary
    parsed["evidence_variables"] = list(predicted[:3])

    # Parse reasoning
    reasoning_match = _REASONING_TAG_RE.search(text)
    if reasoning_match:
        parsed["reasoning"] = (reasoning_match.group(1) or "").strip()
    else:
        semantic_match = _SEMANTIC_TAG_RE.search(text)
        if semantic_match:
            parsed["reasoning"] = (semantic_match.group(1) or "").strip()

    # Parse the newly added explainability fields
    parsed["root_cause_name"] = _extract_tag_text(text, _ROOT_CAUSE_NAME_TAG_RE)
    parsed["numerical_evidence"] = _extract_tag_text(text, _NUMERICAL_EVIDENCE_TAG_RE)
    parsed["temporal_evidence"] = _extract_tag_text(text, _TEMPORAL_EVIDENCE_TAG_RE)
    parsed["type_aware_reasoning"] = _extract_tag_text(text, _TYPE_AWARE_REASONING_TAG_RE)
    parsed["process_relation_reasoning"] = _extract_tag_text(text, _PROCESS_RELATION_REASONING_TAG_RE)
    parsed["why_not_other_candidates"] = _extract_tag_text(text, _WHY_NOT_OTHERS_TAG_RE)
    parsed["uncertainty_analysis"] = _extract_tag_text(text, _UNCERTAINTY_TAG_RE)
    parsed["is_valid_prediction"] = _extract_bool_tag(text, _IS_VALID_PREDICTION_TAG_RE)

    parsed["confidence"] = None
    parsed["parse_method"] = "tag"
    parsed["valid_json"] = False
    return parsed


def _extract_tag_text(text: str, pattern: re.Pattern) -> str:
    """Extract the content of the specified XML tag from the text."""
    match = pattern.search(text)
    if match:
        return (match.group(1) or "").strip()
    return ""


def _extract_bool_tag(text: str, pattern: re.Pattern) -> Optional[bool]:
    """Extract a boolean tag from the text."""
    match = pattern.search(text)
    if match:
        val = (match.group(1) or "").strip().lower()
        if val in ("true", "yes", "1", "correct"):
            return True
        elif val in ("false", "no", "0", "wrong"):
            return False
    return None


def parse_response(raw_text: str) -> Dict[str, Any]:
    """Parse the model's raw response into a structured dict.

    Returned fields (consumed directly by the downstream :class:`Evaluator`):
      - predicted_root_causes: the list of root-cause variables sorted by likelihood
      - primary_root_cause: the most likely root-cause variable
      - root_cause_name: the human-readable variable name (e.g., LIT101)
      - inference_process: a four-step structured, verifiable reasoning process
      - reasoning / numerical_evidence / temporal_evidence and other explainability fields
      - evidence_variables: the evidence variables supporting the judgment
      - confidence: confidence score ([0, 1])
      - valid_json / parse_method: parsing-path markers used for logging and debugging

    Parsing priority (consistent with the module docstring):
      1. XML tag parsing ``<answer>`` -> return immediately on success;
      2. JSON object parsing -> return immediately on success;
      3. Regex fallback (grab all ``V\\d+``) -> as the final fallback.

    This ordering balances "controllable formats" (XML, JSON) with "fault-tolerant
    fallback", maximizing the chance that every response is turned into an evaluable
    structure.
    """
    parsed: Dict[str, Any] = {
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
        "is_valid_prediction": None,
        "confidence": None,
        "valid_json": False,
        "parse_method": "none",
    }

    if not raw_text or not raw_text.strip():
        # Empty response returns an empty structure directly, avoiding later parse exceptions
        parsed["parse_method"] = "empty"
        return parsed

    # 1) Try the XML style first (the format explicitly required by the Prompt)
    tagged = _parse_tagged(raw_text)
    if tagged is not None and tagged.get("predicted_root_causes"):
        # XML parsing succeeded -> return directly, no other method is tried
        return tagged

    # 2) Then try JSON parsing (for models that return JSON)
    candidate_json = _extract_json_object(raw_text)
    obj: Optional[Dict[str, Any]] = None
    if candidate_json:
        obj = _parse_json_safely(candidate_json)

    if obj is not None:
        # JSON parsing succeeded: extract and normalize each field
        parsed["valid_json"] = True
        parsed["parse_method"] = "json"

        # The predicted / primary fields are the core fields used for evaluation
        predicted = _normalize_var_list(obj.get("predicted_root_causes"))
        primary = obj.get("primary_root_cause")
        if isinstance(primary, str):
            primary = normalize_var_name(primary)
            if primary:
                parsed["primary_root_cause"] = primary
        elif isinstance(primary, list) and primary:
            parsed["primary_root_cause"] = _normalize_var_list(primary)[0]

        # Fallback: if primary exists but predicted does not, backfill predicted with primary
        if not predicted and parsed["primary_root_cause"]:
            predicted = [parsed["primary_root_cause"]]
        parsed["predicted_root_causes"] = predicted[:5]

        # Parse the newly added explainability fields (written only, do not affect evaluation logic)
        parsed["root_cause_name"] = str(obj.get("root_cause_name", "") or "")
        parsed["reasoning"] = str(obj.get("reasoning", "") or "")
        parsed["inference_process"] = _normalize_inference_process(
            obj.get("inference_process")
        )
        parsed["numerical_evidence"] = str(obj.get("numerical_evidence", "") or "")
        parsed["temporal_evidence"] = str(obj.get("temporal_evidence", "") or "")
        parsed["type_aware_reasoning"] = str(obj.get("type_aware_reasoning", "") or "")
        parsed["process_relation_reasoning"] = str(obj.get("process_relation_reasoning", "") or "")
        parsed["why_not_other_candidates"] = str(obj.get("why_not_other_candidates", "") or "")
        parsed["uncertainty_analysis"] = str(obj.get("uncertainty_analysis", "") or "")

        # Parse is_valid_prediction (supports bool or string)
        is_valid = obj.get("is_valid_prediction")
        if isinstance(is_valid, bool):
            parsed["is_valid_prediction"] = is_valid
        elif isinstance(is_valid, str):
            parsed["is_valid_prediction"] = is_valid.strip().lower() in ("true", "yes", "1")
        elif is_valid is not None:
            try:
                parsed["is_valid_prediction"] = bool(is_valid)
            except (TypeError, ValueError):
                parsed["is_valid_prediction"] = None

        # evidence_variables: fall back to predicted when empty
        parsed["evidence_variables"] = _normalize_var_list(obj.get("evidence_variables"))
        if not parsed["evidence_variables"]:
            parsed["evidence_variables"] = list(parsed["predicted_root_causes"])
        parsed["confidence"] = _normalize_confidence(obj.get("confidence"))
        return parsed

    # 3) Final fallback: grab all V\d+ identifiers via regex
    parsed["parse_method"] = "regex"
    ordered = []
    for m in _VAR_TOKEN_RE.finditer(raw_text):
        v = m.group(0).upper().replace(" ", "")
        if v not in ordered:
            ordered.append(v)
    if not ordered:
        ordered = _extract_vars_with_regex(raw_text)
    parsed["predicted_root_causes"] = ordered[:5]
    if ordered:
        parsed["primary_root_cause"] = ordered[0]
        parsed["evidence_variables"] = ordered[:3]
    
    # Ensure reasoning is not empty: extract the key reasoning snippet from the raw text
    parsed["reasoning"] = _extract_reasoning_from_text(raw_text)
    if not parsed["reasoning"] and ordered:
        parsed["reasoning"] = f"通过正则提取根因变量: {', '.join(ordered[:3])}"
    
    parsed["confidence"] = None
    return parsed


def _extract_reasoning_from_text(text: str) -> str:
    """Extract a reasoning snippet from the raw text, ensuring the reasoning field is not empty.
    
    Prefers sentences containing the following keywords:
    - English causal cues: because, because of, due to, cause, reason, since
    - Chinese causal cues: 因为, 由于, 原因, 导致, 引起
    - Numeric/anomaly cues: score, anomaly, abnormal, evidence, support
    """
    if not text:
        return ""
    
    text = text.strip()
    
    reason_patterns = [
        re.compile(r"(because\s+[\s\S]*?[.!?。！？])", re.IGNORECASE),
        re.compile(r"(because\s+of\s+[\s\S]*?[.!?。！？])", re.IGNORECASE),
        re.compile(r"(due\s+to\s+[\s\S]*?[.!?。！？])", re.IGNORECASE),
        re.compile(r"(the\s+reason\s+(is\s+)?[\s\S]*?[.!?。！？])", re.IGNORECASE),
        re.compile(r"(it\s+is\s+likely\s+that\s+[\s\S]*?[.!?。！？])", re.IGNORECASE),
        re.compile(r"(最可能的原因[\s\S]*?[。！？])"),
        re.compile(r"(因为[\s\S]*?[。！？])"),
        re.compile(r"(由于[\s\S]*?[。！？])"),
        re.compile(r"(导致[\s\S]*?[。！？])"),
        re.compile(r"(score\s+[\d.]+[\s\S]*?[.!?。！？])", re.IGNORECASE),
        re.compile(r"(anomaly[\s\S]*?[.!?。！？])", re.IGNORECASE),
        re.compile(r"(abnormal[\s\S]*?[.!?。！？])", re.IGNORECASE),
    ]
    
    for pattern in reason_patterns:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()[:500]
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if len(line) > 30 and any(key in line.lower() for key in [
            "reason", "cause", "because", "evidence", "score", 
            "anomaly", "abnormal", "likely", "推测", "推断", "分析"
        ]):
            return line[:500]
    
    return text[:500]


def clamp_to_candidates(
    parsed: Dict[str, Any], candidate_vars: List[str]
) -> Dict[str, Any]:
    """
    Filter the parsed variables, keeping only the items present in `candidate_vars`.
    Variables outside the candidate set are treated as hallucinations and removed.
    """
    canon = {normalize_var_name(c) for c in candidate_vars}

    predicted_raw = parsed.get("predicted_root_causes", []) or []
    filtered = []
    for v in predicted_raw:
        vv = normalize_var_name(v or "")
        if vv and vv in canon and vv not in filtered:
            filtered.append(vv)
    parsed["predicted_root_causes"] = filtered[:5]

    pri = parsed.get("primary_root_cause")
    if pri:
        pri_norm = normalize_var_name(pri)
        parsed["primary_root_cause"] = pri_norm if pri_norm in canon else (filtered[0] if filtered else None)

    evidence = parsed.get("evidence_variables", []) or []
    fe = []
    for v in evidence:
        vv = normalize_var_name(v or "")
        if vv and vv in canon and vv not in fe:
            fe.append(vv)
    parsed["evidence_variables"] = fe or list(parsed["predicted_root_causes"])

    # The evidence variables in the reasoning steps must also obey the same Top-10
    # constraint. Only filter variables; do not rewrite the stage and analysis text
    # given by the model, so the persisted content remains auditable.
    for step in parsed.get("inference_process", []) or []:
        if not isinstance(step, dict):
            continue
        filtered_step_vars: List[str] = []
        for value in step.get("evidence_variables", []) or []:
            normalized = normalize_var_name(value)
            if normalized in canon and normalized not in filtered_step_vars:
                filtered_step_vars.append(normalized)
        step["evidence_variables"] = filtered_step_vars

    return parsed
