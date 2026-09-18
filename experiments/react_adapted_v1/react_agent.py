"""Tool loop and strict label isolation implementation for the ReAct-adapted agent."""

from __future__ import annotations

import json
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.prompt_builder import format_public_numerical_evidence


LABEL_KEYS = {"ground_truth", "gt_vars", "gt_names", "root_tags", "hit_at_1", "hit_at_3", "hit_at_5"}
VALID_ACTIONS = {"get_episode_evidence", "search_domain_knowledge", "submit_ranking"}


def canonical(value: Any) -> str:
    """Consistent with the existing evaluator: uppercase candidate IDs and remove spaces."""
    return str(value or "").strip().upper().replace(" ", "")


def parse_action(text: str) -> Dict[str, Any]:
    """Extract the first complete JSON object from plain text or Markdown-fenced text."""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    decoder = json.JSONDecoder()
    for position, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response_contains_no_json_object")


def candidate_catalog(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expose candidate identity metadata only, without leaking numeric evidence up front."""
    details = case.get("top10_details") or []
    by_id = {canonical(item.get("var")): item for item in details if isinstance(item, dict)}
    result: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(case.get("top10_vars") or [], start=1):
        cid = canonical(candidate)
        detail = by_id.get(cid, {})
        raw_item = detail.get("raw_item") if isinstance(detail.get("raw_item"), dict) else {}
        result.append(
            {
                "rank": rank,
                "candidate_id": cid,
                "name": str(detail.get("name") or ""),
                "type": str(detail.get("type") or ""),
                "stage": str(raw_item.get("stage") or ""),
                "role": str(raw_item.get("role") or ""),
            }
        )
    return result


def assert_label_isolation(value: Any, location: str) -> None:
    """Recursively check the online context to prevent label fields from accidentally reaching the model-visible content."""
    if isinstance(value, dict):
        leaked = LABEL_KEYS.intersection(str(key).lower() for key in value)
        if leaked:
            raise RuntimeError(f"label_isolation_violation@{location}:{sorted(leaked)}")
        for child in value.values():
            assert_label_isolation(child, location)
    elif isinstance(value, list):
        for child in value:
            assert_label_isolation(child, location)


@dataclass
class AgentLimits:
    max_logic_calls: int = 4
    max_completion_tokens: int = 32768
    max_ranking_length: int = 5
    retrieval_top_k: int = 5
    max_query_chars: int = 500


class ReactAdaptedAgent:
    """ReAct loop that lets a single model autonomously choose evidence, retrieval, and submission tools."""

    def __init__(self, model_client: Any, retriever: Any, system_prompt: str, limits: AgentLimits):
        self.model_client = model_client
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.limits = limits

    @staticmethod
    def evidence_view(case: Dict[str, Any]) -> str:
        """Return the numerical evidence view shared with the main experiment and pin its audit hash."""
        return format_public_numerical_evidence(case, compact_evidence=False)

    def _search(self, query: str) -> Dict[str, Any]:
        query = " ".join(str(query or "").split())[: self.limits.max_query_chars]
        if not query:
            return {"ok": False, "error": "empty_search_query", "results": []}
        try:
            vector = self.retriever.embedder.embed_query(query)
            if not vector:
                raise RuntimeError("embedding_returned_empty_vector")
            result = self.retriever.store.query(vector, top_k=self.limits.retrieval_top_k)
            rows = []
            for item in result.get("results", []):
                metadata = item.get("metadata") or {}
                rows.append(
                    {
                        "chunk_id": str(item.get("id") or ""),
                        "source": str(metadata.get("source") or ""),
                        "similarity": item.get("score"),
                        "metadata": metadata,
                        "content": str(item.get("content") or ""),
                    }
                )
            observation = {"ok": True, "query": query, "results": rows}
            assert_label_isolation(observation, "search_domain_knowledge")
            return observation
        except Exception as exc:
            return {"ok": False, "query": query, "error": f"{type(exc).__name__}: {exc}", "results": []}

    def _validate_submission(self, arguments: Any, allowed: Sequence[str]) -> Dict[str, Any]:
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "submit_arguments_not_object"}
        ranking = arguments.get("predicted_root_causes")
        if not isinstance(ranking, list) or not 1 <= len(ranking) <= self.limits.max_ranking_length:
            return {"ok": False, "error": "ranking_length_must_be_1_to_5"}
        normalized = [canonical(value) for value in ranking]
        allowed_set = {canonical(value) for value in allowed}
        if any(not value or value not in allowed_set for value in normalized):
            return {"ok": False, "error": "ranking_contains_non_candidate"}
        if len(set(normalized)) != len(normalized):
            return {"ok": False, "error": "ranking_contains_duplicates"}
        primary = canonical(arguments.get("primary_root_cause"))
        if primary != normalized[0]:
            return {"ok": False, "error": "primary_must_equal_rank_1"}
        confidence = arguments.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                return {"ok": False, "error": "confidence_not_numeric"}
            if not 0.0 <= confidence <= 1.0:
                return {"ok": False, "error": "confidence_out_of_range"}
        return {
            "ok": True,
            "parsed_response": {
                "predicted_root_causes": normalized,
                "primary_root_cause": primary,
                "confidence": confidence,
                "reasoning": str(arguments.get("brief_reason") or "").strip(),
                "valid_json": True,
                "parse_method": "react_tool_submission",
                "is_valid_prediction": True,
            },
        }

    def _build_prompt(self, dataset: str, case: Dict[str, Any], transcript: List[Dict[str, Any]], turn: int) -> str:
        payload = {
            "dataset": dataset,
            "episode_id": str(case.get("case_id")),
            "frozen_candidates": candidate_catalog(case),
            "tools": [
                {"name": "get_episode_evidence", "arguments": {}},
                {"name": "search_domain_knowledge", "arguments": {"query": "string"}},
                {"name": "submit_ranking", "arguments": {"predicted_root_causes": "list[1..5]", "primary_root_cause": "candidate_id", "confidence": "optional 0..1", "brief_reason": "string"}},
            ],
            "action_observation_history": transcript,
            "turn": turn,
            "turns_remaining_including_this": self.limits.max_logic_calls - turn + 1,
            "instruction": "Return one action JSON object. This is the final turn; submit_ranking is required." if turn == self.limits.max_logic_calls else "Return one action JSON object.",
        }
        assert_label_isolation(payload, "model_prompt")
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def run(self, dataset: str, case: Dict[str, Any], run_id: int) -> Dict[str, Any]:
        allowed = [canonical(value) for value in case.get("top10_vars") or []]
        transcript: List[Dict[str, Any]] = []
        calls: List[Dict[str, Any]] = []
        tool_events: List[Dict[str, Any]] = []
        totals = {key: 0 for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "network_attempts")}
        started = time.time()
        stopping_reason = "max_logic_calls_exhausted"
        prediction: Dict[str, Any] = {}

        for turn in range(1, self.limits.max_logic_calls + 1):
            if totals["completion_tokens"] >= self.limits.max_completion_tokens:
                stopping_reason = "completion_budget_exhausted"
                break
            prompt = self._build_prompt(dataset, case, transcript, turn)
            api = self.model_client.chat(prompt=prompt, system_prompt=self.system_prompt)
            call = {
                "turn": turn,
                "success": bool(api.get("success")),
                "content": api.get("content", ""),
                "reasoning_content": api.get("reasoning_content", ""),
                "finish_reason": api.get("finish_reason", ""),
                "raw_response": api.get("raw_response", ""),
                "error_type": api.get("error_type"),
                "error_message": api.get("error_message"),
                "elapsed_s": float(api.get("elapsed", 0.0) or 0.0),
                "prompt_tokens": int(api.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(api.get("completion_tokens", 0) or 0),
                "reasoning_tokens": int(api.get("reasoning_tokens", 0) or 0),
                "total_tokens": int(api.get("total_tokens", 0) or 0),
                "network_attempts": int(api.get("network_attempts", 0) or 0),
                "request_snapshot": api.get("request_snapshot"),
                "request_control": api.get("request_control"),
                "request_fingerprint": api.get("request_fingerprint", ""),
            }
            calls.append(call)
            for key in totals:
                totals[key] += call[key]
            if not call["success"]:
                stopping_reason = f"api_failure:{call['error_type'] or 'unknown'}"
                break
            if str(call["finish_reason"]).lower() == "length":
                observation = {"ok": False, "error": "output_truncated"}
                transcript.append({"turn": turn, "action": None, "observation": observation})
                tool_events.append({"turn": turn, "tool": None, "arguments": {}, "observation": observation})
                continue

            try:
                action_obj = parse_action(call["content"])
                action = str(action_obj.get("action") or "").strip()
                arguments = action_obj.get("arguments") or {}
                action_reason = str(action_obj.get("action_reason") or "").strip()
            except ValueError as exc:
                action, arguments, action_reason = "", {}, ""
                observation = {"ok": False, "error": str(exc)}
                transcript.append({"turn": turn, "action": None, "observation": observation})
                tool_events.append({"turn": turn, "tool": None, "arguments": {}, "observation": observation})
                continue

            if action not in VALID_ACTIONS:
                observation = {"ok": False, "error": "unknown_action", "valid_actions": sorted(VALID_ACTIONS)}
            elif action == "get_episode_evidence":
                evidence_text = self.evidence_view(case)
                observation = {
                    "ok": True,
                    "episode_id": str(case.get("case_id")),
                    "frozen_evidence": evidence_text,
                    "evidence_view_format": "shared_main_prompt_full_v1",
                    "evidence_view_sha256": hashlib.sha256(
                        evidence_text.encode("utf-8")
                    ).hexdigest(),
                }
                assert_label_isolation(observation, "get_episode_evidence")
            elif action == "search_domain_knowledge":
                observation = self._search(str(arguments.get("query") or "") if isinstance(arguments, dict) else "")
            else:
                observation = self._validate_submission(arguments, allowed)
                if observation.get("ok"):
                    prediction = dict(observation["parsed_response"])
                    stopping_reason = "valid_submission"

            event = {"turn": turn, "tool": action or None, "arguments": arguments, "action_reason": action_reason, "observation": observation}
            tool_events.append(event)
            # Once a valid submission succeeds, the answer is no longer fed back as context for the next turn, since the diagnosis has terminated.
            if prediction:
                break
            transcript.append({"turn": turn, "action": action_obj, "observation": observation})

        return {
            "method": "ReAct-adapted",
            "dataset": dataset,
            "model_name": self.model_client.model,
            "episode_id": str(case.get("case_id")),
            "run_id": int(run_id),
            "parsed_response": prediction,
            "completed": bool(prediction),
            "valid_output": bool(prediction),
            "stopping_reason": stopping_reason,
            "logic_calls": len(calls),
            "tool_calls": len([event for event in tool_events if event.get("tool")]),
            "knowledge_search_calls": len([event for event in tool_events if event.get("tool") == "search_domain_knowledge"]),
            "evidence_tool_used": any(event.get("tool") == "get_episode_evidence" for event in tool_events),
            "elapsed_s": time.time() - started,
            **totals,
            "calls": calls,
            "tool_events": tool_events,
        }
