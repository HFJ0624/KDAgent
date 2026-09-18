"""Deterministic fusor of the data-evidence Agent and the RAG knowledge branch."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .iterative_agent import IterativeSelfRefinementAgent
from .response_parser import clamp_to_candidates, parse_response
from .response_validator import validate_response


class DualBranchFusionAgent:
    """Run the data-evidence branch and the RAG branch independently and fuse them deterministically at the candidate level.

    The data branch is responsible for the primary cause, to avoid generic
    knowledge overriding the on-site time series; the RAG branch only supplements
    candidates 2 through 5. The whole process reads only the Top-10 candidates
    and never the ground truth.
    """

    def __init__(
        self,
        model_client,
        min_confidence: float = 0.5,
        max_retries: int = 3,
        timeout: int = 120,
        retry_base_sleep: int = 5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_client = model_client
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_base_sleep = retry_base_sleep
        self.logger = logger or logging.getLogger("dual_branch_fusion_agent")

    @staticmethod
    def _normalize_candidates(values: Any, top10_vars: List[str]) -> List[str]:
        allowed = {str(value).strip().upper() for value in top10_vars if value}
        normalized: List[str] = []
        if not isinstance(values, list):
            return normalized
        for value in values:
            candidate = str(value).strip().upper()
            if candidate and candidate in allowed and candidate not in normalized:
                normalized.append(candidate)
        return normalized

    @staticmethod
    def _valid_primary(parsed: Dict[str, Any], top10_vars: List[str]) -> str:
        primary = str(parsed.get("primary_root_cause") or "").strip().upper()
        allowed = {str(value).strip().upper() for value in top10_vars if value}
        return primary if primary in allowed else ""

    @staticmethod
    def _append_unique(target: List[str], values: List[str], limit: int = 5) -> None:
        for value in values:
            if value not in target:
                target.append(value)
            if len(target) >= limit:
                return

    def _run_rag_branch(
        self,
        case_id: Any,
        run_id: int,
        rag_prompt: str,
        top10_vars: List[str],
    ) -> Dict[str, Any]:
        """Run a single independent RAG inference; format problems stay in the branch audit rather than being silently hidden."""
        api_result = self.model_client.chat(
            prompt=rag_prompt,
            system_prompt=(
                "You are a careful industrial root cause analysis engineer. "
                "Use retrieved knowledge only to explain current evidence and output valid JSON."
            ),
            max_retries=self.max_retries,
            timeout=self.timeout,
            retry_base_sleep=self.retry_base_sleep,
        )
        api_success = bool(api_result.get("success"))
        content = api_result.get("content", "") if api_success else ""
        finish_reason = api_result.get("finish_reason", "") if api_success else ""
        raw_parsed = parse_response(content)
        validation = validate_response(
            raw_parsed,
            top10_vars,
            self.min_confidence,
            api_success=api_success,
            finish_reason=finish_reason,
        )
        parsed = clamp_to_candidates(raw_parsed, top10_vars)
        return {
            "record_type": "agent_iteration",
            "branch": "rag_knowledge",
            "case_id": case_id,
            "run_id": run_id,
            "iteration": 1,
            "prompt_type": "rag_independent",
            "retry_reasons_from_previous_round": [],
            "prompt": rag_prompt,
            "content": content,
            "reasoning_content": api_result.get("reasoning_content", ""),
            "finish_reason": finish_reason,
            "raw_response": api_result.get("raw_response", ""),
            "parsed_response": parsed,
            "raw_parsed_response": raw_parsed,
            "validation_result": validation,
            "useful_information_extracted": {},
            "useful_information_accumulated": {},
            "api_success": api_success,
            "error_type": api_result.get("error_type"),
            "error_message": api_result.get("error_message"),
            "elapsed_s": api_result.get("elapsed", 0.0),
            "prompt_tokens": api_result.get("prompt_tokens", 0),
            "completion_tokens": api_result.get("completion_tokens", 0),
            "reasoning_tokens": api_result.get("reasoning_tokens", 0),
            "total_tokens": api_result.get("total_tokens", 0),
            "was_truncated": finish_reason == "length",
        }

    def run_case(
        self,
        case: Dict[str, Any],
        evidence_prompt: str,
        rag_prompt: str,
        top10_vars: List[str],
        max_iterations: int = 3,
        run_id: int = 1,
    ) -> Dict[str, Any]:
        """Run both branches and return a fused result compatible with the existing Agent main flow."""
        case_id = case.get("case_id")
        evidence_agent = IterativeSelfRefinementAgent(
            model_client=self.model_client,
            min_confidence=self.min_confidence,
            max_retries=self.max_retries,
            timeout=self.timeout,
            retry_base_sleep=self.retry_base_sleep,
            logger=self.logger,
        )
        evidence_result = evidence_agent.run_case(
            case=case,
            initial_prompt=evidence_prompt,
            top10_vars=top10_vars,
            rag_contexts=None,
            max_iterations=max_iterations,
            run_id=run_id,
        )
        evidence_iterations = []
        for item in evidence_result.get("all_iterations", []):
            copied = dict(item)
            copied["branch"] = "data_evidence"
            evidence_iterations.append(copied)

        rag_record = self._run_rag_branch(
            case_id=case_id,
            run_id=run_id,
            rag_prompt=rag_prompt,
            top10_vars=top10_vars,
        )
        rag_parsed = rag_record.get("parsed_response", {}) or {}
        evidence_parsed = evidence_result.get("parsed_response", {}) or {}

        evidence_primary = self._valid_primary(evidence_parsed, top10_vars)
        rag_primary = self._valid_primary(rag_parsed, top10_vars)
        evidence_candidates = self._normalize_candidates(
            evidence_parsed.get("predicted_root_causes"), top10_vars
        )
        rag_candidates = self._normalize_candidates(
            rag_parsed.get("predicted_root_causes"), top10_vars
        )

        # Key fusion rule: the data branch locks only the first slot; the remaining
        # four slots are preferably supplemented by the knowledge branch. This keeps
        # the primary judgment from on-site evidence while using RAG to broaden the
        # coverage of the top-five candidates.
        fused_candidates: List[str] = []
        primary = evidence_primary or rag_primary
        if primary:
            fused_candidates.append(primary)
        self._append_unique(fused_candidates, rag_candidates)
        self._append_unique(fused_candidates, evidence_candidates)
        fused_candidates = fused_candidates[:5]

        base_parsed = evidence_parsed if evidence_primary else rag_parsed
        fused_parsed = dict(base_parsed)
        fused_parsed["primary_root_cause"] = primary or "UNKNOWN"
        fused_parsed["predicted_root_causes"] = fused_candidates
        fused_parsed["valid_json"] = bool(primary and fused_candidates)
        fused_parsed["parse_method"] = "deterministic_dual_branch_fusion"
        fused_parsed["is_valid_prediction"] = bool(primary and fused_candidates)

        agreement = bool(
            evidence_primary and rag_primary and evidence_primary == rag_primary
        )
        supplemented = [
            value
            for value in fused_candidates[1:]
            if value in rag_candidates and value not in evidence_candidates
        ]
        fused_parsed.update(
            {
                "fusion_strategy": "evidence_primary_rag_supplement",
                "evidence_branch_primary": evidence_primary or "UNKNOWN",
                "rag_branch_primary": rag_primary or "UNKNOWN",
                "branch_primary_agreement": agreement,
                "rag_supplemented_candidates": supplemented,
            }
        )

        success = bool(primary and fused_candidates)
        if success and evidence_primary:
            status = "dual_branch_success"
        elif success:
            status = "dual_branch_rag_fallback"
        else:
            status = "dual_branch_fallback_unknown"

        all_iterations = evidence_iterations + [rag_record]
        return {
            "success": success,
            "final_iteration": evidence_result.get("final_iteration", 1),
            "iteration_count": evidence_result.get("iteration_count", 1),
            "retry_count": evidence_result.get("retry_count", 0),
            "validation_passed": success,
            "retry_reasons": [] if success else ["both_branches_without_prediction"],
            "agent_status": status,
            "agent_failed": not success,
            "final_prompt_type": "deterministic_dual_branch_fusion",
            "parsed_response": fused_parsed,
            "raw_parsed_response": fused_parsed,
            "content": evidence_result.get("content", "") or rag_record.get("content", ""),
            "reasoning_content": evidence_result.get("reasoning_content", ""),
            "finish_reason": "stop" if success else evidence_result.get("finish_reason", ""),
            "all_iterations": all_iterations,
            "use_dual_branch_fusion": True,
            "fusion_strategy": "evidence_primary_rag_supplement",
            "evidence_branch_primary": evidence_primary or "UNKNOWN",
            "rag_branch_primary": rag_primary or "UNKNOWN",
            "branch_primary_agreement": agreement,
            "rag_supplemented_candidates": supplemented,
            "evidence_branch_success": bool(evidence_primary),
            "rag_branch_success": bool(rag_primary),
        }
