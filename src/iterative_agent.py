"""Iterative Self-refinement Agent: Implements the core iterative RCA reasoning loop.

Ground-truth labels are only used for offline evaluation and are never used by the Iterative Self-refinement Agent.

This module implements the Iterative Self-refinement Agent pattern:
1. Round 1: Initial RCA reasoning with standard prompt
2. Validation: Check output against structural and content rules
3. Round 2 (if needed): Refinement based on specific failure reasons
4. Round 3 (if still needed): Forced-choice selection with strong constraints
5. Final: Return validated result or UNKNOWN

Key design principles:
- No ground_truth/gt_vars/gt_names access during agent operation
- Each round builds on previous round's validation failures
- Early termination when validation passes
- Final forced-choice round as safety net
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .prompt_builder import COMPACT_OUTPUT_RULES, format_top10_candidates_compact
from .response_parser import parse_response
from .response_validator import validate_response
from .utils import normalize_var_name


class IterativeSelfRefinementAgent:
    """
    Iterative Self-refinement Agent for Root Cause Analysis.
    
    Ground-truth labels are only used for offline evaluation and are never used by the Iterative Self-refinement Agent.
    """

    def __init__(
        self,
        model_client,
        min_confidence: float = 0.5,
        max_retries: int = 3,
        timeout: int = 120,
        retry_base_sleep: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_client = model_client
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_base_sleep = retry_base_sleep
        self.logger = logger or logging.getLogger("iterative_agent")

    def run_case(
        self,
        case: Dict[str, Any],
        initial_prompt: str,
        top10_vars: List[str],
        rag_contexts: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 3,
        run_id: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the iterative self-refinement loop for a single case.
        
        Ground-truth labels are only used for offline evaluation and are never used by the Iterative Self-refinement Agent.
        
        Args:
            case: Case data dictionary
            initial_prompt: The initial RCA prompt for round 1
            top10_vars: List of candidate variable IDs (used for validation only)
            rag_contexts: RAG contexts if available
            max_iterations: Maximum number of iterations (default: 3)
            run_id: Current run ID for logging
        
        Returns:
            Dictionary containing final result and all iteration records
        """
        case_id = case.get("case_id")
        all_iterations: List[Dict[str, Any]] = []
        previous_response: str = ""
        previous_validation: Dict[str, Any] = {}
        accumulated_useful_information: Dict[str, Any] = {}
        final_result: Dict[str, Any] = {}

        for iteration in range(1, max_iterations + 1):
            is_last_iteration = iteration == max_iterations
            
            # Build prompt for this iteration
            if iteration == 1:
                prompt = initial_prompt
                prompt_type = "initial"
                retry_reasons_from_previous = []
            elif not is_last_iteration:
                prompt = self.build_refinement_prompt(
                    case=case,
                    top10_vars=top10_vars,
                    previous_response=previous_response,
                    previous_validation=previous_validation,
                    useful_information=accumulated_useful_information,
                    rag_contexts=rag_contexts,
                )
                prompt_type = "refinement"
                retry_reasons_from_previous = previous_validation.get("retry_reasons", [])
            else:
                prompt = self.build_forced_choice_prompt(
                    case=case,
                    top10_vars=top10_vars,
                    previous_response=previous_response,
                    previous_validation=previous_validation,
                    useful_information=accumulated_useful_information,
                    rag_contexts=rag_contexts,
                )
                prompt_type = "forced_choice"
                retry_reasons_from_previous = previous_validation.get("retry_reasons", [])

            self.logger.info(
                "[run=%s][case=%s][iteration=%d/%d] Using prompt_type=%s",
                run_id, case_id, iteration, max_iterations, prompt_type
            )

            # Call model
            api_result = self.model_client.chat(
                prompt=prompt,
                system_prompt="You are a careful industrial root cause analysis engineer. Output strictly valid JSON only.",
                max_retries=self.max_retries,
                timeout=self.timeout,
                retry_base_sleep=self.retry_base_sleep,
            )

            api_success = api_result.get("success", False)
            content = api_result.get("content", "") if api_success else ""
            reasoning_content = api_result.get("reasoning_content", "") if api_success else ""
            finish_reason = api_result.get("finish_reason", "") if api_success else ""

            # The raw parse result must be validated first. If outside-candidate variables
            # are clamped away first, model hallucinations are silently hidden and
            # the Agent cannot feed the real failure reason back into the next round.
            parsed = parse_response(content)
            validation = validate_response(
                parsed,
                top10_vars,
                self.min_confidence,
                api_success=api_success,
                finish_reason=finish_reason,
            )

            # Keep only the in-candidate variables, valid confidence, and non-empty evidence
            # text from this round's response. These are "clues to re-verify" rather
            # than final conclusions, so the next round must re-judge them together
            # with the failure reason.
            current_useful_information = self.extract_useful_information(parsed, top10_vars)
            accumulated_useful_information = self.merge_useful_information(
                accumulated_useful_information,
                current_useful_information,
            )

            # Record this iteration
            iteration_record = {
                # Distinct from the final case summary record; on resume this entry must not be treated as completed.
                "record_type": "agent_iteration",
                "case_id": case_id,
                "run_id": run_id,
                "iteration": iteration,
                "prompt_type": prompt_type,
                "retry_reasons_from_previous_round": retry_reasons_from_previous,
                "prompt": prompt,
                "content": content,
                "reasoning_content": reasoning_content,
                "finish_reason": finish_reason,
                "raw_response": api_result.get("raw_response", ""),
                "parsed_response": parsed,
                "validation_result": validation,
                "useful_information_extracted": current_useful_information,
                "useful_information_accumulated": accumulated_useful_information,
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
            all_iterations.append(iteration_record)

            # Check validation
            if validation.get("is_valid"):
                self.logger.info(
                    "[run=%s][case=%s][iteration=%d] Validation passed! Stopping.",
                    run_id, case_id, iteration
                )
                final_result = {
                    "success": True,
                    "final_iteration": iteration,
                    "iteration_count": iteration,
                    "retry_count": iteration - 1,
                    "validation_passed": True,
                    "retry_reasons": [],
                    "agent_status": "success",
                    "agent_failed": False,
                    "final_prompt_type": prompt_type,
                    "parsed_response": parsed,
                    # Successful results also keep the raw parsed view for uniform auditing by the main flow.
                    "raw_parsed_response": parsed,
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "finish_reason": finish_reason,
                    "all_iterations": all_iterations,
                }
                return final_result

            # Update previous for next iteration
            previous_response = content
            previous_validation = validation

            self.logger.info(
                "[run=%s][case=%s][iteration=%d] Validation failed: %s",
                run_id, case_id, iteration, validation.get("retry_reasons")
            )

        # All iterations exhausted - return UNKNOWN
        self.logger.info(
            "[run=%s][case=%s] All %d iterations failed. Returning UNKNOWN.",
            run_id, case_id, max_iterations
        )

        # When all rounds fail, the only option is to abstain as UNKNOWN rather than disguise the Top-10 first item as the answer.
        last_iteration = all_iterations[-1] if all_iterations else {}
        final_result = {
            "success": False,
            "final_iteration": max_iterations,
            "iteration_count": max_iterations,
            # The first round is the initial call; only subsequent rounds are retries, so three rounds mean at most two retries.
            "retry_count": max(0, max_iterations - 1),
            "validation_passed": False,
            "retry_reasons": previous_validation.get("retry_reasons", []),
            "agent_status": "fallback_unknown",
            "agent_failed": True,
            "final_prompt_type": "forced_choice",
            "parsed_response": {
                "predicted_root_causes": [],
                "primary_root_cause": "UNKNOWN",
                "root_cause_name": "",
                "confidence": 0.0,
                "reasoning": "The model failed after iterative self-refinement.",
                "inference_process": [],
                "numerical_evidence": "",
                "temporal_evidence": "",
                "type_aware_reasoning": "",
                "process_relation_reasoning": "",
                "why_not_other_candidates": "",
                "evidence_variables": [],
                "uncertainty_analysis": "",
                "is_valid_prediction": False,
                "valid_json": False,
                "parse_method": "fallback",
            },
            # The last round's raw output is still auditable, but it does not participate in evaluation as the final RCA answer.
            "raw_parsed_response": last_iteration.get("parsed_response", {}),
            "content": last_iteration.get("content", ""),
            "reasoning_content": last_iteration.get("reasoning_content", ""),
            "finish_reason": last_iteration.get("finish_reason", ""),
            "all_iterations": all_iterations,
        }
        return final_result

    def build_refinement_prompt(
        self,
        case: Dict[str, Any],
        top10_vars: List[str],
        previous_response: str,
        previous_validation: Dict[str, Any],
        useful_information: Dict[str, Any],
        rag_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build the refinement prompt for round 2 (and subsequent non-final rounds).
        
        Ground-truth labels are only used for offline evaluation and are never used by the Iterative Self-refinement Agent.
        """
        retry_reasons = previous_validation.get("retry_reasons", [])
        reasons_text = "\n".join(f"- {reason}" for reason in retry_reasons)
        useful_text = self.format_useful_information(useful_information)

        top10_text = format_top10_candidates_compact(case)

        rag_text = ""
        if rag_contexts:
            rag_text = "\n".join(
                f"[知识 {i+1}] {ctx.get('content', '')}"
                for i, ctx in enumerate(rag_contexts)
            )

        prompt = f"""[迭代自检 - 第 2 轮]

你上一轮的 RCA 输出未通过有效性检查，失败原因如下：
{reasons_text}

上一轮原始输出：
{previous_response}

从上一轮及更早轮次中提取的可复用信息：
{useful_text}

注意：上述信息只是通过基本约束检查的线索，不代表其根因判断正确。请保留有证据支持的部分，并针对失败原因修正其余内容。

请重新检查并修正你的分析：

## 可用信息

### Top-10 候选变量
{top10_text}

### 工业工艺知识
{rag_text}

## 修正要求

1. **只能从 Top-10 候选变量中选择**，禁止输出候选列表外的变量
2. **predicted_root_causes** 最多 5 个变量，按可能性从高到低排序
3. **primary_root_cause** 必须等于 predicted_root_causes 的第一个元素
4. **confidence** 必须是 0 到 1 之间的数字，且不低于 0.5
5. **所有解释字段不得为空**：
   - root_cause_name：根因变量的真实名称
   - numerical_evidence：异常分数、排名、残差大小等数值证据
   - temporal_evidence：raw/recon/residual/score 在时间窗口内的变化模式
   - type_aware_reasoning：变量类型（continuous/state）分析
   - process_relation_reasoning：通过工艺流程影响其他候选变量的说明
   - why_not_other_candidates：未选择其他候选变量的原因
6. **evidence_variables** 必须是列表，且全部来自 Top-10
7. **inference_process** 必须包含 4 个结构化步骤：候选证据比较、时间先后判断、工艺传播分析、最终根因决策；每步必须包含 step、stage、analysis、evidence_variables
8. **只输出严格合法的 JSON**，不要包含任何其他文字

请针对失败原因进行针对性修正。

{COMPACT_OUTPUT_RULES}
"""
        return prompt.strip()

    def build_forced_choice_prompt(
        self,
        case: Dict[str, Any],
        top10_vars: List[str],
        previous_response: str,
        previous_validation: Dict[str, Any],
        useful_information: Dict[str, Any],
        rag_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build the forced-choice prompt for the final round (round 3).
        
        Ground-truth labels are only used for offline evaluation and are never used by the Iterative Self-refinement Agent.
        """
        retry_reasons = previous_validation.get("retry_reasons", [])
        reasons_text = ", ".join(retry_reasons) if retry_reasons else "多次校验失败"
        useful_text = self.format_useful_information(useful_information)

        top10_text = format_top10_candidates_compact(case)

        rag_text = ""
        if rag_contexts:
            rag_text = "\n".join(
                f"[知识 {i + 1}] {ctx.get('content', '')}"
                for i, ctx in enumerate(rag_contexts)
            )

        prompt = f"""[强制选择 - 第 3 轮（最后一轮）]

前两轮 RCA 输出均未通过有效性检查，失败原因：{reasons_text}

上一轮原始输出：
{previous_response}

前两轮累计提取的可复用信息：
{useful_text}

检索到的工业工艺知识：
{rag_text}

注意：可复用信息只是候选集内的线索，不是已确认答案。请结合 Top-10 时序证据和工业知识完成最后一次独立判断。

这是最后一次机会，必须从以下候选变量中选择最合理的根因：

{top10_text}

## 强制约束

1. **必须从以上 Top-10 候选变量中选择**，即使证据不足也必须选择一个
2. **禁止输出 Top-10 外的任何变量**
3. **predicted_root_causes** 最多 5 个变量
4. **primary_root_cause** 必须等于 predicted_root_causes[0]
5. **confidence** 必须是 0 到 1 之间的数字
6. **所有字段不得为空**：
   - root_cause_name
   - numerical_evidence（即使简短也要说明）
   - temporal_evidence（即使简短也要说明）
   - type_aware_reasoning
   - process_relation_reasoning
   - why_not_other_candidates（即使简短也要说明）
7. **evidence_variables** 只能来自 Top-10
8. **inference_process** 必须包含 4 个结构化步骤：候选证据比较、时间先后判断、工艺传播分析、最终根因决策；每步必须包含 step、stage、analysis、evidence_variables
9. **只输出严格合法的 JSON**，不要包含任何其他文字

{COMPACT_OUTPUT_RULES}
"""
        return prompt.strip()

    @staticmethod
    def extract_useful_information(
        parsed: Dict[str, Any], top10_vars: List[str]
    ) -> Dict[str, Any]:
        """Extract safely reusable structured clues from a failed response.

        Extraction never reads ground-truth labels. Variables must be within the
        Top-10, text evidence must be non-empty, and the confidence must fall in
        [0, 1]. This way it inherits the previous round's valid analysis without
        carrying out-of-candidate hallucinations or illegal fields into later prompts.
        """
        candidates = {
            normalize_var_name(value)
            for value in top10_vars
            if value
        }

        def valid_variables(value: Any, limit: int = 5) -> List[str]:
            if not isinstance(value, list):
                return []
            result: List[str] = []
            for item in value:
                normalized = normalize_var_name(item)
                if normalized in candidates and normalized not in result:
                    result.append(normalized)
                if len(result) >= limit:
                    break
            return result

        useful: Dict[str, Any] = {}
        predicted = valid_variables(parsed.get("predicted_root_causes"))
        if predicted:
            useful["candidate_root_causes"] = predicted

        primary = normalize_var_name(parsed.get("primary_root_cause") or "")
        if primary in candidates:
            useful["primary_candidate"] = primary

        evidence_variables = valid_variables(parsed.get("evidence_variables"), limit=10)
        if evidence_variables:
            useful["evidence_variables"] = evidence_variables

        confidence = parsed.get("confidence")
        if isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 1.0:
            useful["confidence"] = float(confidence)

        # These fields are all recheckable analysis material. Keep only non-empty text
        # and cap its length, to avoid an unbounded prompt from multi-round
        # concatenation or repeatedly carrying meaningless output.
        text_fields = (
            "reasoning",
            "numerical_evidence",
            "temporal_evidence",
            "type_aware_reasoning",
            "process_relation_reasoning",
            "why_not_other_candidates",
            "uncertainty_analysis",
        )
        for field in text_fields:
            value = str(parsed.get(field) or "").strip()
            if value:
                useful[field] = value[:1000]

        # Structured inference steps from failed rounds may still contain recheckable
        # clues. Here we inherit only the stage, analysis summary, and in-candidate
        # evidence variables, without carrying out-of-candidate variables into the next prompt.
        valid_process: List[Dict[str, Any]] = []
        for step in parsed.get("inference_process", []) or []:
            if not isinstance(step, dict):
                continue
            stage = str(step.get("stage") or "").strip()
            analysis = str(step.get("analysis") or "").strip()
            evidence = valid_variables(step.get("evidence_variables"), limit=10)
            if stage and analysis and evidence:
                valid_process.append(
                    {
                        "step": len(valid_process) + 1,
                        "stage": stage,
                        "analysis": analysis[:1000],
                        "evidence_variables": evidence,
                    }
                )
        if valid_process:
            useful["inference_process"] = valid_process[:4]
        return useful

    @staticmethod
    def merge_useful_information(
        accumulated: Dict[str, Any], current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge clues from multiple rounds, preferring the newest content while keeping previously un-replaced information."""
        merged = dict(accumulated)
        for key, value in current.items():
            if isinstance(value, list):
                # inference_process is a list of dicts, so the string-list-oriented
                # dict.fromkeys dedup does not apply; later-generated steps represent
                # the newest corrections and should be replaced as a whole.
                if value and isinstance(value[0], dict):
                    merged[key] = value
                    continue
                old_values = merged.get(key, [])
                old_values = old_values if isinstance(old_values, list) else []
                merged[key] = list(dict.fromkeys(value + old_values))[:10]
            else:
                merged[key] = value
        return merged

    @staticmethod
    def format_useful_information(useful_information: Dict[str, Any]) -> str:
        """Stably serialize the accumulated clues so the next-round prompt is readable and auditable."""
        if not useful_information:
            return "无可复用信息，请仅依据 Top-10 证据重新分析。"
        return json.dumps(useful_information, ensure_ascii=False, indent=2)
