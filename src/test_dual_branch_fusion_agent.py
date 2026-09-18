"""Offline regression tests for the v2 dual-branch fusion Agent."""

import json
import unittest

from src.dual_branch_fusion_agent import DualBranchFusionAgent


def _response(primary, candidates):
    """Build a minimal response that passes the existing Agent structure validation."""
    return json.dumps(
        {
            "predicted_root_causes": candidates,
            "primary_root_cause": primary,
            "root_cause_name": f"设备 {primary}",
            "confidence": 0.8,
            "reasoning": f"{primary} 的异常时间更早，且能够解释其他变量变化。",
            "inference_process": [
                {
                    "step": index,
                    "stage": stage,
                    "analysis": f"第 {index} 步复核 {primary} 的现场证据和传播关系。",
                    "evidence_variables": [primary],
                }
                for index, stage in enumerate(
                    ["候选证据比较", "时间先后判断", "工艺传播分析", "最终根因决策"],
                    start=1,
                )
            ],
            "numerical_evidence": f"{primary} 的异常分数显著升高。",
            "temporal_evidence": f"{primary} 在其他候选之前发生变化。",
            "type_aware_reasoning": f"{primary} 的变量类型符合故障模式。",
            "process_relation_reasoning": f"{primary} 可以沿工艺路径影响下游。",
            "why_not_other_candidates": "其余候选出现更晚或属于传播响应。",
            "evidence_variables": [primary],
        },
        ensure_ascii=False,
    )


class _ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def chat(self, **kwargs):
        self.prompts.append(kwargs.get("prompt", ""))
        return {
            "success": True,
            "content": next(self.responses),
            "finish_reason": "stop",
            "elapsed": 0.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "reasoning_tokens": 0,
            "total_tokens": 150,
        }


class DualBranchFusionAgentTest(unittest.TestCase):
    def test_evidence_primary_is_preserved_and_rag_fills_remaining_slots(self):
        """The data branch locks the first slot, and the RAG branch fills the remaining four in original order."""
        client = _ScriptedClient(
            [
                _response("V1", ["V1", "V2", "V3"]),
                _response("V4", ["V4", "V5", "V6", "V7", "V8"]),
            ]
        )
        agent = DualBranchFusionAgent(client)
        result = agent.run_case(
            # The test input deliberately omits gt_vars to prove fusion does not rely on ground-truth labels.
            case={"case_id": "fusion-1", "top10_vars": [f"V{i}" for i in range(1, 9)]},
            evidence_prompt="DATA_ONLY_PROMPT",
            rag_prompt="RAG_PROMPT",
            top10_vars=[f"V{i}" for i in range(1, 9)],
        )

        parsed = result["parsed_response"]
        self.assertEqual(parsed["primary_root_cause"], "V1")
        self.assertEqual(
            parsed["predicted_root_causes"],
            ["V1", "V4", "V5", "V6", "V7"],
        )
        self.assertEqual(parsed["rag_supplemented_candidates"], ["V4", "V5", "V6", "V7"])
        self.assertFalse(result["branch_primary_agreement"])
        self.assertEqual(client.prompts, ["DATA_ONLY_PROMPT", "RAG_PROMPT"])
        self.assertEqual(
            [item["branch"] for item in result["all_iterations"]],
            ["data_evidence", "rag_knowledge"],
        )

    def test_branch_agreement_deduplicates_primary(self):
        """When both branches agree on the primary cause, keep it only once and continue supplementing by knowledge order."""
        client = _ScriptedClient(
            [
                _response("V1", ["V1", "V2", "V3"]),
                _response("V1", ["V1", "V4", "V5"]),
            ]
        )
        result = DualBranchFusionAgent(client).run_case(
            case={"case_id": "fusion-2"},
            evidence_prompt="DATA",
            rag_prompt="RAG",
            top10_vars=["V1", "V2", "V3", "V4", "V5"],
        )

        self.assertTrue(result["branch_primary_agreement"])
        self.assertEqual(
            result["parsed_response"]["predicted_root_causes"],
            ["V1", "V4", "V5", "V2", "V3"],
        )

    def test_rag_branch_takes_over_when_evidence_agent_exhausts_three_rounds(self):
        """When the data branch fails all three rounds, allow the RAG branch to provide the final primary cause."""
        client = _ScriptedClient(
            ["not json", "not json", "not json", _response("V4", ["V4", "V5"])],
        )
        result = DualBranchFusionAgent(client).run_case(
            case={"case_id": "fusion-3"},
            evidence_prompt="DATA",
            rag_prompt="RAG",
            top10_vars=["V1", "V2", "V3", "V4", "V5"],
        )

        self.assertTrue(result["success"])
        self.assertFalse(result["evidence_branch_success"])
        self.assertTrue(result["rag_branch_success"])
        self.assertEqual(result["agent_status"], "dual_branch_rag_fallback")
        self.assertEqual(result["parsed_response"]["primary_root_cause"], "V4")
        self.assertEqual(len(client.prompts), 4)


if __name__ == "__main__":
    unittest.main()
