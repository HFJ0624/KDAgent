"""Offline regression tests for the iterative self-refinement Agent, without calling a real model API."""

import json
import unittest

from src.iterative_agent import IterativeSelfRefinementAgent
from src.evaluator import build_parsed_rows, compute_metrics, evaluate_case
from src.response_parser import clamp_to_candidates, parse_response
from src.response_validator import validate_response


def _inference_process(primary: str):
    """Build a verifiable four-step inference process, reused by all compliant-response tests."""
    stages = ["候选证据比较", "时间先后判断", "工艺传播分析", "最终根因决策"]
    return [
        {
            "step": index,
            "stage": stage,
            "analysis": f"第 {index} 步根据输入证据分析并复核候选变量 {primary}。",
            "evidence_variables": [primary],
        }
        for index, stage in enumerate(stages, start=1)
    ]


def _valid_response(primary: str) -> str:
    """Build a minimal RCA JSON that passes structural validation, so tests do not depend on prompt details."""
    return json.dumps(
        {
            "predicted_root_causes": [primary],
            "primary_root_cause": primary,
            "root_cause_name": "测试变量",
            "confidence": 0.9,
            "reasoning": "该变量在异常窗口内具有持续且显著的偏离。",
            "inference_process": _inference_process(primary),
            "numerical_evidence": "异常分数排名第一。",
            "temporal_evidence": "残差在窗口开始后持续升高。",
            "type_aware_reasoning": "连续测量变量出现稳定偏移。",
            "process_relation_reasoning": "其变化可解释下游变量的连锁异常。",
            "why_not_other_candidates": "其余候选的异常时间更晚。",
            "evidence_variables": [primary],
        }
    )


class _ScriptedClient:
    """Return preset responses in order, to reliably reproduce the Agent's multi-round behavior."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def chat(self, **kwargs):
        self.prompts.append(kwargs.get("prompt", ""))
        response = next(self.responses)
        if isinstance(response, dict):
            return {"success": True, "elapsed": 0.0, **response}
        return {"success": True, "content": response, "elapsed": 0.0}


class IterativeAgentTest(unittest.TestCase):
    def test_structured_inference_process_is_parsed_and_validated(self):
        """A compliant four-step inference process must fully pass parsing and Agent validation."""
        parsed = parse_response(_valid_response("V001"))
        validation = validate_response(parsed, ["V001"])

        self.assertEqual(len(parsed["inference_process"]), 4)
        self.assertEqual(parsed["inference_process"][3]["step"], 4)
        self.assertTrue(validation["is_valid"])

    def test_truncated_response_fails_validation_even_when_json_is_valid(self):
        """Length truncation takes priority over JSON structure; an incomplete response must go to the next round."""
        parsed = parse_response(_valid_response("V001"))
        validation = validate_response(
            parsed,
            ["V001"],
            finish_reason="length",
        )

        self.assertFalse(validation["is_valid"])
        self.assertIn("output_truncated", validation["retry_reasons"])

    def test_inference_process_is_not_truncated_in_csv_row(self):
        """CSV flattening must fully preserve long analysis text so the paper can review the model's reasoning process."""
        process = _inference_process("V001")
        process[0]["analysis"] = "长推理证据" * 200
        rows = build_parsed_rows(
            [{"parsed_response": {"inference_process": process}}]
        )
        restored = json.loads(rows[0]["inference_process"])

        self.assertEqual(restored[0]["analysis"], process[0]["analysis"])
        self.assertGreater(len(restored[0]["analysis"]), 500)

    def test_empty_prediction_is_miss_without_top1_fallback(self):
        """An empty prediction cannot inherit the TA-RCA Top-1 hit credit."""
        evaluated = evaluate_case(
            {
                "parsed_response": {
                    "predicted_root_causes": [],
                    "primary_root_cause": None,
                },
                "gt_vars": ["V001"],
                "top10_vars": ["V001", "V002"],
            }
        )

        self.assertEqual(evaluated["parsed_response"]["predicted_root_causes"], [])
        self.assertIsNone(evaluated["parsed_response"]["primary_root_cause"])
        self.assertEqual(evaluated["is_hit_at_1"], 0)
        self.assertEqual(evaluated["is_hit_at_3"], 0)
        self.assertEqual(evaluated["is_hit_at_5"], 0)

    def test_outside_candidate_clamped_to_empty_is_still_miss(self):
        """After an outside-candidate answer is clamped, the evaluator must not backfill the candidate Top-1."""
        parsed = clamp_to_candidates(
            {
                "predicted_root_causes": ["V999"],
                "primary_root_cause": "V999",
                "inference_process": _inference_process("V999"),
            },
            ["V001", "V002"],
        )
        evaluated = evaluate_case(
            {
                "parsed_response": parsed,
                "gt_vars": ["V001"],
                "top10_vars": ["V001", "V002"],
            }
        )

        self.assertEqual(evaluated["parsed_response"]["predicted_root_causes"], [])
        self.assertEqual(evaluated["is_hit_at_1"], 0)
        self.assertEqual(evaluated["is_hit_at_3"], 0)
        self.assertEqual(evaluated["is_hit_at_5"], 0)
        self.assertTrue(
            all(
                not step["evidence_variables"]
                for step in evaluated["parsed_response"]["inference_process"]
            )
        )

    def test_metrics_distinguish_unique_cases_and_repeated_records(self):
        """Three repeated runs should report both the unique case count and the final record count."""
        records = []
        for run_id in range(1, 4):
            for case_id in range(2):
                records.append(
                    {
                        "case_id": case_id,
                        "run_id": run_id,
                        "gt_vars": ["V001"],
                        "top10_vars": ["V001"],
                        "parsed_response": {
                            "valid_json": True,
                            "inference_process": _inference_process("V001"),
                        },
                    }
                )

        metrics = compute_metrics(records, "test-model", num_runs=3)
        self.assertEqual(metrics["num_cases"], 2)
        self.assertEqual(metrics["num_records"], 6)
        self.assertEqual(metrics["covered_case_count"], 2)
        self.assertEqual(metrics["covered_record_count"], 6)
        self.assertEqual(metrics["valid_inference_process_rate"], 1.0)
        self.assertEqual(metrics["average_inference_process_steps"], 4.0)

    def test_recovered_agent_truncation_is_history_not_final_truncation(self):
        """Agent records truncated in an intermediate round but ending in a stop must not trigger the final truncation gate."""
        record = {
            "case_id": 5,
            "run_id": 1,
            "finish_reason": "stop",
            # Simulate legacy summary fields persisted before the fix.
            "was_truncated": True,
            "had_truncated_iteration": True,
            "agent_status": "fallback_unknown",
            "parsed_response": {
                "valid_json": False,
                "inference_process": [],
            },
        }

        metrics = compute_metrics([record], "test-model", num_runs=1)
        rows = build_parsed_rows([record])

        self.assertEqual(metrics["truncated_response_count"], 0)
        self.assertEqual(rows[0]["was_truncated"], 0)
        self.assertEqual(rows[0]["had_truncated_iteration"], 1)

    def test_outside_candidate_triggers_refinement_before_success(self):
        """An outside-candidate variable must not be hidden by clamping; it must be the direct cause of second-round correction."""
        agent = IterativeSelfRefinementAgent(
            _ScriptedClient([_valid_response("V999"), _valid_response("V001")])
        )
        result = agent.run_case({"case_id": "case-1", "top10_vars": ["V001"]}, "initial", ["V001"])

        self.assertTrue(result["success"])
        self.assertEqual(result["iteration_count"], 2)
        self.assertIn(
            "predicted_contains_outside_var",
            result["all_iterations"][0]["validation_result"]["retry_reasons"],
        )

    def test_truncated_first_round_is_recovered_by_second_round(self):
        """After the first round is truncated, the Agent must carry the failure reason into the second round and recover."""
        client = _ScriptedClient(
            [
                {
                    "content": _valid_response("V001"),
                    "finish_reason": "length",
                },
                {
                    "content": _valid_response("V001"),
                    "finish_reason": "stop",
                },
            ]
        )
        agent = IterativeSelfRefinementAgent(client)
        result = agent.run_case(
            {"case_id": "case-truncated", "top10_vars": ["V001"]},
            "initial",
            ["V001"],
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["iteration_count"], 2)
        self.assertIn(
            "output_truncated",
            result["all_iterations"][0]["validation_result"]["retry_reasons"],
        )
        self.assertIn("output_truncated", client.prompts[1])
        self.assertEqual(result["finish_reason"], "stop")

    def test_three_failures_remain_unknown_without_top1_fallback(self):
        """After all three rounds fail, the evaluation must keep UNKNOWN and an empty prediction list."""
        agent = IterativeSelfRefinementAgent(_ScriptedClient(["not json"] * 3))
        result = agent.run_case({"case_id": "case-2", "top10_vars": ["V001"]}, "initial", ["V001"])
        evaluated = evaluate_case(
            {
                "parsed_response": result["parsed_response"],
                "gt_vars": ["V001"],
                "top10_vars": ["V001"],
            }
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["retry_count"], 2)
        self.assertEqual(evaluated["parsed_response"]["primary_root_cause"], "UNKNOWN")
        self.assertEqual(evaluated["parsed_response"]["predicted_root_causes"], [])
        self.assertEqual(evaluated["is_hit_at_1"], 0)

    def test_third_round_reuses_valid_clues_and_rag_context(self):
        """The third round must inherit the first two rounds' information and continue carrying RAG process knowledge."""
        first_response = json.dumps(
            {
                "predicted_root_causes": ["V001", "V999"],
                "primary_root_cause": "V001",
                "reasoning": "V001 的异常出现更早，V999 属于候选集外噪声。",
                "evidence_variables": ["V001", "V999"],
                "confidence": 0.7,
            },
            ensure_ascii=False,
        )
        second_response = json.dumps(
            {
                "predicted_root_causes": ["V001"],
                "primary_root_cause": "V001",
                "reasoning": "第二轮继续保留 V001，但解释字段仍不完整。",
                "numerical_evidence": "V001 的异常分数位于候选前列。",
                "evidence_variables": ["V001"],
                "confidence": 0.8,
            },
            ensure_ascii=False,
        )
        client = _ScriptedClient(
            [first_response, second_response, _valid_response("V001")]
        )
        agent = IterativeSelfRefinementAgent(client)
        result = agent.run_case(
            {"case_id": "case-3", "top10_vars": ["V001"]},
            "initial",
            ["V001"],
            rag_contexts=[{"content": "P1 泵阀变化会影响流量与液位。"}],
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["iteration_count"], 3)
        self.assertIn("candidate_root_causes", client.prompts[1])
        self.assertIn("V001", client.prompts[1])
        self.assertNotIn("V999", result["all_iterations"][0]["useful_information_extracted"])
        self.assertIn("第二轮继续保留 V001", client.prompts[2])
        self.assertIn("P1 泵阀变化会影响流量与液位", client.prompts[2])


if __name__ == "__main__":
    unittest.main()
