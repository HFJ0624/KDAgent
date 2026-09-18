"""Tests for model selection and result gating of the formal batch experiment."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_final_experiments import (
    OUTPUT_ROOT,
    build_command,
    parse_args,
    validate_experiment_result,
)


class FinalExperimentRunnerTest(unittest.TestCase):
    def test_multiple_models_and_conditions_preserve_cli_order(self):
        """Multiple models and conditions given on the command line must run serially in original order."""
        args = parse_args(
            [
                "--models",
                "glm-5.2",
                "deepseek-v4-pro",
                "deepseek-v4-flash",
                "--conditions",
                "baseline",
                "only_rag",
            ]
        )

        self.assertEqual(
            args.models,
            ["glm-5.2", "deepseek-v4-pro", "deepseek-v4-flash"],
        )
        self.assertEqual(args.conditions, ["baseline", "only_rag"])
        self.assertEqual(args.temperature, 0.2)
        self.assertEqual(args.max_tokens, 8192)
        self.assertEqual(args.output_root, OUTPUT_ROOT)
        self.assertEqual(OUTPUT_ROOT.name, "final_experiments_frozen_v1")

    def test_v2_condition_is_explicit_and_enables_dual_branch_flag(self):
        """v2 does not enter the old default matrix; the dual-branch flag is only passed when explicitly selected."""
        default_args = parse_args(["--models", "qwen-plus"])
        self.assertNotIn("dual_branch_fusion_agent", default_args.conditions)

        args = parse_args(
            [
                "--models",
                "qwen-plus",
                "--conditions",
                "dual_branch_fusion_agent",
                "--output-root",
                "outputs/final_experiments_frozen_v2",
            ]
        )
        command = build_command(
            "qwen-plus",
            Path("output"),
            1,
            1,
            args,
            use_dual_branch_fusion=1,
        )

        self.assertEqual(
            command[command.index("--use_dual_branch_fusion") + 1],
            "1",
        )

    def test_preflight_can_use_isolated_output_root(self):
        args = parse_args(
            [
                "--models",
                "glm-5.2",
                "--conditions",
                "baseline",
                "--output-root",
                "outputs/thinking_budget_preflight",
                "--case-ids",
                "12",
                "13",
                "16",
            ]
        )

        self.assertEqual(
            args.output_root,
            Path("outputs/thinking_budget_preflight"),
        )
        self.assertEqual(args.case_ids, ["12", "13", "16"])

    def test_built_command_uses_unified_max_tokens(self):
        args = parse_args(["--models", "deepseek-v4-pro", "--conditions", "baseline"])
        command = build_command("deepseek-v4-pro", Path("output"), 0, 0, args)

        self.assertEqual(command[command.index("--max_tokens") + 1], "8192")

    def test_built_command_passes_selected_case_ids(self):
        args = parse_args(
            [
                "--models",
                "deepseek-v4-flash",
                "--conditions",
                "baseline",
                "--case-ids",
                "12",
                "13",
                "16",
            ]
        )
        command = build_command("deepseek-v4-flash", Path("output"), 0, 0, args)

        case_id_index = command.index("--case_ids")
        self.assertEqual(command[case_id_index + 1 :], ["12", "13", "16"])

    def test_result_gate_rejects_truncated_output(self):
        """Any length truncation must prevent the batch from continuing to the next model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir()
            metrics = {
                "num_records": 1,
                "api_failure_count": 0,
                "truncated_response_count": 1,
            }
            with (metrics_dir / "test-model_metrics.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(metrics, file)

            with self.assertRaisesRegex(RuntimeError, "输出截断"):
                validate_experiment_result(output_dir, "test-model", 1)


if __name__ == "__main__":
    unittest.main()
