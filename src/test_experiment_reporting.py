"""Offline regression tests for result reporting to disk and checkpoint resume."""

import csv
import json
import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from src.main import (
    _PROJECT_ROOT,
    _load_case_summary_records,
    _load_completed_keys,
    _summarize_agent_api_status,
    main as experiment_main,
    parse_args,
    run_single_model,
)


class _OutsideCandidateClient:
    """Always output a variable outside the candidates, to confirm an empty prediction does not get a spurious TA-RCA hit."""

    call_count = 0

    def __init__(self, model_config, logger=None):
        self.name = model_config["name"]
        self.provider = model_config["provider"]
        self.base_url = model_config["base_url"]
        self.max_tokens = model_config["max_tokens"]
        self.thinking_budget = model_config.get("thinking_budget")
        self.temperature = float(model_config.get("temperature", 0.0))

    def is_configured(self):
        return True

    def chat(self, **kwargs):
        type(self).call_count += 1
        content = json.dumps(
            {
                "predicted_root_causes": ["V999"],
                "primary_root_cause": "V999",
                "confidence": 0.9,
                "reasoning": "离线测试固定输出候选集外变量。",
                "inference_process": [
                    {
                        "step": index,
                        "stage": stage,
                        "analysis": "离线测试根据固定证据执行结构化判断。",
                        "evidence_variables": ["V999"],
                    }
                    for index, stage in enumerate(
                        ["候选证据比较", "时间先后判断", "工艺传播分析", "最终根因决策"],
                        start=1,
                    )
                ],
            },
            ensure_ascii=False,
        )
        return {
            "success": True,
            "content": content,
            "raw_response": content,
            "finish_reason": "stop",
            "elapsed": 0.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "reasoning_tokens": 0,
            "total_tokens": 150,
        }


class ExperimentReportingTest(unittest.TestCase):
    def _write_jsonl(self, raw_dir, records):
        path = os.path.join(raw_dir, "test-model_run_1.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_resume_loads_complete_latest_case_summaries(self):
        """Resume summaries should exclude Agent rounds and defer to the last result for the same case."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_jsonl(
                raw_dir,
                [
                    {
                        "record_type": "case_summary",
                        "case_id": 0,
                        "run_id": 1,
                        "api_ok": True,
                        "result_version": "first-success",
                    },
                    {
                        "record_type": "agent_iteration",
                        "case_id": 1,
                        "run_id": 1,
                        "api_success": False,
                    },
                    {
                        "record_type": "case_summary",
                        "case_id": 1,
                        "run_id": 1,
                        "api_ok": False,
                        "result_version": "old-failure",
                    },
                    {
                        "record_type": "case_summary",
                        "case_id": 1,
                        "run_id": 1,
                        "api_ok": True,
                        "result_version": "rerun-success",
                    },
                ],
            )

            records = _load_case_summary_records(raw_dir, "test-model")
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1]["result_version"], "rerun-success")
            self.assertEqual(_load_completed_keys(raw_dir, "test-model"), {"1|0", "1|1"})

    def test_agent_api_status_only_fails_when_all_rounds_fail(self):
        """Requests succeeding but failing validation count as Agent failure; only when all three rounds fail is it an API failure."""
        api_ok, error = _summarize_agent_api_status(
            [
                {"api_success": False, "error_type": "Timeout"},
                {"api_success": True},
                {"api_success": False, "error_type": "RateLimit"},
            ]
        )
        self.assertTrue(api_ok)
        self.assertEqual(error, {})

        api_ok, error = _summarize_agent_api_status(
            [
                {"api_success": False, "error_type": "Timeout"},
                {"api_success": False, "error_type": "RateLimit"},
            ]
        )
        self.assertFalse(api_ok)
        self.assertEqual(error["error_type"], "RateLimit")

    def test_main_returns_nonzero_when_initialization_fails(self):
        """Initialization errors such as a missing key must expose a non-zero exit code to the batch."""
        config_path = os.path.join(_PROJECT_ROOT, "configs", "models.yaml")
        argv = [
            "main.py",
            "--model_name",
            "qwen-plus",
            "--config_path",
            config_path,
        ]
        with patch("sys.argv", argv), patch(
            "src.main.run_single_model",
            return_value={"model_name": "qwen-plus", "error": "missing API key"},
        ):
            with self.assertRaises(SystemExit) as raised:
                experiment_main()

        self.assertEqual(raised.exception.code, 1)

    def test_resume_reruns_failed_cases_by_default(self):
        """Resume skips only successful records by default; only with explicit config does it skip failed records."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_jsonl(
                raw_dir,
                [
                    {
                        "record_type": "case_summary",
                        "case_id": 0,
                        "run_id": 1,
                        "api_ok": True,
                    },
                    {
                        "record_type": "case_summary",
                        "case_id": 1,
                        "run_id": 1,
                        "api_ok": False,
                    },
                ],
            )

            self.assertEqual(_load_completed_keys(raw_dir, "test-model"), {"1|0"})
            self.assertEqual(
                _load_completed_keys(raw_dir, "test-model", skip_failed=True),
                {"1|0", "1|1"},
            )

    def test_resume_reruns_truncated_cases_by_default(self):
        """Records that succeed at the API level but have truncated output are still incomplete, so a default resume must re-call them."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_jsonl(
                raw_dir,
                [
                    {
                        "record_type": "case_summary",
                        "case_id": 0,
                        "run_id": 1,
                        "api_ok": True,
                        "finish_reason": "stop",
                        "was_truncated": False,
                    },
                    {
                        "record_type": "case_summary",
                        "case_id": 1,
                        "run_id": 1,
                        "api_ok": True,
                        "finish_reason": "length",
                        "was_truncated": False,
                    },
                    {
                        "record_type": "case_summary",
                        "case_id": 2,
                        "run_id": 1,
                        "api_ok": True,
                        "finish_reason": "stop",
                        "was_truncated": True,
                    },
                    {
                        "record_type": "agent_iteration",
                        "case_id": 3,
                        "run_id": 1,
                        "finish_reason": "length",
                        "was_truncated": True,
                    },
                    {
                        "record_type": "case_summary",
                        "case_id": 3,
                        "run_id": 1,
                        "api_ok": True,
                        "finish_reason": "stop",
                        # Backward compatibility: this field once indicated truncation in any Agent round.
                        "was_truncated": True,
                        "agent_status": "fallback_unknown",
                    },
                ],
            )

            self.assertEqual(
                _load_completed_keys(raw_dir, "test-model"),
                {"1|0", "1|3"},
            )
            self.assertEqual(
                _load_completed_keys(raw_dir, "test-model", skip_failed=True),
                {"1|0", "1|1", "1|2", "1|3"},
            )
            records = _load_case_summary_records(raw_dir, "test-model")
            recovered = next(record for record in records if record["case_id"] == 3)
            self.assertTrue(recovered["had_truncated_iteration"])

    def test_full_baseline_and_resume_keep_all_sixty_records(self):
        """Both the 20x3 baseline and the full resume should stably produce 60 records with no spurious hits."""
        data_path = os.path.join(
            _PROJECT_ROOT,
            "data",
            "swat_s2s_raw_window_fixed",
            "llm_prompt_cases.jsonl",
        )
        config_path = os.path.join(_PROJECT_ROOT, "configs", "models.yaml")

        with tempfile.TemporaryDirectory() as output_dir:
            argv = [
                "main.py",
                "--data_path",
                data_path,
                "--config_path",
                config_path,
                "--model_name",
                "qwen-plus",
                "--output_dir",
                output_dir,
                "--num_runs",
                "3",
                "--max_cases",
                "20",
                "--temperature",
                "0",
                "--max_tokens",
                "4096",
                "--sleep_ms",
                "0",
            ]
            _OutsideCandidateClient.call_count = 0
            with patch("sys.argv", argv), patch(
                "src.main.ModelClient", _OutsideCandidateClient
            ):
                first_metrics = run_single_model(parse_args())

            self.assertEqual(_OutsideCandidateClient.call_count, 60)
            self.assertEqual(first_metrics["num_cases"], 20)
            self.assertEqual(first_metrics["num_records"], 60)
            self.assertEqual(first_metrics["accuracy_at_1"], 0.0)
            self.assertEqual(first_metrics["accuracy_at_3"], 0.0)
            self.assertEqual(first_metrics["accuracy_at_5"], 0.0)
            self.assertFalse(first_metrics["use_rag"])
            self.assertEqual(first_metrics["rag_top_k"], 0)
            self.assertEqual(first_metrics["truncated_response_count"], 0)
            self.assertEqual(first_metrics["average_prompt_tokens"], 100.0)
            self.assertEqual(first_metrics["average_completion_tokens"], 50.0)
            self.assertEqual(first_metrics["average_total_tokens"], 150.0)

            # On the second run with resume, all records that succeeded at the API level
            # should be skipped, but the CSV/metrics still must be rebuilt from the
            # full JSONL into 60 records, rather than becoming an empty report.
            with patch("sys.argv", argv + ["--resume"]), patch(
                "src.main.ModelClient", _OutsideCandidateClient
            ):
                resumed_metrics = run_single_model(parse_args())

            self.assertEqual(_OutsideCandidateClient.call_count, 60)
            self.assertEqual(resumed_metrics["num_cases"], 20)
            self.assertEqual(resumed_metrics["num_records"], 60)
            self.assertEqual(resumed_metrics["total_cases_processed"], 60)
            self.assertEqual(resumed_metrics["api_success_rate"], 1.0)

            csv_path = os.path.join(
                output_dir, "parsed_results", "qwen-plus_parsed.csv"
            )
            with open(csv_path, "r", encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 60)
            self.assertEqual(int(rows[0]["inference_process_step_count"]), 4)
            self.assertEqual(len(json.loads(rows[0]["inference_process"])), 4)
            self.assertEqual(first_metrics["valid_inference_process_rate"], 0.0)

            # Windows does not allow deleting temporary log files still held open by a FileHandler.
            logging.shutdown()

    def test_selected_case_ids_only_call_requested_cases(self):
        """The case_ids preflight must run exactly the specified samples, not degenerate to the first N cases."""
        data_path = os.path.join(
            _PROJECT_ROOT,
            "data",
            "swat_s2s_raw_window_fixed",
            "llm_prompt_cases.jsonl",
        )
        config_path = os.path.join(_PROJECT_ROOT, "configs", "models.yaml")

        with tempfile.TemporaryDirectory() as output_dir:
            argv = [
                "main.py",
                "--data_path",
                data_path,
                "--config_path",
                config_path,
                "--model_name",
                "qwen-plus",
                "--output_dir",
                output_dir,
                "--num_runs",
                "1",
                "--case_ids",
                "12",
                "13",
                "16",
                "--sleep_ms",
                "0",
            ]
            _OutsideCandidateClient.call_count = 0
            with patch("sys.argv", argv), patch(
                "src.main.ModelClient", _OutsideCandidateClient
            ):
                metrics = run_single_model(parse_args())

            self.assertEqual(_OutsideCandidateClient.call_count, 3)
            self.assertEqual(metrics["num_cases"], 3)
            self.assertEqual(metrics["num_records"], 3)
            logging.shutdown()


if __name__ == "__main__":
    unittest.main()
