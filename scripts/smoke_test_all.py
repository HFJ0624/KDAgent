"""End-to-end smoke test that mocks out all models.

Verifies that run_all_models.py can produce all_models_summary.csv.
"""
import argparse
import json
import os
import sys
import logging
from unittest.mock import patch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _build_mock_response(case_id: int) -> dict:
    mod = case_id % 3
    if mod == 0:
        content = json.dumps(
            {
                "predicted_root_causes": ["V001", "V002", "V003"],
                "primary_root_cause": "V001",
                "reasoning": "V001 的异常分数最高。",
                "evidence_variables": ["V001"],
                "confidence": 0.92,
            }
        )
    elif mod == 1:
        content = json.dumps(
            {
                "predicted_root_causes": ["V099", "V002"],
                "primary_root_cause": "V099",
                "reasoning": "幻觉变量。",
                "evidence_variables": ["V099"],
                "confidence": 0.5,
            }
        )
    else:
        content = (
            "根据我的分析，最可能的根因变量是 V001。"
            "其他可能的变量包括 V002 和 V003。"
        )
    return {
        "success": True,
        "content": content,
        "raw_response": json.dumps({"mock": True}, ensure_ascii=False),
        "status_code": 200,
        "error_type": None,
        "error_message": None,
        "elapsed": 0.01,
    }


class _FakeModelClient:
    def __init__(self, model_config, logger=None):
        self.config = model_config
        self.name = model_config.get("name", "mock-model")
        self.provider = model_config.get("provider", "mock")
        self.base_url = model_config.get("base_url", "")
        self.model = model_config.get("model", self.name)
        self.temperature = float(model_config.get("temperature", 0.2))
        self.max_tokens = int(model_config.get("max_tokens", 2048))
        self.thinking_budget = model_config.get("thinking_budget")
        self.api_key_env = model_config.get("api_key_env", "")
        self.logger = logger or logging.getLogger("rca_experiment")
        self._current_case_id: int = 0

    def is_configured(self) -> bool:
        return True

    def identifier(self) -> str:
        return self.name.replace("/", "_").replace(" ", "_")

    def chat(
        self,
        prompt,
        system_prompt="",
        max_retries=3,
        timeout=60,
        retry_base_sleep=None,
    ):
        return _build_mock_response(self._current_case_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/llm_prompt_cases.sample.jsonl")
    parser.add_argument("--config_path", default="configs/models.yaml")
    parser.add_argument("--output_dir", default="outputs_smoke_all")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--sleep_ms", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--prompt_template", default="prompts/rca_prompt_template.txt")
    parser.add_argument("--knowledge_path", default="data/process_knowledge_template.md")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", nargs="*", default=["qwen-plus", "qwen-max"])
    args = parser.parse_args()

    os.environ.setdefault("DASHSCOPE_API_KEY", "mock-key")
    os.environ.setdefault("DEEPSEEK_API_KEY", "mock-key")

    with patch("src.model_client.ModelClient", _FakeModelClient):
        from src.run_all_models import main as run_all_main
        # run_all_models reads sys.argv itself, so it must be rewritten
        import src.run_all_models as runner
        # Replace the call to run_all_models.main:
        # A simpler approach: modify sys.argv to invoke the core logic directly
        orig_argv = sys.argv[:]
        try:
            sys.argv = [
                "run_all_models",
                "--data_path", args.data_path,
                "--config_path", args.config_path,
                "--output_dir", args.output_dir,
                "--num_runs", str(args.num_runs),
                "--max_cases", str(args.max_cases),
                "--sleep_ms", str(args.sleep_ms),
                "--max_retries", str(args.max_retries),
                "--prompt_template", args.prompt_template,
                "--knowledge_path", args.knowledge_path,
            ]
            if args.models:
                sys.argv.extend(["--models"] + list(args.models))
            run_all_main()
        finally:
            sys.argv = orig_argv

    summary_csv = os.path.join(args.output_dir, "metrics", "all_models_summary.csv")
    summary_json = os.path.join(args.output_dir, "metrics", "all_models_summary.json")
    print("汇总 CSV 存在：", os.path.exists(summary_csv))
    print("汇总 JSON 存在：", os.path.exists(summary_json))


if __name__ == "__main__":
    main()
