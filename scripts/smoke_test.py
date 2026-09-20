"""End-to-end smoke test using a mock LLM client (no real API required).

The mock model returns valid JSON / invalid JSON / hallucinated cases,
then the pipeline is verified to correctly produce metrics, the parsed CSV, and the summary table.
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
    """Generate a deterministic mock response based on case_id.

    case_id % 3 == 0 : valid JSON, matches the GT pattern
    case_id % 3 == 1 : valid JSON, but contains a hallucinated variable
    case_id % 3 == 2 : plain text response (used to trigger the regex fallback parser)
    """
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
        "raw_response": json.dumps({"mock": True, "case_id": case_id}, ensure_ascii=False),
        "status_code": 200,
        "error_type": None,
        "error_message": None,
        "elapsed": 0.01,
    }


class _FakeModelClient:
    """A drop-in replacement for ModelClient that makes no network requests."""

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
        # main.run_single_model sets this field before each .chat() call
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
        case_id = self._current_case_id
        self.logger.info("[MOCK] 为 case_id=%s 调用 chat", case_id)
        return _build_mock_response(case_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/llm_prompt_cases.sample.jsonl")
    parser.add_argument("--config_path", default="configs/models.yaml")
    parser.add_argument("--output_dir", default="outputs_smoke")
    parser.add_argument("--model_name", default="qwen-plus")
    parser.add_argument("--num_runs", type=int, default=2)
    parser.add_argument("--max_cases", type=int, default=6)
    parser.add_argument("--sleep_ms", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--prompt_template", default="prompts/rca_prompt_template.txt")
    parser.add_argument("--knowledge_path", default="data/process_knowledge_template.md")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_skip_failed", type=int, default=0)
    parser.add_argument("--request_timeout", type=int, default=120)
    parser.add_argument("--retry_base_sleep", type=int, default=5)
    parser.add_argument("--use_rag", type=int, default=0)
    parser.add_argument("--rag_config", default="configs/rag.yaml")
    parser.add_argument("--rag_top_k", type=int, default=None)
    parser.add_argument("--use_iterative_agent", type=int, default=0)
    parser.add_argument("--max_iterations", type=int, default=3)
    # Keeping the parameter contract consistent with the main program ensures the Agent main flow is
    # covered rather than only the ordinary single-turn flow.
    parser.add_argument("--min_confidence", type=float, default=0.5)
    parser.add_argument("--compact_evidence", type=int, default=0)
    # The remaining flags keep this smoke script in sync with src/main.py so
    # run_single_model always finds every attribute it reads (no drift).
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--thinking_budget", type=int, default=None)
    parser.add_argument("--use_dual_branch_fusion", type=int, default=0)
    parser.add_argument("--case_ids", nargs="*", default=None)
    parser.add_argument("--experiment_version", default="")
    parser.add_argument("--protocol_sha256", default="")
    args = parser.parse_args()

    os.environ.setdefault("DASHSCOPE_API_KEY", "mock-key")

    with patch("src.model_client.ModelClient", _FakeModelClient):
        from src.main import run_single_model

        metrics = run_single_model(args)

    print("\n=== 最终指标 ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    raw_dir = os.path.join(args.output_dir, "raw_responses")
    parsed_csv = os.path.join(args.output_dir, "parsed_results", f"{args.model_name}_parsed.csv")
    metrics_json = os.path.join(args.output_dir, "metrics", f"{args.model_name}_metrics.json")

    checks = [
        os.path.isdir(raw_dir),
        os.path.exists(parsed_csv),
        os.path.exists(metrics_json),
    ]
    print("输出检查（全部应为 True）：", checks)
    if not all(checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
