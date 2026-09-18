"""Offline tests for model reasoning-budget and token usage extraction."""

import json
import unittest

from src.model_client import ModelClient


def _config(**overrides):
    config = {
        "name": "test-model",
        "provider": "openai_compatible",
        "base_url": "https://example.invalid/v1",
        "model": "test-model",
        "max_tokens": 8192,
    }
    config.update(overrides)
    return config


class ModelClientBudgetTest(unittest.TestCase):
    def _run_client(self, config):
        client = ModelClient(config)
        client.api_key = "offline-test-key"
        captured = {}

        def fake_post(url, headers, payload, timeout):
            captured["payload"] = payload
            response = {
                "choices": [
                    {
                        "message": {
                            "content": "{}",
                            "reasoning_content": "精简的内部推理",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 8000,
                    "completion_tokens": 4000,
                    "total_tokens": 12000,
                    "completion_tokens_details": {"reasoning_tokens": 3000},
                },
            }
            return response, 200, json.dumps(response, ensure_ascii=False)

        client._http_post_json = fake_post
        return client, captured, client.chat("prompt", max_retries=1)

    def test_reasoning_model_sends_budget_and_extracts_usage(self):
        """A reasoning model should send a separate budget and fully extract the server token usage."""
        client, captured, result = self._run_client(
            _config(thinking_budget=3072)
        )

        self.assertEqual(captured["payload"]["max_tokens"], 8192)
        self.assertEqual(captured["payload"]["thinking_budget"], 3072)
        self.assertEqual(client.thinking_budget, 3072)
        self.assertEqual(result["prompt_tokens"], 8000)
        self.assertEqual(result["completion_tokens"], 4000)
        self.assertEqual(result["reasoning_tokens"], 3000)
        self.assertEqual(result["total_tokens"], 12000)
        self.assertEqual(result["network_attempts"], 1)

    def test_non_reasoning_model_omits_thinking_budget(self):
        """A model without a configured reasoning budget must not receive an unknown thinking_budget field."""
        client, captured, _ = self._run_client(_config())

        self.assertIsNone(client.thinking_budget)
        self.assertNotIn("thinking_budget", captured["payload"])


if __name__ == "__main__":
    unittest.main()
