"""Synthetic tests for ReAct-adapted that do not call any external API."""

from __future__ import annotations

import unittest

from experiments.react_adapted_v1.react_agent import AgentLimits, ReactAdaptedAgent, parse_action


class FakeModel:
    model = "fake-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.responses = [
            '{"action":"get_episode_evidence","arguments":{},"action_reason":"inspect"}',
            '{"action":"search_domain_knowledge","arguments":{"query":"pump downstream flow"},"action_reason":"check process"}',
            '{"action":"submit_ranking","arguments":{"predicted_root_causes":["V2","V1"],"primary_root_cause":"V2","confidence":0.7,"brief_reason":"evidence and process agree"},"action_reason":"submit"}',
        ]

    def chat(self, **kwargs: object) -> dict:
        prompt = str(kwargs.get("prompt") or "")
        self.prompts.append(prompt)
        if any(token in prompt.lower() for token in ('"gt_vars"', '"ground_truth"', '"root_tags"')):
            raise AssertionError("标签字段进入了模型提示")
        content = self.responses.pop(0)
        return {"success": True, "content": content, "finish_reason": "stop", "elapsed": 0.01, "prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 0, "total_tokens": 15, "network_attempts": 1}


class FakeEmbedder:
    def embed_query(self, _: str) -> list[float]:
        return [1.0, 0.0]


class FakeStore:
    def query(self, _: list[float], top_k: int) -> dict:
        return {"results": [{"id": "c1", "content": "Pump changes downstream flow.", "metadata": {"source": "synthetic.md"}, "score": 0.8}][:top_k]}


class FakeRetriever:
    embedder = FakeEmbedder()
    store = FakeStore()


class ReactAdaptedTest(unittest.TestCase):
    def test_action_parser_accepts_fenced_json(self) -> None:
        self.assertEqual(parse_action('```json\n{"action":"get_episode_evidence","arguments":{}}\n```')["action"], "get_episode_evidence")

    def test_real_tool_loop_and_submission(self) -> None:
        case = {
            "case_id": "SYN-1",
            "top10_vars": ["V1", "V2"],
            "top10_details": [
                {"var": "V1", "name": "LIT101", "type": "continuous", "score": 1.0, "description": "stable"},
                {"var": "V2", "name": "P101", "type": "state", "score": 2.0, "description": "changed"},
            ],
        }
        result = ReactAdaptedAgent(FakeModel(), FakeRetriever(), "test", AgentLimits()).run("synthetic", case, 1)
        self.assertTrue(result["completed"])
        self.assertEqual(result["parsed_response"]["predicted_root_causes"], ["V2", "V1"])
        self.assertEqual(result["logic_calls"], 3)
        self.assertEqual(result["knowledge_search_calls"], 1)


if __name__ == "__main__":
    unittest.main()
