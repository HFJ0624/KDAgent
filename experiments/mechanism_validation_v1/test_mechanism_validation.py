"""Pure-function regression tests for mechanism validation that do not call the API."""

import unittest

from experiments.mechanism_validation_v1.run_mechanism_validation import (
    MODEL_CONFIG,
    acceptable_primary,
    feedback_prompt,
    metric_values,
    normalize_vars,
)
from src.model_client import load_model_configs


class MechanismValidationTest(unittest.TestCase):
    def test_model_configs_are_keyed_by_model_name(self):
        configs = load_model_configs(str(MODEL_CONFIG))
        self.assertIn("qwen-plus", configs)
        self.assertEqual(configs["qwen-plus"]["name"], "qwen-plus")

    def test_normalize_vars_is_stable_and_unique(self):
        self.assertEqual(normalize_vars([" v1 ", "V1", "V 2"]), ["V1", "V2"])

    def test_primary_requires_candidate_membership(self):
        self.assertEqual(acceptable_primary({"primary_root_cause": "v1"}, ["V1"]), "V1")
        self.assertEqual(acceptable_primary({"primary_root_cause": "V9"}, ["V1"]), "")

    def test_multi_label_metrics_use_earliest_hit(self):
        values = metric_values(["V3", "V2", "V1"], ["V1", "V2"])
        self.assertEqual(values["hit_at_1"], 0)
        self.assertEqual(values["hit_at_3"], 1)
        self.assertEqual(values["mrr"], 0.5)

    def test_feedback_prompts_only_differ_by_error_specificity(self):
        snapshot = {
            "case": {"top10": ["V1"], "top_k_variables": [{"var_id": "V1"}]},
            "first": {"prompt": "EVIDENCE", "content": "OLD", "parsed_response": {}},
            "validation": {"retry_reasons": ["invalid_json"]},
        }
        targeted = feedback_prompt(snapshot, "targeted")
        generic = feedback_prompt(snapshot, "generic")
        self.assertIn("OLD", targeted)
        self.assertIn("invalid_json", targeted)
        self.assertIn("OLD", generic)
        self.assertNotIn("invalid_json", generic)


if __name__ == "__main__":
    unittest.main()
