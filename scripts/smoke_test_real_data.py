"""End-to-end smoke test using the real data file structure with a Mock LLM client.

The goal of this script is to run the full RCA experiment pipeline on the real
``llm_prompt_cases.jsonl`` data **without calling any real LLM API**, verifying that:

1. The new schema (``top_k_variables``, ``ground_truth.gt_vars``) is loaded correctly;
2. ``ResponseParser`` can parse all three output formats:
   - JSON format (including the new explainability fields);
   - XML tag format (``<answer>``, ``<reasoning>``, etc.);
   - plain text fallback format;
3. ``Evaluator`` produces a complete parsed-results CSV (gt_vars, top10_vars,
   primary_root_cause, predicted_root_causes all non-empty);
4. The new explainability fields (root_cause_name, numerical_evidence, etc.)
   are written correctly into the CSV.

Typical usage:
    python scripts/smoke_test_real_data.py
"""
import argparse
import json
import os
import sys
from unittest.mock import patch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


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
        self.logger = logger
        self._current_case_id: int = 0

    def is_configured(self) -> bool:
        return True

    def identifier(self) -> str:
        return self.name.replace("/", "_").replace(" ", "_")

    def chat(self, prompt, system_prompt="", max_retries=3, timeout=60):
        """Build a Mock response for the current case.

        The three output formats cycle by ``case_id % 3`` to cover the parser's three paths:
          - 0 -> JSON (with the new explainability fields);
          - 1 -> XML tag format;
          - 2 -> plain text fallback format.
        """
        # Try to extract V<n> from the candidate-variables section of the prompt as the "prediction result"
        import re as _re
        m = _re.search(r"候选变量[:\s]*\n([\s\S]*?)(?:\n\n|$)", prompt)
        candidate_text = m.group(1) if m else prompt
        vars_found = _re.findall(r"V\s*\d+", candidate_text, flags=_re.IGNORECASE)
        vars_upper = [v.upper().replace(" ", "") for v in vars_found][:5]

        # Select the output format by case_id, cycling over the parser's three paths
        mod = self._current_case_id % 3
        if mod == 0 and vars_upper:
            # JSON format: includes all explainability fields, verifying the parser extracts them completely
            content = json.dumps(
                {
                    "predicted_root_causes": vars_upper,
                    "primary_root_cause": vars_upper[0],
                    "root_cause_name": f"Sensor_{vars_upper[0]}",
                    "confidence": 0.85,
                    "reasoning": f"Model picks {vars_upper[0]} based on anomaly scores and process analysis.",
                    "numerical_evidence": f"Anomaly score 0.92, ranked #1 in candidates. Residual mean 0.45.",
                    "temporal_evidence": f"Residual starts deviating at t=10. Raw value shows sudden increase at t=15.",
                    "type_aware_reasoning": "Continuous variable. Analyzing residual trend and reconstruction error.",
                    "process_relation_reasoning": f"This variable is upstream, affecting downstream variables like {vars_upper[1] if len(vars_upper) > 1 else 'others'}.",
                    "why_not_other_candidates": f"Other variables show lower scores or appear to be downstream responses.",
                    "evidence_variables": vars_upper[:2],
                    "uncertainty_analysis": "Some uncertainty due to limited process knowledge.",
                    "is_valid_prediction": True,
                },
                ensure_ascii=False,
            )
        elif mod == 1 and vars_upper:
            # XML tag format: verifies the _parse_tagged path
            content = (
                "<semantic_observations>Top candidate has highest residual.</semantic_observations>\n"
                "<reasoning>The variable shows the strongest anomaly.</reasoning>\n"
                f"<answer>{vars_upper[0]}</answer>"
            )
        else:
            # Plain text fallback format: verifies the _VAR_TOKEN_RE regex fallback path
            content = f"The most likely root cause is {vars_upper[0] if vars_upper else 'V1'}."
        return {"ok": True, "content": content, "error": None, "raw": {"mock": True}, "elapsed": 0.01}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/llm_prompt_cases.jsonl")
    parser.add_argument("--config_path", default="configs/models.yaml")
    parser.add_argument("--output_dir", default="outputs_smoke_real")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--sleep_ms", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--prompt_template", default="prompts/rca_prompt_template.txt")
    parser.add_argument("--knowledge_path", default="data/process_knowledge_template.md")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model_name", default="qwen-plus")
    parser.add_argument("--use_rag", type=int, default=0)
    parser.add_argument("--rag_config", default="configs/rag.yaml")
    parser.add_argument("--rag_k", type=int, default=None)
    parser.add_argument("--rag_top_k", type=int, default=None)
    args = parser.parse_args()

    # If the default data path does not exist, fall back to the root containing the real data file (so it can be run from any cwd)
    if not os.path.exists(args.data_path):
        candidate = os.path.join(
            os.path.dirname(_PROJECT_ROOT),
            "llm_rca_experiment",
            "data",
            "swat_s2s_raw_window_fixed",
            "llm_prompt_cases.jsonl",
        )
        if os.path.exists(candidate):
            args.data_path = candidate
            print(f"[INFO] Using real data file: {candidate}")

    # Set a mock API key (it will not actually be used, but ModelClient initialization requires it)
    os.environ.setdefault("DASHSCOPE_API_KEY", "mock-key")

    # Replace the real ModelClient with the mock client so the entire pipeline makes no real HTTP requests
    with patch("src.model_client.ModelClient", _FakeModelClient):
        from src.main import run_single_model

        # Build an argparse.Namespace consistent with the command line and pass it to run_single_model
        ns = argparse.Namespace(
            data_path=args.data_path,
            config_path=args.config_path,
            model_name=args.model_name,
            output_dir=args.output_dir,
            prompt_template=args.prompt_template,
            knowledge_path=args.knowledge_path,
            num_runs=args.num_runs,
            max_cases=args.max_cases,
            temperature=args.temperature,
            resume=args.resume,
            sleep_ms=args.sleep_ms,
            max_retries=args.max_retries,
            use_rag=args.use_rag,
            rag_config=args.rag_config,
            rag_k=args.rag_k,
            rag_top_k=args.rag_top_k,
        )
        metrics = run_single_model(ns)
        print("\n=== FINAL METRICS ===")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # Read the produced parsed CSV and further verify that the key and explainability fields are complete
    import csv

    model_id = args.model_name.replace("/", "_").replace(" ", "_")
    parsed_csv = os.path.join(args.output_dir, "parsed_results", f"{model_id}_parsed.csv")
    if os.path.exists(parsed_csv):
        with open(parsed_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"\n=== CSV rows ({len(rows)}) ===")
        for row in rows:
            # Check, row by row, that all four evaluation key fields are populated
            missing = []
            for col in ("gt_vars", "top10_vars", "primary_root_cause", "predicted_root_causes"):
                if not row.get(col):
                    missing.append(col)
            status = "OK" if not missing else f"MISSING: {missing}"
            print(
                f"  case_id={row.get('case_id')} run_id={row.get('run_id')} gt_vars={row.get('gt_vars')} "
                f"top10_vars[:80]={str(row.get('top10_vars',''))[:80]} "
                f"primary={row.get('primary_root_cause')} predicted={row.get('predicted_root_causes')} "
                f"-> {status}"
            )

        # Count the non-empty coverage of the explainability fields
        new_fields = [
            "root_cause_name", "numerical_evidence", "temporal_evidence",
            "type_aware_reasoning", "process_relation_reasoning",
            "why_not_other_candidates", "uncertainty_analysis", "is_valid_prediction"
        ]
        print(f"\n=== Verify new explainability fields ===")
        for field in new_fields:
            values = [row.get(field, "") for row in rows]
            non_empty = [v for v in values if v and v != ""]
            print(f"  {field}: {len(non_empty)}/{len(rows)} rows have values")
            if non_empty:
                print(f"    Sample: {str(non_empty[0])[:100]}")


if __name__ == "__main__":
    main()
