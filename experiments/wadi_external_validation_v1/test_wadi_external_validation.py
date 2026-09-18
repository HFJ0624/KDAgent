"""Zero-API tests for the WADI adapter and evaluation orchestration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.wadi_external_validation_v1.run_wadi_external_validation import _method_metrics
from experiments.wadi_external_validation_v1.wadi_adapter import (
    WadiFiles,
    build_episode_manifest,
    contiguous_intervals,
)
from src.data_loader import normalize_case
from src.prompt_builder import build_prompt


class WadiAdapterTest(unittest.TestCase):
    def test_contiguous_intervals_are_not_expanded(self) -> None:
        self.assertEqual(contiguous_intervals([0, 1, 1, 0, 1, 0]), [(1, 2), (4, 4)])

    def test_manifest_uses_mapping_order_and_root_tags_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "normal.csv").write_text(
                "Row,Date,Time,1_FIT_001_PV,1_MV_001_STATUS\n"
                "0,d,t,1.0,0\n",
                encoding="utf-8",
            )
            (root / "attack.csv").write_text(
                "0,1,2,3,4,5\n"
                "Row,Date,Time,1_FIT_001_PV,1_MV_001_STATUS,Attack LABLE (1:No Attack; -1:Attack)\n"
                "0,d,t,1.0,0,1\n"
                "1,d,t,2.0,0,-1\n"
                "2,d,t,2.0,0,-1\n"
                "3,d,t,1.0,1,1\n"
                "4,d,t,1.0,1,-1\n",
                encoding="utf-8",
            )
            (root / "mapping.csv").write_text(
                "attack_id,start_time,end_time,root_tags,affected_tags,notes\n"
                "1,2017-01-01,2017-01-01,1_FIT_001_PV,1_MV_001_STATUS,x\n"
                "2,2017-01-01,2017-01-01,,,y\n",
                encoding="utf-8",
            )
            manifest, catalog = build_episode_manifest(
                WadiFiles(root / "normal.csv", root / "attack.csv", root / "mapping.csv")
            )
            self.assertEqual(len(catalog), 2)
            self.assertEqual(manifest.loc[0, "gt_vars"], "W000")
            self.assertTrue(bool(manifest.loc[0, "included"]))
            self.assertFalse(bool(manifest.loc[1, "included"]))
            self.assertEqual(manifest.loc[1, "exclusion_reason"], "missing_root_tags")

    def test_wadi_system_name_survives_normalization(self) -> None:
        case = normalize_case(
            {
                "case_id": "WADI-E01",
                "system_name": "WADI",
                "task": "WADI RCA",
                "gt_vars": ["W000"],
                "top10_vars": ["W000"],
            }
        )
        prompt = build_prompt(
            case,
            "{process_knowledge}\n{rag_knowledge_block}\n{top10_candidates}",
            rag_contexts=[{"source": "test", "content": "generic relation"}],
        )
        self.assertEqual(case["system_name"], "WADI")
        self.assertIn("WADI 工艺知识库", prompt)
        self.assertNotIn("SWaT 工艺知识库", prompt)

    def test_metrics_support_multi_label_ground_truth(self) -> None:
        records = [
            {
                "run_id": 1,
                "gt_vars": ["W001", "W002"],
                "top10_vars": ["W000", "W002"],
                "parsed_response": {"predicted_root_causes": ["W000", "W002"]},
            }
        ]
        metrics = _method_metrics(records, "baseline")
        self.assertEqual(metrics["Hit@1"], 0.0)
        self.assertEqual(metrics["Hit@3"], 1.0)
        self.assertEqual(metrics["MRR"], 0.5)
        self.assertEqual(metrics["Covered Top-5 Recovery"], 1.0)


if __name__ == "__main__":
    unittest.main()
