"""Static checks for proposal isolation and scoring-label boundaries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

SYSTEM_ROOT = Path(__file__).resolve().parents[1]


class BoundaryTest(unittest.TestCase):
    def test_train_and_scorer_expose_no_evaluation_label_path(self) -> None:
        for relative in ("train/train_screen.py", "train/score_screen.py"):
            source = (SYSTEM_ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("qrels_", source)
            self.assertNotIn("records_test", source)
            self.assertNotIn("candidate_labels.npy", source)

    def test_frozen_resource_identity(self) -> None:
        config = yaml.safe_load(
            (SYSTEM_ROOT / "configs/screen.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["seed"], 20260708)
        self.assertEqual(config["environment"], "myrec-c02")
        self.assertEqual(config["physical_gpu"], 1)
        self.assertEqual(
            config["integrity"]["candidate_manifest_sha256"],
            "94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e",
        )
        self.assertEqual(config["dev_gate"]["evaluator_calls"], 1)
        self.assertLessEqual(config["training"]["max_gpu_hours"], 8.0)


if __name__ == "__main__":
    unittest.main()
