from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.core import (
    write_source_order_scores,
    write_popularity_scores,
)
from myrec.utils.jsonl import iter_jsonl


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class SimpleBaselineTest(unittest.TestCase):
    def test_source_order_preserves_position_and_manifest_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            _write_json(
                standardized / "manifest.json",
                {"dataset_id": "tiny", "dataset_version": "scout-v1"},
            )
            _write_json(
                standardized / "candidate_manifest.json",
                {
                    "entries": [
                        {
                            "split": "dev",
                            "request_id": "r",
                            "candidate_item_ids": ["z", "a"],
                        }
                    ]
                },
            )
            _write_jsonl(
                standardized / "records_dev.jsonl",
                [
                    {
                        "request_id": "r",
                        "candidates": [{"item_id": "z"}, {"item_id": "a"}],
                    }
                ],
            )
            metadata = write_source_order_scores(
                standardized,
                "dev",
                "source",
                runs_dir=root / "runs",
            )
            rows = list(iter_jsonl(root / "runs" / "source" / "scores.jsonl"))
            scores = {row["candidate_item_id"]: row["score"] for row in rows}
            self.assertGreater(scores["z"], scores["a"])
            self.assertEqual(metadata["dataset_id"], "tiny")
            self.assertEqual(metadata["dataset_version"], "scout-v1")

    def test_popularity_stats_are_namespaced_by_dataset_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            _write_json(
                standardized / "manifest.json",
                {"dataset_id": "tiny", "dataset_version": "scout-v1"},
            )
            _write_json(
                standardized / "candidate_manifest.json",
                {
                    "entries": [
                        {"split": "dev", "request_id": "r", "candidate_item_ids": ["a", "b"]}
                    ]
                },
            )
            _write_jsonl(
                standardized / "records_train.jsonl",
                [
                    {
                        "request_id": "t",
                        "candidates": [
                            {"item_id": "a", "clicked": 1},
                            {"item_id": "b", "clicked": 0},
                        ],
                    }
                ],
            )
            _write_jsonl(
                standardized / "records_dev.jsonl",
                [{"request_id": "r", "candidates": [{"item_id": "a"}, {"item_id": "b"}]}],
            )
            metadata = write_popularity_scores(
                standardized,
                "dev",
                "pop",
                runs_dir=root / "runs",
                artifacts_dir=root / "artifacts",
            )
            self.assertIn("scout-v1", metadata["popularity_stats_path"])
            self.assertTrue(Path(metadata["popularity_stats_path"]).exists())


if __name__ == "__main__":
    unittest.main()
