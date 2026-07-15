from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.full_token_training import build_pairwise_examples


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class FullTokenTrainingTest(unittest.TestCase):
    def test_pairwise_examples_are_deterministic_and_mode_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            qrels = root / "qrels.jsonl"
            _write_jsonl(
                records,
                [
                    {
                        "request_id": "r1",
                        "query": "shoe",
                        "history": [{"item_id": "h", "title": "old coat", "event": "click"}],
                        "candidates": [
                            {"item_id": "p", "title": "red shoe"},
                            {"item_id": "n1", "title": "blue shoe"},
                            {"item_id": "n2", "title": "green shoe"},
                        ],
                    }
                ],
            )
            _write_jsonl(
                qrels,
                [{"request_id": "r1", "clicked": ["p"], "purchased": []}],
            )
            full, stats = build_pairwise_examples(
                records,
                qrels,
                input_mode="full",
                history_budget=10,
                negatives_per_positive=2,
                seed=3,
            )
            qc, _ = build_pairwise_examples(
                records,
                qrels,
                input_mode="qc",
                history_budget=10,
                negatives_per_positive=2,
                seed=3,
            )
            self.assertEqual(stats["examples"], 2)
            self.assertIn("old coat", full[0].context)
            self.assertNotIn("old coat", qc[0].context)
            self.assertEqual(qc[0].context, "shoe")
            self.assertEqual(
                [row.negative_document for row in full],
                [row.negative_document for row in qc],
            )


if __name__ == "__main__":
    unittest.main()
