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

    def test_history_dropout_is_deterministic_train_only_and_request_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            qrels = root / "qrels.jsonl"
            _write_jsonl(
                records,
                [
                    {
                        "request_id": request_id,
                        "query": "shoe",
                        "history": [
                            {"item_id": "h", "title": f"history {request_id}", "event": "click"}
                        ],
                        "candidates": [
                            {"item_id": "p", "title": "positive"},
                            {"item_id": "n1", "title": "negative one"},
                            {"item_id": "n2", "title": "negative two"},
                        ],
                    }
                    for request_id in ("r1", "r2", "r3", "r4")
                ],
            )
            _write_jsonl(
                qrels,
                [
                    {"request_id": request_id, "clicked": ["p"], "purchased": []}
                    for request_id in ("r1", "r2", "r3", "r4")
                ],
            )
            left, left_stats = build_pairwise_examples(
                records,
                qrels,
                input_mode="full",
                history_budget=2,
                negatives_per_positive=2,
                seed=9,
                history_dropout_probability=0.5,
            )
            right, right_stats = build_pairwise_examples(
                records,
                qrels,
                input_mode="full",
                history_budget=2,
                negatives_per_positive=2,
                seed=9,
                history_dropout_probability=0.5,
            )
            self.assertEqual(left, right)
            self.assertEqual(left_stats, right_stats)
            contexts_by_request = {}
            for example in left:
                contexts_by_request.setdefault(example.request_id, set()).add(example.context)
            self.assertTrue(all(len(contexts) == 1 for contexts in contexts_by_request.values()))
            dropout = left_stats["history_dropout"]
            self.assertEqual(dropout["history_present_labeled_requests"], 4)
            self.assertEqual(
                dropout["history_dropped_requests"] + dropout["history_retained_requests"],
                4,
            )
            self.assertGreater(dropout["history_dropped_requests"], 0)
            self.assertGreater(dropout["history_retained_requests"], 0)

    def test_qc_rejects_nonzero_history_dropout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            qrels = root / "qrels.jsonl"
            _write_jsonl(
                records,
                [
                    {
                        "request_id": "r",
                        "query": "q",
                        "history": [{"item_id": "h"}],
                        "candidates": [{"item_id": "p"}, {"item_id": "n"}],
                    }
                ],
            )
            _write_jsonl(
                qrels,
                [{"request_id": "r", "clicked": ["p"], "purchased": []}],
            )
            with self.assertRaisesRegex(ValueError, "input_mode=full"):
                build_pairwise_examples(
                    records,
                    qrels,
                    input_mode="qc",
                    history_budget=2,
                    negatives_per_positive=1,
                    seed=1,
                    history_dropout_probability=0.5,
                )


if __name__ == "__main__":
    unittest.main()
