from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.full_token_cross_encoder import (
    _input_fields_used,
    serialize_query_history,
    write_full_token_cross_encoder_scores,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


class _FakePredictor:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, pairs, **kwargs):
        self.batch_sizes.append(len(pairs))
        return [float(len(left) - len(right)) for left, right in pairs]


class FullTokenCrossEncoderTest(unittest.TestCase):
    def test_writes_counterfactual_contract_without_qrels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            _write_json(standardized / "manifest.json", {"dataset_id": "tiny", "dataset_version": "v1"})
            _write_json(
                standardized / "candidate_manifest.json",
                {"entries": [{"split": "dev", "request_id": "r", "candidate_item_ids": ["a", "b"]}]},
            )
            _write_json(standardized / "request_manifest.json", {"entries": [{"request_id": "r"}]})
            _write_jsonl(
                standardized / "records_dev.jsonl",
                [
                    {
                        "request_id": "r",
                        "query": "red shoes",
                        "candidates": [
                            {"item_id": "a", "title": "A"},
                            {"item_id": "b", "title": "BBBB"},
                        ],
                    }
                ],
            )
            assignments = root / "true.jsonl"
            _write_jsonl(
                assignments,
                [
                    {
                        "request_id": "r",
                        "assignment": "true",
                        "history": [{"item_id": "h", "title": "old", "event": "click", "ts": 1}],
                    }
                ],
            )
            metadata = write_full_token_cross_encoder_scores(
                standardized,
                "dev",
                "run",
                "true",
                assignments,
                runs_dir=root / "runs",
                predictor=_FakePredictor(),
                checkpoint_id="fake@1",
            )
            scores = list(iter_jsonl(root / "runs" / "run" / "scores.jsonl"))
            self.assertEqual(len(scores), 2)
            self.assertFalse(metadata["qrels_read"])
            self.assertEqual(metadata["history_condition"], "true")
            self.assertEqual(metadata["checkpoint_id"], "fake@1")
            self.assertEqual(metadata["history_assignment_sha256"], sha256_file(assignments))

    def test_serialization_respects_history_budget(self):
        text = serialize_query_history(
            "q",
            [
                {"title": "old", "event": "click"},
                {"title": "new", "event": "purchase"},
            ],
            history_budget=1,
            serialization_version="query_history_event_text_v1",
        )
        self.assertNotIn("old", text)
        self.assertIn("new", text)

    def test_serialization_includes_optional_history_query(self):
        text = serialize_query_history(
            "target",
            [{"query": "prior terms", "title": "item", "event": "click"}],
            history_budget=1,
            serialization_version="query_history_event_text_v1",
        )
        self.assertIn("prior query: prior terms", text)

    def test_query_only_serialization_is_plain_and_rejects_history(self):
        self.assertEqual(
            _input_fields_used("query_only_text_v1"),
            [
                "query",
                "candidates.title",
                "candidates.brand",
                "candidates.cat",
            ],
        )
        self.assertIn(
            "assigned_history.query",
            _input_fields_used("query_history_event_text_v1"),
        )
        self.assertEqual(
            serialize_query_history(
                "plain query",
                [],
                history_budget=0,
                serialization_version="query_only_text_v1",
            ),
            "plain query",
        )
        with self.assertRaisesRegex(ValueError, "non-empty history"):
            serialize_query_history(
                "q",
                [{"title": "history"}],
                history_budget=1,
                serialization_version="query_only_text_v1",
            )

    def test_request_aligned_batches_do_not_mix_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            _write_json(
                standardized / "manifest.json",
                {"dataset_id": "tiny", "dataset_version": "v1"},
            )
            _write_json(standardized / "candidate_manifest.json", {"entries": []})
            _write_json(standardized / "request_manifest.json", {"entries": []})
            records = []
            assignments = []
            for request_id in ("r1", "r2"):
                records.append(
                    {
                        "request_id": request_id,
                        "query": "q",
                        "candidates": [
                            {"item_id": f"{request_id}-a", "title": "A"},
                            {"item_id": f"{request_id}-b", "title": "B"},
                        ],
                    }
                )
                assignments.append(
                    {"request_id": request_id, "assignment": "null", "history": []}
                )
            _write_jsonl(standardized / "records_dev.jsonl", records)
            assignments_path = root / "null.jsonl"
            _write_jsonl(assignments_path, assignments)
            predictor = _FakePredictor()
            metadata = write_full_token_cross_encoder_scores(
                standardized,
                "dev",
                "run",
                "null",
                assignments_path,
                runs_dir=root / "runs",
                predictor=predictor,
                checkpoint_id="fake@1",
                request_aligned_batches=True,
            )
            self.assertEqual(predictor.batch_sizes, [2, 2])
            self.assertTrue(metadata["request_aligned_batches"])


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    unittest.main()
