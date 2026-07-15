from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.representative_sequence_adapter import (
    OOV_ID,
    QUERY_EVENT_ID,
    QUERY_TOKEN_ID,
    TrainVocabulary,
    audit_standardized_sequence_inputs,
    build_sequence_request,
    serialize_item_content,
)
from myrec.baselines.hstu_pps_adapter import collate_sequence_requests


def _record(request_id: str = "r") -> dict:
    return {
        "request_id": request_id,
        "query": "red shoes",
        "ts": 30,
        "history": [
            {"item_id": "h1", "title": "old bag", "event": "click", "ts": 10},
            {
                "item_id": "h2",
                "title": "red dress",
                "event": "purchase",
                "ts": 20,
            },
        ],
        "candidates": [
            {
                "item_id": "c1",
                "title": "red running shoes",
                "clicked": 1,
                "source_position": 7,
            },
            {"item_id": "c2", "title": "blue hat", "clicked": 0},
        ],
    }


class RepresentativeSequenceAdapterTest(unittest.TestCase):
    def test_vocabulary_is_deterministic_and_train_only(self):
        a = _record("a")
        b = _record("b")
        b["history"][0]["item_id"] = "z"
        forward = TrainVocabulary.fit([a, b])
        reverse = TrainVocabulary.fit([b, a])
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.item_id("dev-only"), OOV_ID)
        self.assertGreater(forward.item_id("c1"), QUERY_TOKEN_ID)

    def test_recent_history_and_query_token_are_causal_sequence(self):
        record = _record()
        vocab = TrainVocabulary.fit([record])
        result = build_sequence_request(record, vocab, history_budget=1)
        self.assertEqual(result.retained_history_count, 1)
        self.assertEqual(result.original_history_count, 2)
        self.assertEqual(result.past_item_ids[-1], QUERY_TOKEN_ID)
        self.assertEqual(result.past_event_ids[-1], QUERY_EVENT_ID)
        self.assertEqual(result.past_timestamps, (20, 30))
        self.assertIn("red dress", result.past_content_texts[0])
        self.assertEqual([c.raw_item_id for c in result.candidates], ["c1", "c2"])

    def test_null_history_is_query_only_not_fake_event(self):
        train = _record("train")
        vocab = TrainVocabulary.fit([train])
        null_record = _record("null")
        null_record["history"] = []
        result = build_sequence_request(null_record, vocab, history_budget=8)
        self.assertEqual(result.retained_history_count, 0)
        self.assertEqual(result.past_item_ids, (QUERY_TOKEN_ID,))
        self.assertEqual(result.past_event_ids, (QUERY_EVENT_ID,))
        self.assertEqual(result.past_content_texts, ("query: red shoes",))

    def test_future_or_unsorted_history_is_rejected(self):
        record = _record()
        vocab = TrainVocabulary.fit([record])
        future = _record("future")
        future["history"][-1]["ts"] = 31
        with self.assertRaisesRegex(ValueError, "future history"):
            build_sequence_request(future, vocab, history_budget=8)
        unsorted = _record("unsorted")
        unsorted["history"][0]["ts"] = 25
        with self.assertRaisesRegex(ValueError, "not chronological"):
            build_sequence_request(unsorted, vocab, history_budget=8)

    def test_candidate_labels_and_positions_never_enter_content(self):
        content = serialize_item_content(
            {
                "item_id": "x",
                "title": "shoe",
                "clicked": 1,
                "purchased": 1,
                "relevance": 3,
                "source_position": 0,
            }
        )
        self.assertEqual(content, "title: shoe")

    def test_vocabulary_can_be_written_as_reproducible_json(self):
        vocab = TrainVocabulary.fit([_record()])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            vocab.write(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["special_item_ids"]["query"], QUERY_TOKEN_ID)
        self.assertEqual(payload["item_to_id"], vocab.item_to_id)

    def test_duplicate_candidates_are_rejected(self):
        record = _record()
        vocab = TrainVocabulary.fit([record])
        record["candidates"][1]["item_id"] = "c1"
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            build_sequence_request(record, vocab, history_budget=8)

    def test_audit_never_requires_or_reads_qrels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            standardized.mkdir()
            (standardized / "manifest.json").write_text(
                json.dumps({"dataset_id": "tiny", "dataset_version": "v1"}),
                encoding="utf-8",
            )
            with (standardized / "records_train.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(_record("train")) + "\n")
            with (standardized / "records_dev.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(_record("dev")) + "\n")
            output = root / "audit.json"
            result = audit_standardized_sequence_inputs(
                standardized, output, history_budget=1
            )
        self.assertFalse(result["qrels_read"])
        self.assertEqual(result["decision"], "pass")
        self.assertEqual(result["splits"]["dev"]["request_count"], 1)

    def test_hstu_collation_preserves_candidate_order_and_query_only_null(self):
        train = _record("train")
        vocab = TrainVocabulary.fit([train])
        null_record = _record("null")
        null_record["history"] = []
        request = build_sequence_request(null_record, vocab, history_budget=8)
        batch = collate_sequence_requests(
            [request],
            lambda text: torch.tensor([len(text), 1.0]),
            content_dim=2,
            max_sequence_length=9,
        )
        self.assertEqual(batch.past_lengths.tolist(), [1])
        self.assertEqual(batch.past_item_ids[0, 0].item(), QUERY_TOKEN_ID)
        self.assertEqual(batch.raw_candidate_item_ids, (("c1", "c2"),))
        self.assertEqual(batch.candidate_mask.tolist(), [[True, True]])

    def test_hstu_collation_rejects_wrong_content_dimension(self):
        record = _record()
        vocab = TrainVocabulary.fit([record])
        request = build_sequence_request(record, vocab, history_budget=1)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            collate_sequence_requests(
                [request],
                lambda text: [0.0],
                content_dim=2,
                max_sequence_length=2,
            )


if __name__ == "__main__":
    unittest.main()
