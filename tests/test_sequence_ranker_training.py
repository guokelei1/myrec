from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.representative_sequence_adapter import TrainVocabulary
from myrec.baselines.sequence_ranker_training import (
    build_labeled_sequence_requests,
    multilabel_listwise_loss,
)


class SequenceRankerTrainingTest(unittest.TestCase):
    def test_multilabel_listwise_loss_matches_probability_mass(self):
        scores = torch.tensor([[0.0, 1.0, 2.0]])
        candidates = torch.tensor([[True, True, True]])
        positives = torch.tensor([[True, False, True]])
        actual = multilabel_listwise_loss(scores, candidates, positives)
        expected = -torch.log((torch.exp(scores[0, 0]) + torch.exp(scores[0, 2])) / torch.exp(scores).sum())
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_qc_and_full_use_same_labels_but_different_history(self):
        record = {
            "request_id": "r",
            "query": "shoe",
            "ts": 20,
            "history": [
                {"item_id": "h", "title": "coat", "event": "click", "ts": 10}
            ],
            "candidates": [
                {"item_id": "p", "title": "shoe"},
                {"item_id": "n", "title": "hat"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            qrels = root / "qrels.jsonl"
            records.write_text(json.dumps(record) + "\n", encoding="utf-8")
            qrels.write_text(
                json.dumps({"request_id": "r", "clicked": ["p"]}) + "\n",
                encoding="utf-8",
            )
            vocabulary = TrainVocabulary.fit([record])
            qc, _ = build_labeled_sequence_requests(
                records, qrels, vocabulary, input_mode="qc", history_budget=8
            )
            full, _ = build_labeled_sequence_requests(
                records, qrels, vocabulary, input_mode="full", history_budget=8
            )
        self.assertEqual(qc[0].positive_mask, full[0].positive_mask)
        self.assertEqual(qc[0].request.retained_history_count, 0)
        self.assertEqual(full[0].request.retained_history_count, 1)


if __name__ == "__main__":
    unittest.main()
