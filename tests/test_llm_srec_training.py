from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.representative_sequence_adapter import (
    TrainVocabulary,
    build_sequence_request,
)
from myrec.baselines.sequence_ranker_training import LabeledSequenceRequest
from myrec.baselines.llm_srec_training import sample_llm_srec_candidates


class LLMSRecTrainingTest(unittest.TestCase):
    def test_sampler_keeps_positive_at_zero_and_uses_in_slate_negatives(self):
        record = {
            "request_id": "r",
            "query": "shoe",
            "ts": 2,
            "history": [],
            "candidates": [
                {"item_id": "a", "title": "a"},
                {"item_id": "b", "title": "b"},
                {"item_id": "c", "title": "c"},
            ],
        }
        vocabulary = TrainVocabulary.fit([record])
        request = build_sequence_request(record, vocabulary, history_budget=1)
        row = LabeledSequenceRequest(request, (False, True, False))
        candidates, positive = sample_llm_srec_candidates(
            row, negatives=1, rng=random.Random(7)
        )
        self.assertEqual(positive, 0)
        self.assertEqual(candidates[0].raw_item_id, "b")
        self.assertIn(candidates[1].raw_item_id, {"a", "c"})


if __name__ == "__main__":
    unittest.main()

