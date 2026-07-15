from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.recoverability_witness import (
    BASE_FEATURE_NAMES,
    EmbeddingStore,
    candidate_features,
)
from myrec.baselines.core import document_text


class RecoverabilityWitnessTest(unittest.TestCase):
    def test_history_changes_only_the_history_feature_suffix(self):
        record = {
            "query": "red shoe",
            "candidates": [],
        }
        candidate = {
            "item_id": "c",
            "title": "red running shoe",
            "brand": "brand-a",
            "cat": ["shoe", "sport", "running"],
        }
        event = {
            "item_id": "h",
            "title": "running sneaker",
            "brand": "brand-a",
            "cat": ["shoe", "sport", "running"],
            "event": "click",
        }
        q_key = "q:red shoe"
        c_key = "d:" + document_text(candidate)
        h_key = "d:" + document_text(event)
        store = EmbeddingStore(
            {q_key: 0, c_key: 1, h_key: 2},
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.8, 0.2],
                    [0.6, 0.4],
                ],
                dtype=np.float32,
            ),
        )
        null = candidate_features(record, candidate, [], store, history_budget=6)
        true = candidate_features(
            record, candidate, [event], store, history_budget=6
        )
        base_count = len(BASE_FEATURE_NAMES)
        np.testing.assert_array_equal(null[:base_count], true[:base_count])
        np.testing.assert_array_equal(null[base_count:], 0.0)
        self.assertTrue(np.any(true[base_count:] != 0.0))


if __name__ == "__main__":
    unittest.main()
