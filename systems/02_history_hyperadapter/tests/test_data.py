"""Packed-data structural and batching tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.data import C02Split, collate_requests, frozen_train_indices, iter_request_batches


def tiny_split() -> C02Split:
    return C02Split(
        split="train",
        request_ids=["r0", "r1", "r2", "r3"],
        candidate_offsets=np.array([0, 2, 4, 6, 8]),
        candidate_embedding_indices=np.arange(8),
        candidate_item_ids=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        candidate_labels=np.array([1, 0, 1, 0, 0, 1, 1, 0]),
        history_offsets=np.array([0, 0, 2, 3, 5]),
        history_embedding_indices=np.array([2, 8, 9, 6, 10]),
        history_item_ids=np.array([3, 9, 10, 7, 11]),
        history_event_weights=np.array([1.0, 0.7, 1.0, 1.5, 0.5], dtype=np.float32),
        history_category_ids=np.array([1, 2, 3, 1, 4]),
        candidate_category_ids=np.arange(8) % 4,
        wrong_request_indices=np.array([0, 2, 3, 1]),
        base_scores=None,
    )


class DataTest(unittest.TestCase):
    def test_complete_history_subsets_ignore_truncation(self) -> None:
        data = tiny_split()
        self.assertEqual(
            [data.structural_subset(index) for index in range(len(data))],
            ["no_history", "repeat", "nonrepeat", "repeat"],
        )
        # r1's exact match is the older event and is excluded at limit=1, but
        # the frozen structural subset remains repeat.
        batch = collate_requests(data, np.array([1]), history_limit=1)
        self.assertFalse(np.asarray(batch["repeat_mask"]).any())
        self.assertEqual(data.structural_subset(1), "repeat")

    def test_collation_masks_and_reverse_corruption(self) -> None:
        data = tiny_split()
        batch = collate_requests(data, np.array([0, 1]), history_limit=2)
        np.testing.assert_array_equal(batch["history_mask"], [[False, False], [True, True]])
        np.testing.assert_array_equal(batch["history_item_ids"][1], [3, 9])
        np.testing.assert_array_equal(batch["shuffled_history_item_ids"][1], [9, 3])
        self.assertTrue(batch["repeat_mask"][1, 0, 0])
        self.assertFalse(batch["candidate_mask"][0, 2:].any())

    def test_frozen_sampling_is_deterministic_and_disjoint(self) -> None:
        data = tiny_split()
        first = frozen_train_indices(data, 11, 1, 1, 0.5)
        second = frozen_train_indices(data, 11, 1, 1, 0.5)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(np.intersect1d(first[0], first[1]).size, 0)

    def test_batch_limits(self) -> None:
        data = tiny_split()
        batches = list(
            iter_request_batches(
                data,
                np.arange(len(data)),
                history_limit=2,
                max_requests=2,
                max_padded_candidates=4,
                max_padded_history=4,
                seed=0,
                shuffle=False,
            )
        )
        self.assertEqual([len(batch) for batch in batches], [2, 2])


if __name__ == "__main__":
    unittest.main()
