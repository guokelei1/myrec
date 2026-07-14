from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.history_response import (
    ResponseCandidate,
    aggregate_history_response,
    request_history_response,
)


class HistoryResponseMetricTest(unittest.TestCase):
    def test_common_mode_shift_has_no_candidate_relative_activity(self):
        row = request_history_response(
            "r1",
            [
                ResponseCandidate("a", true_score=10.0, null_score=0.0, gain=1.0),
                ResponseCandidate("b", true_score=11.0, null_score=1.0, gain=0.0),
            ],
            activity_epsilon=1e-9,
        )
        self.assertEqual(row["common_mode_ratio"], 1.0)
        self.assertEqual(row["response_differential_rms"], 0.0)
        self.assertFalse(row["response_active"])
        self.assertIsNone(row["pairwise_directional_accuracy"])

    def test_direction_and_incremental_ndcg_are_hand_computed(self):
        row = request_history_response(
            "r2",
            [
                ResponseCandidate("a", true_score=3.0, null_score=0.0, gain=1.0),
                ResponseCandidate("b", true_score=1.0, null_score=1.0, gain=0.0),
                ResponseCandidate("c", true_score=0.0, null_score=2.0, gain=0.0),
            ],
            activity_epsilon=0.1,
        )
        self.assertAlmostEqual(row["true_ndcg@10"], 1.0)
        self.assertAlmostEqual(row["null_ndcg@10"], 0.5)
        self.assertAlmostEqual(row["true_minus_null_ndcg@10"], 0.5)
        self.assertEqual(row["pairwise_directional_accuracy"], 1.0)
        self.assertEqual(row["signed_delta_alignment"], 1.0)
        self.assertEqual(row["active_pairs"], 3)
        self.assertAlmostEqual(row["common_mode_ratio"], 1.0 / 39.0)
        self.assertAlmostEqual(row["response_differential_rms"], math.sqrt(38.0 / 9.0))

    def test_wrong_direction_is_negative_alignment(self):
        row = request_history_response(
            "r3",
            [
                ResponseCandidate("a", true_score=-1.0, null_score=0.0, gain=1.0),
                ResponseCandidate("b", true_score=1.0, null_score=0.0, gain=0.0),
            ],
            activity_epsilon=0.1,
        )
        self.assertEqual(row["pairwise_directional_accuracy"], 0.0)
        self.assertEqual(row["signed_delta_alignment"], -1.0)

    def test_true_over_wrong_is_reported(self):
        row = request_history_response(
            "r4",
            [
                ResponseCandidate("a", 2.0, 0.0, 1.0, wrong_score=-1.0),
                ResponseCandidate("b", 0.0, 1.0, 0.0, wrong_score=2.0),
            ],
            activity_epsilon=0.1,
        )
        self.assertGreater(row["true_minus_wrong_ndcg@10"], 0.0)
        self.assertEqual(row["true_over_wrong_signed_alignment"], 2.0)

    def test_aggregate_averages_requests_not_pairs(self):
        positive = request_history_response(
            "p",
            [
                ResponseCandidate("a", 2.0, 0.0, 1.0),
                ResponseCandidate("b", 0.0, 1.0, 0.0),
            ],
            activity_epsilon=0.1,
        )
        harmful = request_history_response(
            "h",
            [
                ResponseCandidate("a", -1.0, 1.0, 1.0),
                ResponseCandidate("b", 2.0, 0.0, 0.0),
                ResponseCandidate("c", 1.0, -1.0, 0.0),
            ],
            activity_epsilon=0.1,
        )
        result = aggregate_history_response([positive, harmful], utility_epsilon=1e-9)
        self.assertEqual(result["num_requests"], 2)
        self.assertEqual(result["active_response_rate"], 1.0)
        self.assertEqual(result["active_response_precision"], 0.5)
        self.assertEqual(result["active_response_harm_rate"], 0.5)
        self.assertEqual(result["mean_pairwise_directional_accuracy"], 0.5)

    def test_unlabeled_direction_does_not_count_against_response_precision(self):
        positive = request_history_response(
            "p",
            [
                ResponseCandidate("a", 2.0, 0.0, 1.0),
                ResponseCandidate("b", 0.0, 1.0, 0.0),
            ],
            activity_epsilon=0.1,
        )
        no_preference = request_history_response(
            "n",
            [
                ResponseCandidate("a", 2.0, 0.0, 0.0),
                ResponseCandidate("b", 0.0, 1.0, 0.0),
            ],
            activity_epsilon=0.1,
        )
        result = aggregate_history_response([positive, no_preference], utility_epsilon=1e-9)
        self.assertEqual(result["num_active_requests"], 2)
        self.assertEqual(result["num_active_direction_requests"], 1)
        self.assertEqual(result["active_response_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
