from __future__ import annotations

import sys
import unittest
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.response_direction_intervention import (
    DirectionInterventionCandidate,
    aggregate_direction_interventions,
    request_direction_intervention,
)


class ResponseDirectionInterventionTest(unittest.TestCase):
    def test_gain_alignment_repairs_wrong_candidate_attribution(self):
        rows = [
            DirectionInterventionCandidate("positive", -2.0, -1.0, 1.0),
            DirectionInterventionCandidate("negative", 1.0, 0.0, 0.0),
        ]
        result = request_direction_intervention(
            "r1", rows, random_permutations=20, seed=7
        )
        self.assertAlmostEqual(result["actual_ndcg@10"], 1.0 / math.log2(3.0))
        self.assertAlmostEqual(result["null_ndcg@10"], 1.0 / math.log2(3.0))
        self.assertEqual(result["aligned_ndcg@10"], 1.0)
        self.assertEqual(
            result["anti_aligned_ndcg@10"], result["actual_ndcg@10"]
        )
        self.assertGreater(result["aligned_minus_actual_ndcg@10"], 0.0)

    def test_random_control_is_deterministic_and_aggregate_is_defined(self):
        candidates = [
            DirectionInterventionCandidate("a", 0.1, 0.0, 1.0),
            DirectionInterventionCandidate("b", 0.2, 0.0, 0.0),
            DirectionInterventionCandidate("c", -0.1, 0.0, 0.0),
        ]
        left = request_direction_intervention(
            "r2", candidates, random_permutations=20, seed=11
        )
        right = request_direction_intervention(
            "r2", candidates, random_permutations=20, seed=11
        )
        self.assertEqual(left, right)
        aggregate = aggregate_direction_interventions([left, right])
        self.assertEqual(aggregate["num_requests"], 2)
        self.assertIsNotNone(aggregate["direction_conversion_efficiency"])


if __name__ == "__main__":
    unittest.main()
