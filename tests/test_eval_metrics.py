import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.metrics import (
    ScoredCandidate,
    mrr,
    ndcg_at_k,
    recall_at_k,
    sort_candidates,
)


class MetricsTest(unittest.TestCase):
    def test_single_positive_hand_computed(self):
        ranked = ["c", "a", "b"]
        positives = {"b"}
        self.assertAlmostEqual(ndcg_at_k(ranked, positives, 10), 0.5)
        self.assertAlmostEqual(mrr(ranked, positives), 1 / 3)
        self.assertAlmostEqual(recall_at_k(ranked, positives, 10), 1.0)

    def test_multiple_positives_hand_computed(self):
        ranked = ["c", "b", "a"]
        positives = {"a", "c"}
        expected_ndcg = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(ranked, positives, 10), expected_ndcg)
        self.assertAlmostEqual(mrr(ranked, positives), 1.0)
        self.assertAlmostEqual(recall_at_k(ranked, positives, 10), 1.0)

    def test_tie_break_is_input_order_invariant(self):
        first = [
            ScoredCandidate("item_a", 1.0),
            ScoredCandidate("item_b", 1.0),
            ScoredCandidate("item_c", 1.0),
        ]
        second = list(reversed(first))
        self.assertEqual(
            [item.item_id for item in sort_candidates("req_tie", first)],
            [item.item_id for item in sort_candidates("req_tie", second)],
        )


if __name__ == "__main__":
    unittest.main()
