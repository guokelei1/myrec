import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.history_identity import (
    BoundedDonorPools,
    Donor,
    Target,
    history_length_bin,
    majority_top_category,
    normalize_query,
    select_donor,
    validate_control_config,
)


def _donor(request_id, user_id, query="query", category="cat", length_bin=4):
    return Donor(
        request_id=request_id,
        user_id=user_id,
        query_key=query,
        request_ts=10,
        history=({"item_id": "h", "cat": [category], "event": "click", "ts": 5},),
        major_category=category,
        length_bin=length_bin,
    )


class HistoryIdentityTest(unittest.TestCase):
    def test_config_validation_rejects_protocol_drift(self):
        config = {
            "source_split": "train",
            "target_split": "dev",
            "static_mixture": {"zscore_scope": "within_request"},
            "wrong_history": {
                "donor_split": "train",
                "require_different_user": True,
                "require_donor_request_before_target_request": True,
                "keep_empty_target_history_empty": True,
                "donor_priority": ["global"],
            },
        }
        with self.assertRaises(ValueError):
            validate_control_config(config)

    def test_normalization_bins_and_majority_category(self):
        self.assertEqual(normalize_query(" Ａ B  女装 "), "ab女装")
        self.assertEqual(history_length_bin(1, [1, 2, 4, 8]), 1)
        self.assertEqual(history_length_bin(3, [1, 2, 4, 8]), 4)
        self.assertEqual(
            majority_top_category(
                [
                    {"cat": ["b"]},
                    {"cat": ["a"]},
                    {"cat": ["b"]},
                    {"cat": ["a"]},
                ]
            ),
            "a",
        )

    def test_selection_prefers_same_query_and_rejects_same_user(self):
        target = Target(
            request_id="target",
            user_id="u1",
            query_key="query",
            request_ts=20,
            history_length=3,
            major_category="cat",
            length_bin=4,
            candidates=({"item_id": "c", "cat": ["cat"]},),
        )
        pools = BoundedDonorPools(max_size=8)
        same_user = _donor("same-user", "u1")
        query_match = _donor("query-match", "u2")
        category_match = _donor("category-match", "u3", query="other")
        pools.offer("query_length", ("query", 4), same_user)
        pools.offer("query_length", ("query", 4), query_match)
        pools.offer("category_length", ("cat", 4), category_match)

        tier, selected = select_donor(target, pools, 20260708)
        self.assertEqual(tier, "query_length")
        self.assertEqual(selected.request_id, "query-match")
        self.assertNotEqual(selected.user_id, target.user_id)

    def test_selection_is_seed_deterministic(self):
        target = Target(
            request_id="target",
            user_id="u1",
            query_key="query",
            request_ts=20,
            history_length=3,
            major_category="cat",
            length_bin=4,
            candidates=({"item_id": "c", "cat": ["cat"]},),
        )
        pools = BoundedDonorPools(max_size=8)
        for index in range(5):
            pools.offer("query_length", ("query", 4), _donor(f"d{index}", f"u{index + 2}"))

        first = select_donor(target, pools, 20260708)[1].request_id
        second = select_donor(target, pools, 20260708)[1].request_id
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
