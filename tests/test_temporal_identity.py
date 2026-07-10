import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.temporal_identity import (
    RecentSnapshotPools,
    TemporalSnapshot,
    TemporalTarget,
    adjudicate_temporal_gate,
    history_age,
    log2_age_gap,
    select_temporal_donor,
    validate_temporal_config,
)


def _history(ts=90, item_id="h"):
    return ({"item_id": item_id, "cat": ["cat"], "event": "click", "ts": ts},)


def _snapshot(
    request_id,
    user_id,
    *,
    query="query",
    request_ts=95,
    latest_ts=90,
    category="cat",
    length_bin=1,
):
    return TemporalSnapshot(
        request_id=request_id,
        user_id=user_id,
        query_key=query,
        request_ts=request_ts,
        latest_event_ts=latest_ts,
        history=_history(latest_ts, request_id),
        major_category=category,
        length_bin=length_bin,
        source_split="train",
    )


def _target(*, user_id="u1", request_ts=100, latest_ts=90):
    return TemporalTarget(
        request_id="target",
        user_id=user_id,
        query_key="query",
        request_ts=request_ts,
        history=_history(latest_ts),
        major_category="cat",
        length_bin=1,
        candidates=({"item_id": "c", "cat": ["cat"]},),
    )


class TemporalIdentityTest(unittest.TestCase):
    def test_history_age_requires_strictly_prior_event(self):
        self.assertEqual(history_age(100, 90), 10)
        with self.assertRaises(ValueError):
            history_age(100, 100)

    def test_log_age_gap_is_hand_computed(self):
        target = _target(request_ts=100, latest_ts=90)  # age 10
        donor = _snapshot("d", "u2", request_ts=99, latest_ts=78)  # age 22 at target
        # (22 + 1) / (10 + 1) = 23 / 11.
        self.assertAlmostEqual(log2_age_gap(target, donor), 1.0641303374197155)

    def test_pool_retains_most_recent_snapshots(self):
        pools = RecentSnapshotPools(max_size=2)
        pools.offer("global", "all", _snapshot("old", "u2", latest_ts=10))
        pools.offer("global", "all", _snapshot("new", "u3", latest_ts=90))
        pools.offer("global", "all", _snapshot("middle", "u4", latest_ts=50))
        self.assertEqual(
            [row.request_id for row in pools.get("global", "all")],
            ["new", "middle"],
        )

    def test_selection_skips_stale_query_for_balanced_category(self):
        target = _target()
        pools = RecentSnapshotPools(max_size=8)
        pools.offer(
            "query_length",
            ("query", 1),
            _snapshot("stale-query", "u2", latest_ts=1),
        )
        pools.offer(
            "category_length",
            ("cat", 1),
            _snapshot("fresh-category", "u3", query="other", latest_ts=89),
        )
        assignment = select_temporal_donor(
            target, pools, seed=20260708, max_log2_age_gap=1.0, top_k=4
        )
        self.assertEqual(assignment.tier, "category_length")
        self.assertEqual(assignment.donor.request_id, "fresh-category")
        self.assertTrue(assignment.balanced)

    def test_selection_rejects_same_user_and_same_time_request(self):
        target = _target()
        pools = RecentSnapshotPools(max_size=8)
        pools.offer(
            "query_length",
            ("query", 1),
            _snapshot("same-user", "u1", latest_ts=90),
        )
        pools.offer(
            "query_length",
            ("query", 1),
            _snapshot("same-time", "u2", request_ts=100, latest_ts=90),
        )
        pools.offer(
            "query_length",
            ("query", 1),
            _snapshot("eligible", "u3", request_ts=99, latest_ts=90),
        )
        assignment = select_temporal_donor(
            target, pools, seed=20260708, max_log2_age_gap=1.0, top_k=4
        )
        self.assertEqual(assignment.donor.request_id, "eligible")

    def test_selection_is_seed_deterministic(self):
        target = _target()
        pools = RecentSnapshotPools(max_size=8)
        for index in range(5):
            pools.offer(
                "query_length",
                ("query", 1),
                _snapshot(f"d{index}", f"u{index + 2}", latest_ts=90 - index),
            )
        first = select_temporal_donor(
            target, pools, seed=20260708, max_log2_age_gap=1.0, top_k=4
        )
        second = select_temporal_donor(
            target, pools, seed=20260708, max_log2_age_gap=1.0, top_k=4
        )
        self.assertEqual(first, second)

    def test_config_validation_rejects_policy_drift(self):
        config = {
            "target_split": "dev",
            "static_mixture": {"zscore_scope": "within_request"},
            "matching": {
                "donor_sources": ["train", "earlier_dev"],
                "require_different_user": True,
                "require_donor_request_strictly_before_target": True,
                "insert_dev_donors_after_same_timestamp_group": True,
                "keep_empty_target_history_empty": True,
                "donor_priority": ["global"],
                "max_log2_age_gap": 2.0,
                "freshness_top_k": 8,
            },
        }
        with self.assertRaises(ValueError):
            validate_temporal_config(config)

    def test_gate_does_not_ignore_failed_same_query_significance(self):
        gate = {
            "min_freshness_balanced_requests": 6000,
            "min_same_query_freshness_balanced_requests": 1000,
            "same_query_min_significant_seeds": 2,
        }
        counts = {
            "freshness_balanced_all_seeds": 7000,
            "same_query_freshness_balanced_all_seeds": 1100,
        }
        freshness = {
            "1": {"delta": 0.03, "ci95": [0.02, 0.04]},
            "2": {"delta": 0.03, "ci95": [0.02, 0.04]},
            "3": {"delta": 0.03, "ci95": [0.02, 0.04]},
        }
        same_query = {
            "1": {"delta": 0.01, "ci95": [-0.001, 0.02]},
            "2": {"delta": 0.01, "ci95": [-0.001, 0.02]},
            "3": {"delta": 0.01, "ci95": [0.001, 0.02]},
        }
        result = adjudicate_temporal_gate(
            gate, counts, freshness, same_query, integrity_passed=True
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["same_query_significant_seed_count"], 1)



if __name__ == "__main__":
    unittest.main()
