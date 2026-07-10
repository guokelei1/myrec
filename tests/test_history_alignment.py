import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.history_alignment import (
    adjudicate_alignment_gate,
    candidate_history_components,
    validate_alignment_config,
)
from myrec.baselines.core import recent_behavior_scores


def _record(history=True):
    return {
        "request_id": "r",
        "history": (
            [
                {
                    "item_id": "a",
                    "cat": ["l1", "l2", "l3"],
                    "event": "purchase",
                },
                {
                    "item_id": "b",
                    "cat": ["l1", "other", "UNKNOWN"],
                    "event": "click",
                },
            ]
            if history
            else []
        ),
        "candidates": [
            {"item_id": "a", "cat": ["l1", "l2", "l3"]},
            {"item_id": "c", "cat": ["l1", "l2", "different"]},
            {"item_id": "d", "cat": ["l1", "new", "UNKNOWN"]},
        ],
    }


def _comparison(delta=0.01, low=0.001, high=0.02):
    return {"delta": delta, "ci95": [low, high], "num_requests": 8119}


def _comparisons():
    return {
        name: {str(seed): _comparison() for seed in (20260708, 20260709, 20260710)}
        for name in (
            "item_vs_d2p",
            "category_vs_d2p",
            "full_vs_item",
            "full_vs_category",
        )
    }


def _gate():
    return {
        "primary_min_significant_seeds": 2,
        "fallback_category_min_relative_gain": 0.02,
    }


def _means(category=0.33, d2p=0.32):
    return {
        str(seed): {"category": category, "d2p": d2p}
        for seed in (20260708, 20260709, 20260710)
    }


class HistoryAlignmentTest(unittest.TestCase):
    def test_components_are_hand_computed_and_sum_to_public_b0b(self):
        record = _record()
        item, category = candidate_history_components(record)
        old_weight = 1.5 / math.sqrt(2.0)
        self.assertAlmostEqual(item["a"], 3.0 * old_weight)
        self.assertEqual(item["c"], 0.0)
        self.assertAlmostEqual(category["a"], old_weight + 0.2)
        self.assertAlmostEqual(category["c"], 0.5 * old_weight + 0.2)
        self.assertAlmostEqual(category["d"], 0.2 * old_weight + 0.2)
        full = recent_behavior_scores(record)
        for item_id in full:
            self.assertAlmostEqual(item[item_id] + category[item_id], full[item_id])

    def test_empty_history_returns_two_zero_maps(self):
        item, category = candidate_history_components(_record(history=False))
        self.assertEqual(item, {"a": 0.0, "c": 0.0, "d": 0.0})
        self.assertEqual(category, item)

    def test_primary_gate_passes_only_when_all_four_families_pass(self):
        result = adjudicate_alignment_gate(
            _gate(), _comparisons(), _means(), integrity_passed=True
        )
        self.assertEqual(result["outcome"], "PRIMARY_PASS")
        self.assertTrue(result["architecture_ready"])

        comparisons = _comparisons()
        comparisons["full_vs_item"] = {
            str(seed): _comparison(low=-0.01) for seed in (20260708, 20260709, 20260710)
        }
        result = adjudicate_alignment_gate(
            _gate(), comparisons, _means(), integrity_passed=True
        )
        self.assertFalse(result["primary_passed"])

    def test_fallback_is_narrow_and_predeclared(self):
        comparisons = _comparisons()
        comparisons["item_vs_d2p"] = {
            str(seed): _comparison(delta=0.0, low=-0.01, high=0.01)
            for seed in (20260708, 20260709, 20260710)
        }
        comparisons["full_vs_item"] = dict(comparisons["item_vs_d2p"])
        result = adjudicate_alignment_gate(
            _gate(), comparisons, _means(category=0.33, d2p=0.32), True
        )
        self.assertEqual(result["outcome"], "FALLBACK_PASS")
        self.assertEqual(
            result["authorized_primitive"],
            "coarse_candidate_history_semantic_matching",
        )

        means = _means(category=0.325, d2p=0.32)
        result = adjudicate_alignment_gate(_gate(), comparisons, means, True)
        self.assertEqual(result["outcome"], "TERMINAL_FAIL")

    def test_integrity_failure_blocks_both_paths(self):
        result = adjudicate_alignment_gate(
            _gate(), _comparisons(), _means(), integrity_passed=False
        )
        self.assertEqual(result["outcome"], "TERMINAL_FAIL")

    def test_config_validation_rejects_test_or_policy_drift(self):
        config = {
            "analysis_id": "c5r3_candidate_history_alignment",
            "status": "locked_before_outcome_evaluation",
            "target_split": "dev",
            "inputs": {
                "records_dev": "records_dev.jsonl",
                "candidate_manifest": "candidate_manifest.json",
                "history_present_ids": "present.txt",
                "history_absent_ids": "absent.txt",
            },
            "seeds": [20260708, 20260709, 20260710],
            "components": {
                "recency": "1/sqrt(reverse_position)",
                "click_weight": 1.0,
                "purchase_weight": 1.5,
                "item_match_weight": 3.0,
                "category_match_semantics": "deepest_exclusive",
                "category_l3_weight": 1.0,
                "category_l2_weight": 0.5,
                "category_l1_weight": 0.2,
            },
            "static_mixture": {"beta": 0.3, "zscore_scope": "within_request"},
            "gate": {
                "bootstrap_samples": 10000,
                "bootstrap_seed": 20260708,
                "primary_min_significant_seeds": 2,
                "fallback_category_require_all_significant_seeds": True,
                "fallback_category_min_relative_gain": 0.02,
                "fallback_relative_gain_definition": "mean_of_per_seed_subset_mean_ratios",
            },
        }
        validate_alignment_config(config)
        config["inputs"]["records_dev"] = "qrels_test.jsonl"
        with self.assertRaises(ValueError):
            validate_alignment_config(config)


if __name__ == "__main__":
    unittest.main()
