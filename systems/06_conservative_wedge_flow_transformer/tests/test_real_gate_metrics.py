from __future__ import annotations

import numpy as np
import pytest

from train.real_gate_metrics import (
    clicked_minus_unclicked,
    compare_primary,
    fold_differences,
    order_change_summary,
    paired_bootstrap,
    request_fold,
)


def test_constant_paired_bootstrap_is_hand_computed_exact() -> None:
    result = paired_bootstrap(
        np.asarray([0.25, 0.25, 0.25]), samples=100, seed=7
    )
    assert result["mean"] == pytest.approx(0.25)
    assert result["percentile_95_ci"] == pytest.approx([0.25, 0.25])


def test_fold_assignment_is_deterministic_and_partitions_requests() -> None:
    request_ids = [f"r{index}" for index in range(100)]
    assignments = [request_fold(value, seed=17, folds=3) for value in request_ids]
    assert assignments == [
        request_fold(value, seed=17, folds=3) for value in request_ids
    ]
    rows = fold_differences(
        request_ids, np.ones(100), seed=17, folds=3
    )
    assert sum(row["requests"] for row in rows) == 100
    assert [row["mean_difference"] for row in rows] == [1.0, 1.0, 1.0]


def test_primary_comparisons_remain_paired() -> None:
    request_ids = [f"r{index}" for index in range(30)]
    primary = np.arange(30, dtype=np.float64) / 100.0
    result = compare_primary(
        request_ids=request_ids,
        primary=primary,
        references={"base": primary - 0.1, "control": primary - 0.05},
        bootstrap_samples=200,
        bootstrap_seed=20260708,
        folds=3,
    )
    assert result["base"]["mean"] == pytest.approx(0.1)
    assert result["control"]["mean"] == pytest.approx(0.05)
    assert all(
        row["mean_difference"] == pytest.approx(0.1)
        for row in result["base"]["hash_folds"]
    )


def test_order_change_summary_distinguishes_order_from_top10_membership() -> None:
    base = [[str(value) for value in range(12)], ["a", "b", "c"]]
    personalized = [
        ["1", "0", *[str(value) for value in range(2, 12)]],
        ["a", "c", "b"],
    ]
    result = order_change_summary(
        base_rankings=base, personalized_rankings=personalized
    )
    assert result["requests_with_any_order_change"] == 2
    assert result["requests_with_top10_membership_change"] == 0


def test_clicked_minus_unclicked_is_request_equal_and_hand_computed() -> None:
    values = clicked_minus_unclicked(
        deltas=[np.asarray([3.0, 1.0, 1.0]), np.asarray([0.0, 4.0])],
        labels=[np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0])],
    )
    assert values.tolist() == [2.0, 4.0]


def test_metric_helpers_reject_unpaired_shapes() -> None:
    with pytest.raises(ValueError):
        clicked_minus_unclicked(
            deltas=[np.asarray([1.0])], labels=[np.asarray([1.0, 0.0])]
        )
    with pytest.raises(ValueError):
        paired_bootstrap(np.asarray([]), samples=10, seed=1)
