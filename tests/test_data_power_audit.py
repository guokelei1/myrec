from __future__ import annotations

import numpy as np
import pytest

from myrec.mechanism.data_power_audit import (
    _assert_partition,
    _category_prefix_overlap,
    _cluster_bootstrap_means,
    _normal_power,
)


def test_category_prefix_overlap_is_hand_computed() -> None:
    assert _category_prefix_overlap(("a", "b"), ("a", "b")) == 1.0
    assert _category_prefix_overlap(("a", "b"), ("a", "c", "d")) == 1 / 3
    assert _category_prefix_overlap(("a",), ("b",)) == 0.0


def test_cluster_bootstrap_is_deterministic_and_cluster_weighted() -> None:
    values = {"a": [1.0, 1.0], "b": [0.0]}
    first = _cluster_bootstrap_means(values, samples=20, seed=7)
    second = _cluster_bootstrap_means(values, samples=20, seed=7)
    np.testing.assert_array_equal(first, second)
    # Every resample is made of two whole clusters, so only these means exist.
    assert set(first.tolist()) <= {0.0, 2 / 3, 1.0}


def test_power_increases_with_effect_and_partition_rejects_overlap() -> None:
    assert _normal_power(0.02, 0.01) > _normal_power(0.005, 0.01)
    _assert_partition({"x": ["a"], "y": ["b"]}, 2)
    with pytest.raises(AssertionError):
        _assert_partition({"x": ["a"], "y": ["a"]}, 2)
