from __future__ import annotations

import numpy as np

from myrec.mechanism.deep_dive_representation_synthesis import (
    region_balanced_accuracy_contrast,
    region_balanced_accuracy_point,
)


def test_region_balanced_accuracy_point_is_mean_state_macro_recall():
    labels = np.asarray(["a", "a", "b", "b"])
    left = np.asarray([[1, 0, 1, 1], [1, 1, 0, 1]], dtype=bool)
    right = np.zeros_like(left)
    # Each state has BA=.75, then states are averaged.
    assert region_balanced_accuracy_point(
        left, right, labels, np.ones(4, dtype=bool)
    ) == 0.75


def test_region_cluster_bootstrap_is_deterministic():
    labels = np.asarray(["a", "a", "b", "b"])
    clusters = np.asarray(["c0", "c1", "c0", "c1"])
    left = np.ones((2, 4), dtype=bool)
    right = np.zeros_like(left)
    mask = np.ones(4, dtype=bool)
    first = region_balanced_accuracy_contrast(
        left, right, labels, clusters, mask, samples=40, seed=3
    )
    second = region_balanced_accuracy_contrast(
        left, right, labels, clusters, mask, samples=40, seed=3
    )
    assert first == second
    assert first["estimate"] == 1.0
