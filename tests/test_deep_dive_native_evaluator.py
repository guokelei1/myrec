from __future__ import annotations

import numpy as np

from myrec.mechanism.deep_dive_native_evaluator import (
    benjamini_hochberg,
    cluster_mean_inference,
)


def test_bh_hand_computed_three_hypotheses():
    observed = benjamini_hochberg([0.01, 0.04, 0.03])
    np.testing.assert_allclose(observed, [0.03, 0.04, 0.04])


def test_cluster_bootstrap_is_deterministic_and_reports_direction():
    values = np.asarray([1.0, 1.0, 2.0, 2.0])
    clusters = np.asarray(["a", "a", "b", "b"])
    first = cluster_mean_inference(values, clusters, samples=100, seed=7)
    second = cluster_mean_inference(values, clusters, samples=100, seed=7)
    assert first == second
    assert first["mean"] == 1.5
    assert first["two_sided_p"] <= 2.0 / 101.0
