import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_deep_dive_rmsnorm_flow.py"
SPEC = importlib.util.spec_from_file_location("rmsnorm_flow", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rmsnorm_replay_is_hand_computed() -> None:
    values = np.asarray([[[3.0, 4.0]]])
    weights = np.asarray([[2.0, 0.5]])
    output = MODULE._rmsnorm(values, weights, epsilon=1.0e-6)
    scale = 1.0 / np.sqrt(12.5 + 1.0e-6)
    np.testing.assert_allclose(output, values * scale * weights[None, :, :])


def test_request_metrics_preserve_common_residual_energy_identity() -> None:
    full = np.asarray(
        [
            [[2.0, 0.0]],
            [[0.0, 2.0]],
        ]
    )
    null = np.zeros_like(full)
    weights = np.ones((1, 2))
    metrics = MODULE._request_norm_metrics(full, null, weights, epsilon=1.0e-6)
    assert metrics["maximum_pre_energy_identity_error"] < 1.0e-12
    assert metrics["maximum_post_energy_identity_error"] < 1.0e-12
    assert np.isclose(metrics["pre_total"][0], 2.0)
    assert np.isclose(metrics["pre_common"][0], 1.0)
    assert np.isclose(metrics["pre_residual"][0], 1.0)


def test_energy_cosine_and_gain_handle_identity_and_zero() -> None:
    assert np.isclose(MODULE._energy_cosine(2.0, 2.0, 2.0), 1.0)
    assert MODULE._energy_cosine(0.0, 0.0, 0.0) is None
    assert np.isclose(MODULE._sqrt_ratio(8.0, 2.0), 2.0)
    assert MODULE._sqrt_ratio(1.0, 0.0) is None


def test_weight_participation_is_one_for_uniform_weights() -> None:
    metrics = MODULE._weight_metrics(np.ones(8))
    assert metrics["rmsnorm_weight_rms"] == 1.0
    assert np.isclose(metrics["rmsnorm_weight_channel_participation_ratio"], 1.0)


def test_accumulator_uses_equal_request_sums() -> None:
    accumulator = MODULE._empty_accumulator()
    metrics = {
        name: np.ones(len(MODULE.STATES))
        for name in (*MODULE.ENERGY_NAMES, *MODULE.DOT_NAMES, *MODULE.RMS_NAMES)
    }
    MODULE._update_accumulator(accumulator, metrics, candidate_count=7)
    MODULE._update_accumulator(accumulator, metrics, candidate_count=3)
    weights = [MODULE._weight_metrics(np.ones(8)) for _ in MODULE.STATES]
    rows = MODULE._finalize_accumulator("q2", "all", accumulator, weights)
    assert rows[1]["requests"] == 2
    assert rows[1]["candidate_rows"] == 10
    assert rows[1]["total_delta_rms_gain"] == 1.0
    assert rows[1]["residual_to_common_gain_ratio"] == 1.0
