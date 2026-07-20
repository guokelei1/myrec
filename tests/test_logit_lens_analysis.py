import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_deep_dive_logit_lens.py"
SPEC = importlib.util.spec_from_file_location("logit_lens", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_final_rmsnorm_is_hand_computed() -> None:
    values = np.asarray([[[3.0, 4.0]]])
    weights = np.asarray([2.0, 0.5])
    result = MODULE._final_rmsnorm(values, weights)
    scale = 1.0 / np.sqrt(12.5 + MODULE.RMS_EPSILON)
    np.testing.assert_allclose(result, values * scale * weights)


def test_request_lens_recomposes_score_delta() -> None:
    full_candidate = np.asarray(
        [
            [[2.0, 0.0]],
            [[0.0, 2.0]],
        ]
    )
    null_candidate = np.zeros_like(full_candidate)
    full_request = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]])
    null_request = np.zeros_like(full_request)
    weights = np.ones(2)
    direction = np.asarray([1.0, -1.0])
    metrics = MODULE._request_lens_metrics(
        full_candidate,
        null_candidate,
        full_request,
        null_request,
        weights,
        direction,
    )
    assert metrics["maximum_lens_identity_error"] < 1.0e-12
    assert np.isclose(metrics["energy_score_common_delta"][0], 0.0)
    assert metrics["energy_score_residual_delta"][0] > 0.0


def test_projection_fraction_is_one_for_exact_readout_direction() -> None:
    accumulator = MODULE._empty_accumulator()
    metrics = {
        **{
            f"energy_{name}": np.ones(len(MODULE.STATES))
            for name in MODULE.ENERGY_NAMES
        },
        **{
            f"dot_{name}": np.ones(len(MODULE.STATES))
            for name in MODULE.DOT_NAMES
        },
        "same_sign_common_history": np.ones(len(MODULE.STATES)),
        "same_sign_common_query": np.ones(len(MODULE.STATES)),
    }
    MODULE._update_accumulator(accumulator, metrics, candidate_count=4)
    rows = MODULE._finalize_accumulator(
        "q2", "all", accumulator, direction_norm_squared=1.0
    )
    assert rows[1]["total_native_direction_energy_fraction"] == 1.0
    assert rows[1]["residual_native_direction_isotropic_multiple"] == MODULE.HIDDEN_SIZE
    assert rows[1]["common_history_score_cosine"] == 1.0


def test_ratio_and_cosine_handle_zero() -> None:
    assert np.isclose(MODULE._rms_ratio(8.0, 2.0), 2.0)
    assert MODULE._rms_ratio(1.0, 0.0) is None
    assert np.isclose(MODULE._energy_cosine(1.0, 2.0, 2.0), 0.5)
    assert MODULE._energy_cosine(0.0, 0.0, 1.0) is None
