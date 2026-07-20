import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_deep_dive_causal_floor.py"
SPEC = importlib.util.spec_from_file_location("causal_floor", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_request_metrics_preserve_candidate_energy_identity() -> None:
    candidate = np.asarray(
        [
            [[3.0, 1.0]],
            [[1.0, 3.0]],
        ]
    )
    common = candidate.mean(axis=0)
    residual = candidate - common[None, :, :]
    history = np.asarray([[2.0, 2.0]])
    query = np.asarray([[1.0, 0.0]])
    metrics = MODULE._request_metrics(candidate, common, residual, history, query)
    assert metrics["maximum_energy_identity_error"] < 1.0e-12
    assert np.isclose(metrics["energy_total"][0], 5.0)
    assert np.isclose(metrics["energy_common"][0], 4.0)
    assert np.isclose(metrics["energy_residual"][0], 1.0)


def test_energy_cosine_and_orthogonal_fraction_are_hand_computed() -> None:
    cosine = MODULE._energy_cosine(1.0, 2.0, 2.0)
    assert np.isclose(cosine, 0.5)
    assert np.isclose(MODULE._orthogonal_fraction(cosine), 0.75)
    assert MODULE._energy_cosine(0.0, 0.0, 1.0) is None
    assert MODULE._orthogonal_fraction(None) is None


def test_rms_ratio_and_vector_cosine_handle_zero() -> None:
    assert np.isclose(MODULE._rms_ratio(8.0, 2.0), 2.0)
    assert MODULE._rms_ratio(1.0, 0.0) is None
    assert np.isclose(
        MODULE._vector_cosine(np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0])),
        1.0,
    )
    assert MODULE._vector_cosine(np.zeros(2), np.ones(2)) is None


def test_accumulator_preserves_equal_request_counts() -> None:
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
        **{
            f"profile_{name}": np.ones((len(MODULE.STATES), MODULE.HIDDEN_SIZE))
            for name in ("common", "history", "query_floor")
        },
        **{
            f"vector_{name}": np.ones((len(MODULE.STATES), MODULE.HIDDEN_SIZE))
            for name in ("common", "history", "query_floor")
        },
    }
    MODULE._update_accumulator(accumulator, metrics, candidate_count=7)
    MODULE._update_accumulator(accumulator, metrics, candidate_count=3)
    rows = MODULE._finalize_accumulator("q2", "all", accumulator)
    assert rows[1]["requests"] == 2
    assert rows[1]["candidate_rows"] == 10
    assert rows[1]["common_rms_over_query_floor"] == 1.0
    assert rows[1]["common_query_floor_cosine"] == 1.0
