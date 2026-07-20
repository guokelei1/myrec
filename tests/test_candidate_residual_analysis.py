import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_deep_dive_candidate_residual.py"
SPEC = importlib.util.spec_from_file_location("candidate_residual_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_candidate_common_residual_energy_identity_is_hand_computed() -> None:
    delta = np.asarray(
        [
            [[3.0, 1.0]],
            [[1.0, 3.0]],
        ]
    )
    common, residual, total, common_energy, residual_energy = (
        MODULE._decompose_candidate_deltas(delta)
    )
    np.testing.assert_allclose(common, [[2.0, 2.0]])
    np.testing.assert_allclose(residual, [[[1.0, -1.0]], [[-1.0, 1.0]]])
    np.testing.assert_allclose(total, [5.0])
    np.testing.assert_allclose(common_energy, [4.0])
    np.testing.assert_allclose(residual_energy, [1.0])
    np.testing.assert_allclose(total, common_energy + residual_energy)


def test_projection_energy_fraction_uses_all_candidate_residual_energy() -> None:
    values = np.asarray([[3.0, 4.0], [0.0, 2.0]])
    basis = np.asarray([[1.0, 0.0]])
    # Projected energy 9; total energy 29.
    assert np.isclose(MODULE._projection_energy_fraction(values, basis), 9.0 / 29.0)


def test_probe_basis_is_raw_coordinate_and_centered() -> None:
    scale = np.asarray([2.0, 1.0])
    coefficient = np.asarray([[2.0, 0.0], [-2.0, 0.0]])
    basis = MODULE._probe_basis(scale, coefficient)
    assert basis.shape == (1, 2)
    assert np.isclose(abs(basis[0, 0]), 1.0)
    assert np.isclose(basis[0, 1], 0.0)


def test_zero_energy_projection_is_defined_as_zero() -> None:
    assert MODULE._projection_energy_fraction(
        np.zeros((2, 3)), np.asarray([[1.0, 0.0, 0.0]])
    ) == 0.0


def test_cosine_handles_parallel_and_zero_vectors() -> None:
    assert np.isclose(MODULE._cosine(np.asarray([1.0, 2.0]), np.asarray([2.0, 4.0])), 1.0)
    assert np.isnan(MODULE._cosine(np.zeros(2), np.ones(2)))
