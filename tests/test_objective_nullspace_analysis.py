from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_deep_dive_objective_nullspace.py"
)
SPEC = importlib.util.spec_from_file_location("objective_nullspace", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("objective", MODULE.OBJECTIVES)
def test_common_shift_is_exact_null_direction(objective: str) -> None:
    result = MODULE._nullspace_metrics([0.2, -0.4, 0.7], [2.0, 0.0, 0.0], objective)
    assert abs(result["common_shift_loss_delta"]) < 1.0e-13
    assert abs(result["gradient_sum"]) < 1.0e-13
    assert result["hessian_times_ones_l2"] < 1.0e-13
    assert abs(result["common_direction_rayleigh"]) < 1.0e-13
    assert result["null_eigenvalue_count"] == 1
    assert result["positive_eigenvalue_count"] == 2


def test_deterministic_scores_are_ordered_and_reproducible() -> None:
    first = MODULE._deterministic_scores("r", ["a", "b", "c"])
    second = MODULE._deterministic_scores("r", ["a", "b", "c"])
    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) == 3
    assert np.all(first >= -1.0) and np.all(first <= 1.0)
