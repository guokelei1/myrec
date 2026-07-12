from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.gate_metrics import clicked_minus_unclicked, compare_primary, retention_bootstrap


def test_compare_primary_uses_paired_differences_and_folds() -> None:
    request_ids = [f"r{index}" for index in range(12)]
    primary = np.linspace(0.2, 0.8, 12)
    result = compare_primary(
        request_ids=request_ids,
        primary=primary,
        references={"base": primary - 0.05},
        samples=1000,
        seed=7,
        folds=3,
    )["base"]
    assert abs(result["mean"] - 0.05) < 1e-12
    assert result["percentile_95_ci"][0] > 0.0
    assert all(row["mean_difference"] > 0.0 for row in result["hash_folds"])


def test_retention_bootstrap_matches_constant_ratio() -> None:
    clean = np.full(30, 0.04)
    corrupt = np.full(30, 0.01)
    result = retention_bootstrap(clean, corrupt, samples=1000, seed=11)
    assert abs(result["retention"] - 0.25) < 1e-12
    assert abs(result["percentile_95_ci"][1] - 0.25) < 1e-12
    assert result["bootstrap_nonpositive_clean_denominator_draws"] == 0


def test_clicked_minus_unclicked_is_request_equal() -> None:
    values = clicked_minus_unclicked(
        [np.asarray([0.3, -0.1]), np.asarray([0.4, 0.2, -0.2])],
        [np.asarray([1.0, 0.0]), np.asarray([1.0, 1.0, 0.0])],
    )
    np.testing.assert_allclose(values, np.asarray([0.4, 0.5]))
