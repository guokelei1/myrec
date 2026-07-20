from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from myrec.mechanism.q0_trajectory_evaluator import (
    _common_implementation_digest,
    trajectory_geometry,
    trajectory_summary_rows,
)


def test_q0_trajectory_geometry_is_hand_computed():
    full = np.zeros((1, 29, 2), dtype=np.float64)
    null = np.zeros_like(full)
    full[:, :, 0] = 1.0
    null[:, :, 1] = 1.0
    result = trajectory_geometry(full, null)
    assert result["delta_l2_per_sqrt_hidden"][0, 13] == pytest.approx(1.0)
    assert result["full_null_cosine"][0, 13] == pytest.approx(0.0)
    assert result["full_rms"][0, 13] == pytest.approx(1 / np.sqrt(2))
    assert result["rms_ratio"][0, 13] == pytest.approx(1.0)


def test_q0_trajectory_summary_has_all_29_fixed_states():
    matrix = np.arange(58, dtype=np.float64).reshape(2, 29)
    rows = trajectory_summary_rows(
        matrix, position="query_end", metric="full_rms", weighting="request"
    )
    assert len(rows) == 29
    assert rows[0]["mean"] == pytest.approx(14.5)
    assert rows[-1]["hidden_state_index"] == 28


def test_q0_trajectory_requires_one_full_null_implementation_digest():
    bundles = {
        "full": SimpleNamespace(
            metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
        ),
        "null": SimpleNamespace(
            metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
        ),
    }
    assert _common_implementation_digest(bundles) == "fixed"
    bundles["null"].metadata["run_contract"]["implementation_digest"] = "drifted"
    with pytest.raises(ValueError, match="differs from run contract"):
        _common_implementation_digest(bundles)
    bundles["null"].metadata["run_contract"]["implementation_digest"] = "fixed"
    bundles["null"].metadata["implementation_identity"]["digest"] = "drifted"
    with pytest.raises(ValueError, match="different implementation digests"):
        _common_implementation_digest(bundles)
