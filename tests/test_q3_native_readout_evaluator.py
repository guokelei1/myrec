from __future__ import annotations

import numpy as np
import pytest

from myrec.mechanism.q3_native_readout_evaluator import (
    _implementation_digest,
    q3_readout_contrast_values,
)


def test_q3_readout_contrast_values_are_hand_computed():
    endpoints = {
        "shared_prompt_same_full_to_null": {"target_margin": np.array([3.0, 5.0])},
        "shared_prompt_cross_full_to_null": {"target_margin": np.array([2.0, 8.0])},
        "baseline_null": {"target_margin": np.array([1.0, 1.5])},
        "baseline_full": {"target_margin": np.array([4.0, 4.5])},
    }
    np.testing.assert_array_equal(
        q3_readout_contrast_values(
            endpoints,
            scope="shared_prompt",
            comparison="same_minus_null",
            endpoint="target_margin",
        ),
        np.array([2.0, 3.5]),
    )
    np.testing.assert_array_equal(
        q3_readout_contrast_values(
            endpoints,
            scope="shared_prompt",
            comparison="same_minus_cross",
            endpoint="target_margin",
        ),
        np.array([1.0, -3.0]),
    )


def test_q3_readout_contrast_rejects_unregistered_axes():
    endpoints = {}
    with pytest.raises(ValueError, match="scope"):
        q3_readout_contrast_values(
            endpoints, scope="best", comparison="same_minus_null", endpoint="ndcg@10"
        )


def test_q3_readout_evaluator_binds_implementation_to_run_contract():
    metadata = {
        "implementation_identity": {"digest": "fixed"},
        "run_contract": {"implementation_digest": "fixed"},
    }
    assert _implementation_digest(metadata) == "fixed"
    metadata["run_contract"]["implementation_digest"] = "drifted"
    with pytest.raises(ValueError, match="differs from run contract"):
        _implementation_digest(metadata)
