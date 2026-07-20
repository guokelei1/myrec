from __future__ import annotations

import numpy as np

from myrec.mechanism.native_readout_evaluator import (
    Q2_READOUT_FAMILY_SIZE,
    q2_readout_contrast_values,
)


def test_q2_readout_registered_contrasts_are_hand_computed():
    endpoints = {
        "input_same_full_to_null": {"target_margin": np.asarray([3.0, 5.0])},
        "input_cross_full_to_null": {"target_margin": np.asarray([2.5, 7.0])},
        "baseline_null": {"target_margin": np.asarray([1.0, 1.0])},
        "baseline_full": {"target_margin": np.asarray([4.0, 4.0])},
    }
    np.testing.assert_array_equal(
        q2_readout_contrast_values(
            endpoints,
            prefix="input",
            comparison="same_minus_null",
            endpoint="target_margin",
        ),
        np.asarray([2.0, 4.0]),
    )
    np.testing.assert_array_equal(
        q2_readout_contrast_values(
            endpoints,
            prefix="input",
            comparison="same_minus_full",
            endpoint="target_margin",
        ),
        np.asarray([-1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        q2_readout_contrast_values(
            endpoints,
            prefix="input",
            comparison="same_minus_cross",
            endpoint="target_margin",
        ),
        np.asarray([0.5, -2.0]),
    )
    assert Q2_READOUT_FAMILY_SIZE == 2 * 3 * 2 == 12
