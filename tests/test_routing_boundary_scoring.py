from __future__ import annotations

import numpy as np
import pytest

from myrec.mechanism.routing_boundary_scoring import (
    GQA_CONDITIONS,
    HEAD_NORM_CONDITIONS,
    _validate_conditions,
)


def test_n17_and_n18_condition_contracts_are_complete_and_disjoint():
    assert len(HEAD_NORM_CONDITIONS) == 16
    assert len(GQA_CONDITIONS) == 10
    assert len(set(HEAD_NORM_CONDITIONS)) == len(HEAD_NORM_CONDITIONS)
    assert len(set(GQA_CONDITIONS)) == len(GQA_CONDITIONS)
    assert HEAD_NORM_CONDITIONS[:2] == ("baseline_full", "baseline_null")
    assert GQA_CONDITIONS[:2] == ("baseline_full", "baseline_null")


def test_routing_condition_validation_requires_finite_complete_vectors():
    values = {name: np.zeros(2, dtype=np.float32) for name in GQA_CONDITIONS}
    _validate_conditions(values, 2, GQA_CONDITIONS)
    values["full_gqa_cyclic_permutation"][1] = np.nan
    with pytest.raises(FloatingPointError, match="cyclic"):
        _validate_conditions(values, 2, GQA_CONDITIONS)


def test_routing_condition_validation_rejects_registration_drift():
    values = {name: np.zeros(1, dtype=np.float32) for name in HEAD_NORM_CONDITIONS}
    with pytest.raises(ValueError, match="condition set"):
        _validate_conditions(values, 1, tuple(HEAD_NORM_CONDITIONS[:-1]))
