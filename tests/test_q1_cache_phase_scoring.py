import numpy as np
import pytest

from myrec.mechanism.q1_cache_phase_scoring import (
    CACHE_PHASE_CONDITIONS,
    _validate_scores,
)


def test_cache_phase_condition_registration_is_fixed():
    assert CACHE_PHASE_CONDITIONS == (
        "native_cache_identity",
        "same_request_rebuild",
        "zero_prefix_cache",
        "wrong_user_prefix_cache",
        "no_cache_rebuild",
    )


def test_cache_phase_score_validation_requires_complete_finite_vectors():
    values = {name: np.zeros(3, dtype=np.float32) for name in CACHE_PHASE_CONDITIONS}
    _validate_scores(values, 3)
    values["zero_prefix_cache"][1] = np.nan
    with pytest.raises(FloatingPointError):
        _validate_scores(values, 3)

