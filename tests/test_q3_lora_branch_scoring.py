from __future__ import annotations

import numpy as np
import pytest

from myrec.mechanism.q3_lora_branch_scoring import (
    LORA_BRANCH_CONDITIONS,
    _validate_conditions,
)


def test_n19_condition_contract_covers_qv_identity_and_controls():
    assert len(LORA_BRANCH_CONDITIONS) == 14
    assert LORA_BRANCH_CONDITIONS[:4] == (
        "baseline_full",
        "baseline_null",
        "full_lora_identity",
        "null_lora_identity",
    )
    assert len(set(LORA_BRANCH_CONDITIONS)) == 14


def test_n19_condition_validation_rejects_nonfinite_scores():
    values = {name: np.zeros(2, dtype=np.float32) for name in LORA_BRANCH_CONDITIONS}
    _validate_conditions(values, 2)
    values["full_lora_zero"][0] = np.inf
    with pytest.raises(FloatingPointError, match="full_lora_zero"):
        _validate_conditions(values, 2)
