from __future__ import annotations

import numpy as np

import pytest

from myrec.mechanism.attention_observation_evaluator import (
    _column_summary,
    _common_implementation_digest,
)


def test_attention_column_summary_preserves_every_head():
    values = np.vstack([np.arange(16), np.arange(16) + 2]).astype(float)
    result = _column_summary(values, 16)
    assert result["rows"] == 2
    assert result["mean"] == (np.arange(16) + 1).astype(float).tolist()
    assert len(result["std"]) == 16


def test_attention_evaluator_requires_one_implementation_digest():
    bundles = {
        "q2": {
            13: (None, {"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}, None),
            20: (None, {"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}, None),
        },
        "q3": {
            13: (None, {"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}, None),
        },
    }
    assert _common_implementation_digest(bundles) == "fixed"
    bundles["q3"][13][1]["run_contract"]["implementation_digest"] = "drifted"
    with pytest.raises(ValueError, match="differs from run contract"):
        _common_implementation_digest(bundles)
    bundles["q3"][13][1]["run_contract"]["implementation_digest"] = "fixed"
    bundles["q3"][13][1]["implementation_identity"]["digest"] = "drifted"
    with pytest.raises(ValueError, match="different implementation digests"):
        _common_implementation_digest(bundles)
