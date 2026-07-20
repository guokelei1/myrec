from __future__ import annotations

import numpy as np
import pytest

from myrec.mechanism.rope_scoring import (
    COMMON_OFFSET_BOUND_RATIO_TOLERANCE,
    ROPE_SCORE_CONDITIONS,
    score_rope_chunk,
)


def test_rope_score_conditions_are_complete_and_unique():
    assert len(ROPE_SCORE_CONDITIONS) == 9
    assert len(set(ROPE_SCORE_CONDITIONS)) == 9
    assert ROPE_SCORE_CONDITIONS[:3] == (
        "baseline_full",
        "zero_phase_identity",
        "common_offset_plus_17_identity",
    )
    assert sum(name.endswith("compression") for name in ROPE_SCORE_CONDITIONS) == 3
    assert sum(name.endswith("expansion") for name in ROPE_SCORE_CONDITIONS) == 3


def test_bfloat_common_offset_registered_bound_scale():
    # 4 * eps(BF16) * a 16-point logit margin is 0.5, so a 0.125
    # quantization shift is within the frozen algebra gate but not a strict
    # wrapper identity.
    assert 4.0 * (2.0**-7) * 16.0 == 0.5
    assert COMMON_OFFSET_BOUND_RATIO_TOLERANCE == 1.0e-4


def test_q3_common_offset_keeps_strict_score_identity_after_signed_aggregation(monkeypatch):
    paths = [
        {"name": "yes", "weight": 1.0},
        {"name": "no", "weight": -1.0},
    ]
    baseline = [np.asarray([-4.0]), np.asarray([-4.0])]
    shifted = [np.asarray([-3.9375]), np.asarray([-4.0625])]

    monkeypatch.setattr("myrec.mechanism.rope_scoring._build_paths", lambda *a, **k: paths)
    values = iter(baseline)
    monkeypatch.setattr(
        "myrec.mechanism.rope_scoring._plain_path", lambda *a, **k: next(values)
    )

    class FakeIntervention:
        def __init__(self, _model, _block, mode):
            self.mode = mode

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def arm(self, *args, **kwargs):
            return None

        def disarm(self):
            return {
                "mode": self.mode,
                "maximum_query_norm_low_precision_ratio": 0.0,
                "maximum_key_norm_low_precision_ratio": 0.0,
            }

    monkeypatch.setattr(
        "myrec.mechanism.rope_scoring.QwenRoPEPhaseIntervention",
        FakeIntervention,
    )

    class FakeModel:
        call = 0

        def __call__(self, **kwargs):
            self.call += 1
            return object()

    condition_values = []
    # zero identity, common offset, then six active modes; each has two paths.
    condition_values.extend(baseline)
    condition_values.extend(shifted)
    condition_values.extend(baseline * 6)
    scored = iter(condition_values)
    monkeypatch.setattr(
        "myrec.mechanism.rope_scoring._path_scores", lambda *a, **k: next(scored)
    )
    for path in paths:
        path.update(
            {
                "positions": np.asarray([[0]]),
                "starts": np.asarray([0]),
                "ends": np.asarray([1]),
                "ids": np.zeros((1, 2), dtype=np.int64),
                "mask": np.ones((1, 2), dtype=np.int64),
                "target": [1],
            }
        )
    result = score_rope_chunk(
        FakeModel(),
        object(),
        object(),
        [{}],
        {"eligible": True},
        {},
        block=13,
        device="cpu",
    )
    # A signed Q3 aggregation must not convert a mechanical score mismatch
    # into an admissible low-precision score perturbation.  The low-precision
    # ratio now belongs only to the vector-norm geometry audit.
    assert result["identity_deltas"]["common_offset_plus_17_identity"] == pytest.approx(0.125)
    assert result["maximum_identity_delta"] == pytest.approx(0.125)
    assert result["common_offset_score_identity_passed"] is False
    assert result["common_offset_low_precision_max_ratio"] == 0.0
    assert result["common_offset_low_precision_passed"] is True
