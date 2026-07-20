import numpy as np
import pytest

from myrec.mechanism.readout_decomposition import (
    condition_decomposition_report,
    decompose_condition_scores,
    summarize_decomposition_contrast,
)


def _scores(left, right):
    return {"r1": {"a": float(left), "b": float(right)}}


def test_request_common_relative_decomposition_is_hand_computed():
    result = decompose_condition_scores(
        _scores(1, 3), ["r1"], {"r1": ["a", "b"]}
    )
    np.testing.assert_allclose(result["common"], [2.0])
    np.testing.assert_allclose(result["relative_rms"], [1.0])
    np.testing.assert_allclose(result["relative_by_request"]["r1"], [-1.0, 1.0])
    assert result["maximum_recomposition_abs_error"] == 0.0
    assert result["maximum_relative_sum_abs_error"] == 0.0


def test_common_shift_is_separated_from_relative_shape_shift():
    candidates = {"r1": ["a", "b"]}
    base = decompose_condition_scores(_scores(1, 3), ["r1"], candidates)
    common_only = decompose_condition_scores(_scores(6, 8), ["r1"], candidates)
    result = summarize_decomposition_contrast(common_only, base, ["r1"])
    assert result["common_shift"]["mean"] == 5.0
    assert result["candidate_relative_shift_rms"]["mean"] == 0.0


def test_decomposition_report_is_qrels_blind_and_nonconfirmatory():
    report = condition_decomposition_report(
        {"left": _scores(2, 6), "right": _scores(1, 3)},
        ["r1"],
        {"r1": ["a", "b"]},
        {"left_minus_right": ("left", "right")},
    )
    assert report["qrels_read"] is False
    assert report["confirmatory_family_membership"] is False
    assert report["algebra"]["maximum_recomposition_abs_error"] == 0.0
    assert report["contrast_summaries"]["left_minus_right"][
        "candidate_relative_shift_rms"
    ]["mean"] == pytest.approx(1.0)
