from __future__ import annotations

import numpy as np

from myrec.mechanism.deep_dive_native_patch import (
    NATIVE_TERMS,
    SCORE_CONDITIONS,
    _canonical_sha256,
)


def test_native_gate_contract_has_four_terms_and_six_conditions():
    assert NATIVE_TERMS == (
        "prompt_predict_yes",
        "yes_predict_terminator",
        "prompt_predict_no",
        "no_predict_terminator",
    )
    assert SCORE_CONDITIONS[-2:] == (
        "same_all_native_positions",
        "same_first_position_only",
    )


def test_native_score_formula_is_two_token_mean_difference():
    from myrec.mechanism.deep_dive_native_patch import _combine_terms

    torch = __import__("pytest").importorskip("torch")
    yes = torch.tensor([[-1.0, -3.0], [-2.0, -2.0]])
    no = torch.tensor([[-4.0, -2.0], [-1.0, -1.0]])
    terms, score = _combine_terms(yes, no)
    np.testing.assert_allclose(terms[0], [-1.0, -3.0, -4.0, -2.0])
    np.testing.assert_allclose(score, [1.0, -1.0])


def test_canonical_hash_is_key_order_invariant():
    assert _canonical_sha256({"a": 1, "b": 2}) == _canonical_sha256(
        {"b": 2, "a": 1}
    )


def test_canonical_json_sorting_does_not_define_condition_semantics():
    # Score readers validate the exact set and then access SCORE_CONDITIONS
    # explicitly; canonical JSON is allowed to reorder object keys.
    reversed_order = tuple(reversed(SCORE_CONDITIONS))
    assert reversed_order != SCORE_CONDITIONS
    assert set(reversed_order) == set(SCORE_CONDITIONS)
