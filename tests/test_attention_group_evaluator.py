import numpy as np
import pytest

from myrec.mechanism.attention_group_evaluator import (
    ACTIVE_GROUP_CONDITIONS,
    ACTIVE_SUPPLEMENTAL_CONDITIONS,
    _audit_frozen_sample_and_eligibility_row,
    _common_implementation_digest,
    _summary,
)


def test_attention_gqa_descriptive_conditions_are_fixed():
    assert ACTIVE_GROUP_CONDITIONS == (
        "history_to_readout_logits_mask",
        "history_to_readout_value_zero",
        "neutral_history_kv",
    )
    assert ACTIVE_SUPPLEMENTAL_CONDITIONS == (
        "query_to_history_logits_mask",
        "query_to_history_value_zero",
        "cross_request_history_summary_kv",
    )


def test_attention_gqa_summary_is_hand_computed_without_significance_selection():
    result = _summary(np.asarray([-2.0, 0.0, 2.0]))
    assert result["rows"] == 3
    assert result["mean"] == 0.0
    assert result["median"] == 0.0
    assert result["mean_absolute"] == pytest.approx(4.0 / 3.0)
    assert "p" not in result


def test_attention_gqa_evaluator_requires_one_implementation_digest():
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


def test_attention_gqa_sample_eligibility_and_ineligible_fallback_are_bound():
    sample = {
        "selection_sha256": "selection",
        "request_id": "r",
        "candidate_item_id": "a",
        "candidate_ordinal": 2,
    }
    row = {
        **sample,
        "donor_request_id": "donor",
        "donor_candidate_item_id": "donor-item",
        "result": {
            "neutral_history_eligible": False,
            "supplemental": {"baseline_full": 0.25},
            "groups": [
                {"conditions": {"neutral_history_kv": 0.25}}
                for _ in range(8)
            ],
        },
    }
    cross_donors = {
        "selection": {
            "donor_request_id": "donor",
            "donor_candidate_item_id": "donor-item",
        }
    }
    _audit_frozen_sample_and_eligibility_row(
        row, sample, {"r": False}, cross_donors
    )

    row["result"]["groups"][3]["conditions"]["neutral_history_kv"] = 0.5
    with pytest.raises(ValueError, match="differs from baseline"):
        _audit_frozen_sample_and_eligibility_row(
            row, sample, {"r": False}, cross_donors
        )

    row["result"]["groups"][3]["conditions"]["neutral_history_kv"] = 0.25
    with pytest.raises(ValueError, match="eligibility differs"):
        _audit_frozen_sample_and_eligibility_row(
            row, sample, {"r": True}, cross_donors
        )

    row["donor_candidate_item_id"] = "drifted"
    with pytest.raises(ValueError, match="donor differs"):
        _audit_frozen_sample_and_eligibility_row(
            row, sample, {"r": False}, cross_donors
        )
