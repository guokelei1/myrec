import numpy as np
import pytest

from myrec.mechanism.mlp_group_evaluator import (
    _audit_mlp_sample_and_donor_row,
    _common_implementation_digest,
    _summary,
)


def test_mlp_summary_is_hand_computed():
    result = _summary(np.asarray([1.0, 2.0, 3.0]))
    assert result["rows"] == 3
    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert result["minimum"] == 1.0
    assert result["maximum"] == 3.0


def test_mlp_evaluator_requires_one_implementation_digest():
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


def test_mlp_sample_and_cross_request_donor_are_bound():
    sample = {
        "request_id": "target",
        "candidate_ordinal": 2,
        "candidate_item_id": "target-item",
    }
    donor = {
        "donor_request_id": "donor",
        "donor_candidate_ordinal": 2,
        "donor_candidate_item_id": "donor-item",
    }
    row = {
        "ordinal": 7,
        **sample,
        **donor,
        "block_zero_based": 13,
    }
    _audit_mlp_sample_and_donor_row(
        row, sample, donor, index=7, block=13
    )

    row["donor_candidate_item_id"] = "drifted"
    with pytest.raises(ValueError, match="donor_candidate_item_id"):
        _audit_mlp_sample_and_donor_row(
            row, sample, donor, index=7, block=13
        )

    row["donor_candidate_item_id"] = "donor-item"
    row["candidate_ordinal"] = 1
    with pytest.raises(ValueError, match="candidate_ordinal"):
        _audit_mlp_sample_and_donor_row(
            row, sample, donor, index=7, block=13
        )
