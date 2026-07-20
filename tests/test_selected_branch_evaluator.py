import pytest

from myrec.mechanism.selected_branch_evaluator import (
    CONTRAST_GROUPS,
    _audit_ineligible_wrong_user_rows,
    _selected_branch_implementation_digest,
    selected_branch_contrast_specs,
)
from myrec.mechanism.selected_branch_scoring import SELECTED_NODES


def test_selected_branch_contrast_family_counts_are_exact():
    specs = selected_branch_contrast_specs()
    counts = {
        group: sum(row["group"] == group for row in specs.values())
        for group in CONTRAST_GROUPS
    }
    assert counts == CONTRAST_GROUPS
    assert len(specs) == 48


def test_selected_branch_evaluator_binds_implementation_to_run_contract():
    metadata = {
        "implementation_identity": {"digest": "fixed"},
        "run_contract": {"implementation_digest": "fixed"},
    }
    assert _selected_branch_implementation_digest(metadata) == "fixed"
    metadata["run_contract"]["implementation_digest"] = "drifted"
    with pytest.raises(ValueError, match="differs from run contract"):
        _selected_branch_implementation_digest(metadata)


def test_ineligible_wrong_user_audit_uses_frozen_not_recomputed_null():
    row = {
        "candidate_item_id": "c1",
        "conditions": {
            "baseline_null": 0.125,
            **{
                f"{node}.wrong_history_to_null": 0.1
                for node in SELECTED_NODES
            },
        },
    }
    _audit_ineligible_wrong_user_rows(
        "r1",
        [row],
        {("r1", "c1"): 0.1},
    )
    row["conditions"][
        f"{SELECTED_NODES[0]}.wrong_history_to_null"
    ] = 0.11
    with pytest.raises(ValueError, match="does not copy frozen null"):
        _audit_ineligible_wrong_user_rows(
            "r1",
            [row],
            {("r1", "c1"): 0.1},
        )
