from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

from train.gate_metrics import compare  # noqa: E402
from train.structure import FEATURE_ROLES, ROLE_COUNTS, load_config, read_json  # noqa: E402


CONFIG = SYSTEM / "configs/train_gate.yaml"


def test_candidate_uses_wholly_fresh_fit_and_outcome_roles() -> None:
    config = load_config(CONFIG, require_selection=True)
    selection = read_json(config["paths"]["selection"])
    c33 = read_json(config["paths"]["c33_selection"])
    c32 = read_json(config["paths"]["c32_selection"])
    assert {name: len(row["indices"]) for name, row in selection["roles"].items()} == ROLE_COUNTS
    flat = [value for row in selection["roles"].values() for value in row["indices"]]
    assert len(flat) == len(set(flat))
    c34_roles = {
        int(value)
        for row in selection["roles"].values()
        for value in row["indices"]
    }
    prior_roles = {
        int(value)
        for prior in (c32, c33)
        for row in prior["roles"].values()
        for value in row["indices"]
    }
    assert c34_roles.isdisjoint(prior_roles)
    assert "delayed_B" not in FEATURE_ROLES
    checks = selection["checks"]
    assert checks["fresh_fit_not_reused"] is True
    assert checks["c33_A_labels_previously_opened_but_not_reused"] is True
    assert checks["c33_delayed_B_and_escrow_not_reused"] is True
    assert checks["c32_c33_target_roles_not_reused"] is True
    assert checks["prior_wrong_donors_may_become_targets_without_prior_target_exposure"] is True
    assert checks["prior_wrong_donors_may_be_reused_as_donors"] is True
    assert checks["c34_internal_A_features_scores_labels_opened"] is False
    assert checks["selection_label_access"] is False


def test_matched_control_and_budget_are_frozen() -> None:
    config = load_config(CONFIG, require_selection=True)
    assert config["model"]["modes"] == [
        "candidate_tangent_cone",
        "standard_target_attention",
        "global_tangent_transport",
    ]
    assert config["model"]["adapter_rank"] == 16
    assert config["training"]["epochs"] == 1
    assert config["training"]["candidate_sampling"] is False
    assert config["training"]["attempts"] == 1
    assert config["training"]["seeds"] == [20260961, 20260962, 20260963]
    assert config["authorization"]["matched_control_training_after_execution_lock"] is True
    assert config["authorization"]["dev"] is False
    assert config["authorization"]["test"] is False


def test_all_references_share_one_hash_fold_partition() -> None:
    request_ids = [f"request-{index}" for index in range(60)]
    primary = np.linspace(-1.0, 1.0, len(request_ids))
    difference = np.linspace(0.01, 0.60, len(request_ids))
    result = compare(
        request_ids,
        primary,
        {
            "first": primary - difference,
            "second": primary - 2.0 * difference,
        },
        samples=100,
        seed=123,
        folds=3,
    )
    first = result["first"]["hash_folds"]
    second = result["second"]["hash_folds"]
    assert [row["requests"] for row in first] == [row["requests"] for row in second]
    for left, right in zip(first, second):
        assert np.isclose(right["mean_difference"], 2.0 * left["mean_difference"])
