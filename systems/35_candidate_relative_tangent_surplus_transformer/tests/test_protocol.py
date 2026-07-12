from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

from train.gate_metrics import compare  # noqa: E402
from train.structure import FEATURE_ROLES, ROLE_COUNTS, load_config, read_json  # noqa: E402


CONFIG = SYSTEM / "configs/train_gate.yaml"


def test_candidate_uses_only_c34_unopened_reserves_for_outcomes() -> None:
    config = load_config(CONFIG, require_selection=True)
    selection = read_json(config["paths"]["selection"])
    c34 = read_json(config["paths"]["c34_selection"])
    assert {name: len(row["indices"]) for name, row in selection["roles"].items()} == ROLE_COUNTS
    flat = [value for row in selection["roles"].values() for value in row["indices"]]
    assert len(flat) == len(set(flat))
    assert selection["roles"]["fit"]["indices"] == c34["roles"]["fit"]["indices"]
    assert selection["roles"]["internal_A"]["indices"] == c34["roles"]["delayed_B"]["indices"]
    assert selection["roles"]["delayed_B"]["indices"] == c34["roles"]["escrow"]["indices"]
    assert set(selection["roles"]["internal_A"]["indices"]).isdisjoint(
        c34["roles"]["internal_A"]["indices"]
    )
    assert "delayed_B" not in FEATURE_ROLES
    checks = selection["checks"]
    assert checks["c34_fit_reused_with_labels_previously_opened"] is True
    assert checks["c34_A_features_scores_opened_but_not_reused"] is True
    assert checks["c34_A_labels_opened"] is False
    assert checks["c34_delayed_B_promoted_to_c35_A_unopened"] is True
    assert checks["c34_escrow_promoted_to_c35_B_unopened"] is True
    assert checks["c35_internal_A_features_scores_labels_opened"] is False
    assert checks["selection_label_access"] is False


def test_matched_control_and_budget_are_frozen() -> None:
    config = load_config(CONFIG, require_selection=True)
    assert config["model"]["modes"] == [
        "candidate_relative_surplus",
        "absolute_tangent_cone",
        "candidate_axis_softmax",
        "global_tangent_transport",
    ]
    assert config["model"]["adapter_rank"] == 16
    assert config["training"]["epochs"] == 1
    assert config["training"]["candidate_sampling"] is False
    assert config["training"]["attempts"] == 1
    assert config["training"]["seeds"] == [20260981, 20260982, 20260983]
    assert config["gate"]["relative_zero_support_candidate_fraction_min"] == 0.10
    assert config["gate"]["relative_mixed_request_fraction_min"] == 0.50
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
