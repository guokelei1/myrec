from __future__ import annotations

from pathlib import Path
import sys


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

from train.structure import FEATURE_ROLES, ROLE_COUNTS, load_config, read_json  # noqa: E402


CONFIG = SYSTEM / "configs" / "train_gate.yaml"


def test_frozen_selection_and_untouched_A_B_boundaries() -> None:
    config = load_config(CONFIG, require_selection=True)
    selection = read_json(config["paths"]["selection"])
    assert {name: len(row["indices"]) for name, row in selection["roles"].items()} == ROLE_COUNTS
    flat = [index for row in selection["roles"].values() for index in row["indices"]]
    assert len(flat) == len(set(flat))
    assert "delayed_B" not in FEATURE_ROLES
    assert selection["checks"]["selection_label_access"] is False
    assert selection["checks"]["c31_A_labels_previously_opened_but_not_reused"] is True
    assert selection["checks"]["c32_internal_A_features_scores_labels_opened"] is False
    assert selection["checks"]["c32_delayed_B_features_scores_labels_opened"] is False
    assert selection["checks"]["donor_user_overlap_zero"] is True
    assert selection["checks"]["c32_code_dev_test_qrels_metrics_read"] is False


def test_only_geometry_changes_from_c31_budget() -> None:
    config = load_config(CONFIG, require_selection=True)
    assert config["model"]["adapter_rank"] == 16
    assert config["model"]["history_temperature"] == 0.1
    assert config["model"]["profile_scale"] == 1.0
    assert config["model"]["correction_scale"] == 2.0
    assert config["training"]["candidate_sampling"] is False
    assert config["training"]["attempts"] == 1
    assert config["training"]["epochs"] == 1
    assert config["authorization"]["dev"] is False
    assert config["authorization"]["test"] is False
