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
    assert selection["checks"]["c30_A_labels_previously_opened_but_not_reused"] is True
    assert selection["checks"]["c31_internal_A_features_scores_labels_opened"] is False
    assert selection["checks"]["c31_delayed_B_features_scores_labels_opened"] is False
    assert selection["checks"]["donor_user_overlap_zero"] is True
    assert selection["checks"]["c31_code_dev_test_qrels_metrics_read"] is False


def test_architecture_and_training_are_not_query_or_label_slices() -> None:
    config = load_config(CONFIG, require_selection=True)
    assert config["model"]["adapter_rank"] == 16
    assert config["training"]["candidate_sampling"] is False
    assert config["training"]["attempts"] == 1
    assert config["training"]["listwise_loss_weight"] == 1.0
    assert config["training"]["direction_loss_weight"] == 1.0
    assert config["authorization"]["dev"] is False
    assert config["authorization"]["test"] is False
    assert config["authorization"]["controls_and_delayed_B_only_after_A1"] is True
