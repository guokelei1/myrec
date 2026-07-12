from __future__ import annotations

from pathlib import Path
import sys


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

from train.structure import FEATURE_ROLES, ROLE_COUNTS, load_config, read_json  # noqa: E402


CONFIG = SYSTEM / "configs" / "train_gate.yaml"


def test_frozen_selection_and_delayed_B_boundary() -> None:
    config = load_config(CONFIG, require_selection=True)
    selection = read_json(config["paths"]["selection"])
    assert {name: len(row["indices"]) for name, row in selection["roles"].items()} == ROLE_COUNTS
    flat = [index for row in selection["roles"].values() for index in row["indices"]]
    assert len(flat) == len(set(flat))
    assert "delayed_B" not in FEATURE_ROLES
    assert selection["checks"]["selection_label_access"] is False
    assert selection["checks"]["donor_user_overlap_zero"] is True
    assert selection["checks"]["c29_code_dev_test_qrels_metrics_read"] is False


def test_design_is_not_a_label_or_query_slice() -> None:
    config = load_config(CONFIG, require_selection=True)
    selection = read_json(config["paths"]["selection"])
    composition = selection["fit_composition"]
    assert composition["authentication_present_filter"] is False
    assert composition["query_or_category_filter"] is False
    assert composition["new_label_free_reserve"] == 6400
    assert config["training"]["candidate_sampling"] is False
    assert config["authorization"]["dev"] is False
    assert config["authorization"]["test"] is False
