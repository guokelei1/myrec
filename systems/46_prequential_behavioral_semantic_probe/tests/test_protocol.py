from pathlib import Path
import json
import sys

import numpy as np
import yaml

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from probe.metrics import bootstrap, clicked_direction, order_change_fraction  # noqa: E402


def load_config() -> dict:
    with (SYSTEM_ROOT / "configs/signal_gate.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_selection_and_time_barriers_are_frozen() -> None:
    config = load_config()
    selection = json.loads((REPO_ROOT / config["paths"]["selection"]).read_text(encoding="utf-8"))
    assert selection["status"] == "frozen_label_free_before_c46_proposal_or_outcome"
    assert len(selection["roles"]["internal_A"]["indices"]) == 600
    assert selection["checks"]["source_strictly_before_outcome"] is True
    assert selection["checks"]["strict_nonrepeat"] is True
    assert selection["checks"]["labels_read"] is False
    assert config["source"]["request_stop_exclusive"] == 40000
    assert selection["provenance"]["outcome_min_index"] == 50000


def test_authorization_and_gate_thresholds_are_explicit() -> None:
    config = load_config()
    assert config["authorization"] == {
        "label_free_selection": True,
        "source_labels_after_proposal_lock": True,
        "A_features_scores_after_proposal_lock": True,
        "A_labels_after_A0_only": True,
        "dev": False,
        "test": False,
        "qrels": False,
        "full_training": False,
    }
    assert config["training"]["steps"] == 500
    assert config["gate"]["primary_minus_wrong_ndcg_min"] == 0.005


def test_metrics_are_paired_and_request_equal() -> None:
    values = np.asarray([1.0, -1.0, 2.0])
    report = bootstrap(values, samples=1000, seed=1)
    assert report["mean"] == float(values.mean())
    scores = [np.asarray([2.0, 1.0]), np.asarray([0.0, 3.0])]
    labels = [np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])]
    assert np.array_equal(clicked_direction(scores, labels), np.asarray([1.0, 3.0]))
    assert order_change_fraction(scores, [row[::-1] for row in scores], [["a", "b"], ["a", "b"]]) == 1.0
