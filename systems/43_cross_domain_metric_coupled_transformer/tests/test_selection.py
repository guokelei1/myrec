from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[3]
SYSTEM = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_c43_A_is_exact_unopened_c37_union() -> None:
    config = yaml.safe_load((SYSTEM / "configs/train_gate.yaml").read_text(encoding="utf-8"))
    selection = load(REPO / config["paths"]["selection"])
    c37 = load(REPO / config["paths"]["c37_selection"])
    c37_g0 = load(REPO / config["paths"]["c37_g0_report"])
    c37_report = load(REPO / config["paths"]["c37_train_report"])
    A = set(selection["roles"]["internal_A"]["indices"])
    expected = set(c37["roles"]["delayed_B"]["indices"]) | set(
        c37["roles"]["escrow"]["indices"]
    )
    old_features = set(
        int(value)
        for value in np.load(
            REPO / Path(config["paths"]["c37_selection"]).parent / "feature_request_indices.npy",
            mmap_mode="r",
        )
    )
    assert len(A) == 1200
    assert A == expected
    assert not A & old_features
    assert c37_g0["delayed_B_features_labels_scores_opened"] is False
    assert c37_g0["escrow_features_or_labels_opened"] is False
    assert c37_report["delayed_B_features_labels_scores_opened"] is False
    assert c37_report["escrow_dev_test_opened"] is False
    assert all(selection["checks"].values())
    assert selection["donor_matching"]["same_length_bin_fraction"] == 1.0
    assert selection["donor_matching"]["same_length_and_time_bucket_fraction"] == 1.0


def test_method_code_has_no_dev_test_or_qrels_input() -> None:
    config = yaml.safe_load((SYSTEM / "configs/train_gate.yaml").read_text(encoding="utf-8"))
    registered_paths = "\n".join(str(value).lower() for value in config["paths"].values())
    for forbidden in ("qrels_dev", "qrels_test", "records_dev", "records_test"):
        assert forbidden not in registered_paths
    for path in (SYSTEM / "train").glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "open(\"qrels" not in text
        assert "open('qrels" not in text
