from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.structure import ROLE_COUNTS, load_config, read_json, sha256_file


def test_preselection_authorization_and_counts() -> None:
    config = load_config(ROOT / "configs" / "train_gate.yaml", require_selection=True)
    assert ROLE_COUNTS["fit"] == 6000
    assert ROLE_COUNTS["internal_A"] == 1200
    assert ROLE_COUNTS["delayed_B"] == 1200
    assert config["authorization"]["escrow"] is False
    assert config["authorization"]["dev"] is False
    assert config["authorization"]["test"] is False
    assert "<seed>" in config["selection"]["hash_payload"]
    assert sha256_file(config["paths"]["selection"]) == config["paths"]["selection_sha256"]
    selection = read_json(config["paths"]["selection"])
    assert selection["checks"]["labels_opened"] is False
    assert selection["checks"]["donor_candidate_overlap_zero"] is True
