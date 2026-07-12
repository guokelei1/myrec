from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.structure import ROLE_COUNTS, load_config, sha256_file


def test_selection_and_authorization_are_frozen() -> None:
    config = load_config(ROOT / "configs" / "train_gate.yaml", require_selection=True)
    assert sha256_file(config["paths"]["selection"]) == config["paths"]["selection_sha256"]
    assert ROLE_COUNTS["internal_A"] == 600
    assert ROLE_COUNTS["escrow"] == 340
    assert config["authorization"]["dev"] is False
    assert config["authorization"]["test"] is False
    assert "<seed>" in config["selection"]["hash_payload"]
    assert config["paths"]["proposal_lock"].endswith("proposal_lock_v2.json")
