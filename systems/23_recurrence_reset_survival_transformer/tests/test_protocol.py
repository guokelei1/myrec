from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.structure import ROLE_COUNTS, load_config, sha256_file


def test_frozen_selection_and_protocol_identity() -> None:
    config = load_config(ROOT / "configs" / "train_gate.yaml", require_frozen_selection=True)
    assert sha256_file(config["paths"]["selection"]) == config["paths"]["selection_sha256"]
    assert ROLE_COUNTS == {
        "fit": 12_000,
        "internal_A": 1_200,
        "delayed_B": 600,
        "escrow": 958,
        "structural_nohistory": 512,
        "structural_nonrepeat": 512,
    }
    assert config["authorization"]["dev"] is False
    assert config["authorization"]["test"] is False
    assert config["authorization"]["soft_anchor_stage_B"] is False
