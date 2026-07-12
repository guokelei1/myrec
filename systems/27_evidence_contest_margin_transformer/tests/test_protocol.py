from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.structure import ROLE_COUNTS, load_config, read_json, sha256_file


def test_selection_and_label_barriers_are_frozen() -> None:
    config = load_config(ROOT / "configs" / "train_gate.yaml", require_selection=True)
    assert ROLE_COUNTS["fit"] == 3000 and ROLE_COUNTS["internal_A"] == 600
    assert sha256_file(config["paths"]["selection"]) == config["paths"]["selection_sha256"]
    selection = read_json(config["paths"]["selection"])
    assert selection["checks"]["c27_internal_A_labels_opened"] is False
    assert selection["checks"]["c27_delayed_B_labels_opened"] is False
    assert config["authorization"]["dev"] is False
    assert config["authorization"]["test"] is False
