from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SELECTION = REPOSITORY_ROOT / "artifacts/c21_contiguous_path_closure_transformer/train_signal_v1/selection.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_label_free_selection_is_exact_and_isolated() -> None:
    value = json.loads(SELECTION.read_text(encoding="utf-8"))
    train = set(value["roles"]["train_fit"]["indices"])
    probe = set(value["roles"]["internal_probe"]["indices"])
    assert len(train) == 9000
    assert len(probe) == 3000
    assert train.isdisjoint(probe)
    assert all(value["checks"].values())
    assert value["labels_opened"] is False
    assert value["outcomes_observed"] is False


def test_selection_hash_matches_frozen_config() -> None:
    import yaml

    config = yaml.safe_load((SYSTEM_ROOT / "configs/train_signal_gate.yaml").read_text(encoding="utf-8"))
    assert sha256(SELECTION) == config["paths"]["selection_sha256"]
    assert sha256(REPOSITORY_ROOT / config["paths"]["c06_selection"]) == config["paths"]["c06_selection_sha256"]
    assert sha256(REPOSITORY_ROOT / config["paths"]["c06_g0_report"]) == config["paths"]["c06_g0_report_sha256"]
