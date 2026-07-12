from __future__ import annotations

from pathlib import Path
import sys


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM / "train"))

from c30_protocol import load_config, read_json  # noqa: E402
from run_continuation import canonical_positions  # noqa: E402


CONFIG = SYSTEM / "configs" / "continuation_gate.yaml"


def test_canonical_positions_ignore_caller_order() -> None:
    item_ids = ["20", "3", "11", "8"]
    identity = canonical_positions(item_ids, [0, 1, 2, 3])
    reverse = canonical_positions(item_ids, [3, 2, 1, 0])
    assert identity == reverse == [2, 0, 1, 3]


def test_continuation_forbids_training_and_threshold_change() -> None:
    config = load_config(CONFIG)
    assert config["training"]["retraining"] is False
    assert config["training"]["optimizer_steps"] == 0
    assert config["authorization"]["threshold_change"] is False
    assert config["gate"]["candidate_permutation_max_abs_difference"] == 1e-6


def test_source_c29_failed_only_permutation() -> None:
    config = load_config(CONFIG)
    report = read_json(config["paths"]["c29_train_report"])
    assert report["internal_A_labels_opened"] is False
    failed = [name for name, passed in report["A0"]["checks"].items() if not passed]
    assert failed == ["candidate_permutation"]
