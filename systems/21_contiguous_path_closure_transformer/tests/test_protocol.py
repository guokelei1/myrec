from __future__ import annotations

from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.path_closure import MODES
from train.materialize_selection import load_config


CONFIG = SYSTEM_ROOT / "configs" / "train_signal_gate.yaml"


def test_frozen_modes_seeds_and_attempt_budget() -> None:
    config = load_config(CONFIG)
    assert tuple(config["training"]["modes"]) == MODES
    assert config["training"]["seeds"] == [20260727, 20260728, 20260729]
    assert config["training"]["epochs"] == 2
    assert config["training"]["attempts"] == 1
    assert config["training"]["candidate_sampling"] is False
    assert config["training"]["corruption_training"] is False


def test_gate_thresholds_and_authorization_are_binding() -> None:
    config = load_config(CONFIG)
    gate = config["gate"]
    assert gate["ndcg10_delta_over_d2p_min"] == 0.001
    assert gate["ndcg10_delta_over_each_control_min"] == 0.0005
    assert gate["corruption_retention_max"] == 0.25
    assert gate["corruption_retention_ci_high_max"] == 0.5
    authorization = config["authorization"]
    assert authorization["compact_fit_label_training_after_lock"] is True
    assert all(
        authorization[name] is False
        for name in (
            "c06_internal_A",
            "c06_internal_B",
            "c06_escrow",
            "original_train_label_array",
            "dev",
            "test",
            "full_transformer_training",
        )
    )


def test_runner_has_no_original_label_or_dev_test_input_name() -> None:
    source = (SYSTEM_ROOT / "train" / "run_train_signal_gate.py").read_text(encoding="utf-8")
    data_source = (SYSTEM_ROOT / "train" / "real_data.py").read_text(encoding="utf-8")
    forbidden_filenames = ("candidate_labels.npy", "qrels_dev.jsonl", "qrels_test.jsonl", "records_dev.jsonl", "records_test.jsonl")
    assert all(name not in source for name in forbidden_filenames)
    assert all(name not in data_source for name in forbidden_filenames)
    assert "src/myrec/eval/metrics.py" == load_config(CONFIG)["paths"]["shared_metric_source"]
    assert (REPOSITORY_ROOT / "src/myrec/eval/metrics.py").is_file()
