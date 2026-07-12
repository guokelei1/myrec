from __future__ import annotations

from pathlib import Path

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]


def test_config_is_synthetic_only_and_frozen_shape() -> None:
    config = yaml.safe_load((SYSTEM_ROOT / "configs/synthetic_gate.yaml").read_text())
    assert config["candidate_id"] == "c22"
    assert config["training"]["modes"] == [
        "filtration",
        "dense",
        "parallel",
        "final_projection",
    ]
    assert config["training"]["seeds"] == [20260730, 20260731, 20260732]
    assert config["training"]["attempts"] == 1
    assert config["authorization"]["synthetic_gpu"] is True
    assert config["authorization"]["repository_data"] is False
    assert config["authorization"]["real_train"] is False
    assert config["authorization"]["dev"] is False
    assert config["authorization"]["test"] is False


def test_source_has_no_repository_data_or_evaluator_path() -> None:
    source_files = list((SYSTEM_ROOT / "model").glob("*.py")) + list((SYSTEM_ROOT / "train").glob("*.py"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    forbidden = (
        "data/standardized",
        "artifacts/analysis",
        "qrels_dev",
        "qrels_test",
        "evaluate_scores.py",
        "records_dev.jsonl",
        "records_test.jsonl",
    )
    assert all(token not in source for token in forbidden)
