from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.run_synthetic_gate import generate, model_from_config, train_model
from train.synthetic import batch_schedule


def test_frozen_config_contains_literal_gate_constants() -> None:
    config = yaml.safe_load((ROOT / "configs" / "synthetic_gate.yaml").read_text())
    assert config["seeds"] == [20260718, 20260719, 20260720]
    assert config["training"]["steps"] == 800
    assert config["attempts"]["learned_runs"] == 1
    assert config["access"] == {
        "repository_data": False,
        "standardized_records": False,
        "labels": "synthetic_only",
        "dev_evaluator_calls": 0,
        "test_access": False,
    }


def test_runner_has_no_external_evidence_or_evaluation_path() -> None:
    source = (ROOT / "train" / "run_synthetic_gate.py").read_text(encoding="utf-8")
    forbidden = (
        "data/standardized",
        "records_train",
        "records_dev",
        "records_test",
        "evaluate_scores",
        "dev_eval_log",
    )
    assert all(value not in source for value in forbidden)


def test_paper_rejected_c17_has_no_executable_tree() -> None:
    c17 = ROOT.parent / "17_evidence_ledger_margin_transformer"
    assert not (c17 / "model").exists()
    assert not (c17 / "train").exists()
    assert not (c17 / "configs").exists()


def test_two_step_runner_smoke_is_finite_on_cpu() -> None:
    config = yaml.safe_load((ROOT / "configs" / "synthetic_gate.yaml").read_text())
    config["data"]["train_requests"] = 64
    config["training"]["steps"] = 2
    config["training"]["batch_size"] = 16
    batch = generate(config, config["seeds"][0], "train")
    schedule = batch_schedule(
        seed=config["seeds"][0], requests=64, steps=2, batch_size=16
    )
    for mode in ("projection", "direct", "soft_penalty"):
        model = model_from_config(config, mode)
        result = train_model(model, batch, schedule, config, torch.device("cpu"))
        assert result["steps"] == 2
        assert result["finite"]
