from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.run_synthetic_gate import generate, model_from_config, train_model
from train.synthetic import batch_schedule


def test_config_literals_and_access_boundary() -> None:
    config = yaml.safe_load((ROOT / "configs" / "synthetic_gate.yaml").read_text())
    assert config["seeds"] == [20260721, 20260722, 20260723]
    assert config["training"]["steps"] == 500
    assert config["attempts"] == {"learned_runs": 1, "post_lock_repairs": 0}
    assert config["access"]["repository_data"] is False
    assert config["access"]["dev_evaluator_calls"] == 0
    assert config["access"]["test_access"] is False


def test_runner_has_no_external_data_or_evaluation_path() -> None:
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


def test_two_step_cpu_smoke_is_finite() -> None:
    config = yaml.safe_load((ROOT / "configs" / "synthetic_gate.yaml").read_text())
    config["data"]["train_requests"] = 64
    config["training"]["steps"] = 2
    config["training"]["batch_size"] = 16
    seed = config["seeds"][0]
    batch = generate(config, seed, "train")
    schedule = batch_schedule(seed=seed, requests=64, steps=2, batch_size=16)
    for mode in ("oriented", "diagonal", "forward", "symmetric", "free_signed"):
        result = train_model(model_from_config(config, mode), batch, schedule, config, torch.device("cpu"))
        assert result["finite"]
        assert result["steps"] == 2
