from pathlib import Path
import sys

import torch
import yaml

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from probe.synthetic import generate, ndcg10, shuffled_history, wrong_history  # noqa: E402


def config() -> dict:
    with (SYSTEM_ROOT / "configs/design_gate.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_generator_is_reproducible_and_has_registered_headroom() -> None:
    cfg = config()
    first = generate(cfg, requests=256, split_offset=200_000)
    second = generate(cfg, requests=256, split_offset=200_000)
    for name in first.__dict__:
        assert torch.equal(getattr(first, name), getattr(second, name))
    base = ndcg10(first.query_only_scores, first.labels).mean()
    oracle = ndcg10(first.oracle_scores, first.labels).mean()
    assert float(oracle - base) >= 0.15
    assert not torch.equal(wrong_history(first), first.history)
    assert not torch.equal(shuffled_history(first), first.history)


def test_authorization_is_data_free_and_thresholds_are_frozen() -> None:
    cfg = config()
    assert cfg["authorization"] == {
        "repository_data": False,
        "repository_labels": False,
        "dev": False,
        "test": False,
        "qrels": False,
        "synthetic_gpu_after_lock": True,
        "real_train_gate": False,
        "full_training": False,
    }
    assert cfg["model"]["modes"] == [
        "innovation",
        "ordinary_delta",
        "factual_state",
        "raw_event",
    ]
    assert cfg["training"]["steps"] == 360
    assert cfg["gate"]["primary_margin_over_each_control_mean_min"] == 0.010
