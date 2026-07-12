from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.query_relay import MODES  # noqa: E402
from probe.synthetic import make_dataset, wrong_history  # noqa: E402


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/design_gate.yaml").read_text())


def test_authorization_and_modes_are_frozen() -> None:
    row = config()
    assert tuple(row["model"]["modes"]) == MODES
    assert row["authorization"]["repository_data"] is False
    assert row["authorization"]["repository_labels"] is False
    assert row["authorization"]["dev"] is False
    assert row["authorization"]["test"] is False
    assert row["authorization"]["qrels"] is False


def test_generator_is_deterministic_and_shift_is_scoped() -> None:
    row = config()
    train_a = make_dataset(row, examples=128, seed=29, split="train")
    train_b = make_dataset(row, examples=128, seed=29, split="train")
    validation = make_dataset(row, examples=128, seed=29, split="validation")
    assert torch.equal(train_a.query_tokens, train_b.query_tokens)
    assert torch.equal(train_a.history_tokens, train_b.history_tokens)
    assert torch.equal(train_a.candidate_tokens, train_b.candidate_tokens)
    assert torch.equal(train_a.query_tokens, validation.query_tokens)
    assert torch.equal(train_a.candidate_tokens, validation.candidate_tokens)
    assert not torch.equal(train_a.history_tokens[:, 5], validation.history_tokens[:, 5])
    keep = torch.tensor([0, 1, 2, 3, 4, 6, 7])
    assert torch.equal(train_a.history_tokens[:, keep], validation.history_tokens[:, keep])


def test_wrong_history_is_cross_example_and_labels_unchanged() -> None:
    row = config()
    data = make_dataset(row, examples=128, seed=31, split="validation")
    wrong = wrong_history(data)
    assert torch.equal(data.labels, wrong.labels)
    assert torch.equal(data.candidate_tokens, wrong.candidate_tokens)
    eligible = data.history_mask.any(-1)
    assert not torch.equal(data.history_tokens[eligible], wrong.history_tokens[eligible])
