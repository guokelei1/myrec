from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.gate_metrics import paired_bootstrap, retention_bootstrap
from train.losses import masked_listwise_loss
from train.run_train_gate import suffix_permutation


def test_listwise_loss_prefers_clicked_candidate() -> None:
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    good = masked_listwise_loss(torch.tensor([[3.0, 0.0, -1.0]]), labels, mask)
    bad = masked_listwise_loss(torch.tensor([[-1.0, 0.0, 3.0]]), labels, mask)
    assert good < bad


def test_bootstrap_and_nonpositive_retention_are_json_safe() -> None:
    row = paired_bootstrap(np.asarray([1.0, 2.0, 3.0]), samples=100, seed=7)
    assert row["mean"] == 2.0
    retention = retention_bootstrap(
        np.asarray([-1.0, 0.0]), np.asarray([1.0, 1.0]), samples=100, seed=7
    )
    assert retention["applicable"] is False
    assert retention["retention"] is None


def test_suffix_shuffle_keeps_anchor_and_prefix_fixed() -> None:
    batch = {
        "repeat_mask_numpy": np.asarray([[[0, 1, 0, 0, 0]]], dtype=bool),
        "history_mask_numpy": np.asarray([[1, 1, 1, 1, 1]], dtype=bool),
        "request_ids": ["r"],
        "candidate_item_ids": np.asarray([[9]], dtype=np.int64),
    }
    permutation = suffix_permutation(
        batch, "c23-suffix-shuffle-v1\\0<request_id>\\0<candidate_item_id>"
    ).numpy()[0, 0]
    assert permutation[0] == 0
    assert permutation[1] == 1
    assert set(permutation[2:].tolist()) == {2, 3, 4}
