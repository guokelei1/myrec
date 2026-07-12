from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.gate_metrics import bootstrap, retention
from train.losses import masked_listwise_loss
from train.real_data import zscore_numpy


def test_listwise_loss_rewards_positive_candidate() -> None:
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    good = masked_listwise_loss(torch.tensor([[2.0, 0.0, -1.0]]), labels, mask)
    bad = masked_listwise_loss(torch.tensor([[-1.0, 0.0, 2.0]]), labels, mask)
    assert good < bad


def test_zscore_and_json_safe_retention() -> None:
    values = np.asarray([[1.0, 2.0, 3.0, 99.0]], dtype=np.float32)
    mask = np.asarray([[1, 1, 1, 0]], dtype=bool)
    result = zscore_numpy(values, mask)
    assert abs(float(result[0, :3].mean())) < 1e-7
    assert result[0, 3] == 0.0
    assert bootstrap(np.asarray([1.0, 2.0]), samples=100, seed=3)["mean"] == 1.5
    row = retention(np.asarray([-1.0, 0.0]), np.ones(2), samples=100, seed=3)
    assert row["applicable"] is False and row["retention"] is None
