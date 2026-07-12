from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.losses import masked_listwise_loss


def test_listwise_loss_prefers_clicked_candidate() -> None:
    mask = torch.ones(1, 3, dtype=torch.bool)
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    good = masked_listwise_loss(torch.tensor([[3.0, 0.0, -1.0]]), labels, mask)
    bad = masked_listwise_loss(torch.tensor([[-1.0, 3.0, 0.0]]), labels, mask)
    assert good < bad


def test_listwise_loss_ignores_padding_and_empty_label_rows() -> None:
    scores = torch.tensor([[1.0, 2.0, 100.0], [2.0, 1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    loss = masked_listwise_loss(scores, labels, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert scores.grad[0].eq(0).all()
    assert scores.grad[:, 2].eq(0).all()
