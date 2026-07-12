from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.run_gate import listwise_loss  # noqa: E402


def test_listwise_loss_matches_hand_result() -> None:
    scores = torch.tensor([[1.0, 0.0, 9.0], [0.0, 0.0, 8.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    actual = listwise_loss(scores, labels, mask)
    first = -torch.log_softmax(torch.tensor([1.0, 0.0]), dim=0)[0]
    second = -torch.log_softmax(torch.tensor([0.0, 0.0]), dim=0).mean()
    assert torch.allclose(actual, (first + second) / 2, atol=1e-7, rtol=0.0)


def test_listwise_loss_requires_positive() -> None:
    try:
        listwise_loss(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool)
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("C57 accepted a label row without positives")
