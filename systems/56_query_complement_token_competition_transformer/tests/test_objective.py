from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.run_signal_gate import listwise_loss, probability_residual  # noqa: E402


def test_probability_residual_is_exact_zero_sum() -> None:
    base = np.asarray([1.0, 0.0, -0.5], dtype=np.float32)
    labels = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    residual = probability_residual(base, labels)
    assert abs(float(residual.sum())) < 1e-7
    shifted = base.astype(np.float64) - base.max()
    expected = np.asarray([0.0, 1.0, 0.0]) - np.exp(shifted) / np.exp(shifted).sum()
    np.testing.assert_allclose(residual, expected, rtol=0.0, atol=1e-7)


def test_masked_listwise_loss_matches_hand_computation() -> None:
    scores = torch.tensor([[1.0, 0.0, 7.0], [0.0, 0.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    actual = listwise_loss(scores, labels, mask)
    expected_first = -torch.log_softmax(torch.tensor([1.0, 0.0]), dim=0)[0]
    expected_second = -torch.log_softmax(torch.tensor([0.0, 0.0]), dim=0).mean()
    expected = (expected_first + expected_second) / 2
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)


def test_listwise_loss_rejects_rows_without_positive() -> None:
    scores = torch.tensor([[1.0, 0.0]])
    labels = torch.zeros_like(scores)
    mask = torch.ones_like(scores, dtype=torch.bool)
    try:
        listwise_loss(scores, labels, mask)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("C56 objective accepted a row without positive labels")
