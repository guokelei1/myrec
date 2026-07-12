from __future__ import annotations

import sys
from pathlib import Path

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.losses import (
    masked_listwise_loss,
    positive_negative_margin,
    ranking_aligned_corruption_loss,
)


def test_masked_listwise_loss_matches_hand_computation() -> None:
    scores = torch.tensor([[2.0, 1.0, -9.0]], requires_grad=True)
    relevance = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False]])
    loss = masked_listwise_loss(scores, relevance, mask)
    expected = torch.log(torch.exp(torch.tensor(2.0)) + torch.exp(torch.tensor(1.0))) - 2.0
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_positive_negative_margin_is_hand_computed() -> None:
    scores = torch.tensor([[3.0, 1.0, 2.0], [4.0, 7.0, 0.0]])
    relevance = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    mask = torch.ones_like(relevance, dtype=torch.bool)
    margin, valid = positive_negative_margin(scores, relevance, mask)
    assert torch.equal(valid, torch.tensor([True, True]))
    assert torch.allclose(margin, torch.tensor([1.5, 5.5]))


def test_corruption_loss_uses_ranking_direction_not_update_norm() -> None:
    relevance = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones_like(relevance, dtype=torch.bool)
    true_scores = torch.tensor([[3.0, 1.0, 0.0]], requires_grad=True)
    weaker_twins = torch.tensor(
        [[[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]]], requires_grad=True
    )
    passed = ranking_aligned_corruption_loss(
        true_scores,
        weaker_twins,
        relevance,
        mask,
        minimum_margin_gap=0.25,
    )
    assert float(passed.detach()) == 0.0

    stronger_twin = torch.tensor([[[4.0, 0.0, 0.0]]])
    failed = ranking_aligned_corruption_loss(
        true_scores,
        stronger_twin,
        relevance,
        mask,
        minimum_margin_gap=0.25,
    )
    assert float(failed.detach()) > 0.0


def test_empty_corruption_subset_returns_differentiable_zero() -> None:
    true_scores = torch.randn(2, 3, requires_grad=True)
    corrupted = torch.randn(2, 4, 3, requires_grad=True)
    relevance = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mask = torch.ones_like(relevance, dtype=torch.bool)
    request_mask = torch.zeros(2, dtype=torch.bool)
    loss = ranking_aligned_corruption_loss(
        true_scores,
        corrupted,
        relevance,
        mask,
        minimum_margin_gap=0.1,
        request_mask=request_mask,
    )
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert true_scores.grad is not None
    assert corrupted.grad is not None
    assert torch.count_nonzero(true_scores.grad) == 0
    assert torch.count_nonzero(corrupted.grad) == 0


def test_zero_twin_dimension_returns_differentiable_zero() -> None:
    true_scores = torch.tensor([[1.0, 0.0]], requires_grad=True)
    corrupted = torch.empty(1, 0, 2, requires_grad=True)
    relevance = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, True]])
    loss = ranking_aligned_corruption_loss(
        true_scores,
        corrupted,
        relevance,
        mask,
        minimum_margin_gap=0.1,
    )
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert true_scores.grad is not None
    assert corrupted.grad is not None


def test_listwise_loss_without_positive_is_finite_zero() -> None:
    scores = torch.randn(2, 3, requires_grad=True)
    relevance = torch.zeros(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    loss = masked_listwise_loss(scores, relevance, mask)
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert scores.grad is not None
