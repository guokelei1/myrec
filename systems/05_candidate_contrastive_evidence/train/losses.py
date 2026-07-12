"""Ranking-aligned and empty-safe objectives for the C05 signal probe."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _differentiable_zero(*values: torch.Tensor) -> torch.Tensor:
    result = values[0].sum() * 0.0
    for value in values[1:]:
        result = result + value.sum() * 0.0
    return result


def masked_listwise_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Multi-positive listwise cross entropy with safe empty-request handling."""

    if scores.ndim != 2 or relevance.shape != scores.shape:
        raise ValueError("scores and relevance must share shape [B, C]")
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask shape mismatch")
    mask = candidate_mask.bool()
    gains = relevance.clamp_min(0.0) * mask.to(relevance.dtype)
    gain_sum = gains.sum(dim=-1)
    valid_request = mask.any(dim=-1) & (gain_sum > 0)
    if not bool(valid_request.any().item()):
        return _differentiable_zero(scores)

    selected_scores = scores[valid_request]
    selected_mask = mask[valid_request]
    selected_gains = gains[valid_request]
    targets = selected_gains / selected_gains.sum(dim=-1, keepdim=True)
    logits = selected_scores.masked_fill(~selected_mask, float("-inf"))
    log_probabilities = F.log_softmax(logits, dim=-1)
    finite_log_probabilities = log_probabilities.masked_fill(~selected_mask, 0.0)
    terms = targets * finite_log_probabilities
    return -terms.sum(dim=-1).mean()


def positive_negative_margin(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-request mean-positive minus mean-negative score and validity."""

    if scores.ndim != 2 or relevance.shape != scores.shape:
        raise ValueError("scores and relevance must share shape [B, C]")
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask shape mismatch")
    mask = candidate_mask.bool()
    positive = mask & (relevance > 0)
    negative = mask & ~positive
    positive_count = positive.sum(dim=-1)
    negative_count = negative.sum(dim=-1)
    valid = (positive_count > 0) & (negative_count > 0)
    positive_mean = (
        scores * positive.to(scores.dtype)
    ).sum(dim=-1) / positive_count.clamp_min(1).to(scores.dtype)
    negative_mean = (
        scores * negative.to(scores.dtype)
    ).sum(dim=-1) / negative_count.clamp_min(1).to(scores.dtype)
    margin = torch.where(valid, positive_mean - negative_mean, torch.zeros_like(positive_mean))
    return margin, valid


def ranking_aligned_corruption_loss(
    true_scores: torch.Tensor,
    corrupted_scores: torch.Tensor,
    relevance: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    minimum_margin_gap: float = 0.0,
    request_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Require true history to beat corruptions in candidate ranking margin.

    `corrupted_scores` has shape `[B, T, C]`, where `T` indexes the frozen
    wrong/shuffle/query-mask/coarse twins.  If no request has both a positive
    and a negative candidate (or `request_mask` selects none), the function
    returns a finite differentiable zero instead of taking `mean(empty)`.
    """

    if true_scores.ndim != 2:
        raise ValueError("true_scores must have shape [B, C]")
    if corrupted_scores.ndim != 3:
        raise ValueError("corrupted_scores must have shape [B, T, C]")
    batch, candidate_count = true_scores.shape
    if corrupted_scores.shape[0] != batch or corrupted_scores.shape[2] != candidate_count:
        raise ValueError("corrupted_scores shape mismatch")
    if relevance.shape != true_scores.shape or candidate_mask.shape != true_scores.shape:
        raise ValueError("relevance/candidate_mask shape mismatch")
    if minimum_margin_gap < 0.0:
        raise ValueError("minimum_margin_gap must be non-negative")

    true_margin, valid = positive_negative_margin(
        true_scores, relevance, candidate_mask
    )
    twin_count = corrupted_scores.shape[1]
    if twin_count == 0:
        return _differentiable_zero(true_scores, corrupted_scores)
    flat_corrupted = corrupted_scores.reshape(batch * twin_count, candidate_count)
    repeated_relevance = relevance[:, None, :].expand(
        -1, twin_count, -1
    ).reshape(batch * twin_count, candidate_count)
    repeated_mask = candidate_mask[:, None, :].expand(
        -1, twin_count, -1
    ).reshape(batch * twin_count, candidate_count)
    corrupted_margin, corrupted_valid = positive_negative_margin(
        flat_corrupted, repeated_relevance, repeated_mask
    )
    corrupted_margin = corrupted_margin.reshape(batch, twin_count)
    corrupted_valid = corrupted_valid.reshape(batch, twin_count).all(dim=-1)
    valid = valid & corrupted_valid
    if request_mask is not None:
        if request_mask.shape != (batch,):
            raise ValueError("request_mask must have shape [B]")
        valid = valid & request_mask.bool()
    if not bool(valid.any().item()):
        return _differentiable_zero(true_scores, corrupted_scores)

    gap = true_margin[:, None] - corrupted_margin
    hinge = F.relu(float(minimum_margin_gap) - gap)
    return hinge[valid].mean()
