"""Minimal request-listwise loss for C06 architecture contracts."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_listwise_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    if scores.ndim != 2 or relevance.shape != scores.shape:
        raise ValueError("scores and relevance must share shape [B, C]")
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask shape mismatch")
    mask = candidate_mask.bool()
    gains = relevance.clamp_min(0.0) * mask.to(relevance.dtype)
    valid = mask.any(dim=-1) & (gains.sum(dim=-1) > 0)
    if not bool(valid.any().item()):
        return scores.sum() * 0.0
    logits = scores[valid].masked_fill(~mask[valid], -torch.inf)
    targets = gains[valid] / gains[valid].sum(dim=-1, keepdim=True)
    log_probabilities = F.log_softmax(logits, dim=-1).masked_fill(
        ~mask[valid], 0.0
    )
    return -(targets * log_probabilities).sum(dim=-1).mean()
