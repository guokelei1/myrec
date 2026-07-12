"""Full-candidate listwise objective for C23."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    if scores.ndim != 2 or labels.shape != scores.shape or candidate_mask.shape != (
        scores.shape
    ):
        raise ValueError("C23 listwise tensors must share [B,C]")
    mask = candidate_mask.bool()
    gains = labels.clamp_min(0.0) * mask.to(labels.dtype)
    valid = mask.any(dim=-1) & gains.sum(dim=-1).gt(0.0)
    if not bool(valid.any()):
        return scores.sum() * 0.0
    logits = scores[valid].masked_fill(~mask[valid], -torch.inf)
    targets = gains[valid] / gains[valid].sum(dim=-1, keepdim=True)
    log_probability = F.log_softmax(logits, dim=-1).masked_fill(~mask[valid], 0.0)
    return -(targets * log_probability).sum(dim=-1).mean()
