from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_listwise_loss(
    scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if scores.shape != labels.shape or scores.shape != mask.shape or scores.ndim != 2:
        raise ValueError("C24 listwise shapes differ")
    gains = labels.clamp_min(0.0) * mask.to(labels.dtype)
    valid = mask.any(-1) & gains.sum(-1).gt(0)
    if not bool(valid.any()):
        return scores.sum() * 0.0
    logits = scores[valid].masked_fill(~mask[valid], -torch.inf)
    target = gains[valid] / gains[valid].sum(-1, keepdim=True)
    logp = F.log_softmax(logits, dim=-1).masked_fill(~mask[valid], 0.0)
    return -(target * logp).sum(-1).mean()
