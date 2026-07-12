"""Centered cross-event affinity covariance for C51."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class AffinityCovarianceOutput:
    covariance: torch.Tensor
    uncentered_second_moment: torch.Tensor
    pearson_control: torch.Tensor
    query_affinity: torch.Tensor
    candidate_affinity: torch.Tensor


def affinity_covariance(
    query: torch.Tensor,
    history: torch.Tensor,
    history_mask: torch.Tensor,
    candidates: torch.Tensor,
    *,
    normalization_epsilon: float = 1e-6,
    variance_epsilon: float = 1e-6,
) -> AffinityCovarianceOutput:
    if query.ndim != 2 or history.ndim != 3 or candidates.ndim != 3:
        raise ValueError("C51 state ranks differ")
    if history.shape[:2] != history_mask.shape:
        raise ValueError("C51 mask shape differs")
    if query.shape[0] != history.shape[0] or query.shape[0] != candidates.shape[0]:
        raise ValueError("C51 batch differs")
    if query.shape[-1] != history.shape[-1] or query.shape[-1] != candidates.shape[-1]:
        raise ValueError("C51 width differs")
    if normalization_epsilon <= 0 or variance_epsilon <= 0:
        raise ValueError("C51 epsilons must be positive")
    q = F.normalize(query, dim=-1, eps=normalization_epsilon)
    h = F.normalize(history, dim=-1, eps=normalization_epsilon)
    c = F.normalize(candidates, dim=-1, eps=normalization_epsilon)
    mask = history_mask.to(h.dtype)
    a = torch.einsum("bhd,bd->bh", h, q) * mask
    b = torch.einsum("bhd,bcd->bhc", h, c) * mask[..., None]
    count = mask.sum(1, keepdim=True)
    safe_count = count.clamp_min(1.0)
    mean_a = a.sum(1, keepdim=True) / safe_count
    mean_b = b.sum(1, keepdim=True) / safe_count[..., None]
    centered_a = (a - mean_a) * mask
    centered_b = (b - mean_b) * mask[..., None]
    denominator = (count - 1.0).clamp_min(1.0)
    covariance = torch.einsum("bh,bhc->bc", centered_a, centered_b) / denominator
    uncentered = torch.einsum("bh,bhc->bc", a, b) / safe_count
    var_a = centered_a.square().sum(1, keepdim=True) / denominator
    var_b = centered_b.square().sum(1) / denominator
    pearson_denominator = torch.sqrt((var_a * var_b).clamp_min(variance_epsilon**2))
    pearson = covariance / pearson_denominator
    usable = count.ge(2.0)
    zeros = torch.zeros_like(covariance)
    covariance = torch.where(usable, covariance, zeros)
    pearson = torch.where(usable & var_a.gt(variance_epsilon) & var_b.gt(variance_epsilon), pearson, zeros)
    uncentered = torch.where(count.gt(0), uncentered, zeros)
    return AffinityCovarianceOutput(
        covariance=covariance,
        uncentered_second_moment=uncentered,
        pearson_control=pearson,
        query_affinity=a,
        candidate_affinity=b,
    )
