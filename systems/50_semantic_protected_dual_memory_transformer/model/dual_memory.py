"""Semantic-direction protection for C50's dual memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DualMemoryReads:
    primary: torch.Tensor
    raw_semantic: torch.Tensor
    innovation: torch.Tensor
    unprojected_sum: torch.Tensor
    orthogonal_innovation: torch.Tensor


def semantic_protected_reads(
    raw_semantic: torch.Tensor,
    innovation: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> DualMemoryReads:
    if raw_semantic.shape != innovation.shape or raw_semantic.ndim != 2:
        raise ValueError("C50 read shapes differ")
    if epsilon <= 0:
        raise ValueError("C50 epsilon must be positive")
    denominator = raw_semantic.square().sum(-1, keepdim=True)
    coefficient = (innovation * raw_semantic).sum(-1, keepdim=True) / denominator.clamp_min(epsilon)
    orthogonal = torch.where(
        denominator.gt(epsilon),
        innovation - coefficient * raw_semantic,
        torch.zeros_like(innovation),
    )
    return DualMemoryReads(
        primary=raw_semantic + orthogonal,
        raw_semantic=raw_semantic,
        innovation=innovation,
        unprojected_sum=raw_semantic + innovation,
        orthogonal_innovation=orthogonal,
    )
