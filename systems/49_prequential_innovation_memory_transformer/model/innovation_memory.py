"""C49 innovation-memory and matched-control reads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class InnovationReads:
    primary: torch.Tensor
    raw_krr: torch.Tensor
    innovation_softmax: torch.Tensor
    delta_net: torch.Tensor
    shuffled_innovation: torch.Tensor
    innovations: torch.Tensor


def innovation_memory_reads(
    query: torch.Tensor,
    keys: torch.Tensor,
    predictions: torch.Tensor,
    mask: torch.Tensor,
    *,
    ridge: float = 1.0,
    softmax_temperature: float = 0.1,
    epsilon: float = 1e-6,
) -> InnovationReads:
    """Read prequential innovation values with KRR and nearest controls.

    Args use `[batch,width]`, `[batch,history,width]`, and `[batch,history]`.
    The DeltaNet control is the fixed beta=1 delta-rule memory evaluated in an
    algebraically equivalent reverse read, avoiding a dense width-by-width
    fast-weight matrix.
    """

    if query.ndim != 2 or keys.ndim != 3 or predictions.shape != keys.shape:
        raise ValueError("C49 memory state ranks differ")
    if mask.shape != keys.shape[:2] or query.shape[0] != keys.shape[0]:
        raise ValueError("C49 memory mask/batch differs")
    if query.shape[-1] != keys.shape[-1]:
        raise ValueError("C49 memory width differs")
    if ridge <= 0 or softmax_temperature <= 0 or epsilon <= 0:
        raise ValueError("C49 memory constants must be positive")
    q = F.normalize(query, dim=-1, eps=epsilon)
    k = F.normalize(keys, dim=-1, eps=epsilon) * mask[..., None].to(keys.dtype)
    p = predictions * mask[..., None].to(predictions.dtype)
    innovations = (k - p) * mask[..., None].to(keys.dtype)
    gram = torch.einsum("bhd,bjd->bhj", k, k)
    eye = torch.eye(keys.shape[1], dtype=keys.dtype, device=keys.device)[None]
    normal = gram + float(ridge) * eye
    kq = torch.einsum("bhd,bd->bh", k, q)
    alpha = torch.linalg.solve(normal, kq[..., None]).squeeze(-1)
    primary = torch.einsum("bhd,bh->bd", innovations, alpha)
    raw = torch.einsum("bhd,bh->bd", k, alpha)

    logits = kq / float(softmax_temperature)
    logits = logits.masked_fill(~mask, -torch.inf)
    any_history = mask.any(1, keepdim=True)
    safe_logits = torch.where(any_history, logits, torch.zeros_like(logits))
    weights = torch.softmax(safe_logits, dim=1) * mask.to(keys.dtype)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(epsilon)
    softmax_read = torch.einsum("bhd,bh->bd", innovations, weights)

    read_query = q
    coefficients = torch.zeros_like(kq)
    for position in range(keys.shape[1] - 1, -1, -1):
        coefficient = torch.einsum("bd,bd->b", k[:, position], read_query)
        coefficient = coefficient * mask[:, position].to(coefficient.dtype)
        coefficients[:, position] = coefficient
        read_query = read_query - coefficient[:, None] * k[:, position]
    delta = torch.einsum("bhd,bh->bd", innovations, coefficients)

    shifted = torch.roll(innovations, shifts=1, dims=1)
    shuffled = torch.einsum("bhd,bh->bd", shifted, alpha)
    zeros = torch.zeros_like(primary)
    return InnovationReads(
        primary=torch.where(any_history, primary, zeros),
        raw_krr=torch.where(any_history, raw, zeros),
        innovation_softmax=torch.where(any_history, softmax_read, zeros),
        delta_net=torch.where(any_history, delta, zeros),
        shuffled_innovation=torch.where(any_history, shuffled, zeros),
        innovations=innovations,
    )
