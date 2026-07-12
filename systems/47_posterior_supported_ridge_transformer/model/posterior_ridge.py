"""Posterior-supported ridge token mixing for C47.

This file contains only the differentiable operator. Dataset masks, registered
base fallbacks, and the eventual Transformer wrapper belong to the staged gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PosteriorRidgeOutput:
    correction: torch.Tensor
    plain_correction: torch.Tensor
    query_write: torch.Tensor
    support: torch.Tensor


def posterior_ridge_mix(
    query: torch.Tensor,
    history: torch.Tensor,
    history_mask: torch.Tensor,
    candidates: torch.Tensor,
    *,
    ridge: float = 1.0,
    mode: str = "posterior_supported",
) -> PosteriorRidgeOutput:
    """Return candidate score coordinates from one exact ridge solve.

    Args:
        query: ``[batch, width]`` query-token states.
        history: ``[batch, history, width]`` history-token states.
        history_mask: valid history positions, ``[batch, history]``.
        candidates: ``[batch, candidates, width]`` candidate-token states.
        ridge: strictly positive normal-equation regularizer.
        mode: ``posterior_supported`` or the Cubit-style ``plain_ridge``
            degeneration.

    The same normalized history geometry produces both ``P q`` and
    ``rho_c = c^T P c``. No learned confidence head appears in this operator.
    """
    if query.ndim != 2 or history.ndim != 3 or candidates.ndim != 3:
        raise ValueError("C47 states must be rank 2/3/3")
    if history.shape[:2] != history_mask.shape:
        raise ValueError("C47 history mask shape differs")
    if query.shape[0] != history.shape[0] or query.shape[0] != candidates.shape[0]:
        raise ValueError("C47 batch dimensions differ")
    if query.shape[-1] != history.shape[-1] or query.shape[-1] != candidates.shape[-1]:
        raise ValueError("C47 state widths differ")
    if ridge <= 0:
        raise ValueError("C47 ridge must be positive")
    if mode not in {"posterior_supported", "plain_ridge"}:
        raise ValueError(f"unknown C47 mode: {mode}")

    q = F.normalize(query, dim=-1, eps=1e-6)
    h = F.normalize(history, dim=-1, eps=1e-6)
    c = F.normalize(candidates, dim=-1, eps=1e-6)
    h = h * history_mask[..., None].to(h.dtype)

    gram = torch.einsum("bhd,bjd->bhj", h, h)
    eye = torch.eye(history.shape[1], dtype=h.dtype, device=h.device)[None]
    normal = gram + float(ridge) * eye

    hq = torch.einsum("bhd,bd->bh", h, q)
    alpha_q = torch.linalg.solve(normal, hq[..., None]).squeeze(-1)
    query_write = torch.einsum("bhd,bh->bd", h, alpha_q)

    hc = torch.einsum("bhd,bcd->bhc", h, c)
    alpha_c = torch.linalg.solve(normal, hc)
    support = (hc * alpha_c).sum(dim=1).clamp(min=0.0, max=1.0)
    present = history_mask.any(dim=1, keepdim=True)
    support = torch.where(present, support, torch.zeros_like(support))
    query_write = torch.where(present, query_write, torch.zeros_like(query_write))

    plain = torch.einsum("bcd,bd->bc", c, query_write)
    correction = plain if mode == "plain_ridge" else support * plain
    correction = torch.where(present, correction, torch.zeros_like(correction))
    return PosteriorRidgeOutput(
        correction=correction,
        plain_correction=plain,
        query_write=query_write,
        support=support,
    )
