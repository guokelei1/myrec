"""Signed event-influence consensus token mixer for C48."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SignedConsensusOutput:
    correction: torch.Tensor
    plain_correction: torch.Tensor
    signed_l1_control: torch.Tensor
    coherence: torch.Tensor
    event_influences: torch.Tensor


def signed_consensus_mix(
    query: torch.Tensor,
    history: torch.Tensor,
    history_mask: torch.Tensor,
    candidates: torch.Tensor,
    *,
    ridge: float = 1.0,
    normalization_epsilon: float = 1e-6,
    influence_epsilon: float = 1e-6,
) -> SignedConsensusOutput:
    """Contract a KRR write by its signed eventwise influence agreement.

    ``z_jc = <h_j,c> [(HH^T + ridge I)^-1 Hq]_j`` decomposes the plain KRR
    candidate correction exactly.  ``|sum z| / sum |z|`` is one only when the
    valid event influences agree in sign and approaches zero under
    cancellation.  It uses no learned confidence head or dataset branch.
    """

    if query.ndim != 2 or history.ndim != 3 or candidates.ndim != 3:
        raise ValueError("C48 states must be rank 2/3/3")
    if history.shape[:2] != history_mask.shape:
        raise ValueError("C48 history mask shape differs")
    if query.shape[0] != history.shape[0] or query.shape[0] != candidates.shape[0]:
        raise ValueError("C48 batch dimensions differ")
    if query.shape[-1] != history.shape[-1] or query.shape[-1] != candidates.shape[-1]:
        raise ValueError("C48 state widths differ")
    if ridge <= 0 or normalization_epsilon <= 0 or influence_epsilon <= 0:
        raise ValueError("C48 positive constants required")
    q = F.normalize(query, dim=-1, eps=normalization_epsilon)
    h = F.normalize(history, dim=-1, eps=normalization_epsilon)
    c = F.normalize(candidates, dim=-1, eps=normalization_epsilon)
    h = h * history_mask[..., None].to(h.dtype)
    gram = torch.einsum("bhd,bjd->bhj", h, h)
    eye = torch.eye(history.shape[1], dtype=h.dtype, device=h.device)[None]
    normal = gram + float(ridge) * eye
    hq = torch.einsum("bhd,bd->bh", h, q)
    alpha_q = torch.linalg.solve(normal, hq[..., None]).squeeze(-1)
    hc = torch.einsum("bhd,bcd->bhc", h, c)
    influences = hc * alpha_q[..., None]
    plain = influences.sum(dim=1)
    total_variation = influences.abs().sum(dim=1)
    present = history_mask.any(dim=1, keepdim=True)
    active = present & total_variation.gt(0)
    coherence = torch.where(
        active,
        plain.abs() / total_variation.clamp_min(influence_epsilon),
        torch.zeros_like(plain),
    ).clamp(min=0.0, max=1.0)
    correction = coherence * plain
    signed_l1 = torch.where(
        active,
        plain / total_variation.clamp_min(influence_epsilon),
        torch.zeros_like(plain),
    )
    correction = torch.where(present, correction, torch.zeros_like(correction))
    plain = torch.where(present, plain, torch.zeros_like(plain))
    signed_l1 = torch.where(present, signed_l1, torch.zeros_like(signed_l1))
    influences = influences * history_mask[..., None].to(influences.dtype)
    return SignedConsensusOutput(
        correction=correction,
        plain_correction=plain,
        signed_l1_control=signed_l1,
        coherence=coherence,
        event_influences=influences,
    )
