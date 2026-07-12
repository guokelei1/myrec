"""History-supported query-concept attention operator for C52."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ConceptAttentionOutput:
    primary_correction: torch.Tensor
    linearized_correction: torch.Tensor
    softmax_correction: torch.Tensor
    evidence: torch.Tensor
    softmax_evidence: torch.Tensor
    base_concept_attention: torch.Tensor
    factual_concept_attention: torch.Tensor


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(torch.bool)
    masked = logits.masked_fill(~mask, -torch.inf)
    maximum = masked.amax(dim=dim, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    values = torch.exp(masked - maximum) * mask.to(logits.dtype)
    return values / values.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def _masked_logsumexp(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked = logits.masked_fill(~mask, -torch.inf)
    output = torch.logsumexp(masked, dim=dim)
    return torch.where(mask.any(dim=dim), output, torch.zeros_like(output))


def history_supported_concept_scores(
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    candidate_tokens: torch.Tensor,
    candidate_token_mask: torch.Tensor,
    history: torch.Tensor,
    *,
    ridge: float = 1.0,
    candidate_token_temperature: float = 0.1,
    query_concept_temperature: float = 0.1,
    history_softmax_temperature: float = 0.1,
    epsilon: float = 1e-6,
) -> ConceptAttentionOutput:
    """Return the fixed C52 primary and exact nearest reductions.

    Shapes are ``query_tokens=[Q,D]``, ``candidate_tokens=[C,L,D]``, and
    ``history=[H,D]``.  All masks are boolean.  Empty history is an exact zero
    intervention.
    """
    if query_tokens.ndim != 2 or candidate_tokens.ndim != 3 or history.ndim != 2:
        raise ValueError("C52 token/history ranks differ")
    if query_mask.shape != query_tokens.shape[:1]:
        raise ValueError("C52 query mask differs")
    if candidate_token_mask.shape != candidate_tokens.shape[:2]:
        raise ValueError("C52 candidate token mask differs")
    if query_tokens.shape[-1] != candidate_tokens.shape[-1] or query_tokens.shape[-1] != history.shape[-1]:
        raise ValueError("C52 hidden dimensions differ")
    if not bool(query_mask.any()) or not bool(candidate_token_mask.any(dim=-1).all()):
        raise ValueError("C52 requires nonempty query and candidate content")
    for name, value in (
        ("ridge", ridge),
        ("candidate_token_temperature", candidate_token_temperature),
        ("query_concept_temperature", query_concept_temperature),
        ("history_softmax_temperature", history_softmax_temperature),
        ("epsilon", epsilon),
    ):
        if value <= 0:
            raise ValueError(f"C52 {name} must be positive")

    q = F.normalize(query_tokens.float(), dim=-1, eps=epsilon)
    c = F.normalize(candidate_tokens.float(), dim=-1, eps=epsilon)
    h = F.normalize(history.float(), dim=-1, eps=epsilon)
    candidate_logits = torch.einsum("qd,cld->cql", q, c) / candidate_token_temperature
    token_mask = candidate_token_mask[:, None, :].expand_as(candidate_logits)
    token_weight = _masked_softmax(candidate_logits, token_mask, dim=-1)
    candidate_match = torch.einsum("cql,cld->cqd", token_weight, c)
    base_alignment = torch.einsum("cqd,qd->cq", candidate_match, q)
    query_axis_mask = query_mask[None, :].expand(candidate_tokens.shape[0], -1)
    base_attention = _masked_softmax(
        base_alignment / query_concept_temperature, query_axis_mask, dim=-1
    )

    if history.shape[0] == 0:
        zeros = base_alignment.new_zeros(base_alignment.shape)
        correction = base_alignment.new_zeros(candidate_tokens.shape[0])
        return ConceptAttentionOutput(
            primary_correction=correction,
            linearized_correction=correction.clone(),
            softmax_correction=correction.clone(),
            evidence=zeros,
            softmax_evidence=zeros.clone(),
            base_concept_attention=base_attention,
            factual_concept_attention=base_attention.clone(),
        )

    gram = h @ h.T
    normal = gram + torch.eye(len(h), dtype=gram.dtype, device=gram.device) * ridge
    alpha = torch.linalg.solve(normal, h @ q.T)
    query_write = (h.T @ alpha).T
    evidence = torch.einsum("cqd,qd->cq", candidate_match, query_write)

    history_logits = (h @ q.T) / history_softmax_temperature
    history_weight = torch.softmax(history_logits, dim=0)
    softmax_write = (h.T @ history_weight).T
    softmax_evidence = torch.einsum("cqd,qd->cq", candidate_match, softmax_write)

    base_partition = query_concept_temperature * _masked_logsumexp(
        base_alignment / query_concept_temperature, query_axis_mask, dim=-1
    )
    factual_partition = query_concept_temperature * _masked_logsumexp(
        (base_alignment + evidence) / query_concept_temperature,
        query_axis_mask,
        dim=-1,
    )
    softmax_partition = query_concept_temperature * _masked_logsumexp(
        (base_alignment + softmax_evidence) / query_concept_temperature,
        query_axis_mask,
        dim=-1,
    )
    factual_attention = _masked_softmax(
        (base_alignment + evidence) / query_concept_temperature,
        query_axis_mask,
        dim=-1,
    )
    return ConceptAttentionOutput(
        primary_correction=factual_partition - base_partition,
        linearized_correction=(base_attention * evidence).sum(dim=-1),
        softmax_correction=softmax_partition - base_partition,
        evidence=evidence,
        softmax_evidence=softmax_evidence,
        base_concept_attention=base_attention,
        factual_concept_attention=factual_attention,
    )
