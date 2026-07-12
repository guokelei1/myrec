"""Minimal Candidate-Contrastive Evidence Budget (CCEB) operator.

This module is the architecture kernel for C05.  It replaces one ordinary
history-to-candidate attention update inside a Transformer ranker.  The
`CCEBProbeRanker` wrapper deliberately uses an externally registered base score
only for the train-internal learnability falsifier; it is not the final proposed
system described in the candidate notes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(value))


@dataclass
class CCEBOutput:
    """Intermediate states needed by the frozen mechanism diagnostics."""

    updated_candidates: torch.Tensor
    update: torch.Tensor
    raw_alignment: torch.Tensor
    contrast: torch.Tensor
    evidence_weights: torch.Tensor
    evidence_l1_mass: torch.Tensor
    history_present: torch.Tensor


@dataclass
class CCEBProbeOutput:
    """Output of the adapter-only train-internal learnability probe."""

    scores: torch.Tensor
    base_scores: torch.Tensor
    score_delta: torch.Tensor
    mechanism: CCEBOutput


class CandidateContrastiveEvidenceBlock(nn.Module):
    """A signed, candidate-centered, L1-budgeted history residual.

    Shapes:
      query:          [B, D]
      candidates:     [B, C, D]
      history:        [B, H, D]
      candidate_mask: [B, C]
      history_mask:   [B, H]
      exact_match:    [B, C, H]

    The block is permutation equivariant over candidates.  With no valid
    history, one valid candidate, or candidate-common event alignment, its
    update is algebraically zero.
    """

    def __init__(
        self,
        input_dim: int,
        evidence_dim: int,
        *,
        dead_zone: float = 0.1,
        exact_bias_init: float = 1.0,
        residual_scale_max: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or evidence_dim <= 0:
            raise ValueError("input_dim and evidence_dim must be positive")
        if dead_zone < 0.0:
            raise ValueError("dead_zone must be non-negative")
        if residual_scale_max <= 0.0:
            raise ValueError("residual_scale_max must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = int(input_dim)
        self.evidence_dim = int(evidence_dim)
        self.residual_scale_max = float(residual_scale_max)

        self.query_norm = nn.LayerNorm(input_dim)
        self.candidate_norm = nn.LayerNorm(input_dim)
        self.history_norm = nn.LayerNorm(input_dim)
        self.query_projection = nn.Linear(input_dim, evidence_dim, bias=False)
        self.candidate_projection = nn.Linear(
            input_dim, evidence_dim, bias=False
        )
        self.history_key_projection = nn.Linear(
            input_dim, evidence_dim, bias=False
        )
        self.history_value_projection = nn.Linear(
            input_dim, input_dim, bias=False
        )
        self.output_projection = nn.Linear(input_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.raw_exact_bias = nn.Parameter(
            torch.tensor(_inverse_softplus(exact_bias_init), dtype=torch.float32)
        )
        # Zero initialization makes the personalized path a true no-op before
        # evidence learning starts.  This is a global step size, not a router.
        self.raw_residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "dead_zone", torch.tensor(float(dead_zone), dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        # Keep the explicitly defined scalar initializations intact.

    @property
    def exact_bias(self) -> torch.Tensor:
        return F.softplus(self.raw_exact_bias)

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.residual_scale_max * torch.tanh(self.raw_residual_scale)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_inputs(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        exact_match: torch.Tensor,
    ) -> None:
        if query.ndim != 2 or query.shape[-1] != self.input_dim:
            raise ValueError(f"query must have shape [B, {self.input_dim}]")
        if candidates.ndim != 3 or candidates.shape[-1] != self.input_dim:
            raise ValueError(
                f"candidates must have shape [B, C, {self.input_dim}]"
            )
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError(f"history must have shape [B, H, {self.input_dim}]")
        batch, candidate_count, _ = candidates.shape
        if query.shape[0] != batch or history.shape[0] != batch:
            raise ValueError("query/candidate/history batch dimensions must match")
        history_count = history.shape[1]
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("candidate_mask shape mismatch")
        if history_mask.shape != (batch, history_count):
            raise ValueError("history_mask shape mismatch")
        if exact_match.shape != (batch, candidate_count, history_count):
            raise ValueError("exact_match shape mismatch")
        if candidate_count == 0:
            raise ValueError("at least one candidate column is required")
        if not bool(candidate_mask.bool().any(dim=1).all().item()):
            raise ValueError("every request must contain at least one valid candidate")

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        exact_match: torch.Tensor,
    ) -> CCEBOutput:
        self._validate_inputs(
            query,
            candidates,
            history,
            candidate_mask,
            history_mask,
            exact_match,
        )
        candidate_mask = candidate_mask.bool()
        history_mask = history_mask.bool()
        pair_mask = candidate_mask[:, :, None] & history_mask[:, None, :]

        q = self.query_projection(self.query_norm(query))
        c = self.candidate_projection(self.candidate_norm(candidates))
        h_key = self.history_key_projection(self.history_norm(history))
        joint = torch.tanh(q[:, None, :] + c)
        raw_alignment = torch.einsum("bcd,bhd->bch", joint, h_key)
        raw_alignment = raw_alignment / math.sqrt(self.evidence_dim)
        raw_alignment = raw_alignment + self.exact_bias.to(
            raw_alignment.dtype
        ) * exact_match.to(raw_alignment.dtype)
        raw_alignment = raw_alignment * pair_mask.to(raw_alignment.dtype)

        valid_candidate_count = candidate_mask.sum(dim=1, keepdim=True).clamp_min(1)
        event_mean = raw_alignment.sum(dim=1) / valid_candidate_count.to(
            raw_alignment.dtype
        )
        contrast = (raw_alignment - event_mean[:, None, :]) * pair_mask.to(
            raw_alignment.dtype
        )

        threshold = self.dead_zone.to(dtype=contrast.dtype, device=contrast.device)
        signed_evidence = F.relu(contrast - threshold) - F.relu(
            -contrast - threshold
        )
        signed_evidence = signed_evidence * pair_mask.to(signed_evidence.dtype)
        denominator = 1.0 + signed_evidence.abs().sum(dim=-1, keepdim=True)
        evidence_weights = signed_evidence / denominator

        history_values = self.history_value_projection(self.history_norm(history))
        aggregate = torch.einsum("bch,bhd->bcd", evidence_weights, history_values)
        update = self.residual_scale.to(aggregate.dtype) * self.output_projection(
            self.dropout(aggregate)
        )
        history_present = history_mask.any(dim=-1)
        valid_update = candidate_mask[:, :, None] & history_present[:, None, None]
        update = update * valid_update.to(update.dtype)
        updated_candidates = candidates + update
        evidence_l1_mass = evidence_weights.abs().sum(dim=-1)

        return CCEBOutput(
            updated_candidates=updated_candidates,
            update=update,
            raw_alignment=raw_alignment,
            contrast=contrast,
            evidence_weights=evidence_weights,
            evidence_l1_mass=evidence_l1_mass,
            history_present=history_present,
        )


class CCEBProbeRanker(nn.Module):
    """Adapter-only probe for the non-repeat signal-existence gate.

    The registered base score is used as an immutable coordinate.  This class
    is intentionally ineligible as the final proposed system; see proposal.md.
    """

    def __init__(
        self,
        input_dim: int,
        evidence_dim: int,
        *,
        dead_zone: float = 0.1,
        exact_bias_init: float = 1.0,
        residual_scale_max: float = 0.5,
        score_delta_max: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if score_delta_max <= 0.0:
            raise ValueError("score_delta_max must be positive")
        self.score_delta_max = float(score_delta_max)
        self.evidence_block = CandidateContrastiveEvidenceBlock(
            input_dim=input_dim,
            evidence_dim=evidence_dim,
            dead_zone=dead_zone,
            exact_bias_init=exact_bias_init,
            residual_scale_max=residual_scale_max,
            dropout=dropout,
        )
        self.score_norm = nn.LayerNorm(input_dim)
        self.score_head = nn.Linear(input_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.score_head.weight)

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        exact_match: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> CCEBProbeOutput:
        if base_scores.shape != candidates.shape[:2]:
            raise ValueError("base_scores shape must match [B, C]")
        mechanism = self.evidence_block(
            query=query,
            candidates=candidates,
            history=history,
            candidate_mask=candidate_mask,
            history_mask=history_mask,
            exact_match=exact_match,
        )
        personalized_logit = self.score_head(
            self.score_norm(mechanism.updated_candidates)
        ).squeeze(-1)
        base_hidden_logit = self.score_head(self.score_norm(candidates)).squeeze(-1)
        raw_delta = personalized_logit - base_hidden_logit
        bounded_delta = self.score_delta_max * torch.tanh(raw_delta)
        active = mechanism.history_present[:, None] & candidate_mask.bool()
        bounded_delta = bounded_delta * active.to(bounded_delta.dtype)
        immutable_base = base_scores.detach()
        personalized_scores = immutable_base + bounded_delta.to(immutable_base.dtype)
        # `where` makes the evidence-empty fallback exact even if future kernels
        # change floating-point evaluation details in the delta path.
        scores = torch.where(active, personalized_scores, immutable_base)
        score_delta = scores - immutable_base
        return CCEBProbeOutput(
            scores=scores,
            base_scores=immutable_base,
            score_delta=score_delta,
            mechanism=mechanism,
        )
