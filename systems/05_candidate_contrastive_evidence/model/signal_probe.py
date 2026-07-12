"""Minimal ordinary target-attention probe for C05 G2a.

This is deliberately a nearest-neighbor falsifier, not the proposed CCEB
mechanism.  It receives frozen, scope-correct D2 states and asks only whether
non-repeat history can improve ranking with one shallow attention update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SignalProbeOutput:
    scores: torch.Tensor
    base_scores: torch.Tensor
    score_delta: torch.Tensor
    attention_weights: torch.Tensor
    updated_candidates: torch.Tensor
    history_present: torch.Tensor


class TargetAttentionSignalProbe(nn.Module):
    """One query/candidate-conditioned positive history-attention update."""

    def __init__(
        self,
        input_dim: int,
        evidence_dim: int,
        *,
        score_delta_max: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or evidence_dim <= 0:
            raise ValueError("input_dim and evidence_dim must be positive")
        if score_delta_max <= 0.0:
            raise ValueError("score_delta_max must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.evidence_dim = int(evidence_dim)
        self.score_delta_max = float(score_delta_max)

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
        self.score_norm = nn.LayerNorm(input_dim)
        self.score_head = nn.Linear(input_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        # The active G2a probe must begin at the registered D2p coordinate.
        # After the first optimizer step opens this residual, gradients reach
        # the upstream attention path on the second and later steps.
        nn.init.zeros_(self.output_projection.weight)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        history_event_weights: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> SignalProbeOutput:
        self._validate(
            query,
            candidates,
            history,
            candidate_mask,
            history_mask,
            history_event_weights,
            base_scores,
        )
        candidate_mask = candidate_mask.bool()
        history_mask = history_mask.bool()
        history_present = history_mask.any(dim=-1)

        # Padded storage may contain arbitrary values.  Remove it before any
        # normalization/projection so NaN * 0 can never contaminate valid rows.
        safe_candidates = torch.where(
            candidate_mask[:, :, None], candidates, torch.zeros_like(candidates)
        )
        safe_history = torch.where(
            history_mask[:, :, None], history, torch.zeros_like(history)
        )
        safe_event_weights = torch.where(
            history_mask,
            history_event_weights,
            torch.zeros_like(history_event_weights),
        )

        q = F.normalize(
            self.query_projection(self.query_norm(query)), dim=-1, eps=1e-6
        )
        c = self.candidate_projection(self.candidate_norm(safe_candidates))
        joint = F.normalize(torch.tanh(q[:, None, :] + c), dim=-1, eps=1e-6)
        history_key = F.normalize(
            self.history_key_projection(self.history_norm(safe_history)),
            dim=-1,
            eps=1e-6,
        )
        alignment = torch.einsum("bcd,bhd->bch", joint, history_key)
        # The frozen recency/action weight is an attention prior.  A shuffle
        # audit permutes event identity while keeping position weights fixed.
        prior = torch.log(safe_event_weights.clamp_min(1e-6))
        alignment = alignment + prior[:, None, :]
        pair_mask = candidate_mask[:, :, None] & history_mask[:, None, :]
        attention = self._safe_masked_softmax(alignment, pair_mask)

        history_value = self.history_value_projection(
            self.history_norm(safe_history)
        )
        context = torch.einsum("bch,bhd->bcd", attention, history_value)
        update = self.output_projection(self.dropout(context))
        active = candidate_mask[:, :, None] & history_present[:, None, None]
        update = torch.where(active, update, torch.zeros_like(update))
        updated = safe_candidates + update

        personalized_logit = self.score_head(self.score_norm(updated)).squeeze(-1)
        unpersonalized_logit = self.score_head(
            self.score_norm(safe_candidates)
        ).squeeze(-1)
        delta = self.score_delta_max * torch.tanh(
            personalized_logit - unpersonalized_logit
        )
        active_score = candidate_mask & history_present[:, None]
        delta = torch.where(active_score, delta, torch.zeros_like(delta))

        immutable_base = base_scores.detach()
        personalized_scores = immutable_base + delta.to(immutable_base.dtype)
        scores = torch.where(active_score, personalized_scores, immutable_base)
        return SignalProbeOutput(
            scores=scores,
            base_scores=immutable_base,
            score_delta=scores - immutable_base,
            attention_weights=attention,
            updated_candidates=updated,
            history_present=history_present,
        )

    @staticmethod
    def _safe_masked_softmax(
        logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        masked = logits.masked_fill(~mask, -torch.inf)
        has_value = mask.any(dim=-1, keepdim=True)
        row_max = masked.amax(dim=-1, keepdim=True)
        row_max = torch.where(has_value, row_max, torch.zeros_like(row_max))
        exponent = torch.exp(logits - row_max) * mask.to(logits.dtype)
        denominator = exponent.sum(dim=-1, keepdim=True)
        return torch.where(
            has_value,
            exponent / denominator.clamp_min(torch.finfo(logits.dtype).tiny),
            torch.zeros_like(exponent),
        )

    def _validate(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        history_event_weights: torch.Tensor,
        base_scores: torch.Tensor,
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
        history_count = history.shape[1]
        if query.shape[0] != batch or history.shape[0] != batch:
            raise ValueError("batch dimensions must match")
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("candidate_mask shape mismatch")
        if history_mask.shape != (batch, history_count):
            raise ValueError("history_mask shape mismatch")
        if history_event_weights.shape != (batch, history_count):
            raise ValueError("history_event_weights shape mismatch")
        if base_scores.shape != (batch, candidate_count):
            raise ValueError("base_scores shape mismatch")
        c_mask = candidate_mask.bool()
        h_mask = history_mask.bool()
        if not bool(c_mask.any(dim=-1).all().item()):
            raise ValueError("every request needs at least one valid candidate")
        if not bool(torch.isfinite(query).all().item()):
            raise ValueError("query contains non-finite values")
        if not bool(torch.isfinite(candidates[c_mask]).all().item()):
            raise ValueError("valid candidates contain non-finite values")
        if h_mask.any() and not bool(torch.isfinite(history[h_mask]).all().item()):
            raise ValueError("valid history contains non-finite values")
        if h_mask.any() and not bool(
            torch.isfinite(history_event_weights[h_mask]).all().item()
        ):
            raise ValueError("valid history weights contain non-finite values")
        if h_mask.any() and not bool((history_event_weights[h_mask] > 0).all().item()):
            raise ValueError("valid history weights must be positive")
        if not bool(torch.isfinite(base_scores[c_mask]).all().item()):
            raise ValueError("valid base scores contain non-finite values")
