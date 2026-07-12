"""Capacity-matched nearest controls for the C06 mechanism gate."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CenteredCrossAttentionOutput:
    scores: torch.Tensor
    base_scores: torch.Tensor
    conservative_score_delta: torch.Tensor
    applied_score_delta: torch.Tensor
    raw_unary_evidence: torch.Tensor
    centered_unary_evidence: torch.Tensor
    attention_weights: torch.Tensor
    history_present: torch.Tensor
    residual_scale: torch.Tensor


class CenteredCrossAttentionProbeRanker(nn.Module):
    """Ordinary candidate-to-history attention with zero-sum final evidence.

    Its trainable parameter count exactly matches the local-Hodge probe when
    both use the same input and evidence dimensions. Thus it separates the new
    trust operator from the simpler benefit of removing score common mode.
    """

    def __init__(
        self,
        input_dim: int,
        evidence_dim: int,
        *,
        score_delta_max: float = 1.0,
        compute_rounds: int = 4,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or evidence_dim <= 0:
            raise ValueError("input_dim and evidence_dim must be positive")
        if score_delta_max <= 0.0:
            raise ValueError("score_delta_max must be positive")
        if compute_rounds <= 0:
            raise ValueError("compute_rounds must be positive")
        self.input_dim = int(input_dim)
        self.evidence_dim = int(evidence_dim)
        self.score_delta_max = float(score_delta_max)
        self.compute_rounds = int(compute_rounds)

        self.query_norm = nn.LayerNorm(input_dim)
        self.candidate_norm = nn.LayerNorm(input_dim)
        self.history_norm = nn.LayerNorm(input_dim)
        self.query_projection = nn.Linear(input_dim, evidence_dim, bias=False)
        self.candidate_projection = nn.Linear(input_dim, evidence_dim, bias=False)
        self.history_projection = nn.Linear(input_dim, evidence_dim, bias=False)
        self.attention_projection = nn.Linear(
            evidence_dim, evidence_dim, bias=False
        )
        self.value_projection = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.raw_residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.score_delta_max * torch.tanh(self.raw_residual_scale)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _safe_masked_softmax(
        logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if logits.shape[-1] == 0:
            return torch.zeros_like(logits)
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

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        history_prior: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> CenteredCrossAttentionOutput:
        self._validate(
            query,
            candidates,
            history,
            candidate_mask,
            history_mask,
            history_prior,
            base_scores,
        )
        candidate_mask = candidate_mask.bool()
        history_mask = history_mask.bool()
        history_present = history_mask.any(dim=-1)
        safe_candidates = torch.where(
            candidate_mask[:, :, None], candidates, torch.zeros_like(candidates)
        )
        safe_history = torch.where(
            history_mask[:, :, None], history, torch.zeros_like(history)
        )
        safe_prior = torch.where(
            history_mask, history_prior, torch.zeros_like(history_prior)
        )

        q = self.query_projection(self.query_norm(query))
        c = self.candidate_projection(self.candidate_norm(safe_candidates))
        h = self.history_projection(self.history_norm(safe_history))
        target = torch.tanh(q[:, None, :] + c)
        pair = torch.tanh(
            target[:, :, None, :] + h[:, None, :, :]
        )
        attention_state = pair
        value_state = pair
        # Four tied rounds contribute the same dominant 16*C*H*r^2 FLOPs as
        # the primary's two factor maps plus six low-rank Hodge contractions.
        # The recurrence is load-bearing: its final states define both logits
        # and values, so no dormant computation is used for FLOP matching.
        for _ in range(self.compute_rounds):
            attention_state = torch.tanh(
                self.attention_projection(attention_state)
            )
            value_state = torch.tanh(self.value_projection(value_state))
        logits = attention_state.mean(dim=-1) * math.sqrt(
            float(self.evidence_dim)
        )
        logits = logits + torch.log(safe_prior.clamp_min(1e-6))[:, None, :]
        pair_mask = candidate_mask[:, :, None] & history_mask[:, None, :]
        attention = self._safe_masked_softmax(logits, pair_mask)
        pair_value = torch.einsum(
            "bchr,bcr->bch", value_state, target
        ) / math.sqrt(float(self.evidence_dim))
        raw = torch.tanh((attention * pair_value).sum(dim=-1))
        active = candidate_mask & history_present[:, None]
        raw = torch.where(active, raw, torch.zeros_like(raw))
        valid_count = candidate_mask.sum(dim=-1).clamp_min(1)
        mean = raw.sum(dim=-1, keepdim=True) / valid_count[:, None].to(raw.dtype)
        centered = torch.where(active, 0.5 * (raw - mean), torch.zeros_like(raw))

        with torch.autocast(device_type=query.device.type, enabled=False):
            scale = self.residual_scale.float()
            conservative_delta = scale * centered.float()
            immutable_base = base_scores.detach()
            scores = torch.where(
                active,
                immutable_base + conservative_delta,
                immutable_base,
            )
            applied_delta = torch.where(
                active, scores - immutable_base, torch.zeros_like(immutable_base)
            )
        return CenteredCrossAttentionOutput(
            scores=scores,
            base_scores=immutable_base,
            conservative_score_delta=conservative_delta,
            applied_score_delta=applied_delta,
            raw_unary_evidence=raw,
            centered_unary_evidence=centered,
            attention_weights=attention,
            history_present=history_present,
            residual_scale=scale,
        )

    def _validate(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        history_prior: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> None:
        if query.ndim != 2 or query.shape[-1] != self.input_dim:
            raise ValueError(f"query must have shape [B, {self.input_dim}]")
        if candidates.ndim != 3 or candidates.shape[-1] != self.input_dim:
            raise ValueError("candidate shape mismatch")
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError("history shape mismatch")
        batch, candidate_count, _ = candidates.shape
        history_count = history.shape[1]
        if candidate_count == 0 or query.shape[0] != batch or history.shape[0] != batch:
            raise ValueError("invalid batch or empty candidate axis")
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("candidate_mask shape mismatch")
        if history_mask.shape != (batch, history_count):
            raise ValueError("history_mask shape mismatch")
        if history_prior.shape != (batch, history_count):
            raise ValueError("history_prior shape mismatch")
        if base_scores.shape != (batch, candidate_count) or base_scores.dtype != torch.float32:
            raise ValueError("base_scores must be FP32 [B, C]")
        c_mask = candidate_mask.bool()
        h_mask = history_mask.bool()
        if not bool(c_mask.any(dim=-1).all().item()):
            raise ValueError("every request needs a valid candidate")
        if not bool(torch.isfinite(query).all().item()):
            raise ValueError("query contains non-finite values")
        if not bool(torch.isfinite(candidates[c_mask]).all().item()):
            raise ValueError("valid candidates contain non-finite values")
        if h_mask.any() and not bool(torch.isfinite(history[h_mask]).all().item()):
            raise ValueError("valid history contains non-finite values")
        if h_mask.any():
            prior = history_prior[h_mask]
            if not bool(torch.isfinite(prior).all().item()):
                raise ValueError("valid history prior contains non-finite values")
            if not bool(((prior > 0.0) & (prior <= 1.0)).all().item()):
                raise ValueError("valid history prior must lie in (0, 1]")
        if not bool(torch.isfinite(base_scores[c_mask]).all().item()):
            raise ValueError("valid base scores contain non-finite values")
