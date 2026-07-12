"""Diagonal-plus-skew temporal affinity operator for personalized ranking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.bool()
    masked = logits.masked_fill(~mask, -torch.inf)
    any_valid = mask.any(dim=dim, keepdim=True)
    safe = torch.where(any_valid, masked, torch.zeros_like(masked))
    probabilities = torch.softmax(safe, dim=dim)
    return torch.where(any_valid, probabilities.masked_fill(~mask, 0.0), torch.zeros_like(probabilities))


def lag_terms(
    query_trace: torch.Tensor,
    candidate_trace: torch.Tensor,
    history_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return diagonal, forward and reverse affinity contractions."""

    if query_trace.ndim != 2 or candidate_trace.ndim != 3:
        raise ValueError("expected query [B,H] and candidate [B,C,H] traces")
    if query_trace.shape[0] != candidate_trace.shape[0] or query_trace.shape[1] != candidate_trace.shape[2]:
        raise ValueError("trace shape mismatch")
    if history_mask.shape != query_trace.shape:
        raise ValueError("history mask mismatch")
    active = history_mask.to(query_trace.dtype)
    a = query_trace * active
    b = candidate_trace * active.unsqueeze(1)
    diagonal = (a.unsqueeze(1) * b).sum(dim=-1)
    if a.shape[1] < 2:
        zero = torch.zeros_like(diagonal)
        return diagonal, zero, zero
    adjacent = (history_mask[:, :-1] & history_mask[:, 1:]).to(a.dtype)
    forward = (
        a[:, None, :-1] * b[:, :, 1:] * adjacent[:, None, :]
    ).sum(dim=-1)
    reverse = (
        a[:, None, 1:] * b[:, :, :-1] * adjacent[:, None, :]
    ).sum(dim=-1)
    return diagonal, forward, reverse


@dataclass(frozen=True)
class OLTOutput:
    scores: torch.Tensor
    base_scores: torch.Tensor
    evidence: torch.Tensor
    diagonal: torch.Tensor
    forward: torch.Tensor
    reverse: torch.Tensor
    oriented_component: torch.Tensor
    query_trace: torch.Tensor
    candidate_trace: torch.Tensor


class OLTRanker(nn.Module):
    VALID_MODES = frozenset({"oriented", "diagonal", "forward", "symmetric", "free_signed"})

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        nhead: int,
        layers: int,
        ffn_dim: int,
        history_slots: int,
        dropout: float,
        affinity_dim: int,
        temperature: float,
        identity_bias: float,
        evidence_scale: float,
        lag_scale_max: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(mode)
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.mode = mode
        self.history_slots = int(history_slots)
        self.temperature = float(temperature)
        self.identity_bias = float(identity_bias)
        self.evidence_scale = float(evidence_scale)
        self.lag_scale_max = float(lag_scale_max)

        self.input_projection = nn.Linear(input_dim, d_model)
        self.query_type = nn.Parameter(torch.empty(d_model))
        self.candidate_type = nn.Parameter(torch.empty(d_model))
        self.history_type = nn.Parameter(torch.empty(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_candidate_transformer = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(d_model)
        )
        self.history_encoder = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )
        self.query_affinity = nn.Linear(d_model, affinity_dim, bias=False)
        self.candidate_affinity = nn.Linear(d_model, affinity_dim, bias=False)
        self.history_query_key = nn.Linear(d_model, affinity_dim, bias=False)
        self.history_candidate_key = nn.Linear(d_model, affinity_dim, bias=False)
        self.evidence_write = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.base_head = nn.Linear(d_model, 1)
        self.diagonal_raw = nn.Parameter(torch.tensor(0.0))
        self.lag_forward_raw = nn.Parameter(torch.tensor(0.0))
        self.lag_reverse_raw = nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_type, std=0.02)
        nn.init.normal_(self.candidate_type, std=0.02)
        nn.init.normal_(self.history_type, std=0.02)
        nn.init.xavier_uniform_(self.base_head.weight)
        nn.init.zeros_(self.base_head.bias)

    def _coefficients(self, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diagonal = F.softplus(self.diagonal_raw) + 1e-4
        forward_raw = self.lag_scale_max * torch.sigmoid(self.lag_forward_raw)
        reverse_raw = self.lag_scale_max * torch.sigmoid(self.lag_reverse_raw)
        if mode == "oriented":
            lag = 0.5 * (forward_raw + reverse_raw)
            return diagonal, lag, -lag
        if mode == "diagonal":
            zero = forward_raw * 0.0 + reverse_raw * 0.0
            return diagonal, zero, zero
        if mode == "forward":
            return diagonal, forward_raw, reverse_raw * 0.0
        if mode == "symmetric":
            lag = 0.25 * (forward_raw + reverse_raw)
            return diagonal, lag, lag
        if mode == "free_signed":
            return (
                diagonal,
                self.lag_scale_max * torch.tanh(self.lag_forward_raw),
                self.lag_scale_max * torch.tanh(self.lag_reverse_raw),
            )
        raise ValueError(mode)

    def query_only(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_token = self.input_projection(query).unsqueeze(1) + self.query_type
        candidate_tokens = self.input_projection(candidates) + self.candidate_type
        tokens = torch.cat((query_token, candidate_tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros(query.shape[0], 1, dtype=torch.bool, device=query.device),
                ~candidate_mask.bool(),
            ),
            dim=1,
        )
        encoded = self.query_candidate_transformer(tokens, src_key_padding_mask=padding)
        query_state = encoded[:, 0]
        candidate_states = encoded[:, 1:]
        base = self.base_head(candidate_states).squeeze(-1).masked_fill(~candidate_mask.bool(), -1e4)
        return base, query_state, candidate_states

    def forward(
        self,
        *,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        identity_relation: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> OLTOutput:
        active_mode = self.mode if mode is None else mode
        if active_mode not in self.VALID_MODES:
            raise ValueError(active_mode)
        batch, candidate_count, _ = candidates.shape
        if history.shape[1] != self.history_slots:
            raise ValueError("history slot mismatch")
        if identity_relation.shape != (batch, candidate_count, self.history_slots):
            raise ValueError("identity relation shape mismatch")
        if candidate_mask is None:
            candidate_mask = torch.ones(batch, candidate_count, dtype=torch.bool, device=query.device)
        base, query_state, candidate_states = self.query_only(query, candidates, candidate_mask)
        history_present = history_mask.bool().any(dim=1)
        history_states = self.input_projection(history) + self.history_type
        history_states = history_states + self.history_encoder(history_states)

        q = F.normalize(self.query_affinity(query_state), dim=-1)
        c = F.normalize(self.candidate_affinity(candidate_states), dim=-1)
        hq = F.normalize(self.history_query_key(history_states), dim=-1)
        hc = F.normalize(self.history_candidate_key(history_states), dim=-1)
        query_logits = torch.einsum("bd,bhd->bh", q, hq) / self.temperature
        candidate_logits = torch.einsum("bcd,bhd->bch", c, hc) / self.temperature
        candidate_logits = candidate_logits + self.identity_bias * identity_relation.to(candidate_logits.dtype)
        query_trace = masked_softmax(query_logits, history_mask.bool(), dim=-1)
        candidate_trace = masked_softmax(
            candidate_logits,
            history_mask[:, None, :].expand_as(candidate_logits),
            dim=-1,
        )
        diagonal, forward, reverse = lag_terms(query_trace, candidate_trace, history_mask)
        diagonal_coefficient, forward_coefficient, reverse_coefficient = self._coefficients(active_mode)
        # Keep the exact raw orientation available for reversal diagnostics;
        # its coefficient is applied only through the selected mode below.
        oriented_component = forward - reverse
        evidence = (
            diagonal_coefficient * diagonal
            + forward_coefficient * forward
            + reverse_coefficient * reverse
        )
        valid_count = candidate_mask.to(evidence.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        evidence = evidence - (
            evidence.masked_fill(~candidate_mask.bool(), 0.0).sum(dim=1, keepdim=True) / valid_count
        )
        evidence = evidence.masked_fill(~candidate_mask.bool(), 0.0)
        write_gate = self.evidence_scale * torch.tanh(evidence).unsqueeze(-1)
        personalized_states = candidate_states + write_gate * self.evidence_write(candidate_states)
        personalized = self.base_head(personalized_states).squeeze(-1)
        personalized = personalized.masked_fill(~candidate_mask.bool(), -1e4)
        scores = torch.where(history_present.unsqueeze(1), personalized, base)
        zero = torch.zeros_like(diagonal)
        return OLTOutput(
            scores=scores,
            base_scores=base,
            evidence=torch.where(history_present.unsqueeze(1), evidence, zero),
            diagonal=torch.where(history_present.unsqueeze(1), diagonal, zero),
            forward=torch.where(history_present.unsqueeze(1), forward, zero),
            reverse=torch.where(history_present.unsqueeze(1), reverse, zero),
            oriented_component=torch.where(history_present.unsqueeze(1), oriented_component, zero),
            query_trace=query_trace,
            candidate_trace=candidate_trace,
        )
