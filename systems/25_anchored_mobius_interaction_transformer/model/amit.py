"""Anchored third-order interaction tokens as the only history residual path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


MODES = ("mobius3", "joint_delta", "pairwise_ch", "trilinear")


@dataclass(frozen=True)
class InteractionOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    anchor_scores: torch.Tensor
    active_request: torch.Tensor
    repeat_request: torch.Tensor


class AnchoredMobiusInteractionTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        ffn_dim: int,
        dropout: float,
        max_history: int,
        score_delta_max: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C25 mode: {mode}")
        if hidden_dim % heads:
            raise ValueError("C25 hidden_dim must divide heads")
        self.mode = mode
        self.max_history = int(max_history)
        self.score_delta_max = float(score_delta_max)
        self.query_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.candidate_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.history_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.potential_norm = nn.LayerNorm(hidden_dim)
        self.potential_in = nn.Linear(hidden_dim, ffn_dim)
        self.potential_out = nn.Linear(ffn_dim, hidden_dim)
        self.read_token = nn.Parameter(torch.zeros(hidden_dim))
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            block, num_layers=layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.read_token, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def _potential(self, value: torch.Tensor) -> torch.Tensor:
        return self.potential_out(
            torch.nn.functional.gelu(self.potential_in(self.potential_norm(value)))
        )

    def interaction_tokens(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        event_weights: torch.Tensor,
        *,
        mode: str | None = None,
    ) -> torch.Tensor:
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in MODES:
            raise ValueError(f"unknown C25 token mode: {selected_mode}")
        batch, candidate_count, input_dim = candidates.shape
        if query.shape != (batch, input_dim) or history.shape[:2] != (
            batch,
            event_weights.shape[1],
        ):
            raise ValueError("C25 interaction input shapes differ")
        if history.shape[-1] != input_dim or event_weights.shape[0] != batch:
            raise ValueError("C25 history/event shapes differ")
        q = self.query_projection(query)[:, None, None, :]
        c = self.candidate_projection(candidates)[:, :, None, :]
        h = self.history_projection(history)[:, None, :, :]
        zero = torch.zeros(1, 1, 1, q.shape[-1], dtype=q.dtype, device=q.device)

        # Every mode evaluates the same lattice branches plus the direct
        # product branch. Only their registered composition differs.
        p_qch = self._potential(q + c + h)
        p_qc = self._potential(q + c)
        p_qh = self._potential(q + h)
        p_ch = self._potential(c + h)
        p_q = self._potential(q)
        p_c = self._potential(c)
        p_h = self._potential(h)
        p_0 = self._potential(zero)
        p_product = self._potential(q * c * h)

        if selected_mode == "mobius3":
            tokens = p_qch - p_qc - p_qh - p_ch + p_q + p_c + p_h - p_0
        elif selected_mode == "joint_delta":
            tokens = p_qch - p_qc
        elif selected_mode == "pairwise_ch":
            tokens = p_ch - p_c - p_h + p_0
        else:
            tokens = p_product - p_0
        compute_match = p_qch + p_qc + p_qh + p_ch + p_q + p_c + p_h + p_0 + p_product
        tokens = tokens + compute_match * 0.0
        return tokens * event_weights[:, None, :, None].to(tokens.dtype)

    def forward(
        self,
        *,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        repeat_mask: torch.Tensor,
        event_weights: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        query_present: torch.Tensor | None = None,
        history_present_override: torch.Tensor | None = None,
    ) -> InteractionOutput:
        batch, candidate_count, input_dim = candidates.shape
        history_count = history.shape[1]
        if query.shape != (batch, input_dim) or history.shape != (
            batch,
            history_count,
            input_dim,
        ):
            raise ValueError("C25 query/candidate/history shape mismatch")
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("C25 candidate mask shape mismatch")
        if history_mask.shape != (batch, history_count):
            raise ValueError("C25 history mask shape mismatch")
        if repeat_mask.shape != (batch, candidate_count, history_count):
            raise ValueError("C25 repeat mask shape mismatch")
        if event_weights.shape != history_mask.shape:
            raise ValueError("C25 event weight shape mismatch")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != (
            candidate_mask.shape
        ):
            raise ValueError("C25 score shape mismatch")
        if history_count > self.max_history:
            raise ValueError("C25 history exceeds registered limit")
        if query_present is None:
            query_present = torch.ones(batch, dtype=torch.bool, device=query.device)

        tokens = self.interaction_tokens(query, candidates, history, event_weights)
        token_mask = history_mask[:, None, :].expand(-1, candidate_count, -1)
        token_mask = token_mask & candidate_mask[:, :, None]
        flat_tokens = tokens.reshape(batch * candidate_count, history_count, -1)
        flat_mask = token_mask.reshape(batch * candidate_count, history_count)
        read = self.read_token.view(1, 1, -1).expand(batch * candidate_count, -1, -1)
        sequence = torch.cat((read, flat_tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros(
                    batch * candidate_count, 1, dtype=torch.bool, device=query.device
                ),
                ~flat_mask,
            ),
            dim=1,
        )
        encoded = self.transformer(sequence, src_key_padding_mask=padding)[:, 0]
        raw = torch.tanh(self.output(self.output_norm(encoded)).squeeze(-1)).reshape(
            batch, candidate_count
        )
        raw = raw * candidate_mask.to(raw.dtype)

        repeat_request = (
            repeat_mask
            & candidate_mask[:, :, None]
            & history_mask[:, None, :]
        ).any(dim=(1, 2))
        history_present = history_mask.any(dim=-1)
        if history_present_override is not None:
            if history_present_override.shape != (batch,):
                raise ValueError("C25 history override shape mismatch")
            history_present = history_present & history_present_override
        active = history_present & ~repeat_request & query_present
        raw = raw * active[:, None].to(raw.dtype)
        weights = candidate_mask.to(raw.dtype)
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        centered = (raw - (raw * weights).sum(dim=-1, keepdim=True) / denominator) * weights
        correction = self.score_delta_max * centered
        anchor = torch.where(repeat_request[:, None], item_only_scores, base_scores)
        scores = (anchor + correction).masked_fill(~candidate_mask, 0.0)
        return InteractionOutput(
            scores=scores,
            correction=correction,
            raw_correction=raw,
            anchor_scores=anchor,
            active_request=active,
            repeat_request=repeat_request,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
