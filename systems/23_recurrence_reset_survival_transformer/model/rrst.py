"""Candidate-local last-recurrence reset inside a causal Transformer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


MODES = (
    "reset_suffix",
    "unreset_history",
    "orderless_suffix",
    "query_independent",
)


def masked_zscore(
    values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Request-local z-score with padded entries fixed to zero."""

    if values.shape != mask.shape:
        raise ValueError("masked_zscore shape mismatch")
    weights = mask.to(values.dtype)
    count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * weights).sum(dim=-1, keepdim=True) / count
    centered = (values - mean) * weights
    variance = centered.square().sum(dim=-1, keepdim=True) / count
    scale = variance.sqrt()
    safe = torch.where(scale > eps, scale, torch.ones_like(scale))
    return (centered / safe) * weights


@dataclass(frozen=True)
class RRSTOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    item_only_scores: torch.Tensor
    repeat_candidate_mask: torch.Tensor
    allowed_event_mask: torch.Tensor
    last_exact_position: torch.Tensor


class RecurrenceResetSurvivalTransformer(nn.Module):
    """Transform only the suffix after a candidate's last exact recurrence."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        ffn_dim: int,
        max_history: int,
        dropout: float,
        score_delta_max: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown RRST mode: {mode}")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.mode = mode
        self.max_history = int(max_history)
        self.score_delta_max = float(score_delta_max)

        self.query_reset = nn.Linear(input_dim, hidden_dim, bias=False)
        self.candidate_reset = nn.Linear(input_dim, hidden_dim, bias=False)
        self.anchor_reset = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_event = nn.Linear(input_dim, hidden_dim, bias=False)
        self.candidate_event = nn.Linear(input_dim, hidden_dim, bias=False)
        self.history_event = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_read = nn.Linear(input_dim, hidden_dim, bias=False)
        self.candidate_read = nn.Linear(input_dim, hidden_dim, bias=False)
        self.event_features = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reset_features = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position = nn.Embedding(max_history + 2, hidden_dim)
        self.reset_token = nn.Parameter(torch.zeros(hidden_dim))
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
            block,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.reset_token, std=0.02)
        nn.init.normal_(self.read_token, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _last_exact(
        repeat_mask: torch.Tensor, history_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if repeat_mask.ndim != 3 or history_mask.ndim != 2:
            raise ValueError("RRST recurrence masks must be [B,C,H] and [B,H]")
        if repeat_mask.shape[0] != history_mask.shape[0] or repeat_mask.shape[2] != (
            history_mask.shape[1]
        ):
            raise ValueError("RRST recurrence mask shape mismatch")
        valid = repeat_mask & history_mask[:, None, :]
        positions = torch.arange(
            repeat_mask.shape[-1], device=repeat_mask.device, dtype=torch.long
        )
        last = torch.where(
            valid,
            positions.view(1, 1, -1),
            torch.full((), -1, device=repeat_mask.device, dtype=torch.long),
        ).amax(dim=-1)
        return last, last.ge(0)

    def _allowed_events(
        self,
        *,
        last: torch.Tensor,
        repeat_candidate: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_count = history_mask.shape[-1]
        positions = torch.arange(history_count, device=history_mask.device).view(1, 1, -1)
        if self.mode == "unreset_history":
            allowed = history_mask[:, None, :] & repeat_candidate[:, :, None]
            coordinates = positions.expand_as(allowed).clamp_max(self.max_history) + 1
        else:
            allowed = (
                history_mask[:, None, :]
                & repeat_candidate[:, :, None]
                & positions.ge(last[:, :, None])
            )
            coordinates = (positions - last[:, :, None]).clamp(0, self.max_history) + 1
        coordinates = torch.where(allowed, coordinates, torch.zeros_like(coordinates))
        return allowed, coordinates

    @staticmethod
    def _gather_anchor(history: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        safe = last.clamp_min(0)
        batch, candidates = safe.shape
        dim = history.shape[-1]
        expanded = history[:, None, :, :].expand(-1, candidates, -1, -1)
        return expanded.gather(
            2, safe[:, :, None, None].expand(batch, candidates, 1, dim)
        ).squeeze(2)

    @staticmethod
    def _apply_suffix_permutation(
        values: torch.Tensor,
        allowed: torch.Tensor,
        suffix_permutation: torch.Tensor | None,
    ) -> torch.Tensor:
        if suffix_permutation is None:
            return values
        if suffix_permutation.shape != allowed.shape:
            raise ValueError("suffix permutation shape mismatch")
        if suffix_permutation.dtype != torch.long:
            raise ValueError("suffix permutation must use torch.long")
        history_count = values.shape[-2]
        if bool((suffix_permutation < 0).any()) or bool(
            (suffix_permutation >= history_count).any()
        ):
            raise ValueError("suffix permutation index out of bounds")
        gathered = values.gather(
            2, suffix_permutation[..., None].expand(*suffix_permutation.shape, values.shape[-1])
        )
        return torch.where(allowed[..., None], gathered, values)

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
        suffix_permutation: torch.Tensor | None = None,
        corrupt_preanchor: bool = False,
    ) -> RRSTOutput:
        batch, candidate_count, input_dim = candidates.shape
        history_count = history.shape[1]
        if query.shape != (batch, input_dim) or history.shape[0] != batch:
            raise ValueError("RRST embedding shape mismatch")
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("RRST candidate mask shape mismatch")
        if history_mask.shape != (batch, history_count):
            raise ValueError("RRST history mask shape mismatch")
        if repeat_mask.shape != (batch, candidate_count, history_count):
            raise ValueError("RRST exact identity mask shape mismatch")
        if event_weights.shape != (batch, history_count):
            raise ValueError("RRST event weight shape mismatch")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != (
            candidate_mask.shape
        ):
            raise ValueError("RRST score shape mismatch")
        if history_count > self.max_history:
            raise ValueError("RRST history exceeds max_history")
        if query_present is None:
            query_present = torch.ones(batch, dtype=torch.bool, device=query.device)
        if query_present.shape != (batch,):
            raise ValueError("RRST query_present shape mismatch")

        last, repeat_candidate = self._last_exact(repeat_mask, history_mask)
        repeat_candidate = repeat_candidate & candidate_mask
        allowed, coordinates = self._allowed_events(
            last=last,
            repeat_candidate=repeat_candidate,
            history_mask=history_mask,
        )
        anchor = self._gather_anchor(history, last)
        anchor = torch.where(repeat_candidate[..., None], anchor, torch.zeros_like(anchor))

        effective_query = query
        if self.mode == "query_independent":
            effective_query = torch.zeros_like(query)
        q_reset = self.query_reset(effective_query)[:, None, :]
        q_event = self.query_event(effective_query)[:, None, None, :]
        q_read = self.query_read(effective_query)[:, None, :]
        c_reset = self.candidate_reset(candidates)
        c_event = self.candidate_event(candidates)[:, :, None, :]
        c_read = self.candidate_read(candidates)

        exact_count = repeat_mask.sum(dim=-1).to(query.dtype)
        anchor_weight = event_weights[:, None, :].expand(-1, candidate_count, -1).gather(
            2, last.clamp_min(0)[..., None]
        ).squeeze(-1)
        suffix_length = allowed.sum(dim=-1).to(query.dtype).sub(1.0).clamp_min(0.0)
        reset_features = torch.stack(
            (
                exact_count / max(1.0, float(self.max_history)),
                anchor_weight,
                suffix_length / max(1.0, float(self.max_history)),
            ),
            dim=-1,
        )
        start = (
            self.reset_token.view(1, 1, -1)
            + q_reset
            + c_reset
            + self.anchor_reset(anchor)
            + self.reset_features(reset_features)
        )

        history_projected = self.history_event(history)[:, None, :, :]
        history_projected = history_projected.expand(-1, candidate_count, -1, -1)
        reverse_position = history_mask.sum(dim=-1, keepdim=True).to(query.dtype) - torch.arange(
            history_count, device=query.device, dtype=query.dtype
        ).view(1, -1)
        reverse_position = reverse_position.clamp_min(1.0)
        feature_values = torch.stack(
            (
                event_weights,
                reverse_position / max(1.0, float(self.max_history)),
                history_mask.to(query.dtype),
            ),
            dim=-1,
        )
        event = (
            history_projected
            + q_event
            + c_event
            + self.event_features(feature_values)[:, None, :, :]
        )
        if self.mode != "orderless_suffix":
            event = event + self.position(coordinates)
        if corrupt_preanchor:
            filler = torch.full_like(event, 1000.0)
            event = torch.where(allowed[..., None], event, filler)
        event = self._apply_suffix_permutation(event, allowed, suffix_permutation)

        read = self.read_token.view(1, 1, -1) + q_read + c_read
        sequence = torch.cat((start[:, :, None, :], event, read[:, :, None, :]), dim=2)
        sequence = sequence.reshape(batch * candidate_count, history_count + 2, -1)
        event_padding = ~allowed
        padding = torch.cat(
            (
                torch.zeros(batch, candidate_count, 1, dtype=torch.bool, device=query.device),
                event_padding,
                torch.zeros(batch, candidate_count, 1, dtype=torch.bool, device=query.device),
            ),
            dim=-1,
        ).reshape(batch * candidate_count, history_count + 2)
        active = torch.nonzero(repeat_candidate.reshape(-1), as_tuple=False).flatten()
        raw_flat = query.new_zeros(batch * candidate_count)
        if active.numel():
            if self.mode == "orderless_suffix":
                causal_mask = None
            else:
                causal_mask = torch.triu(
                    torch.ones(
                        history_count + 2,
                        history_count + 2,
                        dtype=torch.bool,
                        device=query.device,
                    ),
                    diagonal=1,
                )
            encoded = self.transformer(
                sequence.index_select(0, active),
                mask=causal_mask,
                src_key_padding_mask=padding.index_select(0, active),
            )
            raw_active = self.output(self.output_norm(encoded[:, -1, :])).squeeze(-1)
            raw_flat = raw_flat.index_copy(0, active, torch.tanh(raw_active))
        raw = raw_flat.reshape(batch, candidate_count)
        weights = candidate_mask.to(raw.dtype)
        count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        centered = (raw - (raw * weights).sum(dim=-1, keepdim=True) / count) * weights
        correction = self.score_delta_max * centered
        correction = correction * query_present[:, None].to(correction.dtype)

        request_repeat = repeat_candidate.any(dim=-1)
        anchor_scores = torch.where(request_repeat[:, None], item_only_scores, base_scores)
        scores = (anchor_scores + correction).masked_fill(~candidate_mask, 0.0)
        return RRSTOutput(
            scores=scores,
            correction=correction,
            raw_correction=raw,
            item_only_scores=anchor_scores,
            repeat_candidate_mask=repeat_candidate,
            allowed_event_mask=allowed,
            last_exact_position=last,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
