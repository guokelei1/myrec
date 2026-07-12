"""Permutation-equivariant Transformer over exact-recurrence candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


MODES = ("set_attention", "independent", "query_independent")


@dataclass(frozen=True)
class CompetitionOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    anchor_scores: torch.Tensor
    repeat_candidate_mask: torch.Tensor
    multi_repeat_request: torch.Tensor


class MultiRecurrenceCompetitionTransformer(nn.Module):
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
        max_repeat_candidates: int,
        score_delta_max: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C24 mode: {mode}")
        if hidden_dim % heads:
            raise ValueError("C24 hidden_dim must divide heads")
        self.mode = mode
        self.max_history = int(max_history)
        self.max_repeat_candidates = int(max_repeat_candidates)
        self.score_delta_max = float(score_delta_max)
        self.query_token = nn.Parameter(torch.zeros(hidden_dim))
        self.query_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.candidate_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_to_candidate = nn.Linear(input_dim, hidden_dim, bias=False)
        self.feature_projection = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
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
        nn.init.normal_(self.query_token, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _recurrence_features(
        repeat_mask: torch.Tensor,
        history_mask: torch.Tensor,
        event_weights: torch.Tensor,
        base_scores: torch.Tensor,
        max_history: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidates, history = repeat_mask.shape
        valid = repeat_mask & history_mask[:, None, :]
        count = valid.sum(dim=-1)
        reverse = history_mask.sum(dim=-1, keepdim=True).to(event_weights.dtype) - torch.arange(
            history, device=event_weights.device, dtype=event_weights.dtype
        ).view(1, -1)
        reverse = reverse.clamp_min(1.0)
        recency_weight = event_weights / reverse.sqrt()
        mass = 3.0 * (valid.to(event_weights.dtype) * recency_weight[:, None, :]).sum(dim=-1)
        last_reverse = torch.where(
            valid,
            reverse[:, None, :].expand(-1, candidates, -1),
            torch.full((), float(max_history + 1), device=event_weights.device),
        ).amin(dim=-1)
        last_recency = torch.where(count.gt(0), last_reverse.rsqrt(), torch.zeros_like(last_reverse))
        purchase = event_weights.gt(1.25)[:, None, :] & valid
        purchase_share = purchase.sum(dim=-1).to(event_weights.dtype) / count.clamp_min(1).to(
            event_weights.dtype
        )
        features = torch.stack(
            (
                base_scores,
                mass / 10.0,
                count.to(event_weights.dtype) / max(1.0, float(max_history)),
                last_recency,
                purchase_share,
            ),
            dim=-1,
        )
        return features, count.gt(0)

    @staticmethod
    def _independent_mask(length: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(length, length, dtype=torch.bool, device=device)
        mask[0, 0] = False
        positions = torch.arange(1, length, device=device)
        mask[positions, 0] = False
        mask[positions, positions] = False
        return mask

    def forward(
        self,
        *,
        query: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        repeat_mask: torch.Tensor,
        event_weights: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        query_present: torch.Tensor | None = None,
        disable_cross_candidate: bool = False,
    ) -> CompetitionOutput:
        batch, candidate_count, input_dim = candidates.shape
        history_count = history_mask.shape[-1]
        if query.shape != (batch, input_dim):
            raise ValueError("C24 query/candidate shape mismatch")
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("C24 candidate mask shape mismatch")
        if repeat_mask.shape != (batch, candidate_count, history_count):
            raise ValueError("C24 repeat mask shape mismatch")
        if event_weights.shape != (batch, history_count):
            raise ValueError("C24 event weight shape mismatch")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != (
            candidate_mask.shape
        ):
            raise ValueError("C24 score shape mismatch")
        if history_count > self.max_history:
            raise ValueError("C24 history exceeds registered limit")
        if query_present is None:
            query_present = torch.ones(batch, dtype=torch.bool, device=query.device)

        effective_query = query
        if self.mode == "query_independent":
            effective_query = torch.zeros_like(query)
        features, repeat_candidate = self._recurrence_features(
            repeat_mask,
            history_mask,
            event_weights,
            base_scores,
            self.max_history,
        )
        repeat_candidate = repeat_candidate & candidate_mask
        repeat_counts = repeat_candidate.sum(dim=-1)
        multi_repeat = repeat_counts.ge(2)
        max_repeat = max(1, int(repeat_counts.max().item()))
        if max_repeat > self.max_repeat_candidates:
            raise ValueError("C24 repeat candidates exceed registered limit")

        candidate_token = (
            self.candidate_projection(candidates)
            + self.query_to_candidate(effective_query)[:, None, :]
            + self.feature_projection(features)
        )
        hidden = candidate_token.shape[-1]
        packed = candidate_token.new_zeros(batch, max_repeat, hidden)
        packed_mask = torch.zeros(batch, max_repeat, dtype=torch.bool, device=query.device)
        packed_indices = torch.full(
            (batch, max_repeat), -1, dtype=torch.long, device=query.device
        )
        for row in range(batch):
            indices = torch.nonzero(repeat_candidate[row], as_tuple=False).flatten()
            count = int(indices.numel())
            if count:
                packed[row, :count] = candidate_token[row].index_select(0, indices)
                packed_mask[row, :count] = True
                packed_indices[row, :count] = indices

        query_token = self.query_token.view(1, 1, -1) + self.query_projection(
            effective_query
        )[:, None, :]
        sequence = torch.cat((query_token, packed), dim=1)
        padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=query.device),
                ~packed_mask,
            ),
            dim=1,
        )
        use_independent = self.mode == "independent" or disable_cross_candidate
        attention_mask = (
            self._independent_mask(max_repeat + 1, query.device)
            if use_independent
            else None
        )
        encoded = self.transformer(
            sequence, mask=attention_mask, src_key_padding_mask=padding
        )[:, 1:, :]
        packed_raw = self.output(self.output_norm(encoded)).squeeze(-1)
        packed_raw = packed_raw * packed_mask.to(packed_raw.dtype)
        raw = candidates.new_zeros(batch, candidate_count)
        for row in range(batch):
            valid = packed_mask[row]
            if bool(valid.any()):
                raw[row] = raw[row].index_copy(
                    0, packed_indices[row, valid], torch.tanh(packed_raw[row, valid])
                )
        raw = raw * multi_repeat[:, None].to(raw.dtype)
        weights = candidate_mask.to(raw.dtype)
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        centered = (raw - (raw * weights).sum(dim=-1, keepdim=True) / denominator) * weights
        correction = self.score_delta_max * centered
        correction = correction * query_present[:, None].to(correction.dtype)
        request_repeat = repeat_candidate.any(dim=-1)
        anchor = torch.where(request_repeat[:, None], item_only_scores, base_scores)
        scores = (anchor + correction).masked_fill(~candidate_mask, 0.0)
        return CompetitionOutput(
            scores=scores,
            correction=correction,
            raw_correction=raw,
            anchor_scores=anchor,
            repeat_candidate_mask=repeat_candidate,
            multi_repeat_request=multi_repeat,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
