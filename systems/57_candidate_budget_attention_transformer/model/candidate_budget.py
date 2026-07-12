"""Candidate-axis normalized history attention for C57."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


MODES = (
    "candidate_budget",
    "slot_budget_no_null",
    "history_softmax",
    "pooled_history",
    "raw_candidate",
)


@dataclass(frozen=True)
class CandidateBudgetOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    features: torch.Tensor
    allocation: torch.Tensor
    null_allocation: torch.Tensor
    active_request: torch.Tensor


def masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight.unsqueeze(-1)).sum(dim=dim) / weight.sum(
        dim=dim, keepdim=True
    ).clamp_min(1.0)


class CandidateBudgetAttentionTransformer(nn.Module):
    """Allocate each history event across actual candidates and a NULL sink."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        token_layers: int,
        dropout: float,
        max_query_tokens: int,
        max_item_tokens: int,
        max_history: int,
        zero_initial_output: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("C57 hidden dimension must divide heads")
        if min(input_dim, hidden_dim, heads, ffn_dim, token_layers, max_history) <= 0:
            raise ValueError("C57 dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.max_query_tokens = int(max_query_tokens)
        self.max_item_tokens = int(max_item_tokens)
        self.max_history = int(max_history)

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_type = nn.Parameter(torch.zeros(hidden_dim))
        self.candidate_type = nn.Parameter(torch.zeros(hidden_dim))
        self.history_type = nn.Parameter(torch.zeros(hidden_dim))
        self.token_position = nn.Parameter(
            torch.zeros(max(max_query_tokens, max_item_tokens), hidden_dim)
        )
        self.history_position = nn.Parameter(torch.zeros(max_history, hidden_dim))
        token_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_refiner = nn.TransformerEncoder(
            token_layer, num_layers=token_layers, enable_nested_tensor=False
        )
        self.content_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.candidate_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.history_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.history_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.null_logit = nn.Parameter(torch.zeros(heads))
        self.output = nn.Linear(heads, 1, bias=False)
        self.reset_parameters(zero_initial_output=zero_initial_output)

    def reset_parameters(self, *, zero_initial_output: bool) -> None:
        for value in (
            self.query_type,
            self.candidate_type,
            self.history_type,
            self.token_position,
            self.history_position,
        ):
            nn.init.normal_(value, std=0.02)
        nn.init.zeros_(self.null_logit)
        if zero_initial_output:
            nn.init.zeros_(self.output.weight)
        else:
            nn.init.normal_(self.output.weight, std=0.02)

    def _encode(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        type_embedding: torch.Tensor,
        *,
        event_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if states.shape[:-1] != mask.shape or states.shape[-1] != self.input_dim:
            raise ValueError("C57 token input shape differs")
        prefix, length = states.shape[:-2], states.shape[-2]
        if length > self.token_position.shape[0]:
            raise ValueError("C57 token length exceeds maximum")
        flat_states = states.reshape(-1, length, self.input_dim).float()
        flat_mask = mask.reshape(-1, length).bool()
        safe = flat_mask.clone()
        empty = ~safe.any(dim=-1)
        if bool(empty.any()):
            safe[empty, 0] = True
        value = self.input_projection(flat_states)
        value = value + type_embedding + self.token_position[:length]
        if event_position is not None:
            position = event_position.reshape(-1, self.hidden_dim)
            if position.shape[0] != value.shape[0]:
                raise ValueError("C57 event position shape differs")
            value = value + position[:, None]
        encoded = self.token_refiner(value, src_key_padding_mask=~safe)
        encoded = encoded * flat_mask[..., None].to(encoded.dtype)
        return encoded.reshape(*prefix, length, self.hidden_dim)

    def _query_conditioned_items(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        items: torch.Tensor,
        item_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, length, _ = items.shape
        query_summary = masked_mean(query, query_mask, dim=1)
        item_query = query_summary[:, None, None].expand(-1, count, 1, -1).reshape(
            batch * count, 1, self.hidden_dim
        )
        flat_items = items.reshape(batch * count, length, self.hidden_dim)
        flat_mask = item_mask.reshape(batch * count, length).bool()
        safe = flat_mask.clone()
        empty = ~safe.any(dim=-1)
        if bool(empty.any()):
            safe[empty, 0] = True
        attended, _ = self.content_attention(
            item_query,
            flat_items,
            flat_items,
            key_padding_mask=~safe,
            need_weights=False,
        )
        attended = attended[:, 0] * flat_mask.any(dim=-1, keepdim=True).to(attended.dtype)
        return attended.reshape(batch, count, self.hidden_dim)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-1], self.heads, self.head_dim)

    def _candidate_budget(
        self,
        logits: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        include_null: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # logits: [B,C,H,R].  Each event/head normalizes over candidate slots.
        masked = logits.masked_fill(~candidate_mask[:, :, None, None].bool(), -torch.inf)
        if include_null:
            null = self.null_logit.view(1, 1, 1, self.heads).expand(
                logits.shape[0], 1, logits.shape[2], -1
            )
            probability = torch.softmax(torch.cat((masked, null), dim=1), dim=1)
            allocation, null_allocation = probability[:, :-1], probability[:, -1]
            cardinality = candidate_mask.sum(dim=-1).to(logits.dtype) + 1.0
        else:
            allocation = torch.softmax(masked, dim=1)
            null_allocation = logits.new_zeros(logits.shape[0], logits.shape[2], self.heads)
            cardinality = candidate_mask.sum(dim=-1).to(logits.dtype)
        return allocation * cardinality[:, None, None, None], null_allocation

    def _history_axis(
        self,
        logits: torch.Tensor,
        history_mask: torch.Tensor,
        event_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = history_mask[:, None, :, None].bool()
        prior = torch.log(event_weight.clamp_min(1e-12))[:, None, :, None]
        factual = (logits + prior).masked_fill(~valid, -torch.inf)
        null = self.null_logit.view(1, 1, 1, self.heads).expand(
            logits.shape[0], logits.shape[1], 1, -1
        )
        probability = torch.softmax(torch.cat((factual, null), dim=2), dim=2)
        return probability[:, :, :-1], probability[:, :, -1]

    def forward(
        self,
        *,
        query_tokens: torch.Tensor,
        query_token_mask: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        history_tokens: torch.Tensor,
        history_token_mask: torch.Tensor,
        event_weights: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        repeat_request: torch.Tensor,
        query_present: torch.Tensor | None = None,
        mode: str = "candidate_budget",
    ) -> CandidateBudgetOutput:
        if mode not in MODES:
            raise ValueError(f"unknown C57 mode: {mode}")
        if query_tokens.ndim != 3 or candidate_tokens.ndim != 4 or history_tokens.ndim != 4:
            raise ValueError("C57 token ranks differ")
        batch, candidates = candidate_mask.shape
        history_count = history_tokens.shape[1]
        if history_count > self.max_history or candidate_tokens.shape[2] > self.max_item_tokens:
            raise ValueError("C57 maximum length exceeded")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != candidate_mask.shape:
            raise ValueError("C57 score shape differs")
        if repeat_request.shape != (batch,):
            raise ValueError("C57 repeat shape differs")
        if query_present is None:
            query_present = query_token_mask.any(dim=-1)

        query = self._encode(query_tokens, query_token_mask, self.query_type)
        candidate = self._encode(candidate_tokens, candidate_token_mask, self.candidate_type)
        positions = self.history_position[:history_count][None].expand(batch, -1, -1)
        history = self._encode(
            history_tokens,
            history_token_mask,
            self.history_type,
            event_position=positions,
        )
        candidate_vector = self._query_conditioned_items(
            query, query_token_mask, candidate, candidate_token_mask
        )
        history_vector = self._query_conditioned_items(
            query, query_token_mask, history, history_token_mask
        )
        query_vector = masked_mean(query, query_token_mask, dim=1)
        candidate_q = self._heads(self.candidate_query(candidate_vector))
        history_k = self._heads(self.history_key(history_vector))
        history_v = self._heads(self.history_value(history_vector))
        candidate_v = self._heads(self.candidate_value(candidate_vector))
        query_v = self._heads(self.query_value(query_vector))
        logits = torch.einsum("bcrd,bhrd->bchr", candidate_q, history_k) / math.sqrt(
            self.head_dim
        )
        history_mask = history_token_mask.any(dim=-1)
        event_weight = event_weights.float() * history_mask.to(event_weights.dtype)
        event_norm = event_weight / event_weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if mode in {"candidate_budget", "slot_budget_no_null"}:
            allocation, null_allocation = self._candidate_budget(
                logits,
                candidate_mask,
                include_null=mode == "candidate_budget",
            )
            update = torch.einsum(
                "bchr,bh,bhrd->bcrd", allocation, event_norm, history_v
            )
            features = (update * candidate_v).sum(dim=-1) / math.sqrt(self.head_dim)
        elif mode == "history_softmax":
            allocation, null_allocation = self._history_axis(
                logits, history_mask, event_norm
            )
            update = torch.einsum("bchr,bhrd->bcrd", allocation, history_v)
            features = (update * candidate_v).sum(dim=-1) / math.sqrt(self.head_dim)
        elif mode == "pooled_history":
            pooled = torch.einsum("bh,bhrd->brd", event_norm, history_v)
            features = (candidate_v * pooled[:, None]).sum(dim=-1) / math.sqrt(self.head_dim)
            allocation = logits.new_zeros(logits.shape)
            null_allocation = logits.new_zeros(batch, history_count, self.heads)
        else:
            features = (candidate_v * query_v[:, None]).sum(dim=-1) / math.sqrt(self.head_dim)
            allocation = logits.new_zeros(logits.shape)
            null_allocation = logits.new_zeros(batch, history_count, self.heads)

        features = features * candidate_mask[..., None].to(features.dtype)
        raw = self.output(features).squeeze(-1)
        weight = candidate_mask.to(raw.dtype)
        centered = raw - (raw * weight).sum(dim=-1, keepdim=True) / weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        history_present = history_mask.any(dim=-1)
        active = history_present & query_present.bool() & ~repeat_request.bool()
        correction = centered * weight * active[:, None].to(centered.dtype)
        anchor = torch.where(repeat_request[:, None], item_only_scores, base_scores)
        scores = (anchor + correction).masked_fill(~candidate_mask.bool(), 0.0)
        return CandidateBudgetOutput(
            scores=scores,
            correction=correction,
            raw_correction=raw,
            features=features,
            allocation=allocation,
            null_allocation=null_allocation,
            active_request=active,
        )

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())
