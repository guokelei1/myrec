"""Prefix-conditioned event-innovation Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


INNOVATION = "innovation"
ORDINARY_DELTA = "ordinary_delta"
FACTUAL_STATE = "factual_state"
RAW_EVENT = "raw_event"
MODES = (INNOVATION, ORDINARY_DELTA, FACTUAL_STATE, RAW_EVENT)
Mode = Literal["innovation", "ordinary_delta", "factual_state", "raw_event"]


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_multiplier: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ff_multiplier * width),
            nn.GELU(),
            nn.Linear(ff_multiplier * width, width),
        )

    def forward(
        self, values: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.norm1(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        values = values + attended
        return values + self.ffn(self.norm2(values))


@dataclass
class ForwardOutput:
    score: torch.Tensor
    base: torch.Tensor
    correction: torch.Tensor
    event_tokens: torch.Tensor
    factual_states: torch.Tensor
    null_states: torch.Tensor


class PrefixConditionedEventInnovationTransformer(nn.Module):
    """One shared recurrent transition plus a query-candidate event reader."""

    def __init__(
        self,
        *,
        input_dim: int,
        width: int,
        heads: int,
        ff_multiplier: int,
        max_history: int,
        mode: Mode,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C45 mode: {mode}")
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.mode = mode
        self.max_history = int(max_history)
        self.query_projection = nn.Linear(input_dim, width)
        self.candidate_projection = nn.Linear(input_dim, width)
        self.event_projection = nn.Linear(input_dim, width)
        self.initial_state = nn.Parameter(torch.zeros(width))
        # NULL is an immutable absence token.  It cannot learn a second user
        # profile or add unmatched capacity to the primary mode.
        self.register_buffer("null_event", torch.zeros(width), persistent=True)
        self.state_role = nn.Parameter(torch.empty(width))
        self.event_role = nn.Parameter(torch.empty(width))
        self.position = nn.Parameter(torch.empty(max_history, width))
        self.base_read = nn.Parameter(torch.empty(width))
        self.transition = TransformerBlock(width, heads, ff_multiplier)
        self.base_transformer = TransformerBlock(width, heads, ff_multiplier)
        self.evidence_transformer = TransformerBlock(width, heads, ff_multiplier)
        self.evidence_norm = nn.LayerNorm(width)
        self.base_head = nn.Linear(width, 1, bias=False)
        self.correction_head = nn.Linear(width, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_role, std=0.02)
        nn.init.normal_(self.event_role, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.base_read, std=0.02)
        nn.init.normal_(self.initial_state, std=0.02)
        nn.init.normal_(self.correction_head.weight, std=0.02)

    def _transition_once(
        self, state: torch.Tensor, event: torch.Tensor, position: int
    ) -> torch.Tensor:
        position_token = self.position[position].expand_as(state)
        tokens = torch.stack(
            (
                state + self.state_role,
                position_token,
                event + self.event_role,
            ),
            dim=1,
        )
        return self.transition(tokens)[:, 0]

    def encode_events(
        self, history: torch.Tensor, history_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or history_mask.shape != history.shape[:2]:
            raise ValueError("C45 history/mask shape differs")
        if history.shape[1] > self.max_history:
            raise ValueError("C45 history exceeds max_history")
        batch, length, _ = history.shape
        projected = self.event_projection(history)
        state = self.initial_state.expand(batch, -1)
        evidence: list[torch.Tensor] = []
        factual_rows: list[torch.Tensor] = []
        null_rows: list[torch.Tensor] = []
        for position in range(length):
            factual = self._transition_once(state, projected[:, position], position)
            null = self._transition_once(
                state, self.null_event.expand_as(state), position
            )
            if self.mode == INNOVATION:
                token = factual - null
            elif self.mode == ORDINARY_DELTA:
                token = factual - state
            elif self.mode == FACTUAL_STATE:
                token = factual
            else:
                token = projected[:, position]
            token = self.evidence_norm(token)
            present = history_mask[:, position, None]
            evidence.append(torch.where(present, token, torch.zeros_like(token)))
            factual_rows.append(torch.where(present, factual, state))
            null_rows.append(torch.where(present, null, state))
            state = torch.where(present, factual, state)
        if length == 0:
            empty = history.new_zeros(batch, 0, self.initial_state.numel())
            return empty, empty, empty
        return (
            torch.stack(evidence, dim=1),
            torch.stack(factual_rows, dim=1),
            torch.stack(null_rows, dim=1),
        )

    def forward_components(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        query_present: torch.Tensor | None = None,
    ) -> ForwardOutput:
        if query.ndim != 2 or candidates.ndim != 3:
            raise ValueError("C45 query/candidate shape differs")
        if query.shape[0] != candidates.shape[0]:
            raise ValueError("C45 request cardinality differs")
        batch, candidate_count, _ = candidates.shape
        if query_present is None:
            query_present = torch.ones(batch, dtype=torch.bool, device=query.device)
        if query_present.shape != (batch,):
            raise ValueError("C45 query-present shape differs")

        query_state = self.query_projection(query)
        candidate_state = self.candidate_projection(candidates)
        read = self.base_read.view(1, 1, -1).expand(batch, candidate_count, -1)
        base_tokens = torch.stack(
            (
                read,
                query_state[:, None, :].expand(-1, candidate_count, -1),
                candidate_state,
            ),
            dim=2,
        ).reshape(batch * candidate_count, 3, -1)
        base_state = self.base_transformer(base_tokens)[:, 0]
        base = self.base_head(base_state).reshape(batch, candidate_count)

        event_tokens, factual_states, null_states = self.encode_events(
            history, history_mask
        )
        length = event_tokens.shape[1]
        expanded_events = (
            event_tokens[:, None, :, :]
            .expand(-1, candidate_count, -1, -1)
            .reshape(batch * candidate_count, length, -1)
        )
        evidence_input = torch.cat((base_state[:, None, :], expanded_events), dim=1)
        expanded_mask = (
            history_mask[:, None, :]
            .expand(-1, candidate_count, -1)
            .reshape(batch * candidate_count, length)
        )
        key_padding = torch.cat(
            (
                torch.zeros(
                    batch * candidate_count,
                    1,
                    dtype=torch.bool,
                    device=query.device,
                ),
                ~expanded_mask,
            ),
            dim=1,
        )
        personalized = self.evidence_transformer(
            evidence_input, key_padding_mask=key_padding
        )[:, 0]
        correction = self.correction_head(personalized - base_state).reshape(
            batch, candidate_count
        )
        active = history_mask.any(dim=1) & query_present
        correction = torch.where(active[:, None], correction, torch.zeros_like(correction))
        correction = correction - correction.mean(dim=1, keepdim=True)
        score = base + correction
        return ForwardOutput(
            score=score,
            base=base,
            correction=correction,
            event_tokens=event_tokens,
            factual_states=factual_states,
            null_states=null_states,
        )

    def rank(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        query_present: torch.Tensor | None = None,
        repeat_present: torch.Tensor | None = None,
        item_only_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.forward_components(
            query,
            candidates,
            history,
            history_mask,
            query_present=query_present,
        ).score
        if repeat_present is None:
            return output
        if item_only_scores is None or item_only_scores.shape != output.shape:
            raise ValueError("C45 repeat fallback requires item-only scores")
        if repeat_present.shape != (output.shape[0],):
            raise ValueError("C45 repeat-present shape differs")
        return torch.where(repeat_present[:, None], item_only_scores, output)
