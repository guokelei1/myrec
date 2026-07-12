"""Query-complement token transport with history-only candidate competition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


MODES = (
    "query_complement_token",
    "unprojected_token",
    "pooled_complement",
    "raw_candidate",
)


@dataclass(frozen=True)
class QueryComplementOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    carrier: torch.Tensor
    list_message: torch.Tensor
    active_request: torch.Tensor


def masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight.unsqueeze(-1)).sum(dim=dim) / weight.sum(
        dim=dim, keepdim=True
    ).clamp_min(1.0)


class QueryComplementTokenCompetitionTransformer(nn.Module):
    """Form token-level history deltas before a candidate-relative readout.

    The primary correction has no history-free value path.  With no history,
    factual/null token inputs coincide, the carrier is exactly zero, and the
    bias-free list value stream/output remain zero.
    """

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
            raise ValueError("C56 hidden dimension must divide heads")
        if min(input_dim, hidden_dim, ffn_dim, token_layers, max_history) <= 0:
            raise ValueError("C56 dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
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
        self.query_explainer = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.history_transport = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        write_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_write = nn.TransformerEncoder(
            write_layer, num_layers=1, enable_nested_tensor=False
        )
        self.base_projection = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.list_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1, bias=False)
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
        if zero_initial_output:
            nn.init.zeros_(self.output.weight)
        else:
            nn.init.normal_(self.output.weight, std=0.02)

    def _encode(
        self,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        type_embedding: torch.Tensor,
        *,
        event_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_states.shape[:-1] != token_mask.shape:
            raise ValueError("C56 token/mask shape differs")
        if token_states.shape[-1] != self.input_dim:
            raise ValueError("C56 input width differs")
        prefix, length = token_states.shape[:-2], token_states.shape[-2]
        if length > self.token_position.shape[0]:
            raise ValueError("C56 token length exceeds maximum")
        flat_states = token_states.reshape(-1, length, self.input_dim).float()
        flat_mask = token_mask.reshape(-1, length).bool()
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        if bool(empty.any()):
            safe_mask[empty, 0] = True
        value = self.input_projection(flat_states)
        value = value + type_embedding + self.token_position[:length]
        if event_position is not None:
            position = event_position.reshape(-1, self.hidden_dim)
            if position.shape[0] != value.shape[0]:
                raise ValueError("C56 event-position shape differs")
            value = value + position[:, None, :]
        encoded = self.token_refiner(value, src_key_padding_mask=~safe_mask)
        encoded = encoded * flat_mask[..., None].to(encoded.dtype)
        return encoded.reshape(*prefix, length, self.hidden_dim)

    @staticmethod
    def _append_null(
        value: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = value.new_zeros(*value.shape[:-2], 1, value.shape[-1])
        valid = torch.ones(*mask.shape[:-1], 1, dtype=torch.bool, device=mask.device)
        return torch.cat((value, zero), dim=-2), torch.cat((mask.bool(), valid), dim=-1)

    def _query_complements(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        candidates: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        history: torch.Tensor,
        history_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidate_count, item_length, _ = candidates.shape
        history_count = history.shape[1]
        query_context, query_context_mask = self._append_null(query, query_mask)

        candidate_query = candidates.reshape(batch * candidate_count, item_length, -1)
        candidate_context = query_context[:, None].expand(
            -1, candidate_count, -1, -1
        ).reshape(batch * candidate_count, query_context.shape[1], -1)
        candidate_context_mask = query_context_mask[:, None].expand(
            -1, candidate_count, -1
        ).reshape(batch * candidate_count, query_context.shape[1])
        candidate_explained, _ = self.query_explainer(
            candidate_query,
            candidate_context,
            candidate_context,
            key_padding_mask=~candidate_context_mask,
            need_weights=False,
        )
        candidate_perp = (
            candidate_query - candidate_explained
        ).reshape_as(candidates)
        candidate_perp = candidate_perp * candidate_token_mask[..., None].to(
            candidate_perp.dtype
        )

        history_query = history.reshape(batch * history_count, item_length, -1)
        history_context = query_context[:, None].expand(
            -1, history_count, -1, -1
        ).reshape(batch * history_count, query_context.shape[1], -1)
        history_context_mask = query_context_mask[:, None].expand(
            -1, history_count, -1
        ).reshape(batch * history_count, query_context.shape[1])
        history_explained, _ = self.query_explainer(
            history_query,
            history_context,
            history_context,
            key_padding_mask=~history_context_mask,
            need_weights=False,
        )
        history_perp = (history_query - history_explained).reshape_as(history)
        history_perp = history_perp * history_token_mask[..., None].to(
            history_perp.dtype
        )
        return candidate_perp, history_perp

    def _transport_request(
        self,
        candidate_key: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        history_key: torch.Tensor,
        history_token_mask: torch.Tensor,
        event_weights: torch.Tensor,
        *,
        pooled: bool,
    ) -> torch.Tensor:
        candidates, item_length, hidden = candidate_key.shape
        if pooled:
            query = masked_mean(candidate_key, candidate_token_mask, dim=1)[:, None, :]
            history_values = masked_mean(history_key, history_token_mask, dim=1)
            key = history_values
            value = history_values * event_weights[:, None].to(history_values.dtype)
            mask = history_token_mask.any(dim=-1)
        else:
            query = candidate_key
            key = history_key.reshape(-1, hidden)
            weights = event_weights[:, None].expand_as(history_token_mask).reshape(-1)
            value = key * weights[:, None].to(key.dtype)
            mask = history_token_mask.reshape(-1)
        key, mask = self._append_null(key[None], mask[None])
        value, _ = self._append_null(value[None], mask[:, :-1])
        key = key.expand(candidates, -1, -1)
        value = value.expand(candidates, -1, -1)
        padding = ~mask.expand(candidates, -1)
        transported, _ = self.history_transport(
            query, key, value, key_padding_mask=padding, need_weights=False
        )
        if pooled:
            transported = transported.expand(-1, item_length, -1)
        return transported * candidate_token_mask[..., None].to(transported.dtype)

    def _history_transport_batch(
        self,
        candidate_key: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        history_key: torch.Tensor,
        history_token_mask: torch.Tensor,
        event_weights: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        pooled: bool,
    ) -> torch.Tensor:
        output = candidate_key.new_zeros(candidate_key.shape)
        for row in range(candidate_key.shape[0]):
            positions = torch.where(candidate_mask[row].bool())[0]
            if not len(positions):
                continue
            output[row, positions] = self._transport_request(
                candidate_key[row, positions],
                candidate_token_mask[row, positions],
                history_key[row],
                history_token_mask[row],
                event_weights[row],
                pooled=pooled,
            )
        return output

    def _token_carrier(
        self,
        candidates: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        transport: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidate_count, item_length, hidden = candidates.shape
        flat_mask = candidate_token_mask.reshape(batch * candidate_count, item_length)
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        if bool(empty.any()):
            safe_mask[empty, 0] = True
        null_input = candidates.reshape(batch * candidate_count, item_length, hidden)
        factual_input = (candidates + transport).reshape(
            batch * candidate_count, item_length, hidden
        )
        paired = torch.cat((factual_input, null_input), dim=0)
        padding = torch.cat((~safe_mask, ~safe_mask), dim=0)
        encoded = self.token_write(paired, src_key_padding_mask=padding)
        factual, null = encoded.chunk(2, dim=0)
        delta = (factual - null) * flat_mask[..., None].to(encoded.dtype)
        carrier = masked_mean(delta, flat_mask, dim=1)
        return carrier.reshape(batch, candidate_count, hidden)

    def _list_message(
        self,
        content: torch.Tensor,
        carrier: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidates, _ = content.shape
        zero = content.new_zeros(batch, 1, self.hidden_dim)
        key = torch.cat((content, zero), dim=1)
        value = torch.cat((carrier, zero), dim=1)
        valid = torch.cat(
            (
                candidate_mask.bool(),
                torch.ones(batch, 1, dtype=torch.bool, device=content.device),
            ),
            dim=1,
        )
        attention_mask = torch.zeros(
            candidates, candidates + 1, dtype=torch.bool, device=content.device
        )
        attention_mask[:, :candidates] = torch.eye(
            candidates, dtype=torch.bool, device=content.device
        )
        message, _ = self.list_attention(
            content,
            key,
            value,
            attn_mask=attention_mask,
            key_padding_mask=~valid,
            need_weights=False,
        )
        return message * candidate_mask[..., None].to(message.dtype)

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
        mode: str = "query_complement_token",
        edge_ablation: bool = False,
    ) -> QueryComplementOutput:
        if mode not in MODES:
            raise ValueError(f"unknown C56 mode: {mode}")
        if query_tokens.ndim != 3 or candidate_tokens.ndim != 4 or history_tokens.ndim != 4:
            raise ValueError("C56 token ranks differ")
        batch, candidates, item_length, _ = candidate_tokens.shape
        history_count = history_tokens.shape[1]
        if history_count > self.max_history or item_length > self.max_item_tokens:
            raise ValueError("C56 item/history maximum exceeded")
        if candidate_mask.shape != (batch, candidates):
            raise ValueError("C56 candidate mask differs")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != candidate_mask.shape:
            raise ValueError("C56 score shapes differ")
        if repeat_request.shape != (batch,):
            raise ValueError("C56 repeat-request shape differs")
        if query_present is None:
            query_present = query_token_mask.any(dim=-1)

        query = self._encode(query_tokens, query_token_mask, self.query_type)
        candidate = self._encode(
            candidate_tokens, candidate_token_mask, self.candidate_type
        )
        positions = self.history_position[:history_count][None].expand(batch, -1, -1)
        history = self._encode(
            history_tokens,
            history_token_mask,
            self.history_type,
            event_position=positions,
        )
        candidate_perp, history_perp = self._query_complements(
            query,
            query_token_mask,
            candidate,
            candidate_token_mask,
            history,
            history_token_mask,
        )

        if mode == "query_complement_token":
            candidate_key, history_key, pooled = candidate_perp, history_perp, False
            transport = self._history_transport_batch(
                candidate_key,
                candidate_token_mask,
                history_key,
                history_token_mask,
                event_weights,
                candidate_mask,
                pooled=pooled,
            )
        elif mode == "unprojected_token":
            transport = self._history_transport_batch(
                candidate,
                candidate_token_mask,
                history,
                history_token_mask,
                event_weights,
                candidate_mask,
                pooled=False,
            )
        elif mode == "pooled_complement":
            transport = self._history_transport_batch(
                candidate_perp,
                candidate_token_mask,
                history_perp,
                history_token_mask,
                event_weights,
                candidate_mask,
                pooled=True,
            )
        else:
            transport = candidate * candidate_token_mask[..., None].to(candidate.dtype)

        carrier = self._token_carrier(candidate, candidate_token_mask, transport)
        content = masked_mean(candidate, candidate_token_mask, dim=2)
        content = content + self.base_projection(base_scores[..., None].float())
        list_message = self._list_message(content, carrier, candidate_mask)
        relative = carrier - (torch.zeros_like(list_message) if edge_ablation else list_message)
        raw = self.output(self.output_norm(relative)).squeeze(-1)
        weight = candidate_mask.to(raw.dtype)
        raw = raw * weight
        centered = raw - (raw * weight).sum(dim=-1, keepdim=True) / weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        history_present = history_token_mask.any(dim=(-1, -2))
        active = history_present & query_present.bool() & ~repeat_request.bool()
        correction = centered * weight * active[:, None].to(centered.dtype)
        anchor = torch.where(repeat_request[:, None], item_only_scores, base_scores)
        scores = (anchor + correction).masked_fill(~candidate_mask.bool(), 0.0)
        return QueryComplementOutput(
            scores=scores,
            correction=correction,
            raw_correction=raw,
            carrier=carrier,
            list_message=list_message,
            active_request=active,
        )

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())
