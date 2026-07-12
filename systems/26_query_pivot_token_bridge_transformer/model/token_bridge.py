"""Query-token-pivoted candidate/history bridge Transformer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


MODES = ("token_bridge", "generic_token_triadic", "candidate_late", "pooled_history")


@dataclass(frozen=True)
class TokenBridgeOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    anchor_scores: torch.Tensor
    active_request: torch.Tensor
    repeat_request: torch.Tensor


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(torch.bool)
    masked = logits.masked_fill(~mask, -torch.inf)
    maximum = masked.amax(dim=dim, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    values = torch.exp(masked - maximum) * mask.to(logits.dtype)
    return values / values.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class QueryPivotTokenBridgeTransformer(nn.Module):
    def __init__(
        self,
        *,
        embedding_weight: torch.Tensor,
        padding_idx: int,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        token_layers: int,
        history_layers: int,
        ffn_dim: int,
        dropout: float,
        max_query_tokens: int,
        max_item_tokens: int,
        max_history: int,
        score_delta_max: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C26 mode: {mode}")
        if hidden_dim % heads or embedding_weight.ndim != 2:
            raise ValueError("C26 hidden/head/embedding shape differs")
        if embedding_weight.shape[1] != input_dim:
            raise ValueError("C26 embedding input dimension differs")
        self.mode = mode
        self.padding_idx = int(padding_idx)
        self.max_query_tokens = int(max_query_tokens)
        self.max_item_tokens = int(max_item_tokens)
        self.max_history = int(max_history)
        self.score_delta_max = float(score_delta_max)
        self.word_embeddings = nn.Embedding.from_pretrained(
            embedding_weight.detach().clone(), freeze=True, padding_idx=padding_idx
        )
        self.token_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.position_embedding = nn.Parameter(
            torch.empty(max(max_query_tokens, max_item_tokens), hidden_dim)
        )
        token_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_transformer = nn.TransformerEncoder(
            token_block, num_layers=token_layers, enable_nested_tensor=False
        )
        self.bridge_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.read_token = nn.Parameter(torch.zeros(hidden_dim))
        history_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_transformer = nn.TransformerEncoder(
            history_block, num_layers=history_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.read_token, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def _encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        prefix, length = token_ids.shape[:-1], token_ids.shape[-1]
        if length > self.position_embedding.shape[0] or token_ids.shape != attention_mask.shape:
            raise ValueError("C26 token encoder shape/length differs")
        flat_ids = token_ids.reshape(-1, length)
        flat_mask = attention_mask.reshape(-1, length).to(torch.bool)
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        if bool(empty.any()):
            safe_mask[empty, 0] = True
        value = self.token_projection(self.word_embeddings(flat_ids))
        value = value + self.position_embedding[:length].view(1, length, -1)
        encoded = self.token_transformer(value, src_key_padding_mask=~safe_mask)
        encoded = encoded * flat_mask[:, :, None].to(encoded.dtype)
        return encoded.reshape(*prefix, length, encoded.shape[-1])

    @staticmethod
    def _mean_tokens(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(value.dtype)
        return (value * weights[..., None]).sum(dim=-2) / weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)

    def bridge_events(
        self,
        *,
        query_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_content_mask: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        candidate_content_mask: torch.Tensor,
        history_token_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        history_content_mask: torch.Tensor,
        event_weights: torch.Tensor,
        mode: str | None = None,
    ) -> torch.Tensor:
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in MODES:
            raise ValueError(f"unknown C26 bridge mode: {selected_mode}")
        batch, candidates, item_length = candidate_token_ids.shape
        if query_ids.shape[0] != batch or history_token_ids.shape[0] != batch:
            raise ValueError("C26 token batch dimensions differ")
        history = history_token_ids.shape[1]
        if event_weights.shape != (batch, history):
            raise ValueError("C26 event weights differ")

        q = self._encode(query_ids, query_attention_mask)
        c = self._encode(candidate_token_ids, candidate_attention_mask)
        h = self._encode(history_token_ids, history_attention_mask)
        scale = math.sqrt(q.shape[-1])
        candidate_logits = torch.einsum("bqd,bcld->bcql", q, c) / scale
        candidate_mask = candidate_content_mask[:, :, None, :].expand_as(candidate_logits)
        candidate_weights = masked_softmax(candidate_logits, candidate_mask, dim=-1)
        candidate_match = torch.einsum("bcql,bcld->bcqd", candidate_weights, c)
        candidate_strength = (candidate_weights * candidate_logits.masked_fill(~candidate_mask, 0.0)).sum(-1)

        history_logits = torch.einsum("bqd,bhld->bhql", q, h) / scale
        history_match_mask = history_content_mask[:, :, None, :].expand_as(history_logits)
        history_weights = masked_softmax(history_logits, history_match_mask, dim=-1)
        history_match = torch.einsum("bhql,bhld->bhqd", history_weights, h)
        history_strength = (history_weights * history_logits.masked_fill(~history_match_mask, 0.0)).sum(-1)

        q_expanded = q[:, None, None, :, :]
        c_expanded = candidate_match[:, :, None, :, :]
        h_expanded = history_match[:, None, :, :, :]
        query_mask = query_content_mask[:, None, None, :].expand(
            batch, candidates, history, -1
        )
        pivot_logits = candidate_strength[:, :, None, :] + history_strength[:, None, :, :]
        pivot_weights = masked_softmax(pivot_logits, query_mask, dim=-1)

        primary_values = self.bridge_ffn((c_expanded - q_expanded) * (h_expanded - q_expanded))
        primary = (primary_values * pivot_weights[..., None]).sum(dim=-2)
        generic_values = self.bridge_ffn(c_expanded + h_expanded + q_expanded)
        generic = (generic_values * pivot_weights[..., None]).sum(dim=-2)

        candidate_pivot_mask = query_content_mask[:, None, :].expand(
            batch, candidates, -1
        )
        candidate_pivot = masked_softmax(candidate_strength, candidate_pivot_mask, dim=-1)
        candidate_values = self.bridge_ffn(candidate_match - q[:, None, :, :])
        candidate_only = (candidate_values * candidate_pivot[..., None]).sum(dim=-2)
        candidate_only = candidate_only[:, :, None, :].expand(-1, -1, history, -1)

        q_pool = self._mean_tokens(q, query_content_mask)
        c_pool = self._mean_tokens(c, candidate_content_mask)
        h_pool = self._mean_tokens(h, history_content_mask)
        pooled = self.bridge_ffn(
            q_pool[:, None, None, :] + c_pool[:, :, None, :] + h_pool[:, None, :, :]
        )

        if selected_mode == "token_bridge":
            selected = primary
        elif selected_mode == "generic_token_triadic":
            selected = generic
        elif selected_mode == "candidate_late":
            selected = candidate_only
        else:
            selected = pooled
        compute_match = primary + generic + candidate_only + pooled
        selected = selected + compute_match * 0.0
        return selected * event_weights[:, None, :, None].to(selected.dtype)

    def forward(
        self,
        *,
        query_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_content_mask: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        candidate_content_mask: torch.Tensor,
        history_token_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        history_content_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        repeat_mask: torch.Tensor,
        event_weights: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        query_present: torch.Tensor | None = None,
    ) -> TokenBridgeOutput:
        batch, candidates = candidate_mask.shape
        history = history_mask.shape[1]
        if repeat_mask.shape != (batch, candidates, history):
            raise ValueError("C26 repeat mask differs")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != candidate_mask.shape:
            raise ValueError("C26 score shapes differ")
        if history > self.max_history:
            raise ValueError("C26 history exceeds registered maximum")
        if query_present is None:
            query_present = torch.ones(batch, dtype=torch.bool, device=query_ids.device)
        events = self.bridge_events(
            query_ids=query_ids,
            query_attention_mask=query_attention_mask,
            query_content_mask=query_content_mask,
            candidate_token_ids=candidate_token_ids,
            candidate_attention_mask=candidate_attention_mask,
            candidate_content_mask=candidate_content_mask,
            history_token_ids=history_token_ids,
            history_attention_mask=history_attention_mask,
            history_content_mask=history_content_mask,
            event_weights=event_weights,
        )
        event_mask = history_mask[:, None, :].expand(-1, candidates, -1)
        event_mask = event_mask & candidate_mask[:, :, None]
        flat_events = events.reshape(batch * candidates, history, -1)
        flat_mask = event_mask.reshape(batch * candidates, history)
        read = self.read_token.view(1, 1, -1).expand(batch * candidates, -1, -1)
        sequence = torch.cat((read, flat_events), dim=1)
        padding = torch.cat(
            (
                torch.zeros(batch * candidates, 1, dtype=torch.bool, device=query_ids.device),
                ~flat_mask,
            ),
            dim=1,
        )
        encoded = self.history_transformer(sequence, src_key_padding_mask=padding)[:, 0]
        raw = torch.tanh(self.output(self.output_norm(encoded)).squeeze(-1)).reshape(
            batch, candidates
        )
        raw = raw * candidate_mask.to(raw.dtype)
        repeat_request = (
            repeat_mask & candidate_mask[:, :, None] & history_mask[:, None, :]
        ).any(dim=(1, 2))
        active = history_mask.any(dim=-1) & ~repeat_request & query_present
        raw = raw * active[:, None].to(raw.dtype)
        weights = candidate_mask.to(raw.dtype)
        centered = (
            raw - (raw * weights).sum(dim=-1, keepdim=True) / weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
        ) * weights
        correction = self.score_delta_max * centered
        anchor = torch.where(repeat_request[:, None], item_only_scores, base_scores)
        scores = (anchor + correction).masked_fill(~candidate_mask, 0.0)
        return TokenBridgeOutput(
            scores=scores,
            correction=correction,
            raw_correction=raw,
            anchor_scores=anchor,
            active_request=active,
            repeat_request=repeat_request,
        )

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
