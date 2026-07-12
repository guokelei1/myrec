"""History-counterfactual adjacent-edge likelihood Transformer for C61."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


MODES = (
    "counterfactual_edge",
    "factual_edge",
    "ordinary_candidate_attention",
    "candidate_only_edge",
)


@dataclass(frozen=True)
class EdgeLikelihoodOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    likelihood_ratio: torch.Tensor
    pair_logits: torch.Tensor
    edge_mask: torch.Tensor
    transport: torch.Tensor
    base_gap: torch.Tensor
    factual_antisymmetric: torch.Tensor
    null_antisymmetric: torch.Tensor
    active_request: torch.Tensor


def masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight[..., None]).sum(dim=dim) / weight.sum(
        dim=dim, keepdim=True
    ).clamp_min(1.0)


def adjacent_pair_targets(
    labels: torch.Tensor,
    canonical_order: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe = canonical_order.clamp_min(0)
    ordered = torch.gather(labels, 1, safe)
    count = candidate_mask.sum(dim=-1)
    positions = torch.arange(max(0, labels.shape[1] - 1), device=labels.device)[None]
    edge_mask = positions < (count - 1).clamp_min(0)[:, None]
    high, low = ordered[:, :-1].gt(0), ordered[:, 1:].gt(0)
    eligible = edge_mask & (high != low)
    target = low.to(labels.dtype)
    return target, eligible


class CounterfactualEdgeLikelihoodTransformer(nn.Module):
    """Estimate a factual-minus-NULL likelihood ratio for adjacent base errors."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        token_layers: int,
        edge_layers: int,
        dropout: float,
        max_query_tokens: int,
        max_item_tokens: int,
        max_history: int,
        zero_initial_output: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("C61 hidden dimension must divide heads")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_history = int(max_history)
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.token_position = nn.Parameter(
            torch.zeros(max(max_query_tokens, max_item_tokens), hidden_dim)
        )
        self.query_type = nn.Parameter(torch.zeros(hidden_dim))
        self.candidate_type = nn.Parameter(torch.zeros(hidden_dim))
        self.history_type = nn.Parameter(torch.zeros(hidden_dim))
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
        self.token_transformer = nn.TransformerEncoder(
            token_layer, num_layers=token_layers, enable_nested_tensor=False
        )
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.history_norm = nn.LayerNorm(hidden_dim)
        self.edge_read = nn.Parameter(torch.zeros(hidden_dim))
        edge_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.edge_transformer = nn.TransformerEncoder(
            edge_layer, num_layers=edge_layers, enable_nested_tensor=False
        )
        self.history_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.edge_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.edge_head = nn.Linear(hidden_dim, 1, bias=False)
        self.candidate_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.candidate_head = nn.Linear(hidden_dim, 1, bias=False)
        self.reset_parameters(zero_initial_output=zero_initial_output)

    def reset_parameters(self, *, zero_initial_output: bool) -> None:
        for value in (
            self.token_position,
            self.query_type,
            self.candidate_type,
            self.history_type,
            self.history_position,
            self.edge_read,
        ):
            nn.init.normal_(value, std=0.02)
        if zero_initial_output:
            nn.init.zeros_(self.edge_head.weight)
            nn.init.zeros_(self.candidate_head.weight)
        else:
            nn.init.normal_(self.edge_head.weight, std=0.02)
            nn.init.normal_(self.candidate_head.weight, std=0.02)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def _encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.shape[:-1] != mask.shape or tokens.shape[-1] != self.input_dim:
            raise ValueError("C61 contextual token shape differs")
        prefix, length = tokens.shape[:-2], tokens.shape[-2]
        if length > self.token_position.shape[0]:
            raise ValueError("C61 token length exceeds maximum")
        flat_tokens = tokens.reshape(-1, length, self.input_dim).float()
        flat_mask = mask.reshape(-1, length).bool()
        safe = flat_mask.clone()
        empty = ~safe.any(dim=-1)
        if bool(empty.any()):
            safe[empty, 0] = True
        value = self.input_projection(flat_tokens)
        value = value + type_embedding + self.token_position[:length]
        encoded = self.token_transformer(value, src_key_padding_mask=~safe)
        encoded = encoded * flat_mask[..., None].to(encoded.dtype)
        return encoded.reshape(*prefix, length, self.hidden_dim)

    def _edge_state(
        self,
        query: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        edge_count = low.shape[1]
        read = self.edge_read.view(1, 1, -1).expand(query.shape[0], edge_count, -1)
        sequence = torch.stack(
            (
                read,
                query[:, None].expand(-1, edge_count, -1),
                low,
                high,
                low - high,
                low * high,
            ),
            dim=2,
        )
        encoded = self.edge_transformer(
            sequence.reshape(-1, sequence.shape[2], self.hidden_dim)
        )
        return encoded[:, 0].reshape(query.shape[0], edge_count, self.hidden_dim)

    def _attend_history(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe = event_mask.bool().clone()
        empty = ~safe.any(dim=-1)
        if bool(empty.any()):
            safe[empty, 0] = True
        attended, _ = self.history_attention(
            query, history, history, key_padding_mask=~safe, need_weights=False
        )
        return attended * event_mask.any(dim=-1)[:, None, None].to(attended.dtype)

    def _edge_score(self, edge: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        value = edge + attended
        value = self.edge_norm(value + self.edge_ffn(value))
        return self.edge_head(value).squeeze(-1)

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
        canonical_order: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        repeat_request: torch.Tensor,
        mode: str = "counterfactual_edge",
    ) -> EdgeLikelihoodOutput:
        if mode not in MODES:
            raise ValueError(f"unknown C61 mode: {mode}")
        batch, candidates = candidate_mask.shape
        if candidates < 2:
            edge_count = 0
        else:
            edge_count = candidates - 1
        if history_tokens.shape[1] > self.max_history:
            raise ValueError("C61 history exceeds maximum")
        query_encoded = self._encode(query_tokens, query_token_mask, self.query_type)
        candidate_encoded = self._encode(
            candidate_tokens, candidate_token_mask, self.candidate_type
        )
        history_encoded = self._encode(history_tokens, history_token_mask, self.history_type)
        query = masked_mean(query_encoded, query_token_mask.bool(), dim=1)
        candidate = masked_mean(candidate_encoded, candidate_token_mask.bool(), dim=2)
        history = masked_mean(history_encoded, history_token_mask.bool(), dim=2)
        candidate = self.candidate_norm(candidate + self.query_projection(query)[:, None])
        event_mask = history_token_mask.any(dim=-1).bool()
        positions = self.history_position[: history.shape[1]][None]
        history = self.history_norm(history + self.query_projection(query)[:, None] + positions)
        weight = event_weights.float().clamp_min(0.0) * event_mask.to(torch.float32)
        normalized = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        normalized = normalized * event_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(torch.float32)
        history = history * normalized[..., None]

        safe_order = canonical_order.clamp_min(0)
        sorted_candidate = torch.gather(
            candidate, 1, safe_order[..., None].expand(-1, -1, self.hidden_dim)
        )
        sorted_base = torch.gather(base_scores.float(), 1, safe_order)
        count = candidate_mask.sum(dim=-1)
        edge_positions = torch.arange(edge_count, device=base_scores.device)[None]
        edge_mask = edge_positions < (count - 1).clamp_min(0)[:, None]
        high, low = sorted_candidate[:, :-1], sorted_candidate[:, 1:]
        forward_edge = self._edge_state(query, low, high)
        reverse_edge = self._edge_state(query, high, low)
        forward_attended = self._attend_history(forward_edge, history, event_mask)
        reverse_attended = self._attend_history(reverse_edge, history, event_mask)
        zeros_forward = torch.zeros_like(forward_attended)
        zeros_reverse = torch.zeros_like(reverse_attended)
        factual = 0.5 * (
            self._edge_score(forward_edge, forward_attended)
            - self._edge_score(reverse_edge, reverse_attended)
        )
        null = 0.5 * (
            self._edge_score(forward_edge, zeros_forward)
            - self._edge_score(reverse_edge, zeros_reverse)
        )
        candidate_attended = self._attend_history(candidate, history, event_mask)
        candidate_value = candidate + candidate_attended
        candidate_logit = self.candidate_head(
            candidate_value + self.candidate_ffn(candidate_value)
        ).squeeze(-1)
        sorted_candidate_logit = torch.gather(candidate_logit, 1, safe_order)
        ordinary = sorted_candidate_logit[:, 1:] - sorted_candidate_logit[:, :-1]
        if mode == "counterfactual_edge":
            likelihood = factual - null
        elif mode == "factual_edge":
            likelihood = factual
        elif mode == "ordinary_candidate_attention":
            likelihood = ordinary
        else:
            likelihood = null
        likelihood = likelihood * edge_mask.to(likelihood.dtype)
        gap = (sorted_base[:, :-1] - sorted_base[:, 1:]).clamp_min(0.0)
        pair_logits = -gap + likelihood
        baseline = torch.sigmoid(-gap)
        conditioned = torch.sigmoid(pair_logits)
        rate = ((conditioned - baseline) / (1.0 - baseline).clamp_min(1e-12)).clamp(0.0, 1.0)
        query_present = query_token_mask.any(dim=-1).bool()
        active = query_present & event_mask.any(dim=-1) & ~repeat_request.bool()
        transport = rate * gap * edge_mask.to(rate.dtype) * active[:, None].to(rate.dtype)
        correction = torch.zeros_like(base_scores, dtype=torch.float32)
        for row in range(batch):
            edge_total = max(0, int(count[row]) - 1)
            if not edge_total:
                continue
            order = safe_order[row, : int(count[row])]
            sorted_correction = torch.zeros(int(count[row]), device=base_scores.device)
            sorted_correction[:-1] -= transport[row, :edge_total]
            sorted_correction[1:] += transport[row, :edge_total]
            correction[row, order] = sorted_correction
        correction = correction * candidate_mask.to(correction.dtype)
        scores = base_scores.float() + correction
        scores = torch.where(repeat_request[:, None].bool(), item_only_scores.float(), scores)
        scores = scores.masked_fill(~candidate_mask.bool(), 0.0)
        return EdgeLikelihoodOutput(
            scores=scores,
            correction=correction,
            likelihood_ratio=likelihood,
            pair_logits=pair_logits,
            edge_mask=edge_mask,
            transport=transport,
            base_gap=gap,
            factual_antisymmetric=factual * edge_mask.to(factual.dtype),
            null_antisymmetric=null * edge_mask.to(null.dtype),
            active_request=active,
        )
