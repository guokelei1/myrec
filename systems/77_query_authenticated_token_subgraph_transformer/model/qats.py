"""Executable query-authenticated token-subgraph Transformer for C77."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


QUERY = 0
CANDIDATE = 1
HISTORY = 2

MODES = (
    "query_authenticated_subgraph",
    "ungated_full",
    "query_history_filter",
    "query_candidate_filter",
    "pairwise_candidate_history",
)

ATTR_BASE = 10
PAIR_BASE = 32


def structured_anchor_table(
    vocabulary: int,
    attributes: int,
    values_per_attribute: int,
    anchor_dimension: int,
) -> torch.Tensor:
    """Simulate a frozen pretrained semantic coordinate on discrete tokens."""

    if anchor_dimension < attributes + values_per_attribute:
        raise ValueError("C77 anchor dimension is too small")
    table = torch.zeros(vocabulary, anchor_dimension)
    for attribute in range(attributes):
        table[ATTR_BASE + attribute, attribute] = 1.0
        for value in range(values_per_attribute):
            token = PAIR_BASE + attribute * values_per_attribute + value
            table[token, attribute] = 1.0
            table[token, attributes + value] = 1.0
            table[token] = nn.functional.normalize(table[token], dim=0)
    return table


class InteractionBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, states: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(states)
        blocked = (~allowed.bool()).repeat_interleave(self.heads, dim=0)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=blocked,
            need_weights=False,
        )
        states = states + update
        return states + self.ffn(self.ffn_norm(states))


class InteractionTransformer(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        hidden: int,
        heads: int,
        layers: int,
        maximum_length: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary, hidden)
        self.position_embedding = nn.Embedding(maximum_length, hidden)
        self.segment_embedding = nn.Embedding(3, hidden)
        self.layers = nn.ModuleList(
            [InteractionBlock(hidden, heads) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        states = (
            self.token_embedding(token_ids.long())
            + self.position_embedding(positions).unsqueeze(0)
            + self.segment_embedding(segment_ids.long())
        )
        for layer in self.layers:
            states = layer(states, allowed)
        return self.final_norm(states)


@dataclass
class GraphDiagnostics:
    scores: torch.Tensor
    active_fraction: torch.Tensor
    candidate_history_edges: torch.Tensor
    history_candidate_edges: torch.Tensor


def _self_safe(allowed: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    length = allowed.shape[-1]
    eye = torch.eye(length, dtype=torch.bool, device=allowed.device).unsqueeze(0)
    return allowed | (eye & ~active.unsqueeze(2))


def authenticated_graphs(
    token_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    history_present: torch.Tensor,
    anchors: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode not in MODES:
        raise ValueError(f"unknown C77 mode: {mode}")
    anchor = anchors[token_ids.long()]
    similarity = torch.relu(torch.einsum("bld,bmd->blm", anchor, anchor))
    q_mask = segment_ids.eq(QUERY)
    c_mask = segment_ids.eq(CANDIDATE)
    h_mask = segment_ids.eq(HISTORY)
    query_semantic = q_mask & anchor.square().sum(-1).gt(0)
    query_present = query_semantic.any(-1)

    query_key = query_semantic.unsqueeze(1)
    to_query = similarity * query_key
    query_support = to_query.max(-1).values
    pairwise = similarity * c_mask.unsqueeze(2) * h_mask.unsqueeze(1)
    reverse_pairwise = similarity * h_mask.unsqueeze(2) * c_mask.unsqueeze(1)

    # Shared-query triangle for every ordered token pair.  Direct semantic
    # compatibility is required in addition to the shared query anchor.
    left_query = to_query.unsqueeze(2)
    right_query = to_query.unsqueeze(1)
    shared_query = (left_query * right_query).amax(-1)
    triangle = shared_query * similarity

    if mode == "query_authenticated_subgraph":
        c_active = c_mask & (triangle * h_mask.unsqueeze(1)).amax(-1).gt(0)
        h_active = h_mask & (triangle * c_mask.unsqueeze(1)).amax(-1).gt(0)
    elif mode == "ungated_full":
        c_active, h_active = c_mask, h_mask
    elif mode == "query_history_filter":
        c_active = c_mask
        h_active = h_mask & query_support.gt(0)
    elif mode == "query_candidate_filter":
        c_active = c_mask & query_support.gt(0)
        h_active = h_mask
    else:
        c_active = c_mask & pairwise.amax(-1).gt(0)
        h_active = h_mask & reverse_pairwise.amax(-1).gt(0)

    enabled_request = history_present.bool() & query_present
    h_active = h_active & enabled_request.unsqueeze(1)
    c_active = c_active & query_present.unsqueeze(1)
    active = q_mask | c_active | h_active
    full = active.unsqueeze(2) & active.unsqueeze(1)
    full = _self_safe(full, active)
    cross_history = segment_ids.unsqueeze(2).eq(HISTORY) ^ segment_ids.unsqueeze(1).eq(HISTORY)
    cut = full & ~cross_history
    cut = _self_safe(cut, active)
    return full, cut, active


class QueryAuthenticatedTokenSubgraphTransformer(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        vocabulary_size: int,
        hidden_size: int,
        attention_heads: int,
        interaction_layers: int,
        maximum_length: int,
        correction_bound: float,
        attributes: int,
        values_per_attribute: int,
        anchor_dimension: int,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C77 mode: {mode}")
        self.mode = mode
        self.correction_bound = float(correction_bound)
        self.interaction = InteractionTransformer(
            vocabulary_size,
            hidden_size,
            attention_heads,
            interaction_layers,
            maximum_length,
        )
        self.score_head = nn.Linear(hidden_size, 1, bias=False)
        self.register_buffer(
            "semantic_anchors",
            structured_anchor_table(
                vocabulary_size,
                attributes,
                values_per_attribute,
                anchor_dimension,
            ),
            persistent=True,
        )

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        base_scores: torch.Tensor,
        history_present: torch.Tensor,
        repeat_present: torch.Tensor,
        repeat_scores: torch.Tensor,
    ) -> GraphDiagnostics:
        batch, candidates, length = token_ids.shape
        flat_tokens = token_ids.reshape(batch * candidates, length)
        flat_segments = segment_ids.reshape(batch * candidates, length)
        flat_history = history_present[:, None].expand(-1, candidates).reshape(-1)
        full, cut, active = authenticated_graphs(
            flat_tokens,
            flat_segments,
            flat_history,
            self.semantic_anchors,
            self.mode,
        )
        factual = self.interaction(flat_tokens, flat_segments, full)
        counterfactual = self.interaction(flat_tokens, flat_segments, cut)
        delta = self.score_head(factual[:, 0]) - self.score_head(counterfactual[:, 0])
        delta = self.correction_bound * torch.tanh(delta).reshape(batch, candidates)
        delta = delta - delta.mean(-1, keepdim=True)
        personalized = base_scores.detach() + delta
        query_has_semantics = (
            self.semantic_anchors[flat_tokens]
            .square()
            .sum(-1)
            .gt(0)
            .logical_and(flat_segments.eq(QUERY))
            .any(-1)
            .reshape(batch, candidates)
            .all(-1)
        )
        enabled = history_present.bool() & query_has_semantics
        scores = torch.where(enabled.unsqueeze(1), personalized, base_scores.detach())
        scores = torch.where(
            repeat_present.bool().unsqueeze(1), repeat_scores.detach(), scores
        )
        c_to_h = (
            full
            & flat_segments.unsqueeze(2).eq(CANDIDATE)
            & flat_segments.unsqueeze(1).eq(HISTORY)
        ).sum()
        h_to_c = (
            full
            & flat_segments.unsqueeze(2).eq(HISTORY)
            & flat_segments.unsqueeze(1).eq(CANDIDATE)
        ).sum()
        return GraphDiagnostics(
            scores=scores,
            active_fraction=active.float().mean(),
            candidate_history_edges=c_to_h,
            history_candidate_edges=h_to_c,
        )
