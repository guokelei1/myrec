"""C78 event-set-equivariant token interaction Transformer."""

from __future__ import annotations

import torch
from torch import nn

from probe.c77_operator import (
    CANDIDATE,
    HISTORY,
    QUERY,
    GraphDiagnostics,
    InteractionTransformer,
    authenticated_graphs,
    structured_anchor_table,
)


MODES = (
    "query_filtered_set",
    "positional_query_filter",
    "ungated_set",
    "pairwise_set",
    "triadic_set",
)

GRAPH_MODE = {
    "query_filtered_set": "query_candidate_filter",
    "positional_query_filter": "query_candidate_filter",
    "ungated_set": "ungated_full",
    "pairwise_set": "pairwise_candidate_history",
    "triadic_set": "query_authenticated_subgraph",
}


def event_set_position_ids(
    segment_ids: torch.Tensor, event_token_width: int
) -> torch.Tensor:
    batch, length = segment_ids.shape
    ordinary = torch.arange(length, device=segment_ids.device).expand(batch, -1)
    history = segment_ids.eq(HISTORY)
    origin = history.float().argmax(-1).long()
    within = torch.remainder(ordinary - origin[:, None], int(event_token_width))
    return torch.where(history, origin[:, None] + within, ordinary)


class SetInteractionTransformer(InteractionTransformer):
    def __init__(self, *args, set_positions: bool, event_token_width: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_positions = bool(set_positions)
        self.event_token_width = int(event_token_width)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        if self.set_positions:
            position_ids = event_set_position_ids(segment_ids, self.event_token_width)
        else:
            position_ids = torch.arange(token_ids.shape[1], device=token_ids.device).expand_as(token_ids)
        states = (
            self.token_embedding(token_ids.long())
            + self.position_embedding(position_ids.long())
            + self.segment_embedding(segment_ids.long())
        )
        for layer in self.layers:
            states = layer(states, allowed)
        return self.final_norm(states)


class QueryFilteredSetInteractionTransformer(nn.Module):
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
        history_event_token_width: int,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C78 mode: {mode}")
        self.mode = mode
        self.graph_mode = GRAPH_MODE[mode]
        self.correction_bound = float(correction_bound)
        self.interaction = SetInteractionTransformer(
            vocabulary_size,
            hidden_size,
            attention_heads,
            interaction_layers,
            maximum_length,
            set_positions=mode != "positional_query_filter",
            event_token_width=history_event_token_width,
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
        tokens = token_ids.reshape(batch * candidates, length)
        segments = segment_ids.reshape(batch * candidates, length)
        present = history_present[:, None].expand(-1, candidates).reshape(-1)
        full, cut, active = authenticated_graphs(
            tokens,
            segments,
            present,
            self.semantic_anchors,
            self.graph_mode,
        )
        factual = self.interaction(tokens, segments, full)
        counterfactual = self.interaction(tokens, segments, cut)
        delta = self.score_head(factual[:, 0]) - self.score_head(counterfactual[:, 0])
        delta = self.correction_bound * torch.tanh(delta).reshape(batch, candidates)
        delta = delta - delta.mean(-1, keepdim=True)
        personalized = base_scores.detach() + delta
        query_present = (
            self.semantic_anchors[tokens]
            .square()
            .sum(-1)
            .gt(0)
            .logical_and(segments.eq(QUERY))
            .any(-1)
            .reshape(batch, candidates)
            .all(-1)
        )
        enabled = history_present.bool() & query_present
        scores = torch.where(enabled[:, None], personalized, base_scores.detach())
        scores = torch.where(repeat_present.bool()[:, None], repeat_scores.detach(), scores)
        c_to_h = (
            full
            & segments.unsqueeze(2).eq(CANDIDATE)
            & segments.unsqueeze(1).eq(HISTORY)
        ).sum()
        h_to_c = (
            full
            & segments.unsqueeze(2).eq(HISTORY)
            & segments.unsqueeze(1).eq(CANDIDATE)
        ).sum()
        return GraphDiagnostics(
            scores=scores,
            active_fraction=active.float().mean(),
            candidate_history_edges=c_to_h,
            history_candidate_edges=h_to_c,
        )
