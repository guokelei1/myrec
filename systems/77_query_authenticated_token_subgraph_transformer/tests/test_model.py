import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.qats import (  # noqa: E402
    CANDIDATE,
    HISTORY,
    QUERY,
    QueryAuthenticatedTokenSubgraphTransformer,
    authenticated_graphs,
    structured_anchor_table,
)
from probe.c76_surface import NUISANCE_POSITIVE, make_surface  # noqa: E402


def network(mode: str = "query_authenticated_subgraph") -> QueryAuthenticatedTokenSubgraphTransformer:
    torch.manual_seed(5)
    return QueryAuthenticatedTokenSubgraphTransformer(
        mode=mode,
        vocabulary_size=256,
        hidden_size=32,
        attention_heads=4,
        interaction_layers=2,
        maximum_length=46,
        correction_bound=3.0,
        attributes=8,
        values_per_attribute=8,
        anchor_dimension=16,
    )


def batch():
    return make_surface(
        requests=16,
        candidates=8,
        history_events=6,
        attributes=8,
        values_per_attribute=8,
        seed=11,
        split="validation",
    )


def test_primary_excludes_nuisance_and_keeps_bidirectional_ch_edges() -> None:
    surface = batch()
    index = int(torch.nonzero(surface.strata.eq(0), as_tuple=False).flatten()[0])
    tokens = surface.tokens[index]
    segments = surface.segments[index]
    present = surface.history_present[index].expand(8)
    anchors = structured_anchor_table(256, 8, 8, 16)
    full, cut, active = authenticated_graphs(
        tokens, segments, present, anchors, "query_authenticated_subgraph"
    )
    nuisance_positions = tokens.eq(NUISANCE_POSITIVE)
    assert not bool(active[nuisance_positions].any())
    assert bool(
        (
            full
            & segments.unsqueeze(2).eq(CANDIDATE)
            & segments.unsqueeze(1).eq(HISTORY)
        ).any()
    )
    assert bool(
        (
            full
            & segments.unsqueeze(2).eq(HISTORY)
            & segments.unsqueeze(1).eq(CANDIDATE)
        ).any()
    )
    assert not torch.equal(full, cut)


def test_nohistory_querymask_repeat_and_permutation_contracts() -> None:
    model = network()
    surface = batch().subset(torch.arange(4))
    output = model(
        surface.tokens,
        surface.segments,
        surface.base_scores,
        torch.zeros_like(surface.history_present),
        torch.zeros_like(surface.repeat_present),
        surface.repeat_scores,
    )
    assert torch.equal(output.scores, surface.base_scores)
    query_mask = model(
        surface.query_masked_tokens,
        surface.segments,
        surface.base_scores,
        surface.history_present,
        torch.zeros_like(surface.repeat_present),
        surface.repeat_scores,
    )
    assert torch.equal(query_mask.scores, surface.base_scores)
    repeat_scores = torch.randn_like(surface.repeat_scores)
    repeat = model(
        surface.tokens,
        surface.segments,
        surface.base_scores,
        surface.history_present,
        torch.ones_like(surface.repeat_present),
        repeat_scores,
    )
    assert torch.equal(repeat.scores, repeat_scores)
    clean = model(
        surface.tokens,
        surface.segments,
        surface.base_scores,
        surface.history_present,
        surface.repeat_present,
        surface.repeat_scores,
    ).scores
    reversed_scores = model(
        surface.tokens.flip(1),
        surface.segments.flip(1),
        surface.base_scores.flip(1),
        surface.history_present,
        surface.repeat_present,
        surface.repeat_scores.flip(1),
    ).scores.flip(1)
    assert torch.allclose(clean, reversed_scores, atol=1e-6, rtol=0)


def test_unsupported_nuisance_has_zero_personalized_gradient() -> None:
    model = network()
    surface = batch().subset(torch.arange(8))
    output = model(
        surface.tokens,
        surface.segments,
        surface.base_scores,
        surface.history_present,
        torch.zeros_like(surface.repeat_present),
        surface.repeat_scores,
    )
    output.scores[:, 0].sum().backward()
    gradient = model.interaction.token_embedding.weight.grad[NUISANCE_POSITIVE]
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_modes_match_trainable_parameter_count() -> None:
    counts = {
        mode: network(mode).parameter_count()
        for mode in (
            "query_authenticated_subgraph",
            "ungated_full",
            "query_history_filter",
            "query_candidate_filter",
            "pairwise_candidate_history",
        )
    }
    assert len(set(counts.values())) == 1
