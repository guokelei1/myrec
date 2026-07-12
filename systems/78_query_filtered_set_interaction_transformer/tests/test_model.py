import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.qfsi import (  # noqa: E402
    HISTORY,
    QueryFilteredSetInteractionTransformer,
    event_set_position_ids,
)
from probe.c76_surface import NUISANCE_POSITIVE, make_surface  # noqa: E402


def model(mode: str = "query_filtered_set") -> QueryFilteredSetInteractionTransformer:
    torch.manual_seed(7)
    return QueryFilteredSetInteractionTransformer(
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
        history_event_token_width=4,
    )


def batch():
    return make_surface(
        requests=16,
        candidates=8,
        history_events=6,
        attributes=8,
        values_per_attribute=8,
        seed=13,
        split="validation",
    )


def call(network, surface, field="tokens"):
    return network(
        getattr(surface, field),
        surface.segments,
        surface.base_scores,
        surface.history_present,
        surface.repeat_present,
        surface.repeat_scores,
    ).scores


def test_history_events_share_within_event_positions() -> None:
    surface = batch()
    segments = surface.segments[0]
    positions = event_set_position_ids(segments, 4)
    history = segments[0].eq(HISTORY)
    values = positions[0, history].reshape(6, 4)
    assert torch.equal(values, values[0].expand_as(values))


def test_set_modes_are_event_permutation_invariant() -> None:
    surface = batch()
    for mode in ("query_filtered_set", "ungated_set", "pairwise_set", "triadic_set"):
        network = model(mode)
        assert torch.allclose(
            call(network, surface),
            call(network, surface, "shuffled_tokens"),
            atol=1e-6,
            rtol=0,
        )


def test_primary_excludes_nuisance_gradient_and_preserves_fallbacks() -> None:
    surface = batch()
    network = model()
    output = call(network, surface)
    output[:, 0].sum().backward()
    gradient = network.interaction.token_embedding.weight.grad[NUISANCE_POSITIVE]
    assert torch.equal(gradient, torch.zeros_like(gradient))
    nohistory = batch()
    nohistory.history_present[:] = False
    nohistory.repeat_present[:] = False
    nohistory.base_scores = torch.randn_like(nohistory.base_scores)
    assert torch.equal(call(network, nohistory), nohistory.base_scores)
    query = batch()
    query.repeat_present[:] = False
    query.base_scores = torch.randn_like(query.base_scores)
    assert torch.equal(call(network, query, "query_masked_tokens"), query.base_scores)


def test_modes_have_equal_trainable_capacity() -> None:
    counts = {
        mode: model(mode).parameter_count()
        for mode in (
            "query_filtered_set",
            "positional_query_filter",
            "ungated_set",
            "pairwise_set",
            "triadic_set",
        )
    }
    assert len(set(counts.values())) == 1
