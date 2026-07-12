from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.nn import functional as F


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

from model.relative_surplus import (  # noqa: E402
    ABSOLUTE_CONTROL,
    AXIS_SOFTMAX_CONTROL,
    GLOBAL_CONTROL,
    MODES,
    PRIMARY,
    LowRankRelativeSurplusAttention,
)


def make(mode: str, seed: int = 11) -> LowRankRelativeSurplusAttention:
    return LowRankRelativeSurplusAttention(
        dim=8,
        rank=2,
        temperature=0.1,
        profile_scale=1.0,
        correction_scale=2.0,
        seed=seed,
        mode=mode,
    )


def test_all_modes_are_capacity_and_initialization_matched() -> None:
    models = [make(mode) for mode in MODES]
    assert {model.trainable_parameter_count() for model in models} == {32}
    for other in models[1:]:
        assert all(
            torch.equal(first, second)
            for first, second in zip(models[0].state_dict().values(), other.state_dict().values())
        )


def test_relative_surplus_rejects_above_zero_but_below_competitor_evidence() -> None:
    query = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    history = torch.tensor([[0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    candidate_directions = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
        ]
    )
    candidates = torch.zeros(3, 8)
    candidates[:, 0] = 0.6
    candidates[:, 1:3] = 0.8 * candidate_directions
    primary = make(PRIMARY)
    absolute = make(ABSOLUTE_CONTROL)
    _, _, _, _, relative_support = primary.transport_state(query, history, candidates)
    _, _, _, _, absolute_support = absolute.transport_state(query, history, candidates)
    assert bool((absolute_support[:, 0] > 0).all())
    assert relative_support[0, 0] > 0
    assert relative_support[1, 0] < 1e-6
    assert relative_support[2, 0] == 0


def test_axis_softmax_competes_across_candidates_and_global_is_shared() -> None:
    query = F.normalize(torch.randn(8), dim=0)
    history = F.normalize(torch.randn(4, 8), dim=1)
    candidates = F.normalize(torch.randn(6, 8), dim=1)
    axis = make(AXIS_SOFTMAX_CONTROL)
    _, _, axis_displacement, _, allocation = axis.transport_state(
        query, history, candidates
    )
    assert torch.allclose(allocation.sum(dim=0), torch.ones(4), atol=1e-6, rtol=0)
    assert bool((allocation > 0).all())
    global_model = make(GLOBAL_CONTROL)
    _, _, global_displacement, _, _ = global_model.transport_state(
        query, history, candidates
    )
    assert torch.equal(global_displacement, global_displacement[:1].expand_as(global_displacement))
    assert not torch.equal(axis_displacement, global_displacement)


def test_all_modes_are_candidate_permutation_equivariant_and_null_safe() -> None:
    query = F.normalize(torch.randn(8), dim=0)
    history = F.normalize(torch.randn(3, 8), dim=1)
    candidates = F.normalize(torch.randn(5, 8), dim=1)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    for mode in MODES:
        model = make(mode)
        first = model(query, history, candidates)
        second = model(query, history, candidates[permutation])
        assert torch.allclose(second, first[permutation], atol=1e-6, rtol=0)
        null = model(query, torch.empty(0, 8), candidates)
        assert torch.equal(null, torch.zeros_like(null))


def test_every_mode_has_finite_adapter_gradients() -> None:
    query = F.normalize(torch.randn(8), dim=0)
    history = F.normalize(torch.randn(4, 8), dim=1)
    candidates = F.normalize(torch.randn(7, 8), dim=1)
    for mode in MODES:
        model = make(mode)
        loss = model(query, history, candidates).square().mean()
        loss.backward()
        assert all(parameter.grad is not None for parameter in model.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
