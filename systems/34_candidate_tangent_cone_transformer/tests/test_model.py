from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.nn import functional as F


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

from model.tangent_cone import (  # noqa: E402
    GLOBAL_CONTROL,
    MODES,
    PRIMARY,
    TARGET_CONTROL,
    LowRankTangentConeAttention,
)


def make(mode: str, seed: int = 11) -> LowRankTangentConeAttention:
    return LowRankTangentConeAttention(
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


def test_cone_writes_only_positive_query_centered_directions() -> None:
    query = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    candidates = torch.tensor(
        [
            [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    history = torch.tensor(
        [
            [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    model = make(PRIMARY)
    _, _, displacement, _, support = model.transport_state(query, history, candidates)
    assert support[0, 0] > 0.99
    assert support[0, 1] == 0
    assert torch.equal(support[1], torch.zeros_like(support[1]))
    assert torch.equal(displacement[1], torch.zeros_like(displacement[1]))
    assert torch.dot(displacement[0], candidates[0] - candidates[0].dot(query) * query) > 0


def test_controls_isolate_forced_assignment_and_candidate_sharing() -> None:
    query = F.normalize(torch.randn(8), dim=0)
    history = F.normalize(torch.randn(4, 8), dim=1)
    candidates = F.normalize(torch.randn(6, 8), dim=1)
    target = make(TARGET_CONTROL)
    _, _, target_displacement, _, target_mass = target.transport_state(
        query, history, candidates
    )
    assert torch.allclose(target_mass.sum(dim=1), torch.ones(6), atol=1e-6, rtol=0)
    assert bool((target_mass > 0).all())
    global_model = make(GLOBAL_CONTROL)
    _, _, global_displacement, _, _ = global_model.transport_state(
        query, history, candidates
    )
    assert torch.equal(global_displacement, global_displacement[:1].expand_as(global_displacement))
    assert not torch.equal(target_displacement, global_displacement)


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
