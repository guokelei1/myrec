from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.nn import functional as F


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

from model.barycentric_transport import (  # noqa: E402
    GLOBAL_CONTROL,
    MODES,
    PRIMARY,
    RELATIVE_ONLY_CONTROL,
    UNBOUNDED_CONTROL,
    UNCENTERED_CONTROL,
    LowRankBarycentricTransport,
)


def make(mode: str, seed: int = 11) -> LowRankBarycentricTransport:
    return LowRankBarycentricTransport(
        dim=8,
        rank=2,
        temperature=0.1,
        profile_scale=1.0,
        correction_scale=2.0,
        seed=seed,
        mode=mode,
    )


def fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    history = torch.tensor(
        [
            [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    candidates = torch.tensor(
        [
            [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return query, history, candidates


def test_all_modes_are_capacity_and_initialization_matched() -> None:
    models = [make(mode) for mode in MODES]
    assert {model.trainable_parameter_count() for model in models} == {32}
    for other in models[1:]:
        assert all(
            torch.equal(first, second)
            for first, second in zip(
                models[0].state_dict().values(), other.state_dict().values()
            )
        )


def test_primary_preserves_global_barycenter_and_inactive_state() -> None:
    query, history, candidates = fixture()
    state = make(PRIMARY).transport_components(query, history, candidates)
    admitted = state["admitted"]
    assert bool(admitted.any())
    assert bool((~admitted).any())
    assert torch.allclose(
        state["displacement"].mean(dim=0), state["global_write"], atol=1e-7, rtol=0
    )
    assert torch.equal(
        state["displacement"][~admitted],
        state["global_write"][None, :].expand((int((~admitted).sum()), -1)),
    )


def test_primary_trust_region_cannot_reverse_global_write() -> None:
    query, history, candidates = fixture()
    state = make(PRIMARY).transport_components(query, history, candidates)
    alignment = state["displacement"].mv(state["global_write"])
    assert bool((alignment > 0).all())
    global_norm = torch.linalg.vector_norm(state["global_write"])
    residual_max = torch.linalg.vector_norm(
        state["trust_coefficient"] * state["centered_relative"], dim=1
    ).max()
    assert residual_max < global_norm
    assert 0 < float(state["trust_coefficient"]) < 1


def test_controls_remove_one_defining_operator() -> None:
    query, history, candidates = fixture()
    primary = make(PRIMARY).transport_components(query, history, candidates)
    global_only = make(GLOBAL_CONTROL).transport_components(query, history, candidates)
    unbounded = make(UNBOUNDED_CONTROL).transport_components(query, history, candidates)
    uncentered = make(UNCENTERED_CONTROL).transport_components(query, history, candidates)
    relative_only = make(RELATIVE_ONLY_CONTROL).transport_components(
        query, history, candidates
    )
    assert torch.equal(
        global_only["displacement"],
        global_only["global_write"][None, :].expand_as(global_only["displacement"]),
    )
    assert torch.equal(unbounded["trust_coefficient"], torch.tensor(1.0))
    assert torch.allclose(
        unbounded["displacement"].mean(dim=0),
        unbounded["global_write"],
        atol=1e-7,
        rtol=0,
    )
    assert not torch.allclose(
        uncentered["displacement"].mean(dim=0),
        uncentered["global_write"],
        atol=1e-7,
        rtol=0,
    )
    assert torch.equal(relative_only["displacement"], relative_only["raw_relative"])
    assert not torch.equal(primary["displacement"], unbounded["displacement"])


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
