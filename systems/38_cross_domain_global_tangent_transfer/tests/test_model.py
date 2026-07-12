from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.global_tangent import (  # noqa: E402
    MEAN_TANGENT,
    MODES,
    PRIMARY,
    UNPROJECTED,
    LowRankGlobalTangentTransfer,
)


def _model(mode: str, seed: int = 17) -> LowRankGlobalTangentTransfer:
    return LowRankGlobalTangentTransfer(
        dim=12,
        rank=3,
        temperature=0.1,
        profile_scale=1.0,
        correction_scale=2.0,
        seed=seed,
        mode=mode,
    )


def _states() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    return (
        torch.randn(12, generator=generator),
        torch.randn(5, 12, generator=generator),
        torch.randn(9, 12, generator=generator),
    )


def test_modes_have_equal_capacity_and_paired_initialization() -> None:
    models = [_model(mode) for mode in MODES]
    assert {model.trainable_parameter_count() for model in models} == {72}
    for key in models[0].state_dict():
        assert all(torch.equal(models[0].state_dict()[key], model.state_dict()[key]) for model in models[1:])


@pytest.mark.parametrize("mode", MODES)
def test_nohistory_is_exact_zero(mode: str) -> None:
    query, _, candidates = _states()
    correction = _model(mode)(query, torch.empty(0, 12), candidates)
    assert torch.equal(correction, torch.zeros_like(correction))


@pytest.mark.parametrize("mode", MODES)
def test_query_absent_is_exact_zero(mode: str) -> None:
    query, history, candidates = _states()
    correction = _model(mode)(query, history, candidates, query_present=False)
    assert torch.equal(correction, torch.zeros_like(correction))


def test_primary_displacement_is_tangent() -> None:
    query, history, candidates = _states()
    state = _model(PRIMARY).transport_components(query, history, candidates)
    error = torch.dot(state["adapted_query"], state["displacement"]).abs()
    assert float(error) < 1e-6


def test_unprojected_and_mean_are_exact_reductions() -> None:
    query, history, candidates = _states()
    primary = _model(PRIMARY).transport_components(query, history, candidates)
    unprojected = _model(UNPROJECTED).transport_components(query, history, candidates)
    mean = _model(MEAN_TANGENT).transport_components(query, history, candidates)
    assert torch.allclose(primary["weights"], unprojected["weights"])
    assert torch.allclose(unprojected["displacement"], unprojected["profile"])
    assert torch.allclose(mean["weights"], torch.full_like(mean["weights"], 0.2))
    assert not torch.allclose(primary["displacement"], unprojected["displacement"])
    assert not torch.allclose(primary["displacement"], mean["displacement"])


@pytest.mark.parametrize("mode", MODES)
def test_candidate_permutation_equivariance(mode: str) -> None:
    query, history, candidates = _states()
    permutation = torch.tensor([8, 2, 5, 0, 7, 1, 4, 6, 3])
    model = _model(mode)
    expected = model(query, history, candidates)[permutation]
    actual = model(query, history, candidates[permutation])
    assert float((expected - actual).abs().max()) <= 1e-6


@pytest.mark.parametrize("mode", MODES)
def test_adapter_receives_finite_gradients(mode: str) -> None:
    query, history, candidates = _states()
    model = _model(mode)
    loss = model(query, history, candidates).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert any(float(parameter.grad.abs().sum()) > 0 for parameter in model.parameters())
