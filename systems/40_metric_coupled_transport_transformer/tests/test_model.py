from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.metric_coupled import (  # noqa: E402
    MODES,
    MULTIHEAD_COUPLED,
    SHIFTED_LOOP,
    MetricCoupledTransportTransformer,
)


def make(mode: str, seed: int = 7) -> MetricCoupledTransportTransformer:
    return MetricCoupledTransportTransformer(
        dim=16,
        heads=4,
        rank=2,
        temperature=0.2,
        profile_scale=1.0,
        correction_scale=2.0,
        seed=seed,
        mode=mode,
    )


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    return (
        torch.randn(16, generator=generator),
        torch.randn(5, 16, generator=generator),
        torch.randn(7, 16, generator=generator),
    )


def test_modes_have_equal_parameters_and_initial_state() -> None:
    models = [make(mode) for mode in MODES]
    assert {model.trainable_parameter_count() for model in models} == {256}
    reference = models[0].state_dict()
    for model in models[1:]:
        for name, value in model.state_dict().items():
            assert torch.equal(value, reference[name])


@pytest.mark.parametrize("mode", MODES)
def test_exact_fallbacks(mode: str) -> None:
    model = make(mode)
    query, history, candidates = inputs()
    assert torch.equal(model(query, history[:0], candidates), torch.zeros(7))
    assert torch.equal(
        model(query, history, candidates, query_present=False), torch.zeros(7)
    )
    assert torch.equal(
        model(query, history, candidates, repeat_present=True), torch.zeros(7)
    )


@pytest.mark.parametrize("mode", MODES)
def test_candidate_permutation(mode: str) -> None:
    model = make(mode)
    query, history, candidates = inputs()
    permutation = torch.tensor([3, 0, 5, 1, 6, 2, 4])
    expected = model(query, history, candidates)[permutation]
    actual = model(query, history, candidates[permutation])
    assert torch.allclose(actual, expected, atol=2e-7, rtol=0)


def test_loop_assignments_differ() -> None:
    query, history, candidates = inputs()
    primary = make(MULTIHEAD_COUPLED).components(query, history, candidates)
    shifted = make(SHIFTED_LOOP).components(query, history, candidates)
    assert torch.equal(primary["loop_assignment"], torch.arange(4))
    assert torch.equal(shifted["loop_assignment"], torch.tensor([1, 2, 3, 0]))
    assert not torch.equal(primary["correction"], shifted["correction"])


@pytest.mark.parametrize("mode", MODES)
def test_both_factors_receive_gradient(mode: str) -> None:
    model = make(mode)
    query, history, candidates = inputs()
    weights = torch.linspace(-1.0, 1.0, len(candidates))
    loss = (model(query, history, candidates) * weights).sum()
    loss.backward()
    assert model.down.grad is not None and torch.count_nonzero(model.down.grad)
    assert model.up.grad is not None and torch.count_nonzero(model.up.grad)
    assert torch.isfinite(model.down.grad).all()
    assert torch.isfinite(model.up.grad).all()
