from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.semantic_routing import (  # noqa: E402
    MODES,
    COUPLED_CONTENT,
    SEMANTIC_ROUTING,
    SemanticCarrierRoutingTransformer,
    fixed_semantic_correction,
)


def make(mode: str, seed: int = 11) -> SemanticCarrierRoutingTransformer:
    return SemanticCarrierRoutingTransformer(
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
    generator = torch.Generator().manual_seed(115)
    return (
        torch.randn(16, generator=generator),
        torch.randn(6, 16, generator=generator),
        torch.randn(8, 16, generator=generator),
    )


def test_equal_capacity_and_initial_state() -> None:
    models = [make(mode) for mode in MODES]
    assert {model.trainable_parameter_count() for model in models} == {256}
    reference = models[0].state_dict()
    for model in models[1:]:
        for name, value in model.state_dict().items():
            assert torch.equal(value, reference[name])


@pytest.mark.parametrize("mode", MODES)
def test_fallbacks_and_permutation(mode: str) -> None:
    model = make(mode)
    query, history, candidates = inputs()
    permutation = torch.tensor([4, 0, 7, 1, 6, 2, 5, 3])
    expected = model(query, history, candidates)[permutation]
    actual = model(query, history, candidates[permutation])
    assert torch.allclose(actual, expected, atol=2e-7, rtol=0)
    assert torch.equal(model(query, history[:0], candidates), torch.zeros(8))
    assert torch.equal(
        model(query, history, candidates, query_present=False), torch.zeros(8)
    )
    assert torch.equal(
        model(query, history, candidates, repeat_present=True), torch.zeros(8)
    )


def test_primary_profile_is_raw_semantic_convex_combination() -> None:
    model = make(SEMANTIC_ROUTING)
    query, history, candidates = inputs()
    state = model.components(query, history, candidates)
    raw_history = torch.nn.functional.normalize(history, dim=-1)
    expected = torch.einsum("hj,jd->hd", state["attention"], raw_history)
    assert state["semantic_carrier_exact"]
    assert torch.allclose(state["profile"], expected, atol=1e-7, rtol=0)
    assert torch.all(state["attention"] >= 0)
    assert torch.allclose(
        state["attention"].sum(dim=-1), torch.ones(4), atol=2e-7, rtol=0
    )


def test_coupled_control_rewrites_carrier() -> None:
    query, history, candidates = inputs()
    state = make(COUPLED_CONTENT).components(query, history, candidates)
    assert not state["semantic_carrier_exact"]
    raw = torch.nn.functional.normalize(history, dim=-1)
    assert not torch.allclose(state["carrier_history"][0], raw)


@pytest.mark.parametrize("mode", MODES)
def test_both_factors_receive_gradient(mode: str) -> None:
    model = make(mode)
    query, history, candidates = inputs()
    loss = (
        model(query, history, candidates)
        * torch.linspace(-1, 1, len(candidates))
    ).sum()
    loss.backward()
    assert model.down.grad is not None and torch.count_nonzero(model.down.grad)
    assert model.up.grad is not None and torch.count_nonzero(model.up.grad)


def test_parameter_free_semantic_control() -> None:
    query, history, candidates = inputs()
    value = fixed_semantic_correction(
        query,
        history,
        candidates,
        temperature=0.2,
        profile_scale=1.0,
        correction_scale=2.0,
    )
    assert value.shape == (8,) and torch.isfinite(value).all()
