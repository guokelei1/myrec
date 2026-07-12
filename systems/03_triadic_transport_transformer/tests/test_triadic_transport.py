from __future__ import annotations

import math

import torch

from model.triadic_transport import (
    TriadicTransportRanker,
    center_request_residual,
    dustbin_marginals,
    dustbin_transport,
    plan_marginal_error,
)


def _model(dtype: torch.dtype = torch.float64) -> TriadicTransportRanker:
    torch.manual_seed(7)
    model = TriadicTransportRanker(
        input_dim=8,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        ff_dim=16,
        max_history=3,
        transport_dim=4,
        sinkhorn_iterations=80,
        sinkhorn_temperature=0.5,
        cycle_lambda=1.0,
        identity_bonus_floor=4.0,
        dropout=0.0,
    )
    return model.to(dtype=dtype)


def test_hand_computed_one_by_one_uniform_plan() -> None:
    scores = torch.zeros((1, 1), dtype=torch.float64)
    plan = dustbin_transport(scores, 0.0, iterations=20, temperature=1.0)
    expected = torch.full((2, 2), 0.25, dtype=torch.float64)
    torch.testing.assert_close(plan, expected, rtol=0.0, atol=1e-12)
    assert float(plan_marginal_error(plan)) <= 1e-12


def test_rectangular_mass_conservation_and_null_capacity() -> None:
    scores = torch.tensor([[3.0, -1.0, -2.0]], dtype=torch.float64)
    plan = dustbin_transport(scores, 0.25, iterations=120, temperature=0.7)
    log_mu, log_nu = dustbin_marginals(
        1, 3, device=plan.device, dtype=plan.dtype
    )
    torch.testing.assert_close(plan.sum(1), log_mu.exp(), rtol=0.0, atol=1e-10)
    torch.testing.assert_close(plan.sum(0), log_nu.exp(), rtol=0.0, atol=1e-10)
    assert plan[-1, :-1].sum() > 0  # unmatched history has a real dustbin path


def test_identity_cost_atom_monotonically_increases_real_match_mass() -> None:
    base = torch.tensor([[-1.0]], dtype=torch.float64)
    without_identity = dustbin_transport(base, 0.0, iterations=80, temperature=0.5)
    with_identity = dustbin_transport(base + 4.0, 0.0, iterations=80, temperature=0.5)
    assert with_identity[0, 0] > without_identity[0, 0]


def test_masked_padding_gets_zero_weight_and_no_history_is_exact_zero() -> None:
    model = _model()
    query = torch.randn(2, 8, dtype=torch.float64)
    candidate = torch.randn(2, 8, dtype=torch.float64)
    history = torch.randn(2, 3, 8, dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    exact = torch.tensor([[False, True, False], [False, False, False]])
    event_types = torch.tensor([[1, 2, 0], [0, 0, 0]])
    output = model(
        query,
        candidate,
        history,
        mask,
        exact,
        event_types=event_types,
    )
    assert output["event_weights"][0, 2].item() == 0.0
    assert output["event_weights"][1].abs().max().item() == 0.0
    assert output["raw_residual"][1].item() == 0.0
    assert output["trusted_mass"][1].item() == 0.0
    assert output["null_mass"][1].item() == 1.0


def test_main_outputs_are_bounded_and_conserved() -> None:
    model = _model()
    query = torch.randn(3, 8, dtype=torch.float64)
    candidate = torch.randn(3, 8, dtype=torch.float64)
    history = torch.randn(3, 3, 8, dtype=torch.float64)
    mask = torch.tensor([[True, True, True], [True, False, False], [True, True, False]])
    exact = torch.tensor([[False, False, True], [False, False, False], [False, False, False]])
    output = model(query, candidate, history, mask, exact)
    assert torch.isfinite(output["raw_residual"]).all()
    assert torch.all((output["trusted_mass"] >= 0) & (output["trusted_mass"] <= 1))
    assert torch.all((output["null_mass"] >= 0) & (output["null_mass"] <= 1))
    assert float(output["marginal_error"].max()) <= 1e-10
    torch.testing.assert_close(
        output["trusted_mass"] + output["null_mass"],
        torch.ones(3, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_gradients_are_finite() -> None:
    model = _model(dtype=torch.float32)
    query = torch.randn(4, 8)
    candidate = torch.randn(4, 8)
    history = torch.randn(4, 3, 8)
    mask = torch.tensor(
        [[True, True, True], [True, True, False], [True, False, False], [True, True, True]]
    )
    exact = torch.tensor(
        [[True, False, False], [False, False, False], [True, False, False], [False, True, False]]
    )
    event_types = torch.tensor([[1, 2, 1], [1, 2, 0], [2, 0, 0], [1, 1, 2]])
    output = model(
        query,
        candidate,
        history,
        mask,
        exact,
        event_types=event_types,
    )
    loss = (
        output["raw_residual"].square().mean()
        + output["trusted_mass"].mean()
        + output["cycle_gap"].mean()
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_softmax_forces_all_weight_to_real_events() -> None:
    model = _model()
    query = torch.randn(1, 8, dtype=torch.float64)
    candidate = torch.randn(1, 8, dtype=torch.float64)
    history = torch.randn(1, 3, 8, dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])
    exact = torch.zeros((1, 3), dtype=torch.bool)
    output = model(query, candidate, history, mask, exact, operator="softmax")
    assert math.isclose(float(output["event_weights"].sum()), 1.0, abs_tol=1e-12)
    assert output["null_mass"].item() == 0.0


def test_centered_residual_is_signed_and_translation_invariant() -> None:
    raw = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    centered = center_request_residual(raw, 2.0)
    torch.testing.assert_close(centered, torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64))
    shifted = center_request_residual(raw + 17.0, 2.0)
    torch.testing.assert_close(centered, shifted)
