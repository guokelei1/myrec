from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.ecot import protected_margin_violation, project_two_group_order


def test_hand_computed_single_repeat_projection() -> None:
    anchor = torch.zeros(1, 3, dtype=torch.float64)
    proposal = torch.tensor([[0.0, 2.0, -1.0]], dtype=torch.float64)
    repeat = torch.tensor([[True, False, False]])
    output = project_two_group_order(proposal, anchor, repeat)
    expected = torch.tensor([[1.0, 1.0, -1.0]], dtype=torch.float64)
    assert torch.allclose(output, expected, atol=1e-12, rtol=0.0)
    assert torch.allclose(output.sum(1), proposal.sum(1), atol=1e-12, rtol=0.0)
    assert protected_margin_violation(output, anchor, repeat).max() <= 1e-12


def test_two_group_projection_is_feasible_idempotent_and_minimal_witness() -> None:
    anchor = torch.tensor([[0.3, -0.4, 0.2, -0.1]], dtype=torch.float64)
    proposal = anchor + torch.tensor([[-2.0, 1.0, 3.0, -1.0]], dtype=torch.float64)
    repeat = torch.tensor([[True, True, False, False]])
    output = project_two_group_order(proposal, anchor, repeat)
    repeated_delta = (output - anchor)[0, :2]
    nonrepeat_delta = (output - anchor)[0, 2:]
    assert repeated_delta.min() >= nonrepeat_delta.max() - 1e-12
    assert torch.allclose(output.sum(1), proposal.sum(1), atol=1e-12, rtol=0.0)
    second = project_two_group_order(output, anchor, repeat)
    assert torch.allclose(second, output, atol=1e-12, rtol=0.0)

    # Moving the shared active-set threshold in either direction increases L2.
    active = (output - proposal).abs().gt(1e-10)
    candidate = output.clone()
    candidate[active] += torch.tensor([[0.01, 0.01, -0.01, 0.0]])[active]
    assert ((candidate - proposal) ** 2).sum() > ((output - proposal) ** 2).sum()


def test_feasible_and_unconstrained_rows_are_exact_identities() -> None:
    anchor = torch.zeros(3, 3, dtype=torch.float64)
    proposal = torch.tensor(
        [[2.0, 1.0, 0.0], [0.1, -0.2, 0.3], [0.2, -0.1, 0.0]],
        dtype=torch.float64,
    )
    repeat = torch.tensor(
        [[True, False, False], [False, False, False], [True, True, True]]
    )
    output = project_two_group_order(proposal, anchor, repeat)
    assert torch.equal(output, proposal)


def test_projection_commutes_with_candidate_permutation_and_common_shift() -> None:
    generator = torch.Generator().manual_seed(18)
    proposal = torch.randn(7, 6, generator=generator, dtype=torch.float64)
    anchor = torch.randn(7, 6, generator=generator, dtype=torch.float64)
    repeat = torch.zeros(7, 6, dtype=torch.bool)
    repeat[:, 1] = True
    permutation = torch.tensor([5, 2, 0, 3, 1, 4])
    original = project_two_group_order(proposal, anchor, repeat)
    changed = project_two_group_order(
        proposal[:, permutation], anchor[:, permutation], repeat[:, permutation]
    )
    assert torch.allclose(changed, original[:, permutation], atol=1e-12, rtol=0.0)
    shift = torch.arange(7, dtype=torch.float64).unsqueeze(1)
    shifted = project_two_group_order(proposal + shift, anchor + shift, repeat)
    assert torch.allclose(shifted, original + shift, atol=1e-12, rtol=0.0)


def test_active_constraint_has_equal_opposite_correction_and_finite_gradient() -> None:
    proposal = torch.tensor([[0.0, 2.0, -1.0]], requires_grad=True)
    anchor = torch.zeros_like(proposal)
    repeat = torch.tensor([[True, False, False]])
    output = project_two_group_order(proposal, anchor, repeat)
    correction = output - proposal
    assert torch.allclose(correction[0, 0], -correction[0, 1])
    assert correction.sum().abs() < 1e-6
    loss = (output * torch.tensor([[1.0, -0.2, 0.5]])).sum()
    loss.backward()
    assert proposal.grad is not None
    assert torch.isfinite(proposal.grad).all()
    assert proposal.grad.abs().sum() > 0
