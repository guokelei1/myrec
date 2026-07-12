from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.htct import solve_transition_coefficients


def witness(dtype: torch.dtype = torch.float64):
    transitions = torch.tensor([[[1.0, 0.0], [-0.8, 0.6]]], dtype=dtype)
    relation = torch.tensor([[[-0.6, 1.2]]], dtype=dtype)
    mask = torch.ones(1, 2, dtype=torch.bool)
    return relation, transitions, mask


def test_cone_recovers_positive_nonorthogonal_composition_and_rejects_reverse() -> None:
    relation, transitions, mask = witness()
    clean = solve_transition_coefficients(
        relation, transitions, mask, mode="cone", steps=300, ridge=0.0
    )
    reverse = solve_transition_coefficients(
        -relation, transitions, mask, mode="cone", steps=300, ridge=0.0
    )
    torch.testing.assert_close(
        clean.coefficients, torch.tensor([[[1.0, 2.0]]], dtype=torch.float64), atol=1e-6, rtol=0.0
    )
    assert (relation - clean.reconstruction).square().sum() < 1e-10
    assert (-relation - reverse.reconstruction).square().sum() > 0.20
    assert reverse.coefficients.min() >= 0


def test_unconstrained_span_reconstructs_both_signs() -> None:
    relation, transitions, mask = witness()
    for value in (relation, -relation):
        result = solve_transition_coefficients(
            value, transitions, mask, mode="span", steps=300, ridge=0.0
        )
        assert (value - result.reconstruction).square().sum() < 1e-10
    negative = solve_transition_coefficients(
        -relation, transitions, mask, mode="span", steps=300, ridge=0.0
    )
    assert negative.coefficients.min() < 0


def test_one_step_relu_is_not_joint_nnls_solution() -> None:
    relation, transitions, mask = witness()
    one = solve_transition_coefficients(
        relation, transitions, mask, mode="relu1", steps=300, ridge=0.0
    )
    full = solve_transition_coefficients(
        relation, transitions, mask, mode="cone", steps=300, ridge=0.0
    )
    assert one.coefficients[0, 0, 0] == 0
    assert full.coefficients[0, 0, 0] > 0.99
    assert not torch.allclose(one.reconstruction, full.reconstruction)


def test_simplex_coefficients_are_positive_and_sum_to_one() -> None:
    relation, transitions, mask = witness()
    result = solve_transition_coefficients(
        relation, transitions, mask, mode="simplex", steps=5, ridge=0.0
    )
    torch.testing.assert_close(
        result.coefficients.sum(dim=-1), torch.ones(1, 1, dtype=torch.float64)
    )
    assert result.coefficients.min() >= 0
    assert not torch.allclose(result.reconstruction, relation)


def test_projected_objective_is_nonincreasing_and_masked_coefficients_are_zero() -> None:
    generator = torch.Generator().manual_seed(20)
    relation = torch.randn(3, 4, 5, generator=generator, dtype=torch.float64)
    transitions = torch.randn(3, 6, 5, generator=generator, dtype=torch.float64)
    mask = torch.tensor(
        [[True, True, True, False, False, False], [True] * 6, [False] * 6]
    )
    result = solve_transition_coefficients(
        relation, transitions, mask, mode="cone", steps=30, ridge=0.02
    )
    difference = result.objective_trace[1:] - result.objective_trace[:-1]
    assert difference.max() <= 1e-10
    expanded = mask[:, None, :].expand_as(result.coefficients)
    assert torch.equal(result.coefficients.masked_select(~expanded), torch.zeros_like(result.coefficients.masked_select(~expanded)))
    assert result.coefficients.min() >= 0


def test_empty_dictionary_returns_zero_reconstruction() -> None:
    relation = torch.randn(2, 3, 4)
    transitions = torch.randn(2, 5, 4)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    result = solve_transition_coefficients(
        relation, transitions, mask, mode="cone", steps=4, ridge=0.02
    )
    assert torch.equal(result.coefficients, torch.zeros_like(result.coefficients))
    assert torch.equal(result.reconstruction, torch.zeros_like(result.reconstruction))


def test_relation_and_transition_displacements_ignore_common_state_translation() -> None:
    generator = torch.Generator().manual_seed(201)
    query = torch.randn(2, 5, generator=generator, dtype=torch.float64)
    candidates = torch.randn(2, 4, 5, generator=generator, dtype=torch.float64)
    history = torch.randn(2, 6, 5, generator=generator, dtype=torch.float64)
    translation = torch.randn(2, 5, generator=generator, dtype=torch.float64)
    relation = candidates - query[:, None]
    transitions = history[:, 1:] - history[:, :-1]
    shifted_relation = candidates + translation[:, None] - (query + translation)[:, None]
    shifted_transitions = (
        history[:, 1:] + translation[:, None] - (history[:, :-1] + translation[:, None])
    )
    torch.testing.assert_close(relation, shifted_relation, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(transitions, shifted_transitions, atol=1e-12, rtol=0.0)
