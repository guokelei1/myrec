"""Hand-computed and structural tests for the C07 CPU prototype."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.signed_kernel_transformer import (  # noqa: E402
    PairwiseSignedKernel,
    SignedKernelTransformer,
)


def _masks(batch: int = 1, candidates: int = 3, history: int = 1):
    return (
        torch.ones((batch, history), dtype=torch.bool),
        torch.ones((batch, candidates), dtype=torch.bool),
    )


def test_hand_computed_three_candidate_signed_competition() -> None:
    # One history event gives raw evidence [2, 0, -1].  With tau=0.5:
    # pairwise shrunk margins are
    #   c0: (2 -> 1.5, 3 -> 2.5), mean =  2.0
    #   c1: (-2 -> -1.5, 1 -> 0.5), mean = -0.5
    #   c2: (-3 -> -2.5, -1 -> -0.5), mean = -1.5.
    # The null-augmented L1 denominator is 1 + 2 + .5 + 1.5 = 5.
    scores = torch.tensor([[[2.0], [0.0], [-1.0]]], dtype=torch.float64)
    history_mask, candidate_mask = _masks()
    result = PairwiseSignedKernel(threshold=0.5, null_mass=1.0).double()(
        scores, history_mask, candidate_mask
    )
    expected_balance = torch.tensor([[[2.0], [-0.5], [-1.5]]], dtype=torch.float64)
    expected_weight = torch.tensor([[[0.4], [-0.1], [-0.3]]], dtype=torch.float64)
    torch.testing.assert_close(result.balances, expected_balance, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result.weights, expected_weight, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        result.weights.sum(dim=1), torch.zeros((1, 1), dtype=torch.float64), atol=1e-12, rtol=0.0
    )


def test_structural_abstention_on_open_dead_zone() -> None:
    # Every pairwise margin is <= tau, so all attention is exactly zero even
    # though the logits are unequal.  Softmax/sparsemax cannot express this
    # open abstention region without an extra null/gating construction.
    scores = torch.tensor([[[0.20], [0.00], [-0.20]]], dtype=torch.float64)
    history_mask, candidate_mask = _masks()
    result = PairwiseSignedKernel(threshold=0.5, null_mass=1.0).double()(
        scores, history_mask, candidate_mask
    )
    torch.testing.assert_close(result.balances, torch.zeros_like(result.balances))
    torch.testing.assert_close(result.weights, torch.zeros_like(result.weights))


def test_common_mode_shift_is_invisible_and_mass_is_conserved() -> None:
    scores = torch.tensor(
        [[[2.0, -0.5], [0.0, 0.4], [-1.0, 1.2]]], dtype=torch.float64
    )
    history_mask, candidate_mask = _masks(history=2)
    operator = PairwiseSignedKernel(threshold=0.35, null_mass=0.7).double()
    original = operator(scores, history_mask, candidate_mask)
    shifted = operator(
        scores + torch.tensor([[[19.0, -7.0]]], dtype=torch.float64),
        history_mask,
        candidate_mask,
    )
    torch.testing.assert_close(original.weights, shifted.weights, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        original.weights.sum(dim=1), torch.zeros((1, 2), dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    assert original.weights.abs().sum().item() < 1.0


def test_no_history_is_exact_zero_operator() -> None:
    scores = torch.tensor([[[100.0, -100.0], [-50.0, 50.0], [3.0, 7.0]]])
    history_mask = torch.zeros((1, 2), dtype=torch.bool)
    candidate_mask = torch.ones((1, 3), dtype=torch.bool)
    result = PairwiseSignedKernel(0.5, 1.0)(scores, history_mask, candidate_mask)
    assert torch.count_nonzero(result.weights).item() == 0
    assert torch.count_nonzero(result.balances).item() == 0


def test_operator_is_candidate_and_history_permutation_equivariant() -> None:
    scores = torch.tensor(
        [[[2.0, -0.5], [0.0, 0.4], [-1.0, 1.2]]], dtype=torch.float64
    )
    history_mask, candidate_mask = _masks(history=2)
    operator = PairwiseSignedKernel(0.35, 0.7).double()
    reference = operator(scores, history_mask, candidate_mask).weights

    candidate_permutation = torch.tensor([2, 0, 1])
    candidate_result = operator(
        scores[:, candidate_permutation, :],
        history_mask,
        candidate_mask[:, candidate_permutation],
    ).weights
    torch.testing.assert_close(
        candidate_result, reference[:, candidate_permutation, :], atol=1e-12, rtol=0.0
    )

    history_permutation = torch.tensor([1, 0])
    history_result = operator(
        scores[:, :, history_permutation],
        history_mask[:, history_permutation],
        candidate_mask,
    ).weights
    torch.testing.assert_close(
        history_result, reference[:, :, history_permutation], atol=1e-12, rtol=0.0
    )


def test_linear_limit_is_exactly_centered_attention_degeneracy() -> None:
    scores = torch.tensor([[[2.0], [0.0], [-1.0]]], dtype=torch.float64)
    history_mask, candidate_mask = _masks()
    result = PairwiseSignedKernel(threshold=0.0, null_mass=1.0).double()(
        scores, history_mask, candidate_mask
    )
    centered = scores - scores.mean(dim=1, keepdim=True)
    expected_balance = (3.0 / 2.0) * centered
    torch.testing.assert_close(result.balances, expected_balance, atol=1e-12, rtol=0.0)


def test_primary_is_not_a_scalar_gate_times_centered_logits() -> None:
    scores = torch.tensor([[[2.0], [0.0], [-1.0]]], dtype=torch.float64)
    history_mask, candidate_mask = _masks()
    balance = PairwiseSignedKernel(threshold=0.5, null_mass=1.0).double()(
        scores, history_mask, candidate_mask
    ).balances.flatten()
    centered = (scores - scores.mean(dim=1, keepdim=True)).flatten()
    # A scalar gate can only rescale the centered direction.  Nonzero 2x2
    # minors prove these vectors are not collinear.
    minor = balance[0] * centered[1] - balance[1] * centered[0]
    assert abs(minor.item()) > 1e-6


def test_two_candidate_case_is_declared_one_dimensional_degeneracy() -> None:
    scores = torch.tensor([[[2.0], [-1.0]]], dtype=torch.float64)
    history_mask = torch.ones((1, 1), dtype=torch.bool)
    candidate_mask = torch.ones((1, 2), dtype=torch.bool)
    result = PairwiseSignedKernel(threshold=0.5, null_mass=1.0).double()(
        scores, history_mask, candidate_mask
    )
    torch.testing.assert_close(result.balances[:, 0], -result.balances[:, 1])


def test_gradcheck_and_active_gradient_are_finite() -> None:
    scores = torch.tensor([[[2.0], [0.0], [-1.0]]], dtype=torch.float64, requires_grad=True)
    history_mask, candidate_mask = _masks()
    operator = PairwiseSignedKernel(threshold=0.5, null_mass=1.0).double()

    def weights_only(x: torch.Tensor) -> torch.Tensor:
        return operator(x, history_mask, candidate_mask).weights

    assert torch.autograd.gradcheck(weights_only, (scores,), eps=1e-6, atol=1e-5, rtol=1e-4)
    weighted = (weights_only(scores).flatten() * torch.tensor([1.0, -0.5, 0.25])).sum()
    weighted.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad).item() > 0


def test_end_to_end_information_barrier_masks_padding_and_has_no_label_api() -> None:
    torch.manual_seed(20260711)
    model = SignedKernelTransformer(d_model=6, nhead=2, dim_feedforward=12, threshold=0.1)
    model.eval()
    query = torch.randn(1, 6)
    history = torch.randn(1, 3, 6)
    candidates = torch.randn(1, 3, 6)
    history_mask = torch.tensor([[True, False, True]])
    exact = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    reference = model(query, history, candidates, history_mask, exact)
    altered_history = history.clone()
    altered_history[:, 1, :] = 1_000_000.0
    altered_exact = exact.clone()
    altered_exact[:, :, 1] = 1_000_000.0
    altered = model(query, altered_history, candidates, history_mask, altered_exact)
    torch.testing.assert_close(reference.base_logits, altered.base_logits, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(reference.logits, altered.logits, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(reference.signed_weights, altered.signed_weights, atol=1e-6, rtol=0.0)

    parameters = set(inspect.signature(model.forward).parameters)
    forbidden = {"labels", "qrels", "clicked", "purchased", "dataset", "dataset_id"}
    assert parameters.isdisjoint(forbidden)


def test_end_to_end_no_history_equals_query_only_base_exactly() -> None:
    torch.manual_seed(7)
    model = SignedKernelTransformer(d_model=6, nhead=2, dim_feedforward=12, threshold=0.1)
    model.eval()
    query = torch.randn(2, 6)
    history = torch.randn(2, 3, 6) * 100.0
    candidates = torch.randn(2, 4, 6)
    history_mask = torch.zeros((2, 3), dtype=torch.bool)
    exact = torch.ones((2, 4, 3)) * 100.0
    output = model(query, history, candidates, history_mask, exact)
    torch.testing.assert_close(output.logits, output.base_logits, atol=0.0, rtol=0.0)
    assert torch.count_nonzero(output.personalized_delta).item() == 0
    assert torch.count_nonzero(output.signed_weights).item() == 0


def test_end_to_end_permutation_equivariance() -> None:
    torch.manual_seed(11)
    model = SignedKernelTransformer(d_model=6, nhead=2, dim_feedforward=12, threshold=0.1)
    model.eval()
    query = torch.randn(1, 6)
    history = torch.randn(1, 2, 6)
    candidates = torch.randn(1, 3, 6)
    history_mask = torch.ones((1, 2), dtype=torch.bool)
    exact = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
    reference = model(query, history, candidates, history_mask, exact)

    cp = torch.tensor([2, 0, 1])
    candidate_output = model(query, history, candidates[:, cp], history_mask, exact[:, cp])
    torch.testing.assert_close(candidate_output.logits, reference.logits[:, cp], atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        candidate_output.signed_weights, reference.signed_weights[:, cp], atol=1e-6, rtol=0.0
    )

    hp = torch.tensor([1, 0])
    history_output = model(
        query,
        history[:, hp],
        candidates,
        history_mask[:, hp],
        exact[:, :, hp],
    )
    torch.testing.assert_close(history_output.logits, reference.logits, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        history_output.signed_weights,
        reference.signed_weights[:, :, hp],
        atol=1e-6,
        rtol=0.0,
    )


def test_exact_recurrence_feature_activates_candidate_relative_support() -> None:
    torch.manual_seed(13)
    model = SignedKernelTransformer(
        d_model=6,
        nhead=2,
        dim_feedforward=12,
        threshold=0.2,
        exact_match_scale=1.0,
    )
    model.eval()
    with torch.no_grad():
        model.query_projection.weight.zero_()
        model.candidate_projection.weight.zero_()
        model.history_projection.weight.zero_()
    query = torch.randn(1, 6)
    history = torch.randn(1, 1, 6)
    candidates = torch.randn(1, 3, 6)
    exact = torch.tensor([[[1.0], [0.0], [0.0]]])
    output = model(query, history, candidates, torch.ones((1, 1), dtype=torch.bool), exact)
    assert output.evidence_logits[0, 0, 0] > output.evidence_logits[0, 1, 0]
    assert output.signed_weights[0, 0, 0] > 0
    assert output.signed_weights[0, 1, 0] < 0
    torch.testing.assert_close(
        output.signed_weights.sum(dim=1), torch.zeros((1, 1)), atol=1e-7, rtol=0.0
    )


def test_end_to_end_gradient_reaches_all_three_evidence_families() -> None:
    torch.manual_seed(17)
    model = SignedKernelTransformer(
        d_model=6,
        nhead=2,
        dim_feedforward=12,
        threshold=0.01,
        exact_match_scale=2.0,
    )
    model.train()
    query = torch.randn(1, 6, requires_grad=True)
    history = torch.randn(1, 2, 6, requires_grad=True)
    candidates = torch.randn(1, 3, 6, requires_grad=True)
    history_mask = torch.ones((1, 2), dtype=torch.bool)
    exact = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
    output = model(query, history, candidates, history_mask, exact)
    probe = torch.tensor([[1.0, -0.7, 0.2]])
    loss = (output.logits * probe).sum() + output.personalized_delta.square().sum()
    loss.backward()

    for tensor in (query, history, candidates):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad).item() > 0
    assert model.query_projection.weight.grad is not None
    assert torch.count_nonzero(model.query_projection.weight.grad).item() > 0


def test_invalid_normalizer_parameters_fail_closed() -> None:
    with pytest.raises(ValueError):
        PairwiseSignedKernel(threshold=-0.1)
    with pytest.raises(ValueError):
        PairwiseSignedKernel(null_mass=0.0)
