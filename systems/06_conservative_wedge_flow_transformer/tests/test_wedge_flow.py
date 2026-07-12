from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.wedge_flow import (  # noqa: E402
    ConservativeWedgeFlowProbeRanker,
    _clamp_roundoff_nonnegative,
    _cycle_identity_forward_error_bound,
    _repair_materially_negative_cycle_rows,
    explicit_wedge_flow,
    low_rank_hodge_calibration,
)
from train.losses import masked_listwise_loss  # noqa: E402


def _model() -> ConservativeWedgeFlowProbeRanker:
    torch.manual_seed(101)
    return ConservativeWedgeFlowProbeRanker(8, 4, score_delta_max=0.7)


def _inputs(history_count: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(103)
    history_mask = torch.tensor(
        [[True] * history_count, [True] + [False] * (history_count - 1)],
        dtype=torch.bool,
    )
    return {
        "query": torch.randn(2, 8),
        "candidates": torch.randn(2, 4, 8),
        "history": torch.randn(2, history_count, 8),
        "candidate_mask": torch.tensor(
            [[True, True, True, True], [True, True, True, False]],
            dtype=torch.bool,
        ),
        "history_mask": history_mask,
        "history_prior": torch.ones(2, history_count),
        "base_scores": torch.randn(2, 4, dtype=torch.float32),
    }


def _open_scale(model: ConservativeWedgeFlowProbeRanker, fraction: float = 0.8) -> None:
    with torch.no_grad():
        model.raw_residual_scale.fill_(math.atanh(fraction))


def _explicit_local_hodge(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    candidate_mask: torch.Tensor,
    history_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Small-C FP64 oracle that explicitly materializes F, G, C, and T."""

    a = factor_a.double()
    b = factor_b.double()
    candidate_mask = candidate_mask.bool()
    history_mask = history_mask.bool()
    rank = factor_a.shape[-1]
    edge = (
        torch.einsum("bihr,bkhr->bikh", a, b)
        - torch.einsum("bihr,bkhr->bikh", b, a)
    ) / float(2 * rank)
    pair_mask = (
        candidate_mask[:, :, None, None]
        & candidate_mask[:, None, :, None]
        & history_mask[:, None, None, :]
    )
    edge = torch.where(pair_mask, edge, torch.zeros_like(edge))
    valid_count = candidate_mask.sum(dim=-1).clamp_min(1)
    count = valid_count[:, None, None].double()
    active = candidate_mask[:, :, None] & history_mask[:, None, :]
    potential = edge.sum(dim=2) / count
    potential = torch.where(active, potential, torch.zeros_like(potential))
    potential = torch.where(
        active,
        potential - potential.sum(dim=1, keepdim=True) / count,
        torch.zeros_like(potential),
    )
    gradient = potential[:, :, None, :] - potential[:, None, :, :]
    gradient = torch.where(pair_mask, gradient, torch.zeros_like(gradient))
    cycle = edge - gradient
    candidate_gradient_energy = gradient.square().sum(dim=2)
    candidate_cycle_energy = cycle.square().sum(dim=2)
    trust = candidate_gradient_energy / (
        candidate_gradient_energy + candidate_cycle_energy + 1e-12
    )
    trust = torch.where(active, trust, torch.zeros_like(trust))
    trusted_edge = (
        0.5
        * trust[:, :, None, :]
        * trust[:, None, :, :]
        * gradient
    )
    trusted_divergence = trusted_edge.sum(dim=2) / count
    trusted_divergence = torch.where(
        active, trusted_divergence, torch.zeros_like(trusted_divergence)
    )
    trusted_divergence = torch.where(
        active,
        trusted_divergence
        - trusted_divergence.sum(dim=1, keepdim=True) / count,
        torch.zeros_like(trusted_divergence),
    )
    return (
        potential,
        candidate_gradient_energy,
        candidate_cycle_energy,
        trust,
        trusted_edge,
        trusted_divergence,
    )


def _contracted_cycle_rows(
    centered_a: torch.Tensor, centered_b: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    gram_aa = torch.einsum("bchr,bchs->bhrs", centered_a, centered_a)
    gram_bb = torch.einsum("bchr,bchs->bhrs", centered_b, centered_b)
    gram_ba = torch.einsum("bchr,bchs->bhrs", centered_b, centered_a)
    cycle_a = torch.einsum(
        "bchr,bhrs,bchs->bch", centered_a, gram_bb, centered_a
    )
    cycle_b = torch.einsum(
        "bchr,bhrs,bchs->bch", centered_b, gram_aa, centered_b
    )
    cycle_cross = torch.einsum(
        "bchr,bhrs,bchs->bch", centered_a, gram_ba, centered_b
    )
    return (
        (alpha * alpha) * (cycle_a + cycle_b - 2.0 * cycle_cross),
        (alpha * alpha)
        * (cycle_a.abs() + cycle_b.abs() + 2.0 * cycle_cross.abs()),
    )


def _row_primitive_absolute_sum(
    centered_a: torch.Tensor,
    centered_b: torch.Tensor,
    candidate: int,
    alpha: float,
) -> torch.Tensor:
    row_a = centered_a[:, candidate, 0, :]
    row_b = centered_b[:, candidate, 0, :]
    absolute_forward = torch.einsum(
        "br,bcr->bc", row_a.abs(), centered_b[:, :, 0, :].abs()
    )
    absolute_reverse = torch.einsum(
        "br,bcr->bc", row_b.abs(), centered_a[:, :, 0, :].abs()
    )
    return (alpha * alpha) * (
        absolute_forward + absolute_reverse
    ).square().sum(dim=-1)


def test_initialization_is_bit_exact_base_even_with_history() -> None:
    output = _model().eval()(**_inputs())
    assert torch.equal(output.scores, output.base_scores)
    assert torch.count_nonzero(output.conservative_score_delta) == 0
    assert not output.base_scores.requires_grad


def test_true_zero_length_history_is_finite_and_bit_exact() -> None:
    values = _inputs(0)
    output = _model().eval()(**values)
    assert torch.equal(output.scores, values["base_scores"])
    assert output.trusted_event_divergence.shape == (2, 4, 0)
    assert output.candidate_event_trust.shape == (2, 4, 0)
    assert output.event_weights.shape == (2, 0)


def test_local_hodge_matches_explicit_c_squared_oracle() -> None:
    torch.manual_seed(113)
    factor_a = torch.tanh(torch.randn(2, 5, 3, 4))
    factor_b = torch.tanh(torch.randn(2, 5, 3, 4))
    candidate_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    history_mask = torch.tensor(
        [[True, True, False], [True, False, True]]
    )
    (
        potential,
        trust,
        candidate_gradient_energy,
        candidate_cycle_energy,
        trusted_divergence,
        flow_energy,
        gradient_energy,
        cycle_energy,
    ) = low_rank_hodge_calibration(
        factor_a, factor_b, candidate_mask, history_mask
    )
    (
        expected_potential,
        expected_gradient_energy,
        expected_cycle_energy,
        expected_trust,
        trusted_edge,
        expected_divergence,
    ) = _explicit_local_hodge(
        factor_a, factor_b, candidate_mask, history_mask
    )
    assert torch.allclose(
        potential.double(), expected_potential, atol=2e-8, rtol=2e-6
    )
    assert torch.allclose(
        candidate_gradient_energy.double(),
        expected_gradient_energy,
        atol=2e-8,
        rtol=2e-6,
    )
    assert torch.allclose(
        candidate_cycle_energy.double(),
        expected_cycle_energy,
        atol=2e-8,
        rtol=2e-6,
    )
    assert torch.allclose(
        trust.double(), expected_trust, atol=2e-7, rtol=2e-6
    )
    reconstructed_trusted_edge = (
        0.5
        * trust.double()[:, :, None, :]
        * trust.double()[:, None, :, :]
        * (
            potential.double()[:, :, None, :]
            - potential.double()[:, None, :, :]
        )
    )
    pair_mask = (
        candidate_mask[:, :, None, None]
        & candidate_mask[:, None, :, None]
        & history_mask[:, None, None, :]
    )
    reconstructed_trusted_edge = torch.where(
        pair_mask,
        reconstructed_trusted_edge,
        torch.zeros_like(reconstructed_trusted_edge),
    )
    assert torch.allclose(
        reconstructed_trusted_edge,
        trusted_edge,
        atol=2e-7,
        rtol=2e-6,
    )
    assert torch.allclose(
        trusted_divergence.double(),
        expected_divergence,
        atol=2e-8,
        rtol=2e-6,
    )
    assert torch.allclose(
        trusted_edge, -trusted_edge.transpose(1, 2), atol=1e-12, rtol=0
    )
    assert torch.allclose(
        gradient_energy.double(),
        expected_gradient_energy.sum(dim=1),
        atol=2e-8,
        rtol=2e-6,
    )
    assert torch.allclose(
        cycle_energy.double(),
        expected_cycle_energy.sum(dim=1),
        atol=2e-8,
        rtol=2e-6,
    )
    assert torch.allclose(
        flow_energy, gradient_energy + cycle_energy, atol=1e-8, rtol=1e-7
    )


def test_one_and_two_candidate_rows_have_exactly_no_cycle_subspace() -> None:
    torch.manual_seed(127)
    factor_a = torch.tanh(torch.randn(3, 4, 4, 4))
    factor_b = torch.tanh(torch.randn(3, 4, 4, 4))
    candidate_mask = torch.tensor(
        [
            [False, True, False, False],
            [False, True, True, True],
            [True, True, False, False],
        ]
    )
    history_mask = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, False],
            [True, False, True, True],
        ]
    )
    output = low_rank_hodge_calibration(
        factor_a, factor_b, candidate_mask, history_mask
    )
    cycle = output[3]
    assert torch.count_nonzero(cycle[0]) == 0
    assert torch.count_nonzero(cycle[2]) == 0

    expected = _explicit_local_hodge(
        factor_a, factor_b, candidate_mask, history_mask
    )[2]
    assert float(expected[0].abs().max()) <= 1e-30
    assert float(expected[2].abs().max()) <= 1e-30
    assert torch.allclose(
        cycle[1].double(), expected[1], atol=2e-8, rtol=2e-6
    )


def test_nonpotential_cycle_is_representable_but_ranking_null() -> None:
    # These three (a_i, b_i) pairs form a directed 1->2->3->1 circulation.
    # No unary potential can produce a nonzero triangle sum, but divergence
    # correctly discards the pure cycle because it cannot define a ranking.
    factor_a = torch.tensor([[[[1.0]], [[0.0]], [[-1.0]]]])
    factor_b = torch.tensor([[[[0.0]], [[1.0]], [[-1.0]]]])
    edge, divergence = explicit_wedge_flow(
        factor_a,
        factor_b,
        torch.ones(1, 1),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
    )
    circulation = edge[0, 0, 1] + edge[0, 1, 2] + edge[0, 2, 0]
    assert float(circulation) == 1.5
    assert torch.equal(divergence, torch.zeros_like(divergence))
    (
        potential,
        trust,
        candidate_gradient_energy,
        candidate_cycle_energy,
        trusted_divergence,
        flow_energy,
        gradient_energy,
        cycle_energy,
    ) = low_rank_hodge_calibration(
        factor_a,
        factor_b,
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert torch.equal(potential, torch.zeros_like(potential))
    assert torch.equal(trust, torch.zeros_like(trust))
    assert torch.equal(
        candidate_gradient_energy, torch.zeros_like(candidate_gradient_energy)
    )
    assert bool((candidate_cycle_energy > 0).all())
    assert torch.equal(trusted_divergence, torch.zeros_like(trusted_divergence))
    assert float(flow_energy[0, 0]) > 0.0
    assert float(gradient_energy[0, 0]) == 0.0
    assert float(cycle_energy[0, 0]) == float(flow_energy[0, 0])


def test_potential_flow_has_unit_candidate_local_trust() -> None:
    # b_i=1 makes the wedge edge proportional to a_i-a_k, a pure gradient.
    factor_a = torch.tensor([[[[-0.8]], [[0.1]], [[0.7]]]])
    factor_b = torch.ones_like(factor_a)
    (
        _,
        trust,
        _,
        candidate_cycle_energy,
        _,
        flow_energy,
        gradient_energy,
        cycle_energy,
    ) = low_rank_hodge_calibration(
        factor_a,
        factor_b,
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert torch.allclose(trust, torch.ones_like(trust), atol=1e-6)
    assert torch.count_nonzero(candidate_cycle_energy) == 0
    assert torch.count_nonzero(cycle_energy) == 0
    assert torch.allclose(flow_energy, gradient_energy, atol=1e-6)


def test_cycle_sign_only_changes_local_trust_not_ranking_direction() -> None:
    # Dimension 0 is a potential field. Dimension 1 is an orthogonal pure
    # triangle circulation. The circulation leaves divergence unchanged but
    # must lower the Hodge reliability used by the architecture.
    potential_a = torch.tensor([-0.8, 0.1, 0.7])
    potential_b = torch.ones(3)
    cycle_a = torch.tensor([1.0, 0.0, -1.0])
    cycle_b = torch.tensor([0.0, 1.0, -1.0])
    clean_a = torch.stack([potential_a, torch.zeros(3)], dim=-1)[None, :, None, :]
    clean_b = torch.stack([potential_b, torch.zeros(3)], dim=-1)[None, :, None, :]
    mixed_a = torch.stack([potential_a, 0.8 * cycle_a], dim=-1)[None, :, None, :]
    mixed_b = torch.stack([potential_b, 0.8 * cycle_b], dim=-1)[None, :, None, :]
    reversed_a = torch.stack(
        [potential_a, -0.8 * cycle_a], dim=-1
    )[None, :, None, :]
    candidate_mask = torch.ones(1, 3, dtype=torch.bool)
    history_mask = torch.ones(1, 1, dtype=torch.bool)
    clean = low_rank_hodge_calibration(
        clean_a, clean_b, candidate_mask, history_mask
    )
    mixed = low_rank_hodge_calibration(
        mixed_a, mixed_b, candidate_mask, history_mask
    )
    reversed_cycle = low_rank_hodge_calibration(
        reversed_a, mixed_b, candidate_mask, history_mask
    )
    # u is fixed; cycle energy may only lower endpoint-local trust.
    assert torch.allclose(clean[0], mixed[0], atol=1e-7)
    assert torch.all(mixed[1] <= clean[1] + 1e-7)
    assert bool((mixed[1] < clean[1] - 1e-5).any())
    # Reversing C leaves EC, t, and the trusted projected direction unchanged.
    assert torch.allclose(reversed_cycle[0], mixed[0], atol=1e-7)
    assert torch.allclose(reversed_cycle[3], mixed[3], atol=1e-7)
    assert torch.allclose(reversed_cycle[1], mixed[1], atol=1e-7)
    assert torch.allclose(reversed_cycle[4], mixed[4], atol=1e-7)


def test_nearly_equal_factors_match_fp64_explicit_oracle() -> None:
    torch.manual_seed(127)
    common = torch.randn(2, 37, 2, 8)
    factor_a = common + 1e-4 * torch.randn_like(common)
    factor_b = common + 1e-4 * torch.randn_like(common)
    candidate_mask = torch.tensor(
        [[True] * 37, [True] * 31 + [False] * 6], dtype=torch.bool
    )
    history_mask = torch.tensor([[True, True], [True, False]])
    actual = low_rank_hodge_calibration(
        factor_a, factor_b, candidate_mask, history_mask
    )
    expected = _explicit_local_hodge(
        factor_a, factor_b, candidate_mask, history_mask
    )
    for value in actual:
        assert torch.isfinite(value).all()
    for actual_index, expected_index in (
        (0, 0),
        (1, 3),
        (2, 1),
        (3, 2),
        (4, 5),
    ):
        assert torch.allclose(
            actual[actual_index].double(),
            expected[expected_index],
            atol=2e-12,
            rtol=5e-5,
        )


def test_cycle_energy_guard_clamps_roundoff_but_rejects_material_negative() -> None:
    magnitude = torch.ones(2, dtype=torch.float64)
    epsilon = torch.finfo(torch.float64).eps
    roundoff = torch.tensor([-64.0 * epsilon, 1.0], dtype=torch.float64)
    guarded = _clamp_roundoff_nonnegative(
        roundoff,
        magnitude,
        context="test energy",
        contraction_terms=4,
    )
    assert torch.equal(guarded, torch.tensor([0.0, 1.0], dtype=torch.float64))
    with pytest.raises(FloatingPointError, match="negative beyond"):
        _clamp_roundoff_nonnegative(
            torch.tensor([-1e-8], dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            context="test energy",
            contraction_terms=4,
        )


def test_bf16_near_degenerate_fp64_row_with_legal_negative_error_falls_back() -> None:
    # Production dtype/order regression: BF16 factors are promoted and every
    # Gram contraction below is FP64. CPU produces a negative cancellation
    # residue still inside the old guard. We then exercise a negative value
    # outside that guard but inside the independently derived FP64 forward-error
    # bound, matching the only class the fallback is permitted to repair.
    generator = torch.Generator().manual_seed(10830)
    candidates, rank = 3, 8
    factor_a = torch.tanh(
        torch.randn(candidates, rank, generator=generator)
    ).to(torch.bfloat16)
    factor_b = (factor_a.float() + 0.5).to(torch.bfloat16)
    alpha = 1.0 / float(2 * rank)
    a = factor_a.double()[None, :, None, :]
    b = factor_b.double()[None, :, None, :]
    centered_a = a - a.mean(dim=1, keepdim=True)
    centered_b = b - b.mean(dim=1, keepdim=True)
    contracted, magnitude = _contracted_cycle_rows(
        centered_a, centered_b, alpha
    )
    target = 2
    assert candidates >= 3
    assert contracted[0, target, 0] < 0.0
    primitive = _row_primitive_absolute_sum(
        centered_a, centered_b, target, alpha
    )
    _, forward_bound = _cycle_identity_forward_error_bound(
        primitive, candidates + rank
    )
    old_tolerance = (
        32.0
        * float(candidates + rank)
        * torch.finfo(torch.float64).eps
        * magnitude[0, target, 0]
    )
    observed_legal_failure = contracted.clone()
    observed_legal_failure[0, target, 0] = -0.25 * forward_bound[0]
    assert observed_legal_failure[0, target, 0] < -old_tolerance
    active = torch.ones(1, candidates, 1, dtype=torch.bool)
    repaired, _, fallback_count = _repair_materially_negative_cycle_rows(
        observed_legal_failure,
        magnitude,
        centered_a,
        centered_b,
        active,
        torch.zeros_like(active),
        alpha=alpha,
        contraction_terms=candidates + rank,
    )
    expected = _explicit_local_hodge(
        a.float(),
        b.float(),
        torch.ones(1, candidates, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
    )[2]
    assert fallback_count == 1
    assert torch.allclose(
        repaired[0, target, 0],
        expected[0, target, 0],
        atol=1e-15,
        rtol=1e-12,
    )
    assert repaired[0, target, 0] >= 0.0


def test_explicit_cycle_fallback_preserves_oracle_gradients() -> None:
    torch.manual_seed(137)
    candidates, rank = 5, 4
    common = torch.randn(1, candidates, 1, rank, dtype=torch.float64)
    perturbation = torch.randn(1, candidates, 1, rank, dtype=torch.float64)
    factor_a = common.clone().requires_grad_(True)
    factor_b = (common + 1e-9 * perturbation).requires_grad_(True)
    centered_a = factor_a - factor_a.mean(dim=1, keepdim=True)
    centered_b = factor_b - factor_b.mean(dim=1, keepdim=True)
    alpha = 1.0 / float(2 * rank)
    target_candidate = 2
    row_a = centered_a[:, target_candidate, 0, :]
    row_b = centered_b[:, target_candidate, 0, :]
    explicit_edges = alpha * (
        torch.einsum("br,bcr->bc", row_a, centered_b[:, :, 0, :])
        - torch.einsum("br,bcr->bc", row_b, centered_a[:, :, 0, :])
    )
    oracle = explicit_edges.square().sum()

    contracted, magnitude = _contracted_cycle_rows(
        centered_a, centered_b, alpha
    )
    primitive = _row_primitive_absolute_sum(
        centered_a, centered_b, target_candidate, alpha
    )
    _, forward_bound = _cycle_identity_forward_error_bound(
        primitive, candidates + rank
    )
    contracted = contracted.clone()
    contracted[0, target_candidate, 0] = -0.25 * forward_bound.detach()[0]
    assert (contracted[0, target_candidate, 0] - oracle).abs() <= forward_bound[0]
    active = torch.ones_like(contracted, dtype=torch.bool)
    repaired, _, fallback_count = _repair_materially_negative_cycle_rows(
        contracted,
        magnitude,
        centered_a,
        centered_b,
        active,
        torch.zeros_like(active),
        alpha=alpha,
        contraction_terms=candidates + rank,
    )
    repaired_value = repaired[0, target_candidate, 0]
    fallback_gradients = torch.autograd.grad(
        repaired_value, (factor_a, factor_b), retain_graph=True
    )
    oracle_gradients = torch.autograd.grad(oracle, (factor_a, factor_b))
    assert fallback_count == 1
    assert torch.allclose(repaired_value, oracle, atol=1e-15, rtol=1e-12)
    for actual, expected in zip(fallback_gradients, oracle_gradients):
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-10)


def test_cycle_fallback_rejects_material_energy_error_with_correct_magnitude() -> None:
    torch.manual_seed(138)
    candidates, rank = 5, 4
    factor_a = torch.randn(1, candidates, 1, rank, dtype=torch.float64)
    factor_b = torch.randn(1, candidates, 1, rank, dtype=torch.float64)
    centered_a = factor_a - factor_a.mean(dim=1, keepdim=True)
    centered_b = factor_b - factor_b.mean(dim=1, keepdim=True)
    alpha = 1.0 / float(2 * rank)
    contracted, magnitude = _contracted_cycle_rows(
        centered_a, centered_b, alpha
    )
    contracted = contracted.clone()
    contracted[0, 2, 0] = -1.0
    active = torch.ones_like(contracted, dtype=torch.bool)
    with pytest.raises(FloatingPointError, match="independent magnitude consistency"):
        _repair_materially_negative_cycle_rows(
            contracted,
            magnitude,
            centered_a,
            centered_b,
            active,
            torch.zeros_like(active),
            alpha=alpha,
            contraction_terms=candidates + rank,
        )


def test_cycle_fallback_is_mask_safe_and_candidate_permutation_equivariant() -> None:
    torch.manual_seed(141)
    batch, candidates, history, rank = 2, 6, 2, 4
    candidate_mask = torch.tensor(
        [[True] * 6, [True, True, True, True, False, False]]
    )
    history_mask = torch.tensor([[True, True], [True, False]])
    pair_mask = candidate_mask[:, :, None, None] & history_mask[:, None, :, None]
    factor_a = torch.randn(batch, candidates, history, rank, dtype=torch.float64)
    common_shift = torch.randn(batch, 1, history, rank, dtype=torch.float64)
    factor_b = factor_a + common_shift
    safe_a = torch.where(pair_mask, factor_a, torch.zeros_like(factor_a))
    safe_b = torch.where(pair_mask, factor_b, torch.zeros_like(factor_b))
    count = candidate_mask.sum(dim=-1).clamp_min(1)[:, None, None].double()
    mean_a = safe_a.sum(dim=1) / count
    mean_b = safe_b.sum(dim=1) / count
    centered_a = torch.where(
        pair_mask, safe_a - mean_a[:, None, :, :], torch.zeros_like(safe_a)
    )
    centered_b = torch.where(
        pair_mask, safe_b - mean_b[:, None, :, :], torch.zeros_like(safe_b)
    )
    alpha = 1.0 / float(2 * rank)
    contracted, magnitude = _contracted_cycle_rows(
        centered_a, centered_b, alpha
    )
    active = candidate_mask[:, :, None] & history_mask[:, None, :]
    no_cycle = (candidate_mask.sum(dim=-1)[:, None, None] <= 2).expand_as(
        contracted
    )
    targets = [(0, 1, 0), (0, 4, 1), (1, 2, 0)]
    injected = contracted.clone()
    for batch_index, candidate_index, history_index in targets:
        primitive = _row_primitive_absolute_sum(
            centered_a[batch_index : batch_index + 1, :, history_index : history_index + 1],
            centered_b[batch_index : batch_index + 1, :, history_index : history_index + 1],
            candidate_index,
            alpha,
        )
        _, bound = _cycle_identity_forward_error_bound(
            primitive, candidates + rank
        )
        injected[batch_index, candidate_index, history_index] = -0.25 * bound[0]
    # A material negative on an inactive padded row must not trigger repair.
    injected[1, 5, 1] = -1.0
    repaired, _, count_original = _repair_materially_negative_cycle_rows(
        injected,
        magnitude,
        centered_a,
        centered_b,
        active,
        no_cycle,
        alpha=alpha,
        contraction_terms=candidates + rank,
    )
    assert count_original == len(targets)
    assert repaired[1, 5, 1] == -1.0

    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    inverse = torch.argsort(permutation)
    permuted, _, count_permuted = _repair_materially_negative_cycle_rows(
        injected[:, permutation],
        magnitude[:, permutation],
        centered_a[:, permutation],
        centered_b[:, permutation],
        active[:, permutation],
        no_cycle[:, permutation],
        alpha=alpha,
        contraction_terms=candidates + rank,
    )
    assert count_permuted == count_original
    assert torch.allclose(
        permuted[:, inverse], repaired, atol=1e-15, rtol=1e-12
    )


def test_low_rank_diagnostics_record_zero_fallback_on_well_conditioned_rows() -> None:
    torch.manual_seed(139)
    diagnostics: dict[str, int] = {}
    low_rank_hodge_calibration(
        torch.randn(2, 7, 3, 4),
        torch.randn(2, 7, 3, 4),
        torch.ones(2, 7, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool),
        diagnostics=diagnostics,
    )
    assert diagnostics == {"candidate_cycle_energy_fallback_count": 0}


def test_nearly_equal_factors_have_finite_final_output_backward() -> None:
    torch.manual_seed(131)
    common = torch.randn(2, 41, 2, 8)
    factor_a = (common + 1e-4 * torch.randn_like(common)).requires_grad_(True)
    factor_b = (common + 1e-4 * torch.randn_like(common)).requires_grad_(True)
    candidate_mask = torch.ones(2, 41, dtype=torch.bool)
    history_mask = torch.ones(2, 2, dtype=torch.bool)
    trusted = low_rank_hodge_calibration(
        factor_a, factor_b, candidate_mask, history_mask
    )[4]
    contrast = torch.randn_like(trusted)
    loss = (trusted * contrast).sum()
    loss.backward()
    assert factor_a.grad is not None and torch.isfinite(factor_a.grad).all()
    assert factor_b.grad is not None and torch.isfinite(factor_b.grad).all()


def test_divergence_is_zero_sum_and_final_delta_is_bounded() -> None:
    model = _model().eval()
    _open_scale(model, 0.95)
    values = _inputs()
    output = model(**values)
    for row in range(2):
        mask = values["candidate_mask"][row]
        assert abs(float(output.divergence[row, mask].sum().detach())) < 1e-6
        assert abs(
            float(output.conservative_score_delta[row, mask].sum().detach())
        ) < 1e-6
    assert (
        float(output.conservative_score_delta.abs().max().detach())
        <= 0.7 + 1e-7
    )


def test_candidate_common_evidence_cannot_create_common_translation() -> None:
    model = _model().eval()
    _open_scale(model)
    values = _inputs()
    shared = torch.randn(2, 1, 8)
    values["candidates"] = shared.expand(-1, 4, -1).clone()
    output = model(**values)
    assert torch.count_nonzero(output.divergence) == 0
    assert torch.equal(output.scores, values["base_scores"])


def test_single_candidate_has_zero_flow() -> None:
    model = _model().eval()
    _open_scale(model)
    values = _inputs()
    values["candidate_mask"][:, 1:] = False
    output = model(**values)
    assert torch.count_nonzero(output.divergence) == 0


def test_candidate_permutation_is_equivariant() -> None:
    model = _model().eval()
    _open_scale(model)
    values = _inputs()
    output = model(**values)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(values)
    for key in ("candidates", "candidate_mask", "base_scores"):
        permuted[key] = values[key][:, permutation]
    permuted_output = model(**permuted)
    assert torch.allclose(
        permuted_output.scores,
        output.scores[:, permutation],
        atol=1e-7,
        rtol=1e-7,
    )
    assert torch.allclose(
        permuted_output.divergence,
        output.divergence[:, permutation],
        atol=1e-7,
        rtol=1e-6,
    )


def test_masked_nan_never_contaminates_valid_rows() -> None:
    model = _model().eval()
    _open_scale(model)
    values = _inputs()
    values["candidates"][1, 3] = torch.nan
    values["history"][1, 1:] = torch.nan
    output = model(**values)
    assert torch.isfinite(output.scores[values["candidate_mask"]]).all()
    assert torch.isfinite(output.divergence[values["candidate_mask"]]).all()


def test_conservation_statistics_force_fp32_from_low_precision_factors() -> None:
    torch.manual_seed(109)
    for dtype in (torch.float16, torch.bfloat16):
        factor_a = torch.randn(2, 100, 5, 8).to(dtype)
        factor_b = torch.randn(2, 100, 5, 8).to(dtype)
        outputs = low_rank_hodge_calibration(
            factor_a,
            factor_b,
            torch.ones(2, 100, dtype=torch.bool),
            torch.ones(2, 5, dtype=torch.bool),
        )
        assert all(value.dtype == torch.float32 for value in outputs)
        potential, trust, _, cycle, trusted, *_ = outputs
        assert torch.isfinite(potential).all()
        assert torch.isfinite(trust).all()
        assert torch.isfinite(cycle).all()
        assert torch.isfinite(trusted).all()
        assert float(potential.sum(dim=1).abs().max()) < 1e-5
        assert float(trusted.sum(dim=1).abs().max()) < 1e-5


def test_two_steps_open_scale_then_reach_all_evidence_paths() -> None:
    model = _model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    values = _inputs()
    labels = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    nonzero_by_step = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(**values)
        loss = masked_listwise_loss(
            output.scores, labels, values["candidate_mask"]
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        nonzero_by_step.append(
            {
                name.split(".")[0]
                for name, parameter in model.named_parameters()
                if bool((parameter.grad != 0).any().item())
            }
        )
        optimizer.step()
    assert nonzero_by_step[0] == {"raw_residual_scale"}
    assert {
        "query_projection",
        "candidate_projection",
        "history_projection",
        "factor_a_projection",
        "factor_b_projection",
        "raw_residual_scale",
    }.issubset(nonzero_by_step[1])


def test_nested_candidate_pools_remain_finite_and_globally_bounded() -> None:
    # Hodge reliability is intentionally set-conditioned, so C06 makes no
    # per-distractor Lipschitz claim. Nested pools are an empirical audit; the
    # only hard contract is conservation plus the final-score trust region.
    model = _model().eval()
    _open_scale(model, 0.75)
    torch.manual_seed(107)
    query = torch.randn(1, 8)
    history = torch.randn(1, 2, 8)
    old_candidates = torch.randn(1, 5, 8)
    common = {
        "query": query,
        "history": history,
        "history_mask": torch.ones(1, 2, dtype=torch.bool),
        "history_prior": torch.ones(1, 2),
    }
    for candidates in (
        old_candidates,
        torch.cat([old_candidates, torch.randn(1, 1, 8)], dim=1),
    ):
        output = model(
            **common,
            candidates=candidates,
            candidate_mask=torch.ones(1, candidates.shape[1], dtype=torch.bool),
            base_scores=torch.zeros(1, candidates.shape[1]),
        )
        assert torch.isfinite(output.scores).all()
        assert abs(float(output.conservative_score_delta.sum().detach())) < 1e-6
        assert (
            float(output.conservative_score_delta.abs().max().detach())
            <= 0.7 + 1e-7
        )


def test_history_prior_above_one_is_rejected() -> None:
    values = _inputs()
    values["history_prior"][0, 0] = 1.01
    with torch.no_grad():
        try:
            _model()(**values)
        except ValueError as error:
            assert "at most one" in str(error)
        else:
            raise AssertionError("history prior above one must be rejected")
