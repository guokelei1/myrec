"""CPU-only structural tests for C08; no repository data or labels are read."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from reversible_memory import (
    ReversibleCouplingMemory,
    TinyReversibleRanker,
    reversible_coupling_step,
)


def _basis(index: int, width: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    value = torch.zeros(width, dtype=dtype)
    value[index] = 1.0
    return value


def _manual_loop(
    seed: torch.Tensor,
    history_first: torch.Tensor,
    history_second: torch.Tensor,
    history_strength: torch.Tensor,
    probe_first: torch.Tensor,
    probe_second: torch.Tensor,
    probe_strength: torch.Tensor,
) -> torch.Tensor:
    history_gains = torch.tensor([1.0, 0.70], dtype=seed.dtype)
    history_biases = torch.tensor([0.0, 0.0], dtype=seed.dtype)
    probe_gains = torch.tensor([0.80, -0.90], dtype=seed.dtype)
    probe_biases = torch.tensor([0.35, -0.25], dtype=seed.dtype)
    state = seed
    for first, second, strength in zip(
        history_first, history_second, history_strength, strict=True
    ):
        state = reversible_coupling_step(
            state,
            first,
            second,
            strength,
            history_gains,
            history_biases,
        )
    state = reversible_coupling_step(
        state,
        probe_first,
        probe_second,
        probe_strength,
        probe_gains,
        probe_biases,
    )
    for first, second, strength in zip(
        reversed(history_first),
        reversed(history_second),
        reversed(history_strength),
        strict=True,
    ):
        state = reversible_coupling_step(
            state,
            first,
            second,
            strength,
            history_gains,
            history_biases,
            inverse=True,
        )
    state = reversible_coupling_step(
        state,
        probe_first,
        probe_second,
        probe_strength,
        probe_gains,
        probe_biases,
        inverse=True,
    )
    return state - seed


def _toy_batch() -> dict[str, torch.Tensor]:
    return {
        "query_tokens": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "history_tokens": torch.tensor(
            [
                [[6, 7, 0], [8, 9, 0]],
                [[10, 11, 0], [12, 13, 0]],
            ]
        ),
        "history_item_ids": torch.tensor([[3, 5], [4, 7]]),
        "history_mask": torch.tensor([[True, True], [True, True]]),
        "candidate_tokens": torch.tensor(
            [
                [[14, 15, 0], [8, 9, 0], [16, 17, 0]],
                [[18, 19, 0], [20, 21, 0], [12, 13, 0]],
            ]
        ),
        "candidate_item_ids": torch.tensor([[2, 5, 6], [8, 9, 7]]),
    }


def test_reversible_write_undo_and_volume_conservation() -> None:
    dtype = torch.float64
    state = torch.tensor([0.2, -0.1, 0.3, -0.2, 0.4, 0.1], dtype=dtype)
    first = torch.tensor([0.7, -0.2, 0.5], dtype=dtype)
    second = torch.tensor([-0.3, 0.8, 0.1], dtype=dtype)
    strength = torch.tensor(0.61, dtype=dtype)
    gains = torch.tensor([0.9, -0.6], dtype=dtype)
    biases = torch.tensor([0.2, -0.3], dtype=dtype)

    written = reversible_coupling_step(
        state, first, second, strength, gains, biases
    )
    restored = reversible_coupling_step(
        written, first, second, strength, gains, biases, inverse=True
    )
    assert not torch.allclose(written, state)
    assert torch.max(torch.abs(restored - state)).item() < 1e-12

    jacobian = torch.autograd.functional.jacobian(
        lambda value: reversible_coupling_step(
            value, first, second, strength, gains, biases
        ),
        state,
    )
    determinant = torch.linalg.det(jacobian)
    assert determinant.item() == pytest.approx(1.0, abs=1e-10)


def test_history_write_read_information_flow_and_disjoint_support() -> None:
    width = 3
    aligned = _basis(0, width)
    disjoint = _basis(1, width)
    seed = torch.tensor([0.2, 0.1, -0.1, 0.3, -0.2, 0.4], dtype=torch.float64)
    history_first = torch.stack((aligned,))
    history_second = torch.stack((aligned,))
    history_strength = torch.tensor([0.60], dtype=torch.float64)

    aligned_read = _manual_loop(
        seed,
        history_first,
        history_second,
        history_strength,
        aligned,
        aligned,
        torch.tensor(0.70, dtype=torch.float64),
    )
    disjoint_read = _manual_loop(
        seed,
        history_first,
        history_second,
        history_strength,
        disjoint,
        disjoint,
        torch.tensor(0.70, dtype=torch.float64),
    )
    assert torch.linalg.vector_norm(aligned_read).item() > 1e-3
    assert torch.linalg.vector_norm(disjoint_read).item() < 1e-12


def test_endpoint_collision_is_a_nontrivial_reduction_witness() -> None:
    """The loop sees a write map that an ordinary endpoint vector cannot see."""

    width = 3
    axis = _basis(0, width)
    seed = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    history_gains = torch.tensor([1.0, 0.70], dtype=torch.float64)
    history_biases = torch.tensor([0.0, 0.0], dtype=torch.float64)
    strength = torch.tensor(0.60, dtype=torch.float64)
    terminal = reversible_coupling_step(
        seed,
        axis,
        axis,
        strength,
        history_gains,
        history_biases,
    )
    # This particular write fixes z0, so every endpoint-only vector read is
    # identical to empty history at z0.
    assert torch.equal(terminal, seed)

    loop_read = _manual_loop(
        seed,
        torch.stack((axis,)),
        torch.stack((axis,)),
        torch.stack((strength,)),
        axis,
        axis,
        torch.tensor(0.70, dtype=torch.float64),
    )
    assert torch.linalg.vector_norm(loop_read).item() > 0.05

    # Exercise the implemented matched reduction, not only the hand equation.
    memory = ReversibleCouplingMemory(d_model=8, evidence_dim=width).double().eval()
    with torch.no_grad():
        memory.history_gains.copy_(torch.tensor([1.0, 0.70], dtype=torch.float64))
        memory.probe_gains.copy_(torch.tensor([0.80, -0.90], dtype=torch.float64))
        memory.history_biases.copy_(torch.tensor([0.0, 0.0], dtype=torch.float64))
        memory.probe_biases.copy_(torch.tensor([0.35, -0.25], dtype=torch.float64))
    history_axis = axis.view(1, 1, width)
    probe_axis = axis.view(1, 1, width)
    history_mask = torch.ones(1, 1, dtype=torch.bool)
    implemented_loop = memory.interaction_residual_from_axes(
        history_axis,
        history_axis,
        strength.view(1, 1),
        probe_axis,
        probe_axis,
        torch.tensor([[0.70]], dtype=torch.float64),
        history_mask,
        seed=seed,
    )
    ordinary_endpoint = memory.ordinary_residual_from_axes(
        history_axis,
        history_axis,
        strength.view(1, 1),
        probe_axis,
        probe_axis,
        torch.tensor([[0.70]], dtype=torch.float64),
        history_mask,
        seed=seed,
    )
    assert torch.allclose(implemented_loop.view(-1), loop_read, atol=1e-12, rtol=0.0)
    assert torch.equal(ordinary_endpoint, torch.zeros_like(ordinary_endpoint))

    loop_model = TinyReversibleRanker(read_mode="loop")
    ordinary_model = deepcopy(loop_model)
    ordinary_model.memory.read_mode = "ordinary"
    loop_shapes = [(name, tuple(value.shape)) for name, value in loop_model.named_parameters()]
    ordinary_shapes = [
        (name, tuple(value.shape)) for name, value in ordinary_model.named_parameters()
    ]
    assert loop_shapes == ordinary_shapes
    assert sum(value.numel() for value in loop_model.parameters()) == sum(
        value.numel() for value in ordinary_model.parameters()
    )


def test_empty_history_is_exact_zero_and_exact_query_only_fallback() -> None:
    torch.manual_seed(8)
    memory = ReversibleCouplingMemory(d_model=8, evidence_dim=3).eval()
    history = torch.randn(2, 3, 8)
    query = torch.randn(2, 8)
    candidates = torch.randn(2, 4, 8)
    empty_mask = torch.zeros(2, 3, dtype=torch.bool)
    hidden, diagnostics = memory(history, query, candidates, empty_mask)
    assert torch.equal(diagnostics.raw_residual, torch.zeros_like(diagnostics.raw_residual))
    assert torch.equal(hidden, torch.zeros_like(hidden))

    model = TinyReversibleRanker().eval()
    batch = _toy_batch()
    batch["history_mask"] = torch.zeros_like(batch["history_mask"])
    history_scores = model(**batch)
    query_only_scores = model.forward_query_only(
        batch["query_tokens"],
        batch["candidate_tokens"],
        batch["candidate_item_ids"],
    )
    assert torch.equal(history_scores, query_only_scores)


def test_candidate_permutation_equivariance() -> None:
    torch.manual_seed(9)
    model = TinyReversibleRanker().eval()
    batch = _toy_batch()
    original = model(**batch)
    permutation = torch.tensor([2, 0, 1])
    permuted_batch = dict(batch)
    permuted_batch["candidate_tokens"] = batch["candidate_tokens"][:, permutation]
    permuted_batch["candidate_item_ids"] = batch["candidate_item_ids"][:, permutation]
    permuted = model(**permuted_batch)
    assert torch.allclose(permuted, original[:, permutation], atol=1e-6, rtol=1e-6)


def test_score_common_mode_is_removed_before_the_ranker() -> None:
    torch.manual_seed(10)
    memory = ReversibleCouplingMemory(d_model=8, evidence_dim=3).eval()
    raw = torch.randn(2, 4, 6)
    common = torch.randn(2, 1, 6)
    hidden = memory.hidden_from_raw(raw)
    shifted_hidden = memory.hidden_from_raw(raw + common)
    assert torch.allclose(hidden, shifted_hidden, atol=1e-6, rtol=1e-6)

    base_scores = torch.randn(2, 4)
    common_score = torch.randn(2, 1)
    before = torch.argsort(base_scores, dim=1, descending=True)
    after = torch.argsort(base_scores + common_score, dim=1, descending=True)
    assert torch.equal(before, after)


def test_history_order_changes_the_noncommutative_read() -> None:
    width = 3
    first = _basis(0, width)
    # Partially overlapping supports are required: disjoint writes commute by
    # construction and are intentionally rejected as unsupported evidence.
    second = _basis(0, width) + _basis(1, width)
    second = second / torch.linalg.vector_norm(second)
    seed = torch.tensor([0.2, 0.1, -0.1, 0.3, -0.2, 0.4], dtype=torch.float64)
    strengths = torch.tensor([0.65, -0.45], dtype=torch.float64)
    probe_strength = torch.tensor(0.70, dtype=torch.float64)

    forward_order = _manual_loop(
        seed,
        torch.stack((first, second)),
        torch.stack((first, second)),
        strengths,
        first,
        first,
        probe_strength,
    )
    reversed_order = _manual_loop(
        seed,
        torch.stack((second, first)),
        torch.stack((second, first)),
        strengths.flip(0),
        first,
        first,
        probe_strength,
    )
    assert torch.linalg.vector_norm(forward_order - reversed_order).item() > 1e-4


def test_two_optimizer_steps_reach_write_probe_and_transformer() -> None:
    torch.manual_seed(11)
    model = TinyReversibleRanker()
    batch = _toy_batch()
    targets = torch.tensor([1, 2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    before = model.memory.first_axis.weight.detach().clone()
    losses: list[float] = []

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        scores = model(**batch)
        loss = F.cross_entropy(scores, targets)
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in (
            model.memory.first_axis.weight,
            model.memory.query_condition.weight,
            model.memory.residual_to_hidden.weight,
            model.pre_block.linear1.weight,
            model.post_block.linear1.weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum().item() > 0.0
        optimizer.step()
        losses.append(loss.item())

    assert not torch.equal(before, model.memory.first_axis.weight.detach())
    assert all(torch.isfinite(torch.tensor(losses)))
