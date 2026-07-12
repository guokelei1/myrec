from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from model.finite_evidence_memory import (  # noqa: E402
    MODES,
    FiniteEvidenceMemoryTransformer,
    listwise_training_loss,
)


def make_model(*, zero: bool = False) -> FiniteEvidenceMemoryTransformer:
    torch.manual_seed(5)
    return FiniteEvidenceMemoryTransformer(
        input_dim=12,
        hidden_dim=24,
        heads=4,
        ffn_dim=48,
        history_layers=1,
        candidate_layers=1,
        memory_slots=4,
        max_history=8,
        sinkhorn_iterations=5,
        dropout=0.0,
        zero_initial_output=zero,
    ).eval()


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    return {
        "query": torch.randn(3, 12),
        "candidates": torch.randn(3, 6, 12),
        "history": torch.randn(3, 5, 12),
        "history_mask": torch.tensor(
            [[True, True, True, False, False], [True] * 5, [True, False, False, False, False]]
        ),
        "candidate_mask": torch.tensor(
            [[True] * 6, [True, True, True, True, False, False], [True] * 6]
        ),
        "base_scores": torch.randn(3, 6),
        "item_only_scores": torch.randn(3, 6),
        "repeat_request": torch.zeros(3, dtype=torch.bool),
        "query_present": torch.ones(3, dtype=torch.bool),
    }


def test_finite_allocation_conserves_each_valid_event() -> None:
    model = make_model()
    values = batch()
    _, allocation, null, slot_mass = model.build_memory(
        history=values["history"],
        history_mask=values["history_mask"],
        query=values["query"],
        mode="finite_evidence_memory",
    )
    total = allocation.sum(dim=-1) + null
    expected = values["history_mask"].to(total.dtype)
    assert torch.allclose(total, expected, atol=2e-7, rtol=0.0)
    assert torch.all(allocation >= 0)
    assert torch.all(null >= 0)
    assert torch.allclose(slot_mass, allocation.sum(dim=1), atol=0.0, rtol=0.0)


def test_memory_is_exactly_query_independent() -> None:
    model = make_model()
    values = batch()
    first = model.build_memory(
        history=values["history"],
        history_mask=values["history_mask"],
        query=values["query"],
        mode="finite_evidence_memory",
    )
    second = model.build_memory(
        history=values["history"],
        history_mask=values["history_mask"],
        query=torch.randn_like(values["query"]) * 100,
        mode="finite_evidence_memory",
    )
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_fallbacks_and_candidate_permutation() -> None:
    model = make_model()
    values = batch()
    output = model(**values, mode="finite_evidence_memory")
    permutation = torch.tensor([5, 4, 3, 2, 1, 0])
    reversed_values = dict(values)
    for name in ("candidates", "candidate_mask", "base_scores", "item_only_scores"):
        reversed_values[name] = values[name][:, permutation]
    reversed_output = model(**reversed_values, mode="finite_evidence_memory")
    assert torch.allclose(
        output.scores,
        reversed_output.scores[:, permutation],
        atol=2e-6,
        rtol=0.0,
    )

    empty = dict(values)
    empty["history_mask"] = torch.zeros_like(values["history_mask"])
    empty_output = model(**empty, mode="finite_evidence_memory")
    expected_base = values["base_scores"].masked_fill(~values["candidate_mask"], 0.0)
    assert torch.equal(empty_output.scores, expected_base)
    assert torch.count_nonzero(empty_output.allocation) == 0
    assert torch.count_nonzero(empty_output.null_mass) == 0

    repeat = dict(values)
    repeat["repeat_request"] = torch.ones(3, dtype=torch.bool)
    repeat_output = model(**repeat, mode="finite_evidence_memory")
    expected_item = values["item_only_scores"].masked_fill(~values["candidate_mask"], 0.0)
    assert torch.equal(repeat_output.scores, expected_item)


def test_modes_have_equal_parameters_and_distinct_allocations() -> None:
    model = make_model()
    values = batch()
    count = model.parameter_count()
    allocations = {}
    for mode in MODES:
        output = model(**values, mode=mode)
        assert model.parameter_count() == count
        assert torch.isfinite(output.scores).all()
        allocations[mode] = output.allocation
    assert not torch.equal(
        allocations["finite_evidence_memory"],
        allocations["slot_competition_memory"],
    )
    standard_event_mass = allocations["standard_slot_memory"].sum(dim=-1)
    assert bool((standard_event_mass[values["history_mask"]] > 1.0).any())


def test_ranking_gradients_reach_binding_and_ranking_groups() -> None:
    model = make_model(zero=True).train()
    values = batch()
    labels = torch.zeros_like(values["base_scores"])
    labels[:, 0] = 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    active: set[str] = set()
    for _ in range(3):
        output = model(**values, mode="finite_evidence_memory")
        wrong_values = dict(values)
        wrong_values["history"] = values["history"].roll(1, 0)
        wrong = model(**wrong_values, mode="finite_evidence_memory")
        loss, _ = listwise_training_loss(
            output,
            labels,
            values["candidate_mask"],
            wrong_output=wrong,
            base_scores=values["base_scores"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and bool(parameter.grad.ne(0).any()):
                active.add(name)
        optimizer.step()
    required = (
        "history_transformer.",
        "event_key_projection.",
        "event_value_projection.",
        "slot_key_projection.",
        "memory_read_attention.",
        "candidate_transformer.",
        "score_head.",
    )
    assert all(any(name.startswith(prefix) for name in active) for prefix in required)
    assert "break_bias" in active
