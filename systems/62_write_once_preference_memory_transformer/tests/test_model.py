from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from model.write_once_memory import (  # noqa: E402
    MODES,
    WriteOncePreferenceMemoryTransformer,
    listwise_training_loss,
)


def make_model(*, zero: bool = False) -> WriteOncePreferenceMemoryTransformer:
    torch.manual_seed(7)
    return WriteOncePreferenceMemoryTransformer(
        input_dim=12,
        hidden_dim=24,
        heads=4,
        ffn_dim=48,
        history_layers=1,
        candidate_layers=1,
        memory_slots=4,
        max_history=8,
        dropout=0.0,
        zero_initial_output=zero,
    ).eval()


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
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
        "repeat_request": torch.tensor([False, False, False]),
        "query_present": torch.tensor([True, True, True]),
    }


def test_primary_memory_is_query_and_candidate_independent() -> None:
    model = make_model()
    values = batch()
    first, first_mask, _ = model.build_memory(
        history=values["history"],
        history_mask=values["history_mask"],
        query=values["query"],
        mode="write_once_memory",
    )
    second, second_mask, _ = model.build_memory(
        history=values["history"],
        history_mask=values["history_mask"],
        query=torch.randn_like(values["query"]) * 100,
        mode="write_once_memory",
    )
    assert torch.equal(first, second)
    assert torch.equal(first_mask, second_mask)

    conditioned, _, _ = model.build_memory(
        history=values["history"],
        history_mask=values["history_mask"],
        query=torch.randn_like(values["query"]) * 100,
        mode="query_conditioned_writer",
    )
    assert not torch.equal(first, conditioned)


def test_nohistory_and_repeat_are_exact_structural_fallbacks() -> None:
    model = make_model()
    values = batch()
    values["history_mask"] = torch.zeros_like(values["history_mask"])
    output = model(**values, mode="write_once_memory")
    expected_base = values["base_scores"].masked_fill(~values["candidate_mask"], 0.0)
    assert torch.equal(output.scores, expected_base)
    assert torch.count_nonzero(output.correction) == 0
    assert torch.count_nonzero(output.memory) == 0

    values = batch()
    values["repeat_request"] = torch.ones(3, dtype=torch.bool)
    output = model(**values, mode="write_once_memory")
    expected = values["item_only_scores"].masked_fill(~values["candidate_mask"], 0.0)
    assert torch.equal(output.scores, expected)
    assert torch.count_nonzero(output.correction) == 0


def test_candidate_permutation_equivariance_and_centering() -> None:
    model = make_model()
    values = batch()
    output = model(**values, mode="write_once_memory")
    permutation = torch.tensor([5, 4, 3, 2, 1, 0])
    reverse = dict(values)
    for name in ("candidates", "candidate_mask", "base_scores", "item_only_scores"):
        reverse[name] = values[name][:, permutation]
    reversed_output = model(**reverse, mode="write_once_memory")
    assert torch.allclose(
        output.scores,
        reversed_output.scores[:, permutation],
        atol=2e-6,
        rtol=0.0,
    )
    centered = (output.correction * values["candidate_mask"]).sum(dim=-1)
    assert torch.allclose(centered, torch.zeros_like(centered), atol=2e-6, rtol=0.0)


def test_all_modes_share_parameter_count_and_are_finite() -> None:
    model = make_model()
    values = batch()
    count = model.parameter_count()
    assert count > 0
    for mode in MODES:
        output = model(**values, mode=mode)
        assert model.parameter_count() == count
        assert torch.isfinite(output.scores).all()
        assert torch.isfinite(output.memory).all()


def test_ranking_loss_reaches_every_architecture_group() -> None:
    model = make_model(zero=True).train()
    values = batch()
    labels = torch.zeros_like(values["base_scores"])
    labels[:, 0] = 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    active: set[str] = set()
    for _ in range(3):
        output = model(**values, mode="write_once_memory")
        wrong_values = dict(values)
        wrong_values["history"] = values["history"].roll(1, 0)
        wrong = model(**wrong_values, mode="write_once_memory")
        loss, components = listwise_training_loss(
            output,
            labels,
            values["candidate_mask"],
            wrong_output=wrong,
            base_scores=values["base_scores"],
        )
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in components.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and bool(parameter.grad.ne(0).any()):
                active.add(name)
        optimizer.step()
    required = (
        "history_transformer.",
        "slot_write_attention.",
        "memory_read_attention.",
        "candidate_transformer.",
        "score_head.",
    )
    assert all(any(name.startswith(prefix) for name in active) for prefix in required)
