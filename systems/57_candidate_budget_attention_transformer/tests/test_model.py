from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.candidate_budget import CandidateBudgetAttentionTransformer, MODES  # noqa: E402


def make_model(*, zero: bool = False) -> CandidateBudgetAttentionTransformer:
    torch.manual_seed(5)
    return CandidateBudgetAttentionTransformer(
        input_dim=12,
        hidden_dim=16,
        heads=4,
        ffn_dim=32,
        token_layers=1,
        dropout=0.0,
        max_query_tokens=5,
        max_item_tokens=6,
        max_history=4,
        zero_initial_output=zero,
    ).eval()


def batch(*, nohistory: bool = False, repeat: bool = False) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    value = {
        "query_tokens": torch.randn(2, 5, 12),
        "query_token_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool),
        "candidate_tokens": torch.randn(2, 4, 6, 12),
        "candidate_token_mask": torch.tensor(
            [
                [[1, 1, 1, 1, 0, 0]] * 4,
                [[1, 1, 1, 0, 0, 0]] * 3 + [[0, 0, 0, 0, 0, 0]],
            ],
            dtype=torch.bool,
        ),
        "history_tokens": torch.randn(2, 3, 6, 12),
        "history_token_mask": torch.tensor(
            [
                [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]],
                [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            ],
            dtype=torch.bool,
        ),
        "event_weights": torch.tensor([[1.0, 0.8, 0.6], [1.0, 0.7, 0.0]]),
        "candidate_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
        "base_scores": torch.tensor([[1.0, 0.5, -0.2, -0.8], [0.7, 0.1, -0.4, 0.0]]),
        "item_only_scores": torch.tensor([[2.0, 0.4, -0.3, -1.0], [1.2, 0.0, -0.5, 0.0]]),
        "repeat_request": torch.full((2,), repeat, dtype=torch.bool),
    }
    if nohistory:
        value["history_token_mask"].zero_()
        value["event_weights"].zero_()
    return value


def permute(value: dict[str, torch.Tensor], order: torch.Tensor) -> dict[str, torch.Tensor]:
    output = dict(value)
    for name in (
        "candidate_tokens",
        "candidate_token_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        output[name] = value[name][:, order]
    return output


def test_candidate_budget_conserves_event_mass() -> None:
    model = make_model()
    value = batch()
    with torch.inference_mode():
        output = model(**value)
    cardinality = value["candidate_mask"].sum(dim=-1).to(output.allocation.dtype) + 1
    total = output.allocation.sum(dim=1) / cardinality[:, None, None]
    assert torch.allclose(total + output.null_allocation, torch.ones_like(total), atol=1e-6)


def test_nohistory_and_repeat_fallbacks_are_exact() -> None:
    model = make_model()
    nohistory = batch(nohistory=True)
    repeat = batch(repeat=True)
    with torch.inference_mode():
        first = model(**nohistory)
        second = model(**repeat)
    assert torch.equal(first.correction, torch.zeros_like(first.correction))
    assert torch.equal(first.scores, nohistory["base_scores"])
    expected = repeat["item_only_scores"].masked_fill(~repeat["candidate_mask"], 0.0)
    assert torch.equal(second.correction, torch.zeros_like(second.correction))
    assert torch.equal(second.scores, expected)


def test_candidate_permutation_is_equivariant() -> None:
    model = make_model()
    value = batch()
    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(order)
    with torch.inference_mode():
        first = model(**value)
        changed = model(**permute(value, order))
    assert torch.allclose(first.scores, changed.scores[:, inverse], atol=2e-6, rtol=0.0)
    assert torch.allclose(first.features, changed.features[:, inverse], atol=2e-6, rtol=0.0)


def test_axis_and_null_reductions_change_features() -> None:
    model = make_model()
    value = batch()
    with torch.inference_mode():
        primary = model(**value, mode="candidate_budget")
        no_null = model(**value, mode="slot_budget_no_null")
        history = model(**value, mode="history_softmax")
    assert not torch.equal(primary.features, no_null.features)
    assert not torch.equal(primary.features, history.features)


def test_all_modes_share_capacity_and_zero_head_starts_at_base() -> None:
    model = make_model(zero=True)
    value = batch()
    with torch.inference_mode():
        for mode in MODES:
            output = model(**value, mode=mode)
            assert torch.equal(output.correction, torch.zeros_like(output.correction))
            expected = value["base_scores"].masked_fill(~value["candidate_mask"], 0.0)
            assert torch.equal(output.scores, expected)
    assert model.parameter_count() > 0
