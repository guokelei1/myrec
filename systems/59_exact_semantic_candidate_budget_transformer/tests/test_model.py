from __future__ import annotations

import math
from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.semantic_budget import MODES, SemanticCandidateBudgetTransformer  # noqa: E402


def batch(*, nohistory: bool = False, repeat: bool = False) -> dict[str, torch.Tensor]:
    value = {
        "query_tokens": torch.tensor([[[1.0, 0.0]]]),
        "query_token_mask": torch.tensor([[1]], dtype=torch.bool),
        "candidate_tokens": torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]),
        "candidate_token_mask": torch.tensor([[[1], [1]]], dtype=torch.bool),
        "history_tokens": torch.tensor([[[[1.0, 0.0]]]]),
        "history_token_mask": torch.tensor([[[1]]], dtype=torch.bool),
        "event_weights": torch.tensor([[2.0]]),
        "candidate_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        "base_scores": torch.tensor([[0.0, 0.0]]),
        "item_only_scores": torch.tensor([[2.0, -2.0]]),
        "repeat_request": torch.tensor([repeat]),
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


def test_fixed_candidate_budget_matches_hand_computation() -> None:
    model = SemanticCandidateBudgetTransformer()
    with torch.inference_mode():
        output = model(**batch())
    denominator = math.e + 2.0
    expected = torch.tensor([math.e / denominator, 1.0 / denominator])
    assert torch.allclose(output.features["candidate_budget"][0], expected, atol=1e-7)
    assert torch.allclose(output.null_allocation[0, 0], torch.tensor(1.0 / denominator), atol=1e-7)
    assert torch.allclose(output.triadic_logits[0, :, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(output.corrections["candidate_budget"], torch.tensor([[1.0, -1.0]]))


def test_nohistory_and_repeat_fallbacks_are_exact() -> None:
    model = SemanticCandidateBudgetTransformer()
    with torch.inference_mode():
        nohistory = model(**batch(nohistory=True))
        repeat = model(**batch(repeat=True))
    for mode in MODES:
        assert torch.equal(nohistory.corrections[mode], torch.zeros((1, 2)))
        assert torch.equal(nohistory.scores[mode], batch()["base_scores"])
        assert torch.equal(repeat.corrections[mode], torch.zeros((1, 2)))
        assert torch.equal(repeat.scores[mode], batch()["item_only_scores"])


def test_candidate_permutation_is_equivariant() -> None:
    model = SemanticCandidateBudgetTransformer()
    value = batch()
    order = torch.tensor([1, 0])
    inverse = torch.argsort(order)
    with torch.inference_mode():
        first = model(**value)
        second = model(**permute(value, order))
    for mode in MODES:
        assert torch.allclose(first.scores[mode], second.scores[mode][:, inverse], atol=1e-7)
        assert torch.allclose(first.features[mode], second.features[mode][:, inverse], atol=1e-7)


def test_operator_has_no_trainable_gauge() -> None:
    model = SemanticCandidateBudgetTransformer()
    assert model.parameter_count() == 0
    assert list(model.parameters()) == []


def test_constant_semantic_feature_abstains() -> None:
    model = SemanticCandidateBudgetTransformer()
    value = batch()
    value["candidate_tokens"][:] = torch.tensor([1.0, 0.0])
    with torch.inference_mode():
        output = model(**value)
    for mode in MODES:
        assert torch.equal(output.corrections[mode], torch.zeros((1, 2)))

def test_sorted_set_reductions_are_exact_under_large_permutation() -> None:
    model = SemanticCandidateBudgetTransformer()
    torch.manual_seed(91)
    logits = torch.randn(7, 1001, dtype=torch.float32)
    order = torch.randperm(logits.shape[-1])
    inverse = torch.argsort(order)
    first = model._symmetric_softmax(logits)
    second = model._symmetric_softmax(logits[:, order])[:, inverse]
    assert torch.equal(first, second)

    values = torch.randn(1, 1001, dtype=torch.float32)
    mask = torch.ones_like(values, dtype=torch.bool)
    first_z = model._zscore(values, mask)
    second_z = model._zscore(values[:, order], mask)[:, inverse]
    assert torch.equal(first_z, second_z)
