from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.nn import functional as F


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for value in (str(SYSTEM_ROOT), str(REPO_ROOT / "src")):
    sys.path.insert(0, value)

from model.counterfactual_edge import (  # noqa: E402
    MODES,
    CounterfactualEdgeLikelihoodTransformer,
    adjacent_pair_targets,
)


def make_model(*, zero: bool = False) -> CounterfactualEdgeLikelihoodTransformer:
    torch.manual_seed(7)
    return CounterfactualEdgeLikelihoodTransformer(
        input_dim=12,
        hidden_dim=16,
        heads=4,
        ffn_dim=32,
        token_layers=1,
        edge_layers=1,
        dropout=0.0,
        max_query_tokens=4,
        max_item_tokens=5,
        max_history=3,
        zero_initial_output=zero,
    ).eval()


def batch(*, nohistory: bool = False, repeat: bool = False) -> dict[str, torch.Tensor]:
    torch.manual_seed(19)
    value = {
        "query_tokens": torch.randn(1, 4, 12),
        "query_token_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        "candidate_tokens": torch.randn(1, 3, 5, 12),
        "candidate_token_mask": torch.tensor([[[1, 1, 1, 1, 0]] * 3], dtype=torch.bool),
        "history_tokens": torch.randn(1, 2, 5, 12),
        "history_token_mask": torch.tensor([[[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]], dtype=torch.bool),
        "event_weights": torch.tensor([[1.0, 0.6]]),
        "candidate_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
        "canonical_order": torch.tensor([[0, 1, 2]]),
        "base_scores": torch.tensor([[1.0, 0.2, -1.0]]),
        "item_only_scores": torch.tensor([[2.0, 0.0, -2.0]]),
        "repeat_request": torch.tensor([repeat]),
    }
    if nohistory:
        value["history_token_mask"].zero_()
        value["event_weights"].zero_()
    return value


def permute(value: dict[str, torch.Tensor], order: torch.Tensor) -> dict[str, torch.Tensor]:
    output = dict(value)
    inverse = torch.argsort(order)
    for name in (
        "candidate_tokens",
        "candidate_token_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        output[name] = value[name][:, order]
    output["canonical_order"] = inverse[value["canonical_order"]]
    return output


def test_factual_null_and_repeat_fallbacks_are_exact() -> None:
    model = make_model()
    with torch.inference_mode():
        nohistory = model(**batch(nohistory=True))
        repeat = model(**batch(repeat=True))
    assert torch.equal(nohistory.likelihood_ratio, torch.zeros_like(nohistory.likelihood_ratio))
    assert torch.equal(nohistory.scores, batch()["base_scores"])
    assert torch.equal(repeat.scores, batch()["item_only_scores"])
    assert torch.equal(repeat.correction, torch.zeros_like(repeat.correction))


def test_transport_is_conservative_and_capacity_bounded() -> None:
    model = make_model()
    with torch.inference_mode():
        output = model(**batch())
    assert torch.all(output.transport >= 0)
    assert torch.all(output.transport <= output.base_gap + 1e-6)
    assert torch.allclose(output.correction.sum(dim=-1), torch.zeros(1), atol=1e-6)
    assert not torch.equal(output.likelihood_ratio, torch.zeros_like(output.likelihood_ratio))


def test_candidate_permutation_is_equivariant() -> None:
    model = make_model()
    value = batch()
    order = torch.tensor([2, 0, 1])
    inverse = torch.argsort(order)
    with torch.inference_mode():
        first = model(**value)
        second = model(**permute(value, order))
    assert torch.allclose(first.scores, second.scores[:, inverse], atol=2e-6, rtol=0.0)
    assert torch.allclose(first.likelihood_ratio, second.likelihood_ratio, atol=2e-6, rtol=0.0)


def test_zero_initialized_modes_start_at_base() -> None:
    model = make_model(zero=True)
    value = batch()
    with torch.inference_mode():
        for mode in MODES:
            output = model(**value, mode=mode)
            assert torch.equal(output.scores, value["base_scores"])
            assert torch.equal(output.likelihood_ratio, torch.zeros_like(output.likelihood_ratio))


def test_adjacent_pair_targets_and_gradient() -> None:
    model = make_model(zero=True).train()
    value = batch()
    labels = torch.tensor([[0.0, 1.0, 0.0]])
    target, eligible = adjacent_pair_targets(
        labels, value["canonical_order"], value["candidate_mask"]
    )
    assert torch.equal(target, torch.tensor([[1.0, 0.0]]))
    assert bool(eligible.all())
    output = model(**value)
    loss = F.binary_cross_entropy_with_logits(
        output.pair_logits[eligible], target[eligible]
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.edge_head.weight.grad is not None
    assert bool(model.edge_head.weight.grad.ne(0).any())
