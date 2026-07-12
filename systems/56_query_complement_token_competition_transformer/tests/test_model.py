from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.query_complement import (  # noqa: E402
    MODES,
    QueryComplementTokenCompetitionTransformer,
)


def make_model(*, zero: bool = False) -> QueryComplementTokenCompetitionTransformer:
    torch.manual_seed(7)
    return QueryComplementTokenCompetitionTransformer(
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
    torch.manual_seed(11)
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


def permute_candidates(value: dict[str, torch.Tensor], permutation: torch.Tensor) -> dict[str, torch.Tensor]:
    result = dict(value)
    for name in (
        "candidate_tokens",
        "candidate_token_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        result[name] = value[name][:, permutation]
    return result


def test_primary_nohistory_is_exact_base() -> None:
    model = make_model()
    value = batch(nohistory=True)
    with torch.inference_mode():
        output = model(**value)
    assert torch.equal(output.correction, torch.zeros_like(output.correction))
    assert torch.equal(output.scores, value["base_scores"])
    assert torch.equal(output.carrier, torch.zeros_like(output.carrier))
    assert torch.equal(output.list_message, torch.zeros_like(output.list_message))


def test_repeat_is_exact_item_only() -> None:
    model = make_model()
    value = batch(repeat=True)
    with torch.inference_mode():
        output = model(**value)
    expected = value["item_only_scores"].masked_fill(~value["candidate_mask"], 0.0)
    assert torch.equal(output.correction, torch.zeros_like(output.correction))
    assert torch.equal(output.scores, expected)


def test_candidate_permutation_is_equivariant() -> None:
    model = make_model()
    value = batch()
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    with torch.inference_mode():
        first = model(**value)
        changed = model(**permute_candidates(value, permutation))
    assert torch.allclose(first.scores, changed.scores[:, inverse], atol=2e-6, rtol=0.0)
    assert torch.allclose(first.correction, changed.correction[:, inverse], atol=2e-6, rtol=0.0)


def test_history_and_edge_paths_are_numerically_loadable() -> None:
    model = make_model()
    value = batch()
    changed = dict(value)
    changed["history_tokens"] = value["history_tokens"].roll(1, dims=1)
    with torch.inference_mode():
        factual = model(**value)
        wrong = model(**changed)
        independent = model(**value, edge_ablation=True)
    assert not torch.equal(factual.correction, wrong.correction)
    assert not torch.equal(factual.correction, independent.correction)


def test_modes_share_capacity_but_not_operator() -> None:
    model = make_model()
    value = batch()
    with torch.inference_mode():
        outputs = {mode: model(**value, mode=mode).correction for mode in MODES}
    assert len({model.parameter_count() for _ in MODES}) == 1
    assert not torch.equal(outputs["query_complement_token"], outputs["unprojected_token"])
    assert not torch.equal(outputs["query_complement_token"], outputs["pooled_complement"])
    assert not torch.equal(outputs["query_complement_token"], outputs["raw_candidate"])


def test_zero_initialized_head_starts_at_anchor() -> None:
    model = make_model(zero=True)
    value = batch()
    with torch.inference_mode():
        for mode in MODES:
            output = model(**value, mode=mode)
            assert torch.equal(output.correction, torch.zeros_like(output.correction))
            assert torch.equal(output.scores, value["base_scores"].masked_fill(~value["candidate_mask"], 0.0))
