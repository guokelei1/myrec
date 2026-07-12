from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.amit import MODES, AnchoredMobiusInteractionTransformer


def model(mode: str = "mobius3") -> AnchoredMobiusInteractionTransformer:
    torch.manual_seed(7)
    return AnchoredMobiusInteractionTransformer(
        input_dim=8,
        hidden_dim=16,
        heads=4,
        layers=2,
        ffn_dim=32,
        dropout=0.0,
        max_history=5,
        score_delta_max=0.75,
        mode=mode,
    ).eval()


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(8)
    b, c, h, d = 3, 4, 5, 8
    candidate_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool
    )
    history_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool
    )
    repeat = torch.zeros(b, c, h, dtype=torch.bool)
    repeat[1, 1, 1] = True
    return {
        "query": torch.randn(b, d),
        "candidates": torch.randn(b, c, d),
        "history": torch.randn(b, h, d),
        "candidate_mask": candidate_mask,
        "history_mask": history_mask,
        "repeat_mask": repeat,
        "event_weights": torch.tensor(
            [[1.0, 1.5, 1.0, 1.0, 1.5], [1.0, 1.5, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0]]
        ),
        "base_scores": torch.randn(b, c),
        "item_only_scores": torch.randn(b, c),
    }


def test_anchored_mobius_token_is_exactly_null_by_source() -> None:
    values = batch()
    candidate = model()
    query, items, history, weights = (
        values["query"],
        values["candidates"],
        values["history"],
        values["event_weights"],
    )
    for changed in (
        (torch.zeros_like(query), items, history),
        (query, torch.zeros_like(items), history),
        (query, items, torch.zeros_like(history)),
    ):
        tokens = candidate.interaction_tokens(*changed, weights)
        assert float(tokens.abs().max()) <= 1e-6


def test_only_strict_nonrepeat_request_receives_correction() -> None:
    values = batch()
    output = model()(**values)
    assert output.active_request.tolist() == [True, False, False]
    assert bool(output.correction[0].ne(0).any())
    assert torch.equal(output.scores[1, :3], values["item_only_scores"][1, :3])
    assert torch.equal(output.scores[2, :2], values["base_scores"][2, :2])


def test_query_absence_returns_registered_anchor() -> None:
    values = batch()
    values["query_present"] = torch.zeros(3, dtype=torch.bool)
    output = model()(**values)
    expected = torch.stack(
        (values["base_scores"][0], values["item_only_scores"][1], values["base_scores"][2])
    ).masked_fill(~values["candidate_mask"], 0.0)
    assert torch.equal(output.scores, expected)


def test_candidate_permutation_equivariance() -> None:
    values = batch()
    candidate = model()
    clean = candidate(**values).scores
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    changed = dict(values)
    for name in ("candidates", "candidate_mask", "repeat_mask", "base_scores", "item_only_scores"):
        changed[name] = values[name][:, permutation]
    recovered = candidate(**changed).scores[:, inverse]
    assert torch.allclose(clean, recovered, atol=1e-7, rtol=0.0)


def test_modes_match_parameters_and_have_finite_gradients() -> None:
    counts, names = [], []
    for mode in MODES:
        candidate = model(mode).train()
        counts.append(candidate.parameter_count())
        names.append(tuple(name for name, _ in candidate.named_parameters()))
        loss = candidate(**batch()).scores.square().sum()
        loss.backward()
        gradients = [parameter.grad for parameter in candidate.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert len(set(counts)) == 1
    assert len(set(names)) == 1


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        model("dataset_router")
