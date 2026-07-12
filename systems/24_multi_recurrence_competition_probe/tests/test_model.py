from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.competition import MODES, MultiRecurrenceCompetitionTransformer


def model(mode: str = "set_attention") -> MultiRecurrenceCompetitionTransformer:
    torch.manual_seed(3)
    return MultiRecurrenceCompetitionTransformer(
        input_dim=8,
        hidden_dim=16,
        heads=4,
        layers=2,
        ffn_dim=32,
        dropout=0.0,
        max_history=5,
        max_repeat_candidates=8,
        score_delta_max=0.75,
        mode=mode,
    ).eval()


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(4)
    b, c, h, d = 3, 4, 5, 8
    cm = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    hm = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    repeat = torch.zeros(b, c, h, dtype=torch.bool)
    repeat[0, 0, 0] = True
    repeat[0, 2, 3] = True
    repeat[1, 1, 1] = True
    return {
        "query": torch.randn(b, d),
        "candidates": torch.randn(b, c, d),
        "candidate_mask": cm,
        "history_mask": hm,
        "repeat_mask": repeat,
        "event_weights": torch.tensor(
            [[1.0, 1.5, 1.0, 1.0, 1.5], [1.0, 1.5, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0]]
        ),
        "base_scores": torch.randn(b, c),
        "item_only_scores": torch.randn(b, c),
    }


def test_only_multi_repeat_requests_receive_correction() -> None:
    values = batch()
    output = model()(**values)
    assert output.multi_repeat_request.tolist() == [True, False, False]
    assert bool(output.correction[0].ne(0).any())
    assert torch.equal(output.correction[1:], torch.zeros_like(output.correction[1:]))
    assert torch.equal(output.scores[1, :3], values["item_only_scores"][1, :3])
    assert torch.equal(output.scores[2, :2], values["base_scores"][2, :2])


def test_query_absence_returns_registered_anchor() -> None:
    values = batch()
    values["query_present"] = torch.zeros(3, dtype=torch.bool)
    output = model()(**values)
    expected = torch.stack(
        (values["item_only_scores"][0], values["item_only_scores"][1], values["base_scores"][2])
    ).masked_fill(~values["candidate_mask"], 0.0)
    assert torch.equal(output.scores, expected)


def test_cross_edge_ablation_is_behaviorally_distinct() -> None:
    values = batch()
    primary = model()(**values)
    ablated = model()(**values, disable_cross_candidate=True)
    assert not torch.equal(primary.correction[0], ablated.correction[0])


def test_candidate_permutation_equivariance() -> None:
    values = batch()
    clean = model()(**values).scores
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    changed = dict(values)
    for name in ("candidates", "candidate_mask", "repeat_mask", "base_scores", "item_only_scores"):
        changed[name] = values[name][:, permutation]
    recovered = model()(**changed).scores[:, inverse]
    assert torch.allclose(clean, recovered, atol=1e-7, rtol=0.0)


def test_modes_match_parameters_and_gradients_are_finite() -> None:
    counts = []
    names = []
    for mode in MODES:
        candidate = model(mode).train()
        counts.append(candidate.parameter_count())
        names.append(tuple(name for name, _ in candidate.named_parameters()))
        loss = candidate(**batch()).scores.square().sum()
        loss.backward()
        gradients = [p.grad for p in candidate.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert len(set(counts)) == 1
    assert len(set(names)) == 1


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        model("tournament_rebranding")
