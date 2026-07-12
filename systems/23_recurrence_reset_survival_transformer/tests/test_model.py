from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.rrst import MODES, RecurrenceResetSurvivalTransformer, masked_zscore


def make_model(mode: str = "reset_suffix") -> RecurrenceResetSurvivalTransformer:
    torch.manual_seed(7)
    return RecurrenceResetSurvivalTransformer(
        input_dim=8,
        hidden_dim=16,
        heads=4,
        layers=2,
        ffn_dim=32,
        max_history=5,
        dropout=0.0,
        score_delta_max=0.75,
        mode=mode,
    ).eval()


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    batch_size, candidates, history, dim = 2, 4, 5, 8
    candidate_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    history_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    repeat = torch.zeros(batch_size, candidates, history, dtype=torch.bool)
    repeat[0, 0, 1] = True
    repeat[0, 0, 3] = True
    repeat[0, 2, 2] = True
    return {
        "query": torch.randn(batch_size, dim),
        "candidates": torch.randn(batch_size, candidates, dim),
        "history": torch.randn(batch_size, history, dim),
        "candidate_mask": candidate_mask,
        "history_mask": history_mask,
        "repeat_mask": repeat,
        "event_weights": torch.tensor(
            [[1.0, 1.5, 1.0, 1.0, 1.5], [0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "base_scores": torch.randn(batch_size, candidates),
        "item_only_scores": torch.randn(batch_size, candidates),
    }


def test_reset_graph_starts_at_last_exact() -> None:
    output = make_model()(**batch())
    assert output.last_exact_position.tolist() == [[3, -1, 2, -1], [-1, -1, -1, -1]]
    assert output.allowed_event_mask[0, 0].tolist() == [False, False, False, True, True]
    assert output.allowed_event_mask[0, 2].tolist() == [False, False, True, True, True]
    assert not output.allowed_event_mask[0, 1].any()


def test_nohistory_and_nonrepeat_are_exact_base() -> None:
    inputs = batch()
    inputs["repeat_mask"].zero_()
    output = make_model()(**inputs)
    expected = inputs["base_scores"].masked_fill(~inputs["candidate_mask"], 0.0)
    assert torch.equal(output.scores, expected)
    assert torch.equal(output.correction, torch.zeros_like(output.correction))


def test_query_absence_is_exact_item_only() -> None:
    inputs = batch()
    inputs["query_present"] = torch.zeros(2, dtype=torch.bool)
    output = make_model()(**inputs)
    request_repeat = inputs["repeat_mask"].any(dim=-1).any(dim=-1)
    expected = torch.where(
        request_repeat[:, None], inputs["item_only_scores"], inputs["base_scores"]
    ).masked_fill(~inputs["candidate_mask"], 0.0)
    assert torch.equal(output.scores, expected)


def test_masked_preanchor_values_are_exactly_irrelevant() -> None:
    model = make_model()
    inputs = batch()
    clean = model(**inputs)
    corrupt = model(**inputs, corrupt_preanchor=True)
    assert torch.equal(clean.scores, corrupt.scores)
    assert torch.equal(clean.correction, corrupt.correction)


def test_candidate_permutation_equivariance() -> None:
    model = make_model()
    inputs = batch()
    clean = model(**inputs).scores
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    permuted = dict(inputs)
    for name in ("candidates", "candidate_mask", "repeat_mask", "base_scores", "item_only_scores"):
        permuted[name] = inputs[name][:, permutation]
    recovered = model(**permuted).scores[:, inverse]
    assert torch.allclose(clean, recovered, atol=1e-7, rtol=0.0)


def test_all_modes_have_identical_parameters_and_finite_gradients() -> None:
    counts = []
    names = []
    for mode in MODES:
        model = make_model(mode).train()
        counts.append(model.parameter_count())
        names.append(tuple(name for name, _ in model.named_parameters()))
        output = model(**batch())
        loss = output.scores.square().sum()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(value).all() for value in gradients)
    assert len(set(counts)) == 1
    assert len(set(names)) == 1


def test_masked_zscore_hand_computed() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 99.0], [4.0, 4.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    result = masked_zscore(values, mask)
    scale = torch.tensor((2.0 / 3.0) ** 0.5)
    assert torch.allclose(result[0, :3], torch.tensor([-1.0, 0.0, 1.0]) / scale)
    assert torch.equal(result[0, 3:], torch.zeros(1))
    assert torch.equal(result[1], torch.zeros(4))


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        make_model("renamed_attention")
