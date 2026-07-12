from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.contest_margin import MODES, EvidenceContestMarginTransformer


def model(mode: str = "evidence_contest") -> EvidenceContestMarginTransformer:
    torch.manual_seed(17)
    return EvidenceContestMarginTransformer(
        embedding_weight=torch.randn(100, 16),
        padding_idx=0,
        input_dim=16,
        hidden_dim=8,
        heads=2,
        token_layers=1,
        history_layers=1,
        ffn_dim=16,
        dropout=0.0,
        max_query_tokens=5,
        max_item_tokens=6,
        max_history=3,
        pair_delta_max=2.0,
        additive_delta_max=0.75,
        mode=mode,
    ).eval()


def masks(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    attention = ids.ne(0)
    content = attention & ids.ne(1) & ids.ne(2)
    return attention, content


def batch() -> dict[str, torch.Tensor]:
    query = torch.tensor([[1, 3, 4, 2, 0], [1, 5, 2, 0, 0], [1, 6, 2, 0, 0]])
    candidates = torch.tensor(
        [
            [[1, 3, 7, 2, 0, 0], [1, 8, 4, 2, 0, 0], [1, 9, 2, 0, 0, 0]],
            [[1, 5, 10, 2, 0, 0], [1, 11, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[1, 6, 12, 2, 0, 0], [1, 13, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        ]
    )
    history = torch.tensor(
        [
            [[1, 3, 14, 2, 0, 0], [1, 4, 15, 2, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[1, 5, 16, 2, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        ]
    )
    qa, qc = masks(query)
    ca, cc = masks(candidates)
    ha, hc = masks(history)
    candidate_mask = ca.any(-1)
    history_mask = ha.any(-1)
    repeat = torch.zeros(3, 3, 3, dtype=torch.bool)
    repeat[1, 0, 0] = True
    return {
        "query_ids": query,
        "query_attention_mask": qa,
        "query_content_mask": qc,
        "candidate_token_ids": candidates,
        "candidate_attention_mask": ca,
        "candidate_content_mask": cc,
        "history_token_ids": history,
        "history_attention_mask": ha,
        "history_content_mask": hc,
        "candidate_mask": candidate_mask,
        "history_mask": history_mask,
        "repeat_mask": repeat,
        "event_weights": torch.tensor([[1.0, 1.5, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "base_scores": torch.tensor([[1.0, 0.1, -0.8], [0.4, -0.3, 0.0], [0.7, -0.2, 0.0]]),
        "item_only_scores": torch.randn(3, 3),
    }


def test_structural_fallbacks_and_pair_contracts() -> None:
    values = batch()
    output = model()(**values)
    assert output.active_request.tolist() == [True, False, False]
    assert torch.equal(output.scores[1, :2], values["item_only_scores"][1, :2])
    assert torch.equal(output.scores[2, :2], values["base_scores"][2, :2])
    assert float(output.pair_complement_error) < 1e-6
    assert float(output.pair_diagonal_error) == 0.0
    assert torch.allclose(output.pair_delta, -output.pair_delta.transpose(1, 2), atol=1e-7)


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
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    changed = dict(values)
    for name in (
        "candidate_token_ids",
        "candidate_attention_mask",
        "candidate_content_mask",
        "candidate_mask",
        "repeat_mask",
        "base_scores",
        "item_only_scores",
    ):
        changed[name] = values[name][:, permutation]
    recovered = candidate(**changed).scores[:, inverse]
    assert torch.allclose(clean, recovered, atol=1e-6, rtol=0.0)


def test_primary_uses_history_and_candidate_control_does_not() -> None:
    values = batch()
    changed = dict(values)
    changed["history_token_ids"] = values["history_token_ids"].clone()
    changed["history_token_ids"][0, 0, 1:3] = torch.tensor([30, 31])
    primary = model("evidence_contest")
    assert not torch.equal(primary(**values).scores[0], primary(**changed).scores[0])
    control = model("candidate_contest")
    assert torch.allclose(control(**values).scores, control(**changed).scores, atol=1e-7, rtol=0.0)


def test_neutral_contest_preserves_base_order() -> None:
    candidate = model()
    base = torch.tensor([[1.2, -0.4, 0.3, 0.8]])
    mask = torch.ones_like(base, dtype=torch.bool)
    scores, complement, diagonal = candidate.contest_scores(
        base, torch.zeros(1, 4, 4), mask
    )
    assert torch.equal(scores, base)
    assert float(complement) < 1e-7 and float(diagonal) == 0.0


def test_modes_match_parameters_and_have_finite_gradients() -> None:
    counts, names = [], []
    for mode in MODES:
        candidate = model(mode).train()
        counts.append(candidate.parameter_count(trainable_only=True))
        names.append(tuple(name for name, _ in candidate.named_parameters()))
        loss = candidate(**batch()).scores.square().sum()
        loss.backward()
        gradients = [p.grad for p in candidate.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert len(set(counts)) == 1
    assert len(set(names)) == 1


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        model("history_margin_temperature_tuned_on_dev")
