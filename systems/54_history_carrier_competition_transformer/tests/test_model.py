from pathlib import Path
import sys

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.history_carrier import HistoryCarrierCompetitionTransformer  # noqa: E402


def model() -> HistoryCarrierCompetitionTransformer:
    torch.manual_seed(540)
    return HistoryCarrierCompetitionTransformer(
        input_dim=8, hidden_dim=16, heads=4, ffn_dim=32,
        dropout=0.0, max_history=4,
    ).eval()


def inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(541)
    return {
        "query": torch.randn(2, 8),
        "history": torch.randn(2, 3, 8),
        "history_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "candidates": torch.randn(2, 5, 8),
        "candidate_mask": torch.tensor([[True] * 4 + [False], [True] * 5]),
        "base_scores": torch.randn(2, 5),
    }


def test_nohistory_is_exact_base_and_zero_carrier() -> None:
    values = inputs(); values["history_mask"] = torch.zeros_like(values["history_mask"])
    out = model()(**values)
    expected = values["base_scores"].masked_fill(~values["candidate_mask"], 0.0)
    assert torch.equal(out.scores, expected)
    assert torch.count_nonzero(out.correction) == 0
    assert torch.count_nonzero(out.carrier) == 0
    assert torch.count_nonzero(out.list_message) == 0


def test_candidate_permutation_equivariance() -> None:
    values = inputs(); ranker = model(); out = ranker(**values)
    permutation = torch.tensor([4, 2, 0, 3, 1]); changed = dict(values)
    for name in ("candidates", "candidate_mask", "base_scores"):
        changed[name] = changed[name][:, permutation]
    permuted = ranker(**changed)
    assert torch.allclose(out.scores[:, permutation], permuted.scores, atol=2e-6)


def test_cross_candidate_carrier_is_functional() -> None:
    values = inputs(); ranker = model()
    primary = ranker(**values, mode="history_carrier")
    independent = ranker(**values, mode="independent_carrier")
    assert torch.isfinite(primary.scores).all()
    assert torch.isfinite(independent.scores).all()
    assert not torch.allclose(primary.scores, independent.scores)


def test_history_changes_carrier_but_not_raw_candidate_control() -> None:
    values = inputs(); ranker = model(); changed = dict(values)
    changed["history"] = values["history"].roll(1, dims=0)
    primary = ranker(**values, mode="history_carrier")
    other = ranker(**changed, mode="history_carrier")
    raw = ranker(**values, mode="raw_candidate")
    raw_other = ranker(**changed, mode="raw_candidate")
    assert not torch.allclose(primary.correction, other.correction)
    assert torch.equal(raw.correction, raw_other.correction)


def test_distractor_changes_primary_not_independent_target_raw_score() -> None:
    values = {name: value[:1].clone() for name, value in inputs().items()}
    values["candidate_mask"][:] = True
    ranker = model(); changed = dict(values)
    changed["candidates"] = values["candidates"].clone()
    changed["candidates"][:, 1] += 3.0
    primary = ranker(**values, mode="history_carrier")
    primary_changed = ranker(**changed, mode="history_carrier")
    independent = ranker(**values, mode="independent_carrier")
    independent_changed = ranker(**changed, mode="independent_carrier")
    assert not torch.allclose(primary.raw_correction[:, 0], primary_changed.raw_correction[:, 0])
    assert torch.allclose(independent.raw_correction[:, 0], independent_changed.raw_correction[:, 0], atol=2e-6)


def test_gradients_are_finite_and_active() -> None:
    values = inputs(); ranker = model().train()
    out = ranker(**values)
    loss = (out.scores * torch.arange(5, dtype=out.scores.dtype)).sum()
    loss.backward()
    gradients = [value.grad for value in ranker.parameters() if value.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value) for value in gradients)
