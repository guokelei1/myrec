from __future__ import annotations

from pathlib import Path
import sys

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.concept_attention import history_supported_concept_scores  # noqa: E402


def fixture() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(52)
    q = torch.randn(3, 8)
    qm = torch.tensor([True, True, True])
    c = torch.randn(4, 5, 8)
    cm = torch.tensor([[True, True, True, False, False]] * 4)
    h = torch.randn(3, 8)
    return q, qm, c, cm, h


def test_empty_history_is_exact_noop() -> None:
    q, qm, c, cm, _ = fixture()
    out = history_supported_concept_scores(q, qm, c, cm, torch.empty(0, 8))
    assert torch.count_nonzero(out.primary_correction) == 0
    assert torch.equal(out.base_concept_attention, out.factual_concept_attention)


def test_candidate_and_history_permutation_equivariance() -> None:
    q, qm, c, cm, h = fixture()
    out = history_supported_concept_scores(q, qm, c, cm, h)
    cp = torch.tensor([3, 1, 0, 2])
    hp = torch.tensor([2, 0, 1])
    changed_c = history_supported_concept_scores(q, qm, c[cp], cm[cp], h)
    changed_h = history_supported_concept_scores(q, qm, c, cm, h[hp])
    assert torch.allclose(out.primary_correction[cp], changed_c.primary_correction, atol=2e-6)
    assert torch.allclose(out.primary_correction, changed_h.primary_correction, atol=2e-6)


def test_primary_is_not_linearized_reduction() -> None:
    q, qm, c, cm, h = fixture()
    out = history_supported_concept_scores(q, qm, c, cm, h)
    assert not torch.allclose(out.primary_correction, out.linearized_correction)
    assert torch.isfinite(out.primary_correction).all()


def test_gradients_are_finite() -> None:
    q, qm, c, cm, h = fixture()
    q.requires_grad_(True)
    c.requires_grad_(True)
    h.requires_grad_(True)
    out = history_supported_concept_scores(q, qm, c, cm, h)
    out.primary_correction.sum().backward()
    for value in (q.grad, c.grad, h.grad):
        assert value is not None and torch.isfinite(value).all()
