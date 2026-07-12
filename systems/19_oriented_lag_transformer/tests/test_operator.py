from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.olt import lag_terms, masked_softmax


def test_hand_computed_lag_terms() -> None:
    a = torch.tensor([[0.75, 0.25, 0.0]], dtype=torch.float64)
    b = torch.tensor(
        [[[0.0, 0.8, 0.2], [0.6, 0.4, 0.0]]], dtype=torch.float64
    )
    mask = torch.ones(1, 3, dtype=torch.bool)
    diagonal, forward, reverse = lag_terms(a, b, mask)
    assert torch.allclose(diagonal, torch.tensor([[0.2, 0.55]], dtype=torch.float64))
    assert torch.allclose(forward, torch.tensor([[0.65, 0.3]], dtype=torch.float64))
    assert torch.allclose(reverse, torch.tensor([[0.0, 0.15]], dtype=torch.float64))


def test_reversal_preserves_diagonal_swaps_lags_and_negates_orientation() -> None:
    generator = torch.Generator().manual_seed(19)
    a = torch.softmax(torch.randn(5, 7, generator=generator, dtype=torch.float64), dim=-1)
    b = torch.softmax(torch.randn(5, 4, 7, generator=generator, dtype=torch.float64), dim=-1)
    mask = torch.ones(5, 7, dtype=torch.bool)
    diagonal, forward, reverse = lag_terms(a, b, mask)
    changed_diagonal, changed_forward, changed_reverse = lag_terms(
        a.flip(1), b.flip(2), mask.flip(1)
    )
    assert torch.allclose(changed_diagonal, diagonal, atol=1e-12, rtol=0.0)
    assert torch.allclose(changed_forward, reverse, atol=1e-12, rtol=0.0)
    assert torch.allclose(changed_reverse, forward, atol=1e-12, rtol=0.0)
    assert torch.allclose(
        changed_forward - changed_reverse,
        -(forward - reverse),
        atol=1e-12,
        rtol=0.0,
    )


def test_self_trace_has_zero_skew_component() -> None:
    generator = torch.Generator().manual_seed(191)
    a = torch.softmax(torch.randn(9, 6, generator=generator, dtype=torch.float64), dim=-1)
    _, forward, reverse = lag_terms(a, a.unsqueeze(1), torch.ones(9, 6, dtype=torch.bool))
    assert torch.allclose(forward, reverse, atol=1e-12, rtol=0.0)


def test_masked_softmax_is_zero_for_empty_and_normalized_for_present() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    mask = torch.tensor([[True, False, True], [False, False, False]])
    values = masked_softmax(logits, mask, dim=-1)
    assert values[0, 1] == 0
    assert torch.allclose(values[0].sum(), torch.tensor(1.0))
    assert torch.equal(values[1], torch.zeros(3))
