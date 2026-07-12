from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import episode_value, memory_value, score_memory  # noqa: E402


def test_choice_gradient_is_selected_minus_query_expectation() -> None:
    query = torch.tensor([1.0, 0.0], dtype=torch.float64)
    slate = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    value = episode_value(
        query, slate, 1, mode="choice_gradient", temperature=1.0, epsilon=1e-12
    )
    weights = torch.softmax(torch.tensor([1.0, 0.0], dtype=torch.float64), dim=0)
    expected = slate[1] - weights @ slate
    expected = expected / expected.norm()
    torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)


def test_empty_memory_returns_base_exactly() -> None:
    query = torch.tensor([1.0, 0.0], dtype=torch.float64)
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    memory = memory_value(
        query,
        torch.empty(0, 2, dtype=torch.float64),
        torch.empty(0, 2, dtype=torch.float64),
        temperature=0.1,
        epsilon=1e-12,
    )
    scores, correction = score_memory(
        query, candidates, memory, correction_scale=1.0, epsilon=1e-12
    )
    torch.testing.assert_close(scores, torch.tensor([1.0, 0.0], dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(correction, torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)


def test_candidate_permutation_equivariance() -> None:
    torch.manual_seed(71)
    query = torch.randn(8, dtype=torch.float64)
    candidates = torch.randn(9, 8, dtype=torch.float64)
    memory = torch.randn(8, dtype=torch.float64)
    first, _ = score_memory(query, candidates, memory, correction_scale=1.0, epsilon=1e-12)
    reverse, _ = score_memory(query, candidates.flip(0), memory, correction_scale=1.0, epsilon=1e-12)
    torch.testing.assert_close(first, reverse.flip(0), atol=0, rtol=0)
