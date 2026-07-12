from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "src"))

from train.run_train_gate import ndcg_rows, uniform_correction  # noqa: E402


def test_uniform_control_fallback_and_permutation() -> None:
    generator = torch.Generator().manual_seed(18)
    query = torch.randn(16, generator=generator)
    history = torch.randn(5, 16, generator=generator)
    candidates = torch.randn(7, 16, generator=generator)
    config = {"model": {"profile_scale": 1.0, "correction_scale": 2.0}}
    assert torch.equal(
        uniform_correction(query, history[:0], candidates, config),
        torch.zeros(7),
    )
    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    expected = uniform_correction(query, history, candidates, config)[permutation]
    actual = uniform_correction(query, history, candidates[permutation], config)
    assert torch.allclose(actual, expected, atol=2e-7, rtol=0)


def test_shared_ndcg_hand_computed_top_positive() -> None:
    value = ndcg_rows(
        ["r"],
        [["a", "b", "c"]],
        [np.asarray([3.0, 2.0, 1.0])],
        [np.asarray([1.0, 0.0, 0.0])],
    )
    assert np.array_equal(value, np.asarray([1.0]))
