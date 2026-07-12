from pathlib import Path
import sys

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from probe.run_signal_gate import probability_residual, standardize_base  # noqa: E402


def test_standardize_base_is_affine_invariant() -> None:
    mask = torch.tensor([[True, True, True, False]])
    base = torch.tensor([[1.0, 2.0, 4.0, 99.0]])
    changed = 3.0 * base + 7.0
    assert torch.allclose(standardize_base(base, mask), standardize_base(changed, mask), atol=1e-6)


def test_probability_residual_is_zero_sum_and_hand_computed() -> None:
    mask = torch.tensor([[True, True]])
    base = torch.zeros(1, 2)
    labels = torch.tensor([[1.0, 0.0]])
    target = probability_residual(base, labels, mask)
    assert torch.allclose(target, torch.tensor([[0.5, -0.5]]))
    assert torch.equal(target.sum(dim=-1), torch.zeros(1))


def test_padding_is_zero() -> None:
    mask = torch.tensor([[True, True, False]])
    target = probability_residual(
        torch.tensor([[0.0, 0.0, 100.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]), mask,
    )
    assert target[0, 2] == 0
