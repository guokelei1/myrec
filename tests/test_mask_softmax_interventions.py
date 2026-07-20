import pytest
import torch

from myrec.mechanism.mask_softmax_interventions import (
    MASK_SOFTMAX_MODES,
    apply_softmax_temperature,
    apply_visibility_mask,
)


def test_visibility_mask_preserves_allowed_logits_and_masks_disallowed():
    logits = torch.tensor([[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]])
    allowed = torch.tensor([[[True, False, True], [True, True, False]]])
    masked = apply_visibility_mask(logits, allowed)
    assert torch.isfinite(masked[0, 0, :, [0, 2]]).all()
    assert torch.isneginf(masked[0, 0, :, 1]).all()
    assert torch.isneginf(masked[0, 1, :, 2]).all()


def test_visibility_mask_rejects_empty_row():
    logits = torch.zeros(1, 1, 2, 3)
    allowed = torch.zeros(1, 1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="every key"):
        apply_visibility_mask(logits, allowed)


def test_temperature_preserves_negative_infinity_and_changes_finite_logits():
    logits = torch.tensor([[[[1.0, float("-inf"), -2.0]]]])
    half = apply_softmax_temperature(logits, 0.5)
    double = apply_softmax_temperature(logits, 2.0)
    assert torch.isneginf(half[0, 0, 0, 1])
    assert torch.isneginf(double[0, 0, 0, 1])
    torch.testing.assert_close(half[0, 0, 0, 0], torch.tensor(0.5))
    torch.testing.assert_close(double[0, 0, 0, 2], torch.tensor(-4.0))


def test_temperature_requires_positive_scalar():
    with pytest.raises(ValueError, match="positive"):
        apply_softmax_temperature(torch.zeros(1, 1, 1, 2), 0.0)


def test_modes_are_registered():
    assert set(MASK_SOFTMAX_MODES) == {
        "identity",
        "prefix_history_swap",
        "candidate_visibility_swap",
        "temperature_half",
        "temperature_double",
    }
