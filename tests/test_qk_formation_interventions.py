import pytest
import torch

from myrec.mechanism.qk_formation_interventions import (
    QK_FORMATION_MODES,
    apply_qk_formation_operator,
)


def test_qk_identity_preserves_masked_tensor_and_storage():
    logits = torch.tensor([[[[1.0, float("-inf"), 3.0], [2.0, 4.0, float("-inf")]]]])
    identity = apply_qk_formation_operator(logits, "identity")
    assert identity.data_ptr() == logits.data_ptr()
    torch.testing.assert_close(identity, logits)


@pytest.mark.parametrize("mode", ("centered_scale_half", "centered_scale_double", "sign_flip"))
def test_qk_centered_controls_preserve_mask_entries_and_row_mean(mode):
    logits = torch.tensor([[[[1.0, float("-inf"), 3.0], [2.0, 4.0, float("-inf")]]]])
    result = apply_qk_formation_operator(logits, mode)
    assert torch.isneginf(result[0, 0, 0, 1])
    assert torch.isneginf(result[0, 0, 1, 2])
    finite_before = logits[0, 0, 0, [0, 2]].mean()
    finite_after = result[0, 0, 0, [0, 2]].mean()
    torch.testing.assert_close(finite_after, finite_before)


def test_qk_random_is_deterministic_and_head_rms_matched():
    logits = torch.tensor([[[[1.0, float("-inf"), 4.0], [2.0, 5.0, 8.0]]]])
    keys = [["request/a"]]
    first = apply_qk_formation_operator(logits, "head_preserving_random", identity_keys=keys)
    second = apply_qk_formation_operator(logits, "head_preserving_random", identity_keys=keys)
    torch.testing.assert_close(first, second)
    for head in range(2):
        before = logits[0, 0, head]
        after = first[0, 0, head]
        mask = torch.isfinite(before)
        torch.testing.assert_close(
            (after[mask] - after[mask].mean()).pow(2).mean(),
            (before[mask] - before[mask].mean()).pow(2).mean(),
            atol=1e-5,
            rtol=1e-5,
        )
    assert torch.isneginf(first[0, 0, 0, 1])


def test_qk_random_requires_identity_keys_and_invalid_mode_rejected():
    logits = torch.zeros(1, 1, 1, 2)
    with pytest.raises(ValueError, match="identity keys"):
        apply_qk_formation_operator(logits, "head_preserving_random")
    with pytest.raises(ValueError, match="mode"):
        apply_qk_formation_operator(logits, "bad")


def test_modes_are_registered():
    assert set(QK_FORMATION_MODES) == {
        "identity",
        "centered_scale_half",
        "centered_scale_double",
        "sign_flip",
        "head_preserving_random",
    }
