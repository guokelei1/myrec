import pytest
import torch

from myrec.mechanism.residual_addition_interventions import apply_increment_operator


def test_increment_identity_zero_and_scales():
    increment = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    assert apply_increment_operator(increment, "identity").data_ptr() == increment.data_ptr()
    torch.testing.assert_close(apply_increment_operator(increment, "zero"), torch.zeros_like(increment))
    torch.testing.assert_close(apply_increment_operator(increment, "scale_half"), increment * 0.5)
    torch.testing.assert_close(apply_increment_operator(increment, "scale_double"), increment * 2.0)
    torch.testing.assert_close(apply_increment_operator(increment, "sign_flip"), -increment)


def test_increment_random_direction_is_deterministic_and_rms_matched():
    increment = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    keys = [["a", "b"]]
    first = apply_increment_operator(increment, "output_norm_matched_random", identity_keys=keys)
    second = apply_increment_operator(increment, "output_norm_matched_random", identity_keys=keys)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        first.float().pow(2).mean(-1), increment.float().pow(2).mean(-1), atol=1e-6, rtol=1e-6
    )


def test_increment_random_requires_identity_keys_and_invalid_mode_rejected():
    increment = torch.ones(1, 1, 2)
    with pytest.raises(ValueError, match="identity keys"):
        apply_increment_operator(increment, "output_norm_matched_random")
    with pytest.raises(ValueError, match="mode"):
        apply_increment_operator(increment, "invalid")
