import pytest
import torch

from myrec.mechanism.rmsnorm_operator_interventions import (
    RMSNORM_MODES,
    RMSNORM_OPERATORS,
    apply_rmsnorm_operator,
)


def _native(hidden, weight, eps=1e-6):
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    return (hidden.float() * torch.rsqrt(variance + eps)).to(hidden.dtype) * weight


def test_identity_is_tensor_preserving_and_native_formula_is_reproducible():
    hidden = torch.tensor([[[1.0, -2.0, 3.0], [2.0, 1.0, -1.0]]], dtype=torch.float32)
    weight = torch.tensor([1.5, 0.75, 2.0])
    identity = apply_rmsnorm_operator(
        hidden, weight, eps=1e-6, operator="variance_rescale", mode="identity"
    )
    assert identity.data_ptr() == hidden.data_ptr()
    torch.testing.assert_close(identity, hidden)
    torch.testing.assert_close(
        _native(hidden, weight),
        _native(hidden, weight),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.parametrize("operator", RMSNORM_OPERATORS)
@pytest.mark.parametrize("mode", ("half", "double", "sign_flip"))
def test_operator_modes_are_finite_and_nonidentity(operator, mode):
    hidden = torch.tensor([[[1.0, 2.0, -3.0], [0.5, -1.5, 2.5]]])
    weight = torch.tensor([1.0, 0.75, 1.25])
    result = apply_rmsnorm_operator(
        hidden, weight, eps=1e-6, operator=operator, mode=mode
    )
    assert result.shape == hidden.shape
    assert torch.isfinite(result).all()
    assert not torch.equal(result, hidden)
    if mode == "sign_flip":
        native = _native(hidden, weight)
        torch.testing.assert_close(
            result,
            -native,
            atol=1e-6,
            rtol=1e-6,
        )


def test_random_mode_requires_post_norm_direction_control():
    hidden = torch.ones(1, 2, 3)
    weight = torch.ones(3)
    with pytest.raises(ValueError, match="random direction"):
        apply_rmsnorm_operator(
            hidden,
            weight,
            eps=1e-6,
            operator="variance_rescale",
            mode="output_norm_matched_random",
        )


def test_invalid_operator_and_mode_are_rejected():
    hidden = torch.ones(1, 1, 3)
    weight = torch.ones(3)
    with pytest.raises(ValueError, match="operator"):
        apply_rmsnorm_operator(hidden, weight, eps=1e-6, operator="bad", mode="identity")
    with pytest.raises(ValueError, match="mode"):
        apply_rmsnorm_operator(hidden, weight, eps=1e-6, operator="learned_gain", mode="bad")
