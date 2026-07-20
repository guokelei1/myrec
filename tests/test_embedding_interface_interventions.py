import pytest
import torch

from myrec.mechanism.embedding_interface_interventions import (
    EMBEDDING_MODES,
    apply_embedding_operator,
)


def test_embedding_identity_preserves_storage_and_zero_is_exact():
    embedding = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    identity = apply_embedding_operator(embedding, "identity")
    assert identity.data_ptr() == embedding.data_ptr()
    torch.testing.assert_close(apply_embedding_operator(embedding, "zero"), torch.zeros_like(embedding))


@pytest.mark.parametrize("mode", ("scale_half", "scale_double", "sign_flip"))
def test_embedding_linear_controls(mode):
    embedding = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    result = apply_embedding_operator(embedding, mode)
    factor = {"scale_half": 0.5, "scale_double": 2.0, "sign_flip": -1.0}[mode]
    torch.testing.assert_close(result, embedding * factor)


def test_embedding_random_control_is_identity_bound():
    embedding = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    keys = [["request/a", "request/b"]]
    first = apply_embedding_operator(embedding, "output_norm_matched_random", identity_keys=keys)
    second = apply_embedding_operator(embedding, "output_norm_matched_random", identity_keys=keys)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        first.float().pow(2).mean(dim=-1),
        embedding.float().pow(2).mean(dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_embedding_random_requires_keys_and_invalid_mode_rejected():
    embedding = torch.ones(1, 1, 3)
    with pytest.raises(ValueError, match="identity keys"):
        apply_embedding_operator(embedding, "output_norm_matched_random")
    with pytest.raises(ValueError, match="mode"):
        apply_embedding_operator(embedding, "bad")


def test_embedding_modes_are_registered():
    assert set(EMBEDDING_MODES) == {
        "identity",
        "zero",
        "scale_half",
        "scale_double",
        "sign_flip",
        "output_norm_matched_random",
    }
