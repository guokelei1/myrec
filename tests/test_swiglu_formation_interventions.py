from __future__ import annotations

import pytest

from myrec.mechanism.swiglu_formation_interventions import (
    SWIGLU_MODES,
    SWIGLU_OPERATORS,
    QwenSwiGLUFormationPatch,
)


@pytest.mark.parametrize("operator", SWIGLU_OPERATORS)
def test_swiglu_identity_hook_fires_once(operator):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    positions = torch.tensor([[1, 3]])
    with torch.no_grad(), QwenSwiGLUFormationPatch(model, 0, operator, "identity") as patch:
        patch.arm(positions, sequence_length=ids.shape[1])
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        result = patch.disarm()
    assert result["fire_count"] == 1


def test_swiglu_mode_registration_is_fixed():
    assert SWIGLU_MODES == (
        "identity",
        "zero",
        "scale_half",
        "scale_double",
        "sign_flip",
        "output_norm_matched_random",
    )

