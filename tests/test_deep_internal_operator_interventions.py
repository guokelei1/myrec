from __future__ import annotations

import pytest

from myrec.mechanism.residual_composition_interventions import (
    QwenResidualCompositionPatch,
)
from myrec.mechanism.rmsnorm_interventions import QwenRMSNormPatch


def _tiny_qwen():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    attention_mask = torch.ones_like(input_ids)
    positions = torch.tensor([[4, 2], [4, 1]])
    return torch, model, input_ids, attention_mask, positions


@pytest.mark.parametrize("branch", ["attention", "mlp"])
def test_residual_identity_is_exact(branch):
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    with torch.no_grad():
        baseline = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        with QwenResidualCompositionPatch(model, 13, branch, "identity") as patch:
            patch.arm(positions, sequence_length=input_ids.shape[1])
            patched = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            summary = patch.disarm()
    torch.testing.assert_close(patched, baseline, rtol=0.0, atol=0.0)
    assert summary["maximum_applied_delta"] == 0.0


def test_residual_operator_changes_registered_branch():
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    with torch.no_grad():
        baseline = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        with QwenResidualCompositionPatch(model, 13, "attention", "zero") as patch:
            patch.arm(positions, sequence_length=input_ids.shape[1])
            changed = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            summary = patch.disarm()
    assert not torch.equal(changed, baseline)
    assert summary["selected_positions"] == positions.numel()


@pytest.mark.parametrize("scope,block", [("input", 13), ("post_attention", 13), ("final", None)])
def test_rmsnorm_identity_is_exact(scope, block):
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    with torch.no_grad():
        baseline = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        with QwenRMSNormPatch(model, scope, block=block, mode="identity") as patch:
            patch.arm(positions, sequence_length=input_ids.shape[1])
            patched = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            summary = patch.disarm()
    torch.testing.assert_close(patched, baseline, rtol=0.0, atol=0.0)
    assert summary["maximum_applied_delta"] == 0.0


def test_rmsnorm_gain_operator_changes_registered_rows():
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    with torch.no_grad():
        baseline = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        with QwenRMSNormPatch(
            model, "post_attention", block=13, mode="gain_sign_flip"
        ) as patch:
            patch.arm(positions, sequence_length=input_ids.shape[1])
            changed = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            summary = patch.disarm()
    assert not torch.equal(changed, baseline)
    assert summary["maximum_applied_delta"] > 0.0


def test_operator_validation_rejects_invalid_scope_and_branch():
    torch, model, _input_ids, _mask, _positions = _tiny_qwen()
    with pytest.raises(ValueError, match="unsupported residual branch"):
        QwenResidualCompositionPatch(model, 13, "rmsnorm", "identity")
    with pytest.raises(ValueError, match="final RMSNorm does not take a block"):
        QwenRMSNormPatch(model, "final", block=13, mode="identity")
