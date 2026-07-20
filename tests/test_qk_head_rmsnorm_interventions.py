from __future__ import annotations

import pytest

from myrec.mechanism.qk_head_rmsnorm_interventions import QwenQKHeadRMSNormPatch


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
    positions = torch.tensor([[4], [4]])
    starts = torch.tensor([0, 0])
    ends = torch.tensor([3, 2])
    return torch, model, input_ids, attention_mask, positions, starts, ends


@pytest.mark.parametrize("component", ["q", "k"])
def test_qk_head_norm_identity_is_exact(component):
    torch, model, input_ids, attention_mask, positions, starts, ends = _tiny_qwen()
    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        with QwenQKHeadRMSNormPatch(model, 13, component, "identity") as patch:
            patch.arm(positions, starts, ends, sequence_length=input_ids.shape[1])
            patched = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            summary = patch.disarm()
    torch.testing.assert_close(patched, baseline, rtol=0.0, atol=0.0)
    assert summary["maximum_applied_delta"] == 0.0


def test_qk_head_norm_variance_operator_changes_registered_rows():
    torch, model, input_ids, attention_mask, positions, starts, ends = _tiny_qwen()
    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        with QwenQKHeadRMSNormPatch(model, 13, "q", "variance_scale_double") as patch:
            patch.arm(positions, starts, ends, sequence_length=input_ids.shape[1])
            changed = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            summary = patch.disarm()
    assert not torch.equal(changed, baseline)
    assert summary["selected_positions"] == positions.numel()
    assert summary["maximum_applied_delta"] > 0.0
