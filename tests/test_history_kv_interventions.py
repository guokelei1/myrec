from __future__ import annotations

import pytest

from myrec.mechanism.history_kv_interventions import QwenHistoryKVIntervention


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
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids)
    starts = torch.tensor([1, 1])
    ends = torch.tensor([3, 3])
    return torch, model, ids, mask, starts, ends


def test_same_history_kv_patch_is_exact_identity():
    torch, model, ids, mask, starts, ends = _tiny_qwen()
    with torch.no_grad():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with QwenHistoryKVIntervention(model, 13) as intervention:
            intervention.arm_capture(starts, ends)
            captured_output = model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
            keys, values = intervention.disarm_capture()
            intervention.arm_patch(starts, ends, keys, values)
            patched = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            intervention.disarm_patch()
    torch.testing.assert_close(captured_output, baseline, rtol=0.0, atol=0.0)
    torch.testing.assert_close(patched, baseline, rtol=0.0, atol=0.0)


def test_zero_history_kv_donor_changes_finite_output():
    torch, model, ids, mask, starts, ends = _tiny_qwen()
    with torch.no_grad(), QwenHistoryKVIntervention(model, 27) as intervention:
        intervention.arm_capture(starts, ends)
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        keys, values = intervention.disarm_capture()
        zero_keys = [torch.zeros_like(value) for value in keys]
        zero_values = [torch.zeros_like(value) for value in values]
        intervention.arm_patch(starts, ends, zero_keys, zero_values)
        observed = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        intervention.disarm_patch()
    assert torch.isfinite(observed).all()
    assert not torch.equal(observed, baseline)
