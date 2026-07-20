from __future__ import annotations

import pytest

from myrec.mechanism.mlp_feature_formation import (
    MLP_FEATURE_STAGES,
    QwenMLPFeatureObserver,
    summarize_mlp_feature_pair,
)


def test_mlp_feature_observer_captures_real_swiglu_identity():
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
    with torch.no_grad(), QwenMLPFeatureObserver(model, 0) as observer:
        observer.arm(positions, sequence_length=ids.shape[1])
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        result = observer.disarm()
    assert set(result["captures"]) == set(MLP_FEATURE_STAGES)
    assert result["captures"]["product"].shape == (1, 2, 64)
    assert result["product_recomposition_low_precision_ratio"] <= 1.0


def test_mlp_feature_delta_decomposition_is_hand_exact():
    torch = pytest.importorskip("torch")
    a_null = torch.arange(1, 17, dtype=torch.float32)[None, :]
    u_null = torch.arange(17, 33, dtype=torch.float32)[None, :]
    a_full = a_null + 2.0
    u_full = u_null - 3.0
    full = {
        "gate_pre": a_full - 0.5,
        "gate_activated": a_full,
        "up": u_full,
        "product": a_full * u_full,
    }
    null = {
        "gate_pre": a_null - 0.5,
        "gate_activated": a_null,
        "up": u_null,
        "product": a_null * u_null,
    }
    result = summarize_mlp_feature_pair(
        full, null, groups=(tuple(range(16)),)
    )
    assert result["groups"] == 1
    assert result["maximum_product_delta_recomposition_abs_error"] == 0.0
    assert result["maximum_actual_product_quantization_abs_error"] == 0.0
    decomposition = result["positions"][0]["groups"][0][
        "product_delta_decomposition"
    ]
    expected_gate = (2.0 * u_null).square().mean().sqrt().item()
    expected_up = (a_null * -3.0).square().mean().sqrt().item()
    expected_interaction = torch.full_like(a_null, -6.0).square().mean().sqrt().item()
    assert decomposition["gate_change_times_null_up"]["rms"] == pytest.approx(expected_gate)
    assert decomposition["null_gate_times_up_change"]["rms"] == pytest.approx(expected_up)
    assert decomposition["gate_up_interaction"]["rms"] == pytest.approx(expected_interaction)


def test_mlp_feature_pair_rejects_product_or_group_drift():
    torch = pytest.importorskip("torch")
    values = torch.ones(1, 16)
    full = {name: values.clone() for name in MLP_FEATURE_STAGES}
    null = {name: values.clone() for name in MLP_FEATURE_STAGES}
    full["product"][0, 0] = 2.0
    with pytest.raises(ValueError, match="product identity exceeds precision bound"):
        summarize_mlp_feature_pair(full, null, groups=(tuple(range(16)),))
    full["product"][0, 0] = 1.0
    with pytest.raises(ValueError, match="do not partition"):
        summarize_mlp_feature_pair(full, null, groups=((0,),))
