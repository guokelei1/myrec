from __future__ import annotations

import numpy as np
import pytest

from myrec.mechanism.mlp_group_scoring import (
    _permutation_control,
    _residual_geometry,
    capture_mlp_paths,
    patch_mlp_paths,
)


def test_mlp_group_capture_patch_and_residual_geometry_are_mechanical():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=10_000,
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
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    path = {
        "name": "prompt",
        "target": [],
        "weight": 1.0,
        "ids": ids,
        "mask": torch.ones_like(ids),
        "positions": torch.tensor([[3], [3]]),
    }
    with torch.no_grad():
        captured = capture_mlp_paths(model, [path], block=13)
        patched = patch_mlp_paths(
            model, [path], captured["products"], block=13, group_id=0
        )
    np.testing.assert_array_equal(patched[0], captured["path_scores"][0])
    geometry = _residual_geometry(captured["nodes"])
    assert geometry["recomposition_max_abs_error"] < 1.0e-5
    permutation = _permutation_control(model, 13, captured["products"])
    assert permutation["maximum_bound_ratio"] <= 1.0
    assert permutation["recomposition_dtype"] == "float32"
    assert permutation["bound_reference_dtype"] == "float32"


def test_mlp_permutation_bound_uses_native_product_dtype():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
    )
    model = transformers.Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    products = [torch.randn(1, 1, 3072, dtype=torch.bfloat16)]
    result = _permutation_control(model, 13, products)
    assert result["recomposition_dtype"] == "float32"
    assert result["bound_reference_dtype"] == "bfloat16"
    assert result["maximum_bound_ratio"] <= 1.0
