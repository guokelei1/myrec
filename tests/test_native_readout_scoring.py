from __future__ import annotations

import pytest

from myrec.mechanism.native_readout_scoring import (
    capture_q2_native_readout,
    decompose_request_scores,
    score_q2_with_final_node_patch,
)


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
        tie_word_embeddings=True,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.ones_like(ids)
    positions = torch.tensor([[3], [3]])
    return torch, model, ids, mask, positions


def test_q2_native_score_is_exact_tied_readout_direction():
    torch, model, ids, mask, positions = _tiny_qwen()
    with torch.no_grad():
        result = capture_q2_native_readout(
            model, ids, mask, positions, yes_token_id=9, no_token_id=10
        )
    torch.testing.assert_close(
        result["algebra_score"], result["native_score"], rtol=1e-5, atol=1e-6
    )
    assert result["algebra_low_precision_max_ratio"] <= 1.0
    assert result["final_rmsnorm_input"].shape == (2, 32)
    assert result["final_rmsnorm_output"].shape == (2, 32)


@pytest.mark.parametrize("node_id", ["final_rmsnorm_input", "final_rmsnorm_output"])
def test_same_q2_final_node_patch_is_exact_identity(node_id):
    torch, model, ids, mask, positions = _tiny_qwen()
    with torch.no_grad():
        result = capture_q2_native_readout(
            model, ids, mask, positions, yes_token_id=9, no_token_id=10
        )
        observed = score_q2_with_final_node_patch(
            model,
            ids,
            mask,
            positions,
            result[node_id],
            node_id=node_id,
            yes_token_id=9,
            no_token_id=10,
        )
    torch.testing.assert_close(observed, result["native_score"], rtol=0, atol=0)


def test_request_common_relative_decomposition_is_exact():
    torch = pytest.importorskip("torch")
    result = decompose_request_scores(torch.tensor([1.0, 3.0, 5.0]))
    assert result["common"].item() == pytest.approx(3.0)
    assert result["relative"].tolist() == pytest.approx([-2.0, 0.0, 2.0])
    assert result["recomposition_max_abs_error"] == 0.0
    assert result["relative_sum_abs_error"] == 0.0
