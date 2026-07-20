from __future__ import annotations

import numpy as np
import pytest

from myrec.mechanism.deep_dive_representation_runtime import (
    ALL_HIDDEN_STATE_INDICES,
    AllResidualStateCapture,
    write_all_state_shard,
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
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.ones_like(input_ids)
    positions = torch.tensor([[3, 1], [2, 0]])
    return torch, model, input_ids, mask, positions


def test_all_residual_capture_has_exact_29_state_order():
    torch, model, input_ids, mask, positions = _tiny_qwen()
    with torch.no_grad(), AllResidualStateCapture(model) as capture:
        values = capture.capture(input_ids, mask, positions)
    assert values.shape == (2, 2, len(ALL_HIDDEN_STATE_INDICES), 32)
    assert capture.capture_calls == 1
    assert np.isfinite(values).all()


def test_all_state_shard_round_trip_and_identity(tmp_path):
    path = tmp_path / "shard_00000.npz"
    request = np.arange(2 * 2 * 29 * 4, dtype=np.float32).reshape(2, 2, 29, 4)
    candidate = np.arange(3 * 29 * 4, dtype=np.float32).reshape(3, 29, 4)
    row = write_all_state_shard(
        path,
        request_ids=["r0", "r1"],
        normalized_queries=["q0", "q1"],
        request_activations=request,
        candidate_offsets=[0, 1, 3],
        candidate_ids=["c0", "c1", "c2"],
        candidate_activations=candidate,
    )
    assert row["request_count"] == 2
    assert row["candidate_count"] == 3
    with np.load(path, allow_pickle=False) as payload:
        assert payload["hidden_state_indices"].tolist() == list(range(29))
        assert payload["request_activations"].dtype == np.float16
        assert payload["candidate_offsets"].tolist() == [0, 1, 3]


def test_all_state_shard_rejects_incomplete_state_axis(tmp_path):
    with pytest.raises(ValueError, match="invalid shape"):
        write_all_state_shard(
            tmp_path / "bad.npz",
            request_ids=["r"],
            normalized_queries=["q"],
            request_activations=np.zeros((1, 2, 5, 4), dtype=np.float32),
            candidate_offsets=[0, 0],
            candidate_ids=[],
            candidate_activations=np.zeros((0, 29, 4), dtype=np.float32),
        )


def test_empty_candidate_axis_is_valid_only_as_a_low_level_fixture(tmp_path):
    row = write_all_state_shard(
        tmp_path / "empty.npz",
        request_ids=["r"],
        normalized_queries=["q"],
        request_activations=np.zeros((1, 2, 29, 4), dtype=np.float32),
        candidate_offsets=[0, 0],
        candidate_ids=[],
        candidate_activations=np.zeros((0, 29, 4), dtype=np.float32),
    )
    assert row["candidate_count"] == 0
