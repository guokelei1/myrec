from __future__ import annotations

import torch

from myrec.mechanism.attention_head_observation import (
    QwenAttentionHeadObserver,
    projected_contribution_metrics,
)
from myrec.mechanism.attention_observation_runtime import covering_token_span
from myrec.mechanism.attention_observation_runtime import _summarize_path


def test_projected_contribution_uses_true_per_head_columns():
    contribution = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])
    total = contribution.clone()
    weight = torch.eye(4)
    norm, cosine = projected_contribution_metrics(contribution, total, weight)
    assert norm.shape == (1, 1, 2)
    assert torch.allclose(norm, torch.tensor([[[1.0, 2.0]]]))
    assert torch.allclose(cosine, torch.ones_like(cosine))


def test_zero_projected_contribution_has_finite_zero_cosine():
    contribution = torch.zeros(1, 1, 2, 2)
    total = torch.ones_like(contribution)
    norm, cosine = projected_contribution_metrics(contribution, total, torch.eye(4))
    assert torch.equal(norm, torch.zeros_like(norm))
    assert torch.equal(cosine, torch.zeros_like(cosine))


def test_projected_contribution_accepts_fp32_observation_with_bf16_o_proj():
    contribution = torch.tensor(
        [[[[1.0, -0.5], [0.25, 2.0]]]], dtype=torch.float32
    )
    weight = torch.eye(4, dtype=torch.bfloat16)
    norm, cosine = projected_contribution_metrics(
        contribution, contribution.clone(), weight
    )
    assert norm.dtype == torch.float32
    assert torch.allclose(
        norm, torch.tensor([[[5.0**0.5 / 2.0, 65.0**0.5 / 4.0]]])
    )
    assert torch.allclose(cosine, torch.ones_like(cosine))


def test_qwen_observer_records_fp32_manual_reconstruction():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
    )
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11]], dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)
    capture = torch.tensor([[1, 2, 5], [1, 2, 5]], dtype=torch.long)
    queries = {
        "history_summary": torch.tensor([[2], [2]], dtype=torch.long),
        "native_readout": torch.tensor([[5], [5]], dtype=torch.long),
    }
    spans = {
        "query": (
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([2, 2], dtype=torch.long),
        ),
        "history": (
            torch.tensor([2, 2], dtype=torch.long),
            torch.tensor([4, 4], dtype=torch.long),
        ),
        "candidate": (
            torch.tensor([4, 4], dtype=torch.long),
            torch.tensor([6, 6], dtype=torch.long),
        ),
    }
    with torch.inference_mode(), QwenAttentionHeadObserver(model, 0) as observer:
        observer.arm(capture, queries, spans, sequence_length=6)
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        result = observer.disarm()
    observed = result["observations"]
    assert observed["manual_reconstruction_dtype"] == "float32"
    assert observed["manual_selected_row_native_max_abs_error"] < 1.0e-5
    assert set(observed["scopes"]) == {"history_summary", "native_readout"}


def test_covering_token_span_accepts_contiguous_utf8_fallback_overlap():
    offsets = [(0, 2), (1, 2), (2, 4), (4, 5)]
    assert covering_token_span(offsets, 1, 4) == (0, 3)


def test_attention_summary_selects_registered_row_from_native_candidate_batch():
    metric = torch.stack(
        [torch.zeros(1, 16), torch.full((1, 16), 7.0)],
        dim=0,
    )
    captures = {}
    for kind in ("q", "k"):
        for stage in ("pre_norm", "post_norm", "post_rope"):
            captures[f"{kind}_{stage}"] = torch.stack(
                [torch.zeros(1, 2, 3), torch.full((1, 2, 3), 5.0)],
                dim=0,
            )
    observed = {
        "captures": captures,
        "observations": {
            "scopes": {
                "native_readout": {
                    "history": {"attention_mass": metric},
                }
            },
            "manual_selected_row_native_max_abs_error": 0.0,
            "manual_selected_row_native_low_precision_ratio": 0.0,
        },
    }
    result = _summarize_path(observed, observed, selected_batch_row=1)
    assert result["observations"]["native_readout"]["history"][
        "attention_mass"
    ]["query_head"] == [7.0] * 16
    assert torch.allclose(
        torch.tensor(result["qk_geometry"]["q"]["pre_norm"]["full_norm"]),
        torch.full((1, 2), 5.0 * (3.0**0.5)),
    )
