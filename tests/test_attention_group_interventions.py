from __future__ import annotations

import pytest

from myrec.mechanism.attention_group_interventions import (
    QwenAttentionGQAIntervention,
    QwenHistoryKVGroupIntervention,
)
from myrec.mechanism.attention_group_scoring import (
    _neutral_history_paths,
    _selected_cross_donors,
)
from myrec.mechanism.deep_dive_assignments import CONTENT_NEUTRAL_TOKEN_ID
from myrec.mechanism.history_kv_interventions import QwenHistoryKVIntervention


def _tiny_frozen_topology_qwen():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.ones_like(ids)
    positions = torch.tensor([[4]])
    starts = torch.tensor([1])
    ends = torch.tensor([3])
    return torch, model, ids, mask, positions, starts, ends


@pytest.mark.parametrize("mode", ["history_logits_mask", "history_value_edge_zero"])
def test_one_gqa_group_attention_intervention_is_finite(mode):
    torch, model, ids, mask, positions, starts, ends = _tiny_frozen_topology_qwen()
    with torch.no_grad(), QwenAttentionGQAIntervention(
        model, 13, mode, 3
    ) as intervention:
        intervention.arm(positions, starts, ends, sequence_length=ids.shape[1])
        output = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        summary = intervention.disarm()
    assert torch.isfinite(output).all()
    assert summary["gqa_group"] == 3
    assert summary["query_heads"] == [6, 7]
    assert summary["maximum_applied_delta"] > 0
    assert summary["manual_baseline_native_max_abs_error"] < 1.0e-6


@pytest.mark.parametrize("mode", ["history_logits_mask", "history_value_edge_zero"])
def test_gqa_intervention_changes_only_registered_group_and_query_rows(mode):
    torch, model, ids, mask, positions, starts, ends = _tiny_frozen_topology_qwen()
    o_proj = model.model.layers[13].self_attn.o_proj
    baseline_observed = {}

    def observe_baseline(_module, inputs):
        baseline_observed["pre_o"] = inputs[0].detach().clone()

    baseline_handle = o_proj.register_forward_pre_hook(observe_baseline)
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        baseline_handle.remove()

    intervened_observed = {}

    def observe_intervened(_module, inputs):
        intervened_observed["pre_o"] = inputs[0].detach().clone()

    with torch.no_grad(), QwenAttentionGQAIntervention(
        model, 13, mode, 3
    ) as intervention:
        intervened_handle = o_proj.register_forward_pre_hook(observe_intervened)
        try:
            intervention.arm(
                positions, starts, ends, sequence_length=ids.shape[1]
            )
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            summary = intervention.disarm()
        finally:
            intervened_handle.remove()

    baseline = baseline_observed["pre_o"].reshape(1, ids.shape[1], 16, 2)
    observed = intervened_observed["pre_o"].reshape(1, ids.shape[1], 16, 2)
    rows = torch.arange(ids.shape[0])[:, None]
    query_heads = torch.tensor(summary["query_heads"], device=observed.device)
    other_heads = torch.tensor(
        sorted(set(range(16)) - set(summary["query_heads"])),
        device=observed.device,
    )
    assert not torch.equal(
        observed[rows, positions][:, :, query_heads],
        baseline[rows, positions][:, :, query_heads],
    )
    torch.testing.assert_close(
        observed[rows, positions][:, :, other_heads],
        baseline[rows, positions][:, :, other_heads],
        rtol=0.0,
        atol=0.0,
    )
    unselected = torch.ones(ids.shape, dtype=torch.bool, device=observed.device)
    unselected[rows, positions] = False
    torch.testing.assert_close(
        observed[unselected], baseline[unselected], rtol=0.0, atol=0.0
    )


def test_one_gqa_group_restore_is_exact_score_identity():
    torch, model, ids, mask, positions, starts, ends = _tiny_frozen_topology_qwen()
    with torch.no_grad():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with QwenAttentionGQAIntervention(
            model, 13, "mask_then_restore_output", 5
        ) as intervention:
            intervention.arm(positions, starts, ends, sequence_length=ids.shape[1])
            observed = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            summary = intervention.disarm()
    torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
    assert summary["maximum_returned_delta"] == 0.0


def test_one_history_kv_group_same_patch_is_exact_identity():
    torch, model, ids, mask, _positions, starts, ends = _tiny_frozen_topology_qwen()
    with torch.no_grad():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with QwenHistoryKVIntervention(model, 13) as capture:
            capture.arm_capture(starts, ends)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            donors = capture.disarm_capture()
        with QwenHistoryKVGroupIntervention(model, 13, 4) as patch:
            patch.arm_patch(starts, ends, *donors)
            observed = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            patch.disarm_patch()
    torch.testing.assert_close(observed, baseline, rtol=0, atol=0)


def test_neutral_history_uses_each_native_batch_rows_padded_span():
    torch = pytest.importorskip("torch")
    ids = torch.arange(16).reshape(2, 8)
    paths = [
        {
            "name": "prompt",
            "selected_batch_row": 1,
            "full": {
                "ids": ids,
                "spans": {
                    "history": (torch.tensor([2, 3]), torch.tensor([4, 5]))
                },
                "left_padding": torch.tensor([1, 2]),
            },
        }
    ]
    result = _neutral_history_paths(
        paths,
        {
            "history_span_start": 1,
            "history_span_end_exclusive": 3,
            "history_span_tokens": 2,
        },
    )
    observed = result[0]["full"]["ids"]
    assert result[0]["selected_batch_row"] == 1
    assert torch.equal(
        observed[0, 2:4],
        torch.full((2,), CONTENT_NEUTRAL_TOKEN_ID),
    )
    assert torch.equal(
        observed[1, 3:5],
        torch.full((2,), CONTENT_NEUTRAL_TOKEN_ID),
    )
    assert observed[0, 1] == ids[0, 1]
    assert observed[1, 5] == ids[1, 5]


def test_cross_summary_kv_replaces_only_selected_recipient_batch_row():
    torch = pytest.importorskip("torch")
    recipient_keys = tuple(torch.full((1,), float(index)) for index in range(2))
    recipient_values = tuple(
        torch.full((1,), float(index + 10)) for index in range(2)
    )
    donor_keys = tuple(torch.full((1,), float(index + 20)) for index in range(3))
    donor_values = tuple(torch.full((1,), float(index + 30)) for index in range(3))
    result = _selected_cross_donors(
        [(recipient_keys, recipient_values)],
        [(donor_keys, donor_values)],
        recipient_row=1,
        donor_row=2,
    )
    keys, values = result[0]
    assert torch.equal(keys[0], recipient_keys[0])
    assert torch.equal(values[0], recipient_values[0])
    assert torch.equal(keys[1], donor_keys[2])
    assert torch.equal(values[1], donor_values[2])
