from __future__ import annotations

from types import SimpleNamespace

import pytest

from myrec.mechanism.attention_edge_interventions import (
    QwenAttentionEdgeIntervention,
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
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids)
    positions = torch.tensor([4, 4])
    starts = torch.tensor([1, 1])
    ends = torch.tensor([3, 3])
    return torch, model, ids, mask, positions, starts, ends


def test_zero_delta_attention_wrapper_is_exact_identity():
    torch, model, ids, mask, positions, starts, ends = _tiny_qwen()
    with torch.no_grad():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with QwenAttentionEdgeIntervention(
            model, 13, "zero_additive_delta"
        ) as intervention:
            intervention.arm(positions, starts, ends, sequence_length=ids.shape[1])
            observed = model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
            summary = intervention.disarm()
    torch.testing.assert_close(observed, baseline, rtol=0.0, atol=0.0)
    assert summary["maximum_applied_delta"] == 0.0


@pytest.mark.parametrize("mode", ["history_logits_mask", "history_value_edge_zero"])
def test_attention_edge_intervention_is_finite_and_local(mode):
    torch, model, ids, mask, positions, starts, ends = _tiny_qwen()
    with torch.no_grad(), QwenAttentionEdgeIntervention(model, 13, mode) as intervention:
        intervention.arm(positions, starts, ends, sequence_length=ids.shape[1])
        output = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        summary = intervention.disarm()
    assert torch.isfinite(output).all()
    assert summary["history_tokens"] == 4
    assert summary["maximum_applied_delta"] > 0.0
    assert summary["manual_baseline_native_max_abs_error"] < 1.0e-6


@pytest.mark.parametrize("mode", ["history_logits_mask", "history_value_edge_zero"])
def test_attention_edge_changes_only_registered_readout_query_rows(mode):
    torch, model, ids, mask, positions, starts, ends = _tiny_qwen()
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

    with torch.no_grad(), QwenAttentionEdgeIntervention(
        model, 13, mode
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

    baseline = baseline_observed["pre_o"]
    observed = intervened_observed["pre_o"]
    rows = torch.arange(ids.shape[0])
    assert not torch.equal(observed[rows, positions], baseline[rows, positions])
    untouched = torch.ones(ids.shape, dtype=torch.bool, device=observed.device)
    untouched[rows, positions] = False
    torch.testing.assert_close(
        observed[untouched], baseline[untouched], rtol=0.0, atol=0.0
    )
    assert summary["maximum_applied_delta"] > 0.0


def test_two_native_query_positions_are_supported():
    torch, model, ids, mask, positions, starts, ends = _tiny_qwen()
    positions = torch.stack((positions - 1, positions), dim=1)
    with torch.no_grad(), QwenAttentionEdgeIntervention(
        model, 27, "history_logits_mask"
    ) as intervention:
        intervention.arm(positions, starts, ends, sequence_length=ids.shape[1])
        output = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        summary = intervention.disarm()
    assert torch.isfinite(output).all()
    assert summary["query_positions_per_row"] == 2
    assert summary["manual_baseline_native_max_abs_error"] < 1.0e-6


def test_mask_then_restore_executes_mask_but_returns_native_output():
    torch, model, ids, mask, positions, starts, ends = _tiny_qwen()
    with torch.no_grad():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with QwenAttentionEdgeIntervention(
            model, 13, "mask_then_restore_output"
        ) as intervention:
            intervention.arm(positions, starts, ends, sequence_length=ids.shape[1])
            observed = model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
            summary = intervention.disarm()
    torch.testing.assert_close(observed, baseline, rtol=0.0, atol=0.0)
    assert summary["maximum_applied_delta"] > 0.0
    assert summary["maximum_returned_delta"] == 0.0


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        # Equal logits over values [1,4,10]: excluding history value 4 and
        # renormalizing leaves mean([1,10]) = 5.5.
        ("history_logits_mask", 5.5),
        # Value-edge zero keeps the original 1/3 probabilities and removes
        # only the history contribution: (1+0+10)/3 = 11/3.
        ("history_value_edge_zero", 11.0 / 3.0),
    ],
)
def test_attention_edge_modes_have_distinct_hand_computed_semantics(mode, expected):
    torch = pytest.importorskip("torch")
    query = torch.zeros((1, 1, 3, 1), dtype=torch.float32)
    key = torch.zeros_like(query)
    value = torch.tensor([[[[1.0], [4.0], [10.0]]]])
    baseline = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
    baseline[:, 2, 0, 0] = 5.0

    intervention = object.__new__(QwenAttentionEdgeIntervention)
    intervention.original_function = lambda *_args, **_kwargs: (baseline, None)
    intervention.block = 13
    intervention.mode = mode
    intervention.positions = torch.tensor([[2]])
    intervention.history_starts = torch.tensor([1])
    intervention.history_ends = torch.tensor([2])
    intervention.sequence_length = 3
    intervention.fire_count = 0
    intervention.last_summary = {}
    module = SimpleNamespace(
        layer_idx=13,
        scaling=1.0,
        num_key_value_groups=1,
    )
    observed, _weights = intervention._wrapper(
        module, query, key, value, attention_mask=None
    )
    assert observed[0, 2, 0, 0].item() == pytest.approx(expected)
    assert intervention.fire_count == 1
    assert intervention.last_summary[
        "manual_baseline_native_max_abs_error"
    ] == pytest.approx(0.0)
