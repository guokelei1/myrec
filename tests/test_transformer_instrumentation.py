from __future__ import annotations

import pytest

from myrec.mechanism.attention_instrumentation import (
    SCOPED_INTERFACE_KEY,
    QwenAttentionInterfaceAudit,
)
from myrec.mechanism.transformer_instrumentation import (
    BLOCK_NODE_IDS,
    FINAL_NODE_IDS,
    NodeSpec,
    QwenNodeCallAudit,
    QwenNodeCapture,
    QwenNodePatch,
    QwenPostAttentionStatePatch,
    _resolve_node_module,
    _tensor_output,
    canonical_deep_dive_specs,
    resolve_qwen_backbone,
    rms_matched_random_direction,
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
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    attention_mask = torch.ones_like(input_ids)
    positions = torch.tensor([[4, 2], [4, 1]])
    return torch, model, input_ids, attention_mask, positions


def test_node_spec_validation_and_canonical_order():
    with pytest.raises(ValueError, match="unsupported"):
        NodeSpec("not_a_node", 0)
    with pytest.raises(ValueError, match="block=None"):
        NodeSpec("final_rmsnorm_output", 27)
    with pytest.raises(ValueError, match="zero-based"):
        NodeSpec("block_output_residual", 28)
    specs = canonical_deep_dive_specs([13, 27])
    assert len(specs) == 2 * len(BLOCK_NODE_IDS) + len(FINAL_NODE_IDS)
    assert len({spec.key for spec in specs}) == len(specs)


def test_capture_exposes_every_registered_hookable_node():
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    specs = canonical_deep_dive_specs([13])
    with torch.no_grad(), QwenNodeCapture(model, specs) as capture:
        output, values = capture.capture_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            positions=positions,
            model_kwargs={"logits_to_keep": 1},
        )
    assert output.logits.shape == (2, 1, 64)
    assert set(values) == {spec.key for spec in specs}
    for spec in specs:
        value = values[spec.key]
        assert value.shape[:2] == positions.shape
        assert torch.isfinite(value).all()
    assert values[NodeSpec("mlp_swiglu_product", 13).key].shape[-1] == 64
    assert values[NodeSpec("attention_head_output_pre_o", 13).key].shape[-1] == 32


@pytest.mark.parametrize(
    "spec",
    [
        NodeSpec("block_input_residual", 13),
        NodeSpec("input_rmsnorm_output", 13),
        NodeSpec("q_post_norm_pre_rope", 13),
        NodeSpec("attention_head_output_pre_o", 13),
        NodeSpec("attention_o_projection", 13),
        NodeSpec("post_attention_rmsnorm_output", 13),
        NodeSpec("mlp_swiglu_product", 13),
        NodeSpec("mlp_down_projection", 13),
        NodeSpec("block_output_residual", 13),
        NodeSpec("final_rmsnorm_input", None),
        NodeSpec("final_rmsnorm_output", None),
    ],
)
def test_identity_patch_is_numerically_exact(spec: NodeSpec):
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    with torch.no_grad():
        baseline = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        with QwenNodeCapture(model, [spec]) as capture:
            _output, values = capture.capture_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                positions=positions,
            )
        with QwenNodePatch(model, spec) as patch:
            patch.arm(
                positions,
                values[spec.key],
                sequence_length=input_ids.shape[1],
            )
            patched = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            patch.disarm()
    torch.testing.assert_close(patched, baseline, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "spec",
    [
        NodeSpec("block_input_residual", 13),
        NodeSpec("input_rmsnorm_output", 13),
        NodeSpec("attention_o_projection", 13),
        NodeSpec("post_attention_rmsnorm_output", 13),
        NodeSpec("mlp_down_projection", 13),
        NodeSpec("block_output_residual", 13),
    ],
)
def test_selected_node_patch_hits_exact_boundary_and_only_selected_tokens(
    spec: NodeSpec,
):
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    backbone = resolve_qwen_backbone(model)
    module, hook_kind = _resolve_node_module(backbone, spec)

    def observe(target):
        if hook_kind == "input":
            def observe_input(_module, inputs):
                target["tensor"] = inputs[0].detach().clone()

            return module.register_forward_pre_hook(observe_input)

        def observe_output(_module, _inputs, output):
            target["tensor"] = _tensor_output(output, spec.key).detach().clone()

        return module.register_forward_hook(observe_output)

    baseline_observed = {}
    baseline_handle = observe(baseline_observed)
    try:
        with torch.no_grad(), QwenNodeCapture(model, [spec]) as capture:
            _output, captured = capture.capture_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                positions=positions,
            )
    finally:
        baseline_handle.remove()

    desired = captured[spec.key] + 0.25
    patched_observed = {}
    with torch.no_grad(), QwenNodePatch(model, spec) as patch:
        patched_handle = observe(patched_observed)
        try:
            patch.arm(
                positions,
                desired,
                sequence_length=input_ids.shape[1],
            )
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            patch.disarm()
        finally:
            patched_handle.remove()

    rows = torch.arange(input_ids.shape[0])[:, None]
    baseline_tensor = baseline_observed["tensor"]
    patched_tensor = patched_observed["tensor"]
    torch.testing.assert_close(
        patched_tensor[rows, positions], desired, rtol=0.0, atol=0.0
    )
    untouched = torch.ones(
        patched_tensor.shape[:2], dtype=torch.bool, device=patched_tensor.device
    )
    untouched[rows, positions] = False
    torch.testing.assert_close(
        patched_tensor[untouched],
        baseline_tensor[untouched],
        rtol=0.0,
        atol=0.0,
    )


def test_patch_changes_only_armed_forward_and_checks_fire_count():
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    spec = NodeSpec("block_output_residual", 13)
    with torch.no_grad(), QwenNodeCapture(model, [spec]) as capture:
        _output, values = capture.capture_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            positions=positions,
        )
    replacement = values[spec.key] + 1.0
    with torch.no_grad(), QwenNodePatch(model, spec) as patch:
        patch.arm(positions, replacement, sequence_length=input_ids.shape[1])
        changed = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        patch.disarm()
        with pytest.raises(RuntimeError, match="unarmed"):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    baseline = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    assert not torch.equal(changed, baseline)


def test_resolve_backbone_and_rms_matched_control():
    torch, model, _input_ids, _attention_mask, _positions = _tiny_qwen()
    backbone = resolve_qwen_backbone(model)
    assert len(backbone.layers) == 28
    values = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
            [[3.0, 1.0, 2.0], [5.0, 4.0, 1.0]],
        ]
    )
    keys = [["r0:p0", "r0:p1"], ["r1:p0", "r1:p1"]]
    left = rms_matched_random_direction(
        values, seed=20260718, identity_keys=keys
    )
    right = rms_matched_random_direction(
        values, seed=20260718, identity_keys=keys
    )
    torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        left.float().pow(2).mean(-1).sqrt(),
        values.float().pow(2).mean(-1).sqrt(),
    )


def test_node_call_audit_records_every_decoder_phase():
    torch, model, input_ids, attention_mask, _positions = _tiny_qwen()
    with torch.no_grad(), QwenNodeCallAudit(model) as audit:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        model(input_ids=input_ids[:, :2], attention_mask=attention_mask[:, :2], use_cache=False)
        result = audit.result()
    assert result["calls_per_block"] == 2
    assert result["all_blocks_identical"] is True
    assert result["block_0_input_shapes"] == [[2, 5, 32], [2, 2, 32]]


def test_generic_patch_rejects_post_attention_residual_semantics():
    _torch_module, model, _input_ids, _attention_mask, _positions = _tiny_qwen()
    with pytest.raises(ValueError, match="composition-safe"):
        with QwenNodePatch(model, NodeSpec("post_attention_residual", 13)):
            pass


def test_composition_safe_post_attention_state_patch_is_exact():
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    spec = NodeSpec("post_attention_residual", 13)
    with torch.no_grad(), QwenNodeCapture(model, [spec]) as capture:
        baseline, values = capture.capture_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            positions=positions,
        )
    with torch.no_grad(), QwenPostAttentionStatePatch(model, 13) as patch:
        patch.arm(
            positions,
            values[spec.key],
            sequence_length=input_ids.shape[1],
        )
        patched = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        patch.disarm()
    torch.testing.assert_close(patched.logits, baseline.logits, rtol=0.0, atol=0.0)


def test_composition_safe_post_attention_state_patch_changes_to_exact_target():
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    spec = NodeSpec("post_attention_residual", 13)
    layer = resolve_qwen_backbone(model).layers[13]
    rows = torch.arange(input_ids.shape[0])[:, None]
    baseline_observed = {}

    def observe_baseline_norm_input(_module, inputs):
        baseline_observed["post_attention_state_full"] = (
            inputs[0].detach().clone()
        )

    def observe_baseline_block_output(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        baseline_observed["block_output_full"] = tensor.detach().clone()

    baseline_handles = [
        layer.post_attention_layernorm.register_forward_pre_hook(
            observe_baseline_norm_input
        ),
        layer.register_forward_hook(observe_baseline_block_output),
    ]
    try:
        with torch.no_grad(), QwenNodeCapture(model, [spec]) as capture:
            _baseline, values = capture.capture_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                positions=positions,
            )
    finally:
        for handle in baseline_handles:
            handle.remove()

    desired = values[spec.key] + 0.25
    observed = {}

    def observe_norm_input(_module, inputs):
        observed["post_attention_state_full"] = inputs[0].detach().clone()
        observed["post_attention_state"] = observed[
            "post_attention_state_full"
        ][rows, positions]

    def observe_mlp_increment(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        observed["mlp_increment"] = tensor[rows, positions].detach().clone()

    def observe_block_output(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        observed["block_output_full"] = tensor.detach().clone()
        observed["block_output"] = observed["block_output_full"][rows, positions]

    with torch.no_grad(), QwenPostAttentionStatePatch(model, 13) as patch:
        handles = [
            layer.post_attention_layernorm.register_forward_pre_hook(
                observe_norm_input
            ),
            layer.mlp.down_proj.register_forward_hook(observe_mlp_increment),
            layer.register_forward_hook(observe_block_output),
        ]
        try:
            patch.arm(positions, desired, sequence_length=input_ids.shape[1])
            changed = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
            patch.disarm()
        finally:
            for handle in handles:
                handle.remove()
    torch.testing.assert_close(
        observed["post_attention_state"], desired, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        observed["block_output"],
        desired + observed["mlp_increment"],
        rtol=0.0,
        atol=0.0,
    )
    untouched = torch.ones(
        input_ids.shape, dtype=torch.bool, device=observed["block_output"].device
    )
    untouched[rows, positions] = False
    for key in ("post_attention_state_full", "block_output_full"):
        torch.testing.assert_close(
            observed[key][untouched],
            baseline_observed[key][untouched],
            rtol=0.0,
            atol=0.0,
        )
    assert torch.isfinite(changed.logits).all()
    assert not torch.equal(changed.logits, _baseline.logits)


def test_random_control_is_batch_and_resume_stable():
    torch = pytest.importorskip("torch")
    values = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6) + 1.0
    keys = [["a", "b"], ["c", "d"]]
    together = rms_matched_random_direction(
        values, seed=7, identity_keys=keys
    )
    separate = torch.cat(
        [
            rms_matched_random_direction(
                values[index : index + 1],
                seed=7,
                identity_keys=[keys[index]],
            )
            for index in range(2)
        ],
        dim=0,
    )
    torch.testing.assert_close(together, separate, rtol=0.0, atol=0.0)


def test_scoped_attention_wrapper_is_noop_and_restores_backend():
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    backbone = resolve_qwen_backbone(model)
    original = backbone.layers[0].self_attn.config._attn_implementation
    assert SCOPED_INTERFACE_KEY not in ALL_ATTENTION_FUNCTIONS
    with torch.no_grad():
        baseline = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        with QwenAttentionInterfaceAudit(model) as audit:
            audit.arm(positions, sequence_length=input_ids.shape[1])
            wrapped = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            captured = audit.disarm()
            assert backbone.layers[0].self_attn.config._attn_implementation == original
        assert backbone.layers[0].self_attn.config._attn_implementation == original
    assert SCOPED_INTERFACE_KEY not in ALL_ATTENTION_FUNCTIONS
    torch.testing.assert_close(wrapped, baseline, rtol=0.0, atol=0.0)
    assert set(captured) == set(range(28))
    assert captured[13]["post_rope_query"].shape == (2, 2, 4, 8)
    assert captured[13]["post_rope_key"].shape == (2, 2, 2, 8)


def test_scoped_attention_wrapper_audits_multiple_native_calls():
    torch, model, input_ids, attention_mask, _positions = _tiny_qwen()
    with torch.no_grad():
        baseline = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        with QwenAttentionInterfaceAudit(model) as audit:
            audit.arm_all_calls()
            first = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            second = model(
                input_ids=input_ids[:, :2],
                attention_mask=attention_mask[:, :2],
                use_cache=False,
            ).logits
            calls = audit.disarm_all_calls()
    torch.testing.assert_close(first, baseline, rtol=0.0, atol=0.0)
    assert second.shape[1] == 2
    assert calls["calls_per_block"] == 2
    assert calls["all_blocks_identical"] is True
    assert calls["block_0_shapes"][0]["query"] == [2, 4, 5, 8]


def test_post_rope_preserves_qk_norm_at_selected_positions():
    torch, model, input_ids, attention_mask, positions = _tiny_qwen()
    specs = [
        NodeSpec("q_post_norm_pre_rope", 13),
        NodeSpec("k_post_norm_pre_rope", 13),
    ]
    with torch.no_grad(), QwenNodeCapture(model, specs) as capture, QwenAttentionInterfaceAudit(model) as audit:
        capture.arm(positions, sequence_length=input_ids.shape[1])
        audit.arm(positions, sequence_length=input_ids.shape[1])
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        pre = capture.disarm()
        post = audit.disarm()
    torch.testing.assert_close(
        pre[specs[0].key].float().pow(2).sum(-1).sqrt(),
        post[13]["post_rope_query"].float().pow(2).sum(-1).sqrt(),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        pre[specs[1].key].float().pow(2).sum(-1).sqrt(),
        post[13]["post_rope_key"].float().pow(2).sum(-1).sqrt(),
        rtol=2e-5,
        atol=2e-5,
    )
