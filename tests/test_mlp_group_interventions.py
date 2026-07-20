from __future__ import annotations

import pytest

from myrec.mechanism.mlp_group_interventions import (
    MLP_GROUPS,
    QwenMLPGroupCapture,
    QwenMLPGroupPatch,
    exact_permutation_recomposition,
    frozen_mlp_groups,
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
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.ones_like(ids)
    positions = torch.tensor([[3], [3]])
    return torch, model, ids, mask, positions


def test_frozen_mlp_groups_are_balanced_partition():
    groups = frozen_mlp_groups(3072)
    assert len(groups) == MLP_GROUPS
    assert {len(value) for value in groups} == {192}
    assert sorted(index for value in groups for index in value) == list(range(3072))


def test_same_group_patch_is_exact_identity():
    torch, model, ids, mask, positions = _tiny_qwen()
    with torch.no_grad():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with QwenMLPGroupCapture(model, 13) as capture:
            capture.arm(positions)
            captured_output = model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
            donor = capture.disarm()
        with QwenMLPGroupPatch(model, 13, [0, 5]) as patch:
            patch.arm(positions, donor)
            observed = model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
            patch.disarm()
    torch.testing.assert_close(captured_output, baseline, rtol=0.0, atol=0.0)
    torch.testing.assert_close(observed, baseline, rtol=0.0, atol=0.0)


def test_group_patch_changes_only_registered_groups_at_selected_tokens():
    torch, model, ids, mask, positions = _tiny_qwen()
    layer = model.model.layers[13]
    baseline_observed = {}

    def observe_baseline(_module, inputs):
        baseline_observed["product"] = inputs[0].detach().clone()

    with torch.no_grad(), QwenMLPGroupCapture(model, 13) as capture:
        baseline_handle = layer.mlp.down_proj.register_forward_pre_hook(
            observe_baseline
        )
        try:
            capture.arm(positions)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            captured = capture.disarm()
        finally:
            baseline_handle.remove()

    donor = captured + 0.25
    patched_observed = {}

    def observe_patched(_module, inputs):
        patched_observed["product"] = inputs[0].detach().clone()

    with torch.no_grad(), QwenMLPGroupPatch(model, 13, [0, 5]) as patch:
        patched_handle = layer.mlp.down_proj.register_forward_pre_hook(
            observe_patched
        )
        try:
            patch.arm(positions, donor)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            patch.disarm()
        finally:
            patched_handle.remove()

    baseline = baseline_observed["product"]
    observed = patched_observed["product"]
    rows = torch.arange(ids.shape[0])[:, None]
    group_indices = patch.indices.to(observed.device)
    other_indices = torch.tensor(
        sorted(set(range(observed.shape[-1])) - set(group_indices.tolist())),
        dtype=torch.long,
        device=observed.device,
    )
    torch.testing.assert_close(
        observed[rows, positions][:, :, group_indices],
        donor[:, :, group_indices],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        observed[rows, positions][:, :, other_indices],
        baseline[rows, positions][:, :, other_indices],
        rtol=0.0,
        atol=0.0,
    )
    unselected = torch.ones(ids.shape, dtype=torch.bool, device=observed.device)
    unselected[rows, positions] = False
    torch.testing.assert_close(
        observed[unselected], baseline[unselected], rtol=0.0, atol=0.0
    )


def test_permutation_and_inverse_columns_recompose_exactly():
    torch = pytest.importorskip("torch")
    product = torch.randn(2, 3, 64)
    weight = torch.randn(32, 64)
    permutation = torch.randperm(64)
    original, recomposed, _ = exact_permutation_recomposition(
        product, weight, permutation
    )
    torch.testing.assert_close(recomposed, original, rtol=1.0e-5, atol=1.0e-5)
