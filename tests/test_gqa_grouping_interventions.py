import pytest
import torch

from myrec.mechanism.gqa_grouping_interventions import (
    FROZEN_KV_HEADS,
    FROZEN_QUERY_HEADS,
    cyclic_group_permutation_mapping,
    grouping_control,
    native_query_to_kv_mapping,
    remap_kv_for_query_heads,
    within_group_rotation_mapping,
)


def test_native_and_fixed_alternative_mappings_are_complete():
    native = native_query_to_kv_mapping()
    cyclic = cyclic_group_permutation_mapping()
    rotated = within_group_rotation_mapping()
    assert len(native) == len(cyclic) == len(rotated) == FROZEN_QUERY_HEADS
    assert set(native) == set(range(FROZEN_KV_HEADS))
    assert all(0 <= value < FROZEN_KV_HEADS for value in cyclic + rotated)
    assert cyclic != native
    assert rotated != native


def test_remap_preserves_inputs_and_expands_exact_head_values():
    key = torch.arange(1 * 8 * 3 * 2, dtype=torch.float32).reshape(1, 8, 3, 2)
    value = key + 100.0
    key_before, value_before = key.clone(), value.clone()
    remapped_key, remapped_value = remap_kv_for_query_heads(
        key,
        value,
        native_query_to_kv_mapping(),
        query_heads=16,
        kv_heads=8,
    )
    assert remapped_key.shape == (1, 16, 3, 2)
    torch.testing.assert_close(key, key_before)
    torch.testing.assert_close(value, value_before)
    for query_head, kv_head in enumerate(native_query_to_kv_mapping()):
        torch.testing.assert_close(remapped_key[:, query_head], key[:, kv_head])
        torch.testing.assert_close(remapped_value[:, query_head], value[:, kv_head])


@pytest.mark.parametrize("mode", ("identity", "group_permutation", "group_rotation"))
def test_grouping_control_reports_auditable_contract(mode):
    key = torch.randn(2, 8, 4, 3)
    value = torch.randn(2, 8, 4, 3)
    remapped_key, remapped_value, summary = grouping_control(key, value, mode=mode)
    assert remapped_key.shape[1] == FROZEN_QUERY_HEADS
    assert remapped_value.shape == remapped_key.shape
    assert summary["input_key_unchanged"] is True
    assert summary["input_value_unchanged"] is True
    assert len(summary["mapping"]) == FROZEN_QUERY_HEADS


def test_gqa_mapping_rejects_wrong_head_contract():
    key = torch.randn(1, 7, 2, 3)
    value = torch.randn(1, 7, 2, 3)
    with pytest.raises(ValueError, match="head count"):
        remap_kv_for_query_heads(
            key,
            value,
            native_query_to_kv_mapping(),
            query_heads=16,
            kv_heads=8,
        )
