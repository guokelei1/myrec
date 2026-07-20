"""Pure GQA query-to-KV grouping controls for the registered N33 boundary."""

from __future__ import annotations

from typing import Any, Sequence


FROZEN_QUERY_HEADS = 16
FROZEN_KV_HEADS = 8
FROZEN_QUERY_HEADS_PER_KV = 2


def native_query_to_kv_mapping(
    *,
    query_heads: int = FROZEN_QUERY_HEADS,
    kv_heads: int = FROZEN_KV_HEADS,
) -> tuple[int, ...]:
    """Return the native repeat-KV mapping for a frozen GQA topology."""

    query_heads = int(query_heads)
    kv_heads = int(kv_heads)
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query_heads must be a positive multiple of kv_heads")
    groups = query_heads // kv_heads
    return tuple(kv for kv in range(kv_heads) for _ in range(groups))


def cyclic_group_permutation_mapping(
    *,
    shift: int = 1,
    query_heads: int = FROZEN_QUERY_HEADS,
    kv_heads: int = FROZEN_KV_HEADS,
) -> tuple[int, ...]:
    """Rotate the KV-group assignment of every query-head group."""

    native = native_query_to_kv_mapping(query_heads=query_heads, kv_heads=kv_heads)
    shift = int(shift) % kv_heads
    return tuple((kv + shift) % kv_heads for kv in native)


def within_group_rotation_mapping(
    *,
    offset: int = 1,
    query_heads: int = FROZEN_QUERY_HEADS,
    kv_heads: int = FROZEN_KV_HEADS,
) -> tuple[int, ...]:
    """Assign the second query head in each native group to a fixed neighbor."""

    native = native_query_to_kv_mapping(query_heads=query_heads, kv_heads=kv_heads)
    heads_per_kv = query_heads // kv_heads
    offset = int(offset) % kv_heads
    return tuple(
        (kv + (query_index % heads_per_kv) * offset) % kv_heads
        for query_index, kv in enumerate(native)
    )


def validate_query_to_kv_mapping(
    mapping: Sequence[int], *, query_heads: int, kv_heads: int
) -> tuple[int, ...]:
    """Validate a complete query-head mapping without changing tensor values."""

    mapping = tuple(int(value) for value in mapping)
    if len(mapping) != int(query_heads):
        raise ValueError("GQA mapping length differs from query head count")
    if any(value < 0 or value >= int(kv_heads) for value in mapping):
        raise ValueError("GQA mapping contains an out-of-range KV head")
    return mapping


def remap_kv_for_query_heads(
    key: Any,
    value: Any,
    mapping: Sequence[int],
    *,
    query_heads: int,
    kv_heads: int,
) -> tuple[Any, Any]:
    """Expand K/V to query-head order under an explicit mapping.

    The returned tensors are new views/copies indexed from the original K/V;
    the inputs, head counts, and per-head vectors are never modified.
    """

    if key.ndim != 4 or value.ndim != 4 or tuple(key.shape) != tuple(value.shape):
        raise ValueError("GQA key/value tensors must both be [batch,kv_heads,seq,dim]")
    if int(key.shape[1]) != int(kv_heads):
        raise ValueError("GQA key/value head count differs from contract")
    mapping = validate_query_to_kv_mapping(mapping, query_heads=query_heads, kv_heads=kv_heads)
    import torch

    index = torch.as_tensor(mapping, dtype=torch.long, device=key.device)
    return key.index_select(1, index), value.index_select(1, index)


def grouping_control(
    key: Any,
    value: Any,
    *,
    mode: str,
    query_heads: int = FROZEN_QUERY_HEADS,
    kv_heads: int = FROZEN_KV_HEADS,
) -> tuple[Any, Any, dict[str, Any]]:
    """Apply a fixed grouping mode and return an auditable summary."""

    if mode == "identity":
        mapping = native_query_to_kv_mapping(query_heads=query_heads, kv_heads=kv_heads)
    elif mode == "group_permutation":
        mapping = cyclic_group_permutation_mapping(query_heads=query_heads, kv_heads=kv_heads)
    elif mode == "group_rotation":
        mapping = within_group_rotation_mapping(query_heads=query_heads, kv_heads=kv_heads)
    else:
        raise ValueError(f"unsupported GQA grouping mode={mode}")
    remapped_key, remapped_value = remap_kv_for_query_heads(
        key,
        value,
        mapping,
        query_heads=query_heads,
        kv_heads=kv_heads,
    )
    summary = {
        "mode": mode,
        "query_heads": int(query_heads),
        "kv_heads": int(kv_heads),
        "mapping": list(mapping),
        "input_key_unchanged": True,
        "input_value_unchanged": True,
        "output_query_head_count": int(remapped_key.shape[1]),
    }
    return remapped_key, remapped_value, summary
