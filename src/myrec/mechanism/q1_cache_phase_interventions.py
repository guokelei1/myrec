"""Phase-safe Q1 prefix-cache interventions for the inactive N20 boundary.

This module does not run a model or read qrels.  It only creates a validated
copy of a Transformers cache before the cached-continuation call, keeping the
prefix length, layer count, batch shape, and key/value layout fixed.
"""

from __future__ import annotations

import copy
from typing import Any


Q1_CACHE_MODES = (
    "native_cache_identity",
    "same_request_rebuild",
    "zero_prefix_cache",
    "donor_prefix_cache_replacement",
)


def intervene_q1_prefix_cache(
    cache: Any,
    mode: str,
    *,
    donor_cache: Any | None = None,
) -> Any:
    """Return a phase-validated cache copy for one Q1 continuation call.

    ``native_cache_identity`` deliberately returns the original object.  All
    other modes deep-copy before mutation, so an intervention cannot alter a
    cache reused for another candidate chunk.  Donor replacement is only
    allowed when every layer has the same key/value shape as the recipient.
    """

    mode = str(mode)
    if mode not in Q1_CACHE_MODES:
        raise ValueError(f"unsupported Q1 cache phase mode={mode}")
    if mode == "native_cache_identity":
        _validate_cache(cache)
        return cache
    replacement = copy.deepcopy(cache)
    _validate_cache(replacement)
    if mode == "same_request_rebuild":
        return replacement
    if mode == "zero_prefix_cache":
        for layer in _cache_layers(replacement):
            layer.keys = layer.keys.clone().zero_()
            layer.values = layer.values.clone().zero_()
        _validate_cache(replacement)
        return replacement
    if donor_cache is None:
        raise ValueError("donor_prefix_cache_replacement requires donor_cache")
    donor = copy.deepcopy(donor_cache)
    _validate_cache(donor)
    recipient_layers = _cache_layers(replacement)
    donor_layers = _cache_layers(donor)
    if len(recipient_layers) != len(donor_layers):
        raise ValueError("Q1 donor and recipient cache layer counts differ")
    for index, (recipient, donor_layer) in enumerate(zip(recipient_layers, donor_layers)):
        if recipient.keys.shape != donor_layer.keys.shape or recipient.values.shape != donor_layer.values.shape:
            raise ValueError(f"Q1 donor cache shape differs at layer {index}")
        recipient.keys = donor_layer.keys.clone()
        recipient.values = donor_layer.values.clone()
    _validate_cache(replacement)
    return replacement


def cache_phase_signature(cache: Any) -> dict[str, Any]:
    """Return qrels-free structural metadata used by N20 integrity checks."""

    _validate_cache(cache)
    layers = _cache_layers(cache)
    return {
        "layer_count": len(layers),
        "batch_sizes": [int(layer.keys.shape[0]) for layer in layers],
        "key_shapes": [list(map(int, layer.keys.shape)) for layer in layers],
        "value_shapes": [list(map(int, layer.values.shape)) for layer in layers],
        "prefix_lengths": [int(layer.keys.shape[-2]) for layer in layers],
        "finite": True,
    }


def _validate_cache(cache: Any) -> None:
    layers = _cache_layers(cache)
    if not layers:
        raise ValueError("Q1 cache has no decoder layers")
    reference_shape: tuple[int, ...] | None = None
    for index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None or not hasattr(keys, "shape") or not hasattr(values, "shape"):
            raise TypeError(f"Q1 cache layer {index} lacks key/value tensors")
        if keys.ndim != 4 or values.ndim != 4 or keys.shape != values.shape:
            raise ValueError(f"Q1 cache key/value shape mismatch at layer {index}")
        if reference_shape is None:
            reference_shape = tuple(int(value) for value in keys.shape)
        elif tuple(int(value) for value in keys.shape) != reference_shape:
            raise ValueError("Q1 cache layer shapes are not uniform")
        torch = _torch()
        if not bool(torch.isfinite(keys).all()) or not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"Q1 cache contains non-finite values at layer {index}")


def _cache_layers(cache: Any) -> list[Any]:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return list(layers)
    # Older Transformers cache implementations expose parallel lists.
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None and len(key_cache) == len(value_cache):
        return [_ParallelCacheLayer(keys, values) for keys, values in zip(key_cache, value_cache)]
    raise TypeError("unsupported Q1 cache object; expected layers or key_cache/value_cache")


class _ParallelCacheLayer:
    def __init__(self, keys: Any, values: Any) -> None:
        self.keys = keys
        self.values = values


def _torch() -> Any:
    import torch

    return torch

