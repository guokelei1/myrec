from __future__ import annotations

import torch
import pytest

from myrec.mechanism.q1_cache_phase_interventions import (
    cache_phase_signature,
    intervene_q1_prefix_cache,
)


def _cache():
    transformers = pytest.importorskip("transformers")
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer in range(3):
        key = torch.full((2, 2, 4, 4), float(layer + 1))
        value = torch.full((2, 2, 4, 4), float(layer + 11))
        cache.update(key, value, layer_idx=layer)
    return cache


def test_q1_cache_identity_and_rebuild_preserve_structure():
    cache = _cache()
    identity = intervene_q1_prefix_cache(cache, "native_cache_identity")
    rebuilt = intervene_q1_prefix_cache(cache, "same_request_rebuild")
    assert identity is cache
    assert rebuilt is not cache
    assert cache_phase_signature(rebuilt) == cache_phase_signature(cache)


def test_q1_cache_zero_and_donor_replacement_are_isolated():
    cache = _cache()
    donor = _cache()
    donor.layers[0].keys.fill_(99.0)
    zero = intervene_q1_prefix_cache(cache, "zero_prefix_cache")
    replaced = intervene_q1_prefix_cache(
        cache, "donor_prefix_cache_replacement", donor_cache=donor
    )
    assert torch.count_nonzero(zero.layers[0].keys) == 0
    assert torch.all(replaced.layers[0].keys == 99.0)
    assert torch.all(cache.layers[0].keys == 1.0)

