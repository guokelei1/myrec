from __future__ import annotations

import torch
from torch import nn

from cpdlr.model import PrefixDeltaRanker


def _operator() -> PrefixDeltaRanker:
    model = PrefixDeltaRanker.__new__(PrefixDeltaRanker)
    nn.Module.__init__(model)
    model.mode = "paired_delta"
    model.delta_clip = 100.0
    model.tangent_epsilon = 1.0e-6
    return model


def test_empty_prefix_has_exact_zero_delta_and_null_final() -> None:
    model = _operator()
    null = torch.tensor([[0.2, 0.7, -0.1]])
    mask = torch.ones_like(null, dtype=torch.bool)
    present = torch.tensor([False])
    output = model.combine(null.clone(), null, mask, present)
    torch.testing.assert_close(output["raw_delta"], torch.zeros_like(null), rtol=0, atol=0)
    torch.testing.assert_close(output["final"], null, rtol=0, atol=0)


def test_candidate_order_tangent_is_centered_and_orthogonal_to_null() -> None:
    model = _operator()
    null = torch.tensor([[0.0, 1.0, 3.0, -2.0]])
    factual = torch.tensor([[1.0, 0.0, 5.0, -1.0]])
    mask = torch.ones_like(null, dtype=torch.bool)
    tangent, _ = model.tangent_delta(factual, null, mask, torch.tensor([True]))
    centered_null = null - null.mean(dim=-1, keepdim=True)
    assert abs(float(tangent.sum())) < 1e-5
    assert abs(float((tangent * centered_null).sum())) < 1e-4


def test_candidate_mask_blocks_padded_logit() -> None:
    model = _operator()
    null = torch.tensor([[0.0, 1.0, 99.0]])
    factual = null + 1.0
    mask = torch.tensor([[True, True, False]])
    output = model.combine(factual, null, mask, torch.tensor([True]))
    assert output["final"][0, 2] < -1e30
