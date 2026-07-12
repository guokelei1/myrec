from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


MODULE = Path(__file__).resolve().parents[1] / "model/signed_consensus.py"
SPEC = importlib.util.spec_from_file_location("c48_signed_consensus", MODULE)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def fixture():
    q = torch.tensor([[1.0, 0.4, -0.2]], requires_grad=True)
    h = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    m = torch.tensor([[True, True, False]])
    c = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]], requires_grad=True)
    return q, h, m, c


def test_no_history_is_exact_zero():
    q, h, _, c = fixture()
    out = MOD.signed_consensus_mix(q, h, torch.zeros(1, 3, dtype=torch.bool), c)
    assert torch.equal(out.correction, torch.zeros_like(out.correction))
    assert torch.equal(out.coherence, torch.zeros_like(out.coherence))


def test_exact_decomposition_and_contraction():
    out = MOD.signed_consensus_mix(*fixture())
    assert torch.allclose(out.event_influences.sum(1), out.plain_correction, atol=1e-7)
    assert bool((out.coherence >= 0).all() and (out.coherence <= 1).all())
    assert bool((out.correction.abs() <= out.plain_correction.abs() + 1e-7).all())


def test_single_aligned_event_has_unit_consensus():
    q = torch.tensor([[1.0, 0.0]])
    h = torch.tensor([[[1.0, 0.0]]])
    m = torch.tensor([[True]])
    c = torch.tensor([[[1.0, 0.0]]])
    out = MOD.signed_consensus_mix(q, h, m, c)
    assert out.coherence.item() == 1.0
    assert torch.equal(out.correction, out.plain_correction)


def test_candidate_and_history_permutations():
    q, h, m, c = fixture()
    expected = MOD.signed_consensus_mix(q, h, m, c)
    co = torch.tensor([2, 0, 1])
    candidate = MOD.signed_consensus_mix(q, h, m, c[:, co])
    assert torch.allclose(expected.correction[:, co], candidate.correction, atol=1e-7)
    ho = torch.tensor([1, 2, 0])
    history = MOD.signed_consensus_mix(q, h[:, ho], m[:, ho], c)
    assert torch.allclose(expected.correction, history.correction, atol=1e-7)


def test_gradients_reach_query_history_and_candidates():
    q, h, m, c = fixture()
    out = MOD.signed_consensus_mix(q, h, m, c)
    (out.correction.square().sum() + out.coherence.sum()).backward()
    for value in (q, h, c):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())
        assert bool(value.grad.ne(0).any())
