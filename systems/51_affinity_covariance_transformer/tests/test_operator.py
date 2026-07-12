from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


MODULE = Path(__file__).resolve().parents[1] / "model/affinity_covariance.py"
SPEC = importlib.util.spec_from_file_location("c51_operator", MODULE)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def fixture():
    q = torch.tensor([[1.0, 0.2, 0.0]])
    h = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]])
    m = torch.tensor([[True, True, True]])
    c = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    return q, h, m, c


def test_no_or_single_history_has_zero_covariance():
    q, h, _, c = fixture()
    empty = MOD.affinity_covariance(q, h, torch.zeros(1, 3, dtype=torch.bool), c)
    single = MOD.affinity_covariance(q, h, torch.tensor([[True, False, False]]), c)
    assert torch.equal(empty.covariance, torch.zeros_like(empty.covariance))
    assert torch.equal(single.covariance, torch.zeros_like(single.covariance))


def test_candidate_and_history_permutations():
    q, h, m, c = fixture()
    expected = MOD.affinity_covariance(q, h, m, c)
    co = torch.tensor([2, 0, 1])
    candidate = MOD.affinity_covariance(q, h, m, c[:, co])
    assert torch.allclose(expected.covariance[:, co], candidate.covariance, atol=1e-7)
    ho = torch.tensor([1, 2, 0])
    history = MOD.affinity_covariance(q, h[:, ho], m[:, ho], c)
    assert torch.allclose(expected.covariance, history.covariance, atol=1e-7)


def test_hand_computed_centered_covariance():
    q, h, m, c = fixture()
    output = MOD.affinity_covariance(q, h, m, c)
    qn = torch.nn.functional.normalize(q, dim=-1)
    hn = torch.nn.functional.normalize(h, dim=-1)
    cn = torch.nn.functional.normalize(c, dim=-1)
    a = torch.einsum("bhd,bd->bh", hn, qn)[0]
    b = torch.einsum("bhd,bcd->bhc", hn, cn)[0]
    expected = ((a - a.mean())[:, None] * (b - b.mean(0))).sum(0) / 2
    assert torch.allclose(output.covariance[0], expected, atol=1e-7)


def test_gradients_are_finite_and_nonzero():
    q, h, m, c = fixture()
    q.requires_grad_(); h.requires_grad_(); c.requires_grad_()
    output = MOD.affinity_covariance(q, h, m, c)
    (output.covariance.square().sum() + output.pearson_control.square().sum()).backward()
    for value in (q, h, c):
        assert value.grad is not None and bool(torch.isfinite(value.grad).all()) and bool(value.grad.ne(0).any())
