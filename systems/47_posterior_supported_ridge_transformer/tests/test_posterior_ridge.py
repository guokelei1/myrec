from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).resolve().parents[1] / "model/posterior_ridge.py"
SPEC = importlib.util.spec_from_file_location("c47_posterior_ridge", MODULE)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
posterior_ridge_mix = MOD.posterior_ridge_mix


def tensors():
    query = torch.tensor([[1.0, 0.5, -0.25]], requires_grad=True)
    history = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, False]])
    candidates = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]],
        requires_grad=True,
    )
    return query, history, mask, candidates


def test_nohistory_is_exact_zero():
    q, h, _, c = tensors()
    out = posterior_ridge_mix(q, h, torch.zeros(1, 3, dtype=torch.bool), c)
    assert torch.equal(out.correction, torch.zeros_like(out.correction))
    assert torch.equal(out.query_write, torch.zeros_like(out.query_write))
    assert torch.equal(out.support, torch.zeros_like(out.support))


def test_support_bounds_and_contraction():
    q, h, m, c = tensors()
    out = posterior_ridge_mix(q, h, m, c)
    assert bool((out.support >= 0).all() and (out.support <= 1).all())
    assert bool((out.correction.abs() <= out.plain_correction.abs() + 1e-7).all())


def test_orthogonal_candidate_has_zero_support_and_write():
    q, h, m, c = tensors()
    out = posterior_ridge_mix(q, h, m, c)
    assert out.support[0, 1].item() == 0.0
    assert out.correction[0, 1].item() == 0.0


def test_candidate_permutation_equivariance():
    q, h, m, c = tensors()
    order = torch.tensor([2, 0, 1])
    a = posterior_ridge_mix(q, h, m, c)
    b = posterior_ridge_mix(q, h, m, c[:, order])
    assert torch.allclose(a.correction[:, order], b.correction, atol=1e-7, rtol=0)
    assert torch.allclose(a.support[:, order], b.support, atol=1e-7, rtol=0)


def test_history_permutation_invariance():
    q, h, m, c = tensors()
    order = torch.tensor([1, 2, 0])
    a = posterior_ridge_mix(q, h, m, c)
    b = posterior_ridge_mix(q, h[:, order], m[:, order], c)
    assert torch.allclose(a.correction, b.correction, atol=1e-7, rtol=0)
    assert torch.allclose(a.support, b.support, atol=1e-7, rtol=0)


def test_duplicate_aligned_evidence_raises_support():
    q = torch.tensor([[1.0, 0.0]])
    c = torch.tensor([[[1.0, 0.0]]])
    one = posterior_ridge_mix(
        q,
        torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
        torch.tensor([[True, False]]),
        c,
    )
    two = posterior_ridge_mix(
        q,
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        torch.tensor([[True, True]]),
        c,
    )
    assert two.support.item() > one.support.item()


def test_finite_nonzero_gradients_reach_all_state_groups():
    q, h, m, c = tensors()
    out = posterior_ridge_mix(q, h, m, c)
    loss = out.correction.square().sum() + out.support.sum()
    loss.backward()
    for value in (q, h, c):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())
        assert bool(value.grad.ne(0).any())


def test_plain_mode_is_exact_mean_write():
    q, h, m, c = tensors()
    out = posterior_ridge_mix(q, h, m, c, mode="plain_ridge")
    assert torch.equal(out.correction, out.plain_correction)
