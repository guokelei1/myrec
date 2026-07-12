from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


MODULE = Path(__file__).resolve().parents[1] / "model/dual_memory.py"
SPEC = importlib.util.spec_from_file_location("c50_dual", MODULE)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_orthogonal_component_is_orthogonal_and_semantic_is_preserved():
    semantic = torch.tensor([[1.0, 0.0, 0.0]])
    innovation = torch.tensor([[2.0, 3.0, 0.0]])
    output = MOD.semantic_protected_reads(semantic, innovation)
    assert torch.allclose((output.orthogonal_innovation * semantic).sum(-1), torch.zeros(1))
    assert torch.equal(output.primary, torch.tensor([[1.0, 3.0, 0.0]]))


def test_parallel_innovation_reduces_exactly_to_raw():
    semantic = torch.tensor([[1.0, 2.0]])
    output = MOD.semantic_protected_reads(semantic, 3.0 * semantic)
    assert torch.allclose(output.primary, semantic, atol=1e-6)


def test_zero_semantic_abstains_from_unanchored_innovation():
    semantic = torch.zeros(2, 3)
    innovation = torch.randn(2, 3)
    output = MOD.semantic_protected_reads(semantic, innovation)
    assert torch.equal(output.primary, semantic)
    assert torch.equal(output.orthogonal_innovation, semantic)


def test_finite_gradients_reach_both_reads():
    semantic = torch.tensor([[1.0, 0.2, -0.1]], requires_grad=True)
    innovation = torch.tensor([[0.2, 0.8, 0.3]], requires_grad=True)
    output = MOD.semantic_protected_reads(semantic, innovation)
    output.primary.square().sum().backward()
    for value in (semantic, innovation):
        assert value.grad is not None and bool(torch.isfinite(value.grad).all()) and bool(value.grad.ne(0).any())
