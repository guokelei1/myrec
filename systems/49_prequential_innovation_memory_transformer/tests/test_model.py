from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PRED = load("c49_predictor", "model/predictor.py")
MEM = load("c49_memory", "model/innovation_memory.py")


def test_predictor_shapes_and_no_prefix_zero():
    model = PRED.PrequentialSemanticTransformer(input_dim=5, width=8, heads=2, layers=1, ff_multiplier=2, max_history=4, temperature=0.1)
    values = torch.randn(3, 2, 5)
    mask = torch.tensor([[True, True], [True, False], [False, False]])
    output = model.predict_next(values, mask)
    assert output.shape == (3, 8)
    assert torch.equal(output[2], torch.zeros(8))
    assert model.encode_items(torch.randn(7, 5)).shape == (7, 8)


def fixture():
    query = torch.tensor([[1.0, 0.3, -0.2]], requires_grad=True)
    keys = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    predictions = torch.tensor([[[0.2, 0.1, 0.0], [0.1, 0.2, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    return query, keys, predictions, mask


def test_no_history_all_reads_exact_zero():
    q, k, p, _ = fixture()
    out = MEM.innovation_memory_reads(q, k, p, torch.zeros(1, 3, dtype=torch.bool))
    for value in (out.primary, out.raw_krr, out.innovation_softmax, out.delta_net, out.shuffled_innovation):
        assert torch.equal(value, torch.zeros_like(value))


def test_zero_prediction_reduces_primary_to_raw_krr():
    q, k, _, m = fixture()
    out = MEM.innovation_memory_reads(q, k, torch.zeros_like(k), m)
    assert torch.allclose(out.primary, out.raw_krr, atol=1e-7)


def test_history_permutation_preserves_krr_but_not_sequential_controls():
    q, k, p, m = fixture()
    order = torch.tensor([1, 0, 2])
    first = MEM.innovation_memory_reads(q, k, p, m)
    second = MEM.innovation_memory_reads(q, k[:, order], p[:, order], m[:, order])
    assert torch.allclose(first.primary, second.primary, atol=1e-7)
    assert torch.allclose(first.raw_krr, second.raw_krr, atol=1e-7)


def test_finite_gradients_reach_query_keys_and_predictions():
    q, k, p, m = fixture()
    out = MEM.innovation_memory_reads(q, k, p, m)
    loss = sum(value.square().sum() for value in (out.primary, out.raw_krr, out.innovation_softmax, out.delta_net))
    loss.backward()
    for value in (q, k, p):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())
        assert bool(value.grad.ne(0).any())
