from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.metric_coupled import (  # noqa: E402
    MODES,
    MULTIHEAD_COUPLED,
    SELECTION_ONLY,
    SHIFTED_LOOP,
    SINGLE_WIDE_COUPLED,
    MetricCoupledTransportTransformer,
)


def build(mode: str) -> MetricCoupledTransportTransformer:
    return MetricCoupledTransportTransformer(
        dim=32,
        heads=4,
        rank=2,
        temperature=0.1,
        profile_scale=1.0,
        correction_scale=2.0,
        seed=17,
        mode=mode,
        init_std=0.01,
    )


def test_operator_is_exact_c40_source_and_capacity_matched() -> None:
    source = ROOT / "model/metric_coupled.py"
    c40 = ROOT.parent / "40_metric_coupled_transport_transformer/model/metric_coupled.py"
    assert source.read_bytes() == c40.read_bytes()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "93ad470eb165e175c6999499dd160758ef6c9775bb45210e5ecf2cf54a5e6d2d"
    )
    models = [build(mode) for mode in MODES]
    assert {model.trainable_parameter_count() for model in models} == {512}
    assert all(
        torch.equal(models[0].down, model.down) and torch.equal(models[0].up, model.up)
        for model in models[1:]
    )


def test_loop_assignments_permutation_and_fallbacks() -> None:
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(32, generator=generator)
    history = torch.randn(7, 32, generator=generator)
    candidates = torch.randn(11, 32, generator=generator)
    expected = {
        MULTIHEAD_COUPLED: [0, 1, 2, 3],
        SINGLE_WIDE_COUPLED: [0],
        SELECTION_ONLY: [-1, -1, -1, -1],
        SHIFTED_LOOP: [1, 2, 3, 0],
    }
    order = torch.tensor([10, 1, 7, 3, 4, 5, 6, 2, 8, 9, 0])
    for mode in MODES:
        model = build(mode).eval()
        first = model(query, history, candidates)
        assert torch.equal(first, model(query, history, candidates))
        assert torch.allclose(first[order], model(query, history, candidates[order]), atol=1e-6)
        assert model.components(query, history, candidates)["loop_assignment"].tolist() == expected[mode]
        assert torch.equal(model(query, history[:0], candidates), torch.zeros(11))
        assert torch.equal(
            model(query, history, candidates, query_present=False), torch.zeros(11)
        )
        assert torch.equal(
            model(query, history, candidates, repeat_present=True), torch.zeros(11)
        )
