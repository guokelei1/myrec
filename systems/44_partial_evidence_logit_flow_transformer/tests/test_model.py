from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.partial_logit_flow import (  # noqa: E402
    FORCED_LOGIT_FLOW,
    GLOBAL_VECTOR_WRITE,
    MODES,
    PARTIAL_LOGIT_FLOW,
    PARTIAL_VECTOR_WRITE,
    PartialEvidenceLogitFlowTransformer,
)


def build(mode: str) -> PartialEvidenceLogitFlowTransformer:
    return PartialEvidenceLogitFlowTransformer(
        dim=8,
        heads=2,
        rank=2,
        temperature=0.1,
        profile_scale=1.0,
        correction_scale=2.0,
        seed=11,
        mode=mode,
        init_std=0.01,
    )


def inputs():
    query = F.normalize(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]), dim=0)
    candidates = F.normalize(
        torch.tensor(
            [
                [1.0, 1.0, 0, 0, 0, 0, 0, 0],
                [1.0, 0, 1.0, 0, 0, 0, 0, 0],
                [1.0, 0, 0, 1.0, 0, 0, 0, 0],
            ]
        ),
        dim=-1,
    )
    history = torch.tensor(
        [
            [0.0, 1.0, 0, 0, 0, 0, 0, 0],
            [0.0, 0, 0, 0, 1.0, 0, 0, 0],
        ]
    )
    return query, history, candidates


def test_partial_mass_zero_sum_and_signal_routing() -> None:
    model = build(PARTIAL_LOGIT_FLOW).eval()
    query, history, candidates = inputs()
    state = model.components(query, history, candidates)
    assert torch.allclose(
        state["candidate_mass"].sum(dim=1) + state["null_mass"],
        torch.ones_like(state["null_mass"]),
        atol=1e-6,
        rtol=0,
    )
    assert float(state["correction"].sum().abs()) <= 1e-6
    assert float(state["candidate_mass"][:, 0, 0].mean()) > 0.5
    assert float(state["null_mass"][:, 1].mean()) > 0.5


def test_capacity_controls_permutation_and_fallbacks() -> None:
    models = [build(mode).eval() for mode in MODES]
    assert {model.trainable_parameter_count() for model in models} == {64}
    assert all(
        torch.equal(models[0].down, model.down) and torch.equal(models[0].up, model.up)
        for model in models[1:]
    )
    query, history, candidates = inputs()
    order = torch.tensor([2, 0, 1])
    for model in models:
        first = model(query, history, candidates)
        assert torch.equal(first, model(query, history, candidates))
        assert torch.allclose(first[order], model(query, history, candidates[order]), atol=1e-6)
        assert torch.equal(model(query, history[:0], candidates), torch.zeros(3))
        assert torch.equal(
            model(query, history, candidates, query_present=False), torch.zeros(3)
        )
        assert torch.equal(
            model(query, history, candidates, repeat_present=True), torch.zeros(3)
        )
    assert build(FORCED_LOGIT_FLOW).components(query, history, candidates)["null_mass"].eq(0).all()
    assert build(PARTIAL_VECTOR_WRITE).components(query, history, candidates)["candidate_mass"] is not None
    assert build(GLOBAL_VECTOR_WRITE).components(query, history, candidates)["candidate_mass"] is None
