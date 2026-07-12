from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.nn import functional as F


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

from model.query_transport import CONTROL, PRIMARY, LowRankQueryTransport  # noqa: E402


def make(mode: str, seed: int = 11) -> LowRankQueryTransport:
    return LowRankQueryTransport(
        dim=8,
        rank=2,
        temperature=0.1,
        profile_scale=1.0,
        correction_scale=2.0,
        seed=seed,
        mode=mode,
    )


def test_primary_and_control_are_capacity_and_initialization_matched() -> None:
    primary = make(PRIMARY)
    control = make(CONTROL)
    assert primary.trainable_parameter_count() == control.trainable_parameter_count() == 32
    assert all(
        torch.equal(first, second)
        for first, second in zip(primary.state_dict().values(), control.state_dict().values())
    )
    assert primary.uses_tangent_projection is True
    assert control.uses_tangent_projection is False


def test_primary_write_is_orthogonal_and_control_is_not_forced_orthogonal() -> None:
    query = F.normalize(torch.randn(8), dim=0)
    history = F.normalize(torch.randn(4, 8), dim=1)
    primary = make(PRIMARY)
    control = make(CONTROL)
    q_primary, tangent, _ = primary.transport_state(query, history)
    q_control, profile, _ = control.transport_state(query, history)
    assert abs(float(torch.dot(q_primary, tangent))) < 1e-6
    assert abs(float(torch.dot(q_control, profile))) > 1e-4


def test_both_modes_are_candidate_permutation_equivariant_and_null_safe() -> None:
    query = F.normalize(torch.randn(8), dim=0)
    history = F.normalize(torch.randn(3, 8), dim=1)
    candidates = F.normalize(torch.randn(5, 8), dim=1)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    for mode in (PRIMARY, CONTROL):
        model = make(mode)
        first = model(query, history, candidates)
        second = model(query, history, candidates[permutation])
        assert torch.allclose(second, first[permutation], atol=1e-6, rtol=0)
        null = model(query, torch.empty(0, 8), candidates)
        assert torch.equal(null, torch.zeros_like(null))
