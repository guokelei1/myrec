from __future__ import annotations

import torch

from myrec.mechanism.candidate_gap_scoring import (
    CANDIDATE_GAP_CONDITIONS,
    CANDIDATE_GAP_MODES,
    CANDIDATE_GAP_NODES,
    PERTURBATION_FRACTION,
    candidate_gap_direction_tensors,
    perturb_state,
)


def test_candidate_gap_direction_shapes_and_norm_matching() -> None:
    full = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 3.0, 2.0]]],
        dtype=torch.float32,
    )
    null = torch.zeros_like(full)
    directions = candidate_gap_direction_tensors(full, null)
    assert tuple(directions) == CANDIDATE_GAP_MODES
    reference_norm = (full - null).norm(dim=-1)
    for direction in directions.values():
        assert direction.shape == full.shape
        assert torch.isfinite(direction).all()
        assert torch.allclose(direction.norm(dim=-1), reference_norm, atol=1e-5)


def test_candidate_gap_orthogonal_direction_is_orthogonal_to_full_null() -> None:
    full = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]],
        dtype=torch.float32,
    )
    null = torch.zeros_like(full)
    directions = candidate_gap_direction_tensors(full, null)
    delta = full - null
    dot = (directions["orthogonal_direction"] * delta).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-5)


def test_candidate_gap_perturbation_fraction_and_condition_registry() -> None:
    full = torch.ones((2, 1, 4), dtype=torch.float32)
    direction = torch.full_like(full, 2.0)
    observed = perturb_state(full, direction)
    assert torch.allclose(observed, full + PERTURBATION_FRACTION * direction)
    expected = (
        "baseline_full",
        "baseline_null",
        *(f"{node}_{mode}" for node in CANDIDATE_GAP_NODES for mode in CANDIDATE_GAP_MODES),
        *(f"{node}_full_identity" for node in CANDIDATE_GAP_NODES),
    )
    assert CANDIDATE_GAP_CONDITIONS == expected
