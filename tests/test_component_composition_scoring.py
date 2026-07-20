from __future__ import annotations

import pytest
import torch

from myrec.mechanism.component_composition_scoring import (
    COMPOSITION_CONDITIONS,
    COMPOSITION_NODES,
    _assert_path_geometry,
    _validate_scores,
    composition_conditions,
)


def test_composition_condition_order_is_frozen() -> None:
    assert composition_conditions() == COMPOSITION_CONDITIONS
    assert COMPOSITION_NODES == (
        "attention_o_projection",
        "mlp_down_projection",
    )


def test_neutral_path_preserves_registered_geometry() -> None:
    ids = torch.tensor([[1, 2, 3, 4]])
    neutral = torch.tensor([[1, 9, 9, 4]])
    common = {
        "mask": torch.ones_like(ids),
        "positions": torch.tensor([[3]]),
        "starts": torch.tensor([1]),
        "ends": torch.tensor([3]),
    }
    _assert_path_geometry({"ids": ids, **common}, {"ids": neutral, **common})


def test_neutral_path_rejects_changed_positions() -> None:
    ids = torch.tensor([[1, 2, 3, 4]])
    common = {
        "mask": torch.ones_like(ids),
        "positions": torch.tensor([[3]]),
        "starts": torch.tensor([1]),
        "ends": torch.tensor([3]),
    }
    changed = {**common, "positions": torch.tensor([[2]])}
    with pytest.raises(RuntimeError, match="changed positions"):
        _assert_path_geometry({"ids": ids, **common}, {"ids": ids, **changed})


def test_validate_scores_requires_all_finite_conditions() -> None:
    values = {name: torch.zeros(2) for name in COMPOSITION_CONDITIONS}
    _validate_scores(values, 2)
    values["joint_attention_mlp_neutral_removal"][1] = float("nan")
    with pytest.raises(FloatingPointError):
        _validate_scores(values, 2)
