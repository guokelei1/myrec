from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import BehaviorRelationTransformer  # noqa: E402


def make_model() -> BehaviorRelationTransformer:
    torch.manual_seed(4)
    return BehaviorRelationTransformer(
        input_dim=8, width=16, heads=4, layers=1, ffn_dim=32,
        dropout=0.0, score_bound=4.0,
    ).double()


def test_zero_source_and_target_are_exact_nulls() -> None:
    model = make_model()
    source = torch.randn(5, 8, dtype=torch.float64)
    target = torch.randn(5, 8, dtype=torch.float64)
    zero = torch.zeros_like(source)
    torch.testing.assert_close(
        model.anchored_score(zero, target), torch.zeros(5, dtype=torch.float64),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(
        model.anchored_score(source, zero), torch.zeros(5, dtype=torch.float64),
        atol=0, rtol=0,
    )


def test_determinism_and_gradients() -> None:
    model = make_model()
    source = torch.randn(7, 8, dtype=torch.float64)
    target = torch.randn(7, 8, dtype=torch.float64)
    first = model.anchored_score(source, target)
    second = model.anchored_score(source, target)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    first.square().mean().backward()
    groups = {"input_projection": False, "relation_token": False, "role": False,
              "transformer": False, "output_norm": False, "score_head": False}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        active = bool(parameter.grad.abs().sum() > 0)
        for group in groups:
            if name == group or name.startswith(group + "."):
                groups[group] |= active
    assert all(groups.values()), groups


def test_parameter_count_is_input_width_specific_only() -> None:
    first = make_model()
    second = make_model()
    assert first.parameter_count == second.parameter_count
