from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from model.cross_validated_fast_weight import (  # noqa: E402
    MODES,
    CrossValidatedFastWeightTransformer,
)


def make_model() -> CrossValidatedFastWeightTransformer:
    torch.manual_seed(71)
    return CrossValidatedFastWeightTransformer(
        input_dim=8,
        hidden_dim=12,
        projection_ffn_dim=24,
        heads=3,
        dropout=0.0,
        initial_inner_step=0.2,
    ).eval()


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(72)
    b, h, c, d = 2, 5, 6, 8
    return {
        "query": torch.randn(b, d),
        "candidates": torch.randn(b, c, d),
        "candidate_keys": torch.tensor(
            [[41, 9, 31, 2, 27, 15], [71, 65, 61, 59, 53, 47]], dtype=torch.int64
        ),
        "history_keys": torch.randn(b, h, d),
        "history_values": torch.randn(b, h, d),
        "history_mask": torch.ones(b, h, dtype=torch.bool),
        "candidate_mask": torch.ones(b, c, dtype=torch.bool),
        "base_scores": torch.randn(b, c),
        "item_only_scores": torch.randn(b, c),
        "repeat_request": torch.zeros(b, dtype=torch.bool),
        "query_present": torch.ones(b, dtype=torch.bool),
    }


def test_modes_have_one_parameterization() -> None:
    model = make_model()
    values = batch()
    count = model.parameter_count()
    for mode in MODES:
        output = model(**values, mode=mode)
        assert output.scores.shape == values["base_scores"].shape
        assert model.parameter_count() == count


def test_nohistory_and_repeat_are_exact() -> None:
    model = make_model()
    values = batch()
    empty = dict(values)
    empty["history_mask"] = torch.zeros_like(values["history_mask"])
    output = model(**empty, mode="cross_validated_write")
    assert torch.equal(output.scores, values["base_scores"])
    repeat = dict(values)
    repeat["repeat_request"] = torch.ones_like(values["repeat_request"])
    output = model(**repeat, mode="cross_validated_write")
    assert torch.equal(output.scores, values["item_only_scores"])


def test_candidate_permutation_is_bit_exact() -> None:
    model = make_model()
    values = batch()
    output = model(**values, mode="cross_validated_write")
    permutation = torch.tensor([5, 4, 3, 2, 1, 0])
    changed = dict(values)
    for name in (
        "candidates",
        "candidate_keys",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        changed[name] = values[name][:, permutation]
    changed_output = model(**changed, mode="cross_validated_write")
    assert torch.equal(output.scores, changed_output.scores[:, permutation])


def test_single_event_cross_validation_abstains() -> None:
    model = make_model()
    values = batch()
    values["history_mask"][:, 1:] = False
    output = model(**values, mode="cross_validated_write")
    assert torch.equal(output.event_weight, torch.zeros_like(output.event_weight))
    assert torch.equal(output.scores, values["base_scores"])


def test_exact_write_backpropagates_through_transformer_and_step() -> None:
    model = make_model().train()
    values = batch()
    output = model(**values, mode="cross_validated_write")
    loss = output.raw_delta.square().mean() + output.fast_weight.square().mean()
    loss.backward()
    active = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(parameter.grad.ne(0).any())
    }
    assert any(name.startswith("pair_transformer.") for name in active)
    assert any(name.startswith("key_head.") for name in active)
    assert any(name.startswith("value_head.") for name in active)
    assert "inner_step_logit" in active
