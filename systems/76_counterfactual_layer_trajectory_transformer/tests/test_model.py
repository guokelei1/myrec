import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.cltt import (  # noqa: E402
    CANDIDATE,
    HISTORY,
    QUERY,
    CounterfactualLayerTrajectoryTransformer,
    carrier_scaled_difference,
    full_and_cut_masks,
)


def model(mode: str = "counterfactual_trajectory") -> CounterfactualLayerTrajectoryTransformer:
    torch.manual_seed(3)
    return CounterfactualLayerTrajectoryTransformer(
        mode=mode,
        vocabulary_size=160,
        hidden_size=32,
        attention_heads=4,
        backbone_layers=3,
        trajectory_layers=1,
        trajectory_heads=4,
        maximum_length=12,
        correction_bound=2.0,
    )


def inputs() -> tuple[torch.Tensor, ...]:
    tokens = torch.tensor(
        [[
            [1, 10, 2, 11, 32, 2, 12, 33, 2, 13, 34, 2],
            [1, 10, 2, 11, 35, 2, 12, 33, 2, 13, 34, 2],
        ]]
    )
    segment_row = [QUERY] * 3 + [CANDIDATE] * 3 + [HISTORY] * 6
    segments = torch.tensor([[segment_row, segment_row]])
    return (
        tokens,
        segments,
        torch.tensor([[0.2, 0.1]]),
        torch.tensor([True]),
        torch.tensor([False]),
        torch.tensor([[0.0, 0.0]]),
    )


def test_nohistory_is_exact_base_and_repeat_is_exact_item_only() -> None:
    network = model()
    values = list(inputs())
    values[3] = torch.tensor([False])
    output = network(*values)
    assert torch.equal(output.scores, values[2])
    values = list(inputs())
    values[4] = torch.tensor([True])
    values[5] = torch.tensor([[0.0, 5.0]])
    output = network(*values)
    assert torch.equal(output.scores, values[5])


def test_cut_mask_isolates_history_and_difference_does_not_amplify_zero() -> None:
    segment = torch.tensor([[QUERY, QUERY, CANDIDATE, HISTORY]])
    full, cut = full_and_cut_masks(segment, torch.tensor([True]))
    assert bool(full.all())
    assert not bool(cut[0, 0, 3]) and not bool(cut[0, 3, 0])
    states = torch.randn(2, 4, 8)
    assert torch.equal(carrier_scaled_difference(states, states), torch.zeros_like(states))


def test_candidate_permutation_and_primary_gradients() -> None:
    network = model()
    values = inputs()
    output = network(*values)
    reversed_values = list(values)
    for index in (0, 1, 2, 5):
        reversed_values[index] = reversed_values[index].flip(1)
    reversed_output = network(*reversed_values).scores.flip(1)
    assert torch.allclose(output.scores, reversed_output, atol=1e-6, rtol=0)
    output.scores[0, 0].backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for name, parameter in network.named_parameters()
        if name.startswith("backbone.")
    )
    assert any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for name, parameter in network.named_parameters()
        if name.startswith("trajectory.")
    )


def test_all_modes_have_identical_parameter_count() -> None:
    counts = {
        mode: model(mode).parameter_count()
        for mode in (
            "counterfactual_trajectory",
            "final_logit_delta",
            "final_hidden_delta",
            "factual_trajectory",
            "ordinary_full",
        )
    }
    assert len(set(counts.values())) == 1
