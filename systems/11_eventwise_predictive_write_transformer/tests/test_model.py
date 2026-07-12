from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import (
    EventwisePredictiveWriteTransformer,
    bounded_zero_sum_write,
    eventwise_token_innovations,
)


MODES = [
    "eventwise_predictive",
    "pooled_c10",
    "centered_attention",
    "scalar_logit",
    "eventwise_hidden",
    "base",
]


def inputs():
    query = torch.tensor([[1, 4], [1, 5]])
    candidates = torch.tensor(
        [
            [[20, 4, 8], [21, 4, 9], [22, 5, 8]],
            [[23, 5, 9], [24, 5, 10], [25, 4, 9]],
        ]
    )
    history = torch.tensor(
        [
            [[20, 4, 8], [26, 5, 9], [27, 4, 10]],
            [[28, 4, 8], [29, 5, 10], [23, 5, 9]],
        ]
    )
    mask = torch.ones((2, 3), dtype=torch.bool)
    return query, candidates, history, mask


def model(mode="eventwise_predictive"):
    torch.manual_seed(41)
    return EventwisePredictiveWriteTransformer(
        vocab_size=64,
        candidate_token_count=3,
        max_history_events=4,
        d_model=16,
        nhead=4,
        lm_layers=1,
        integrator_layers=1,
        dim_feedforward=32,
        max_lm_sequence_length=8,
        dropout=0.0,
        max_write_norm=0.7,
        mode=mode,
    ).eval()


@pytest.mark.parametrize("mode", MODES)
def test_no_history_is_bitwise_base(mode):
    ranker = model(mode)
    query, candidates, history, mask = inputs()
    mask.zero_()
    output = ranker(
        query_tokens=query,
        candidate_tokens=candidates,
        history_tokens=history,
        history_mask=mask,
    )
    assert torch.equal(output.scores, output.base_scores)
    assert torch.isfinite(output.hidden_write).all()


def test_base_path_is_structurally_history_blind():
    ranker = model("base")
    query, candidates, history, mask = inputs()
    first = ranker(
        query_tokens=query,
        candidate_tokens=candidates,
        history_tokens=history,
        history_mask=mask,
    )
    second = ranker(
        query_tokens=query,
        candidate_tokens=candidates,
        history_tokens=history.flip(1) + 1,
        history_mask=mask,
    )
    assert torch.equal(first.scores, second.scores)


@pytest.mark.parametrize("mode", MODES)
def test_candidate_permutation_equivariance(mode):
    ranker = model(mode)
    query, candidates, history, mask = inputs()
    permutation = torch.tensor([2, 0, 1])
    first = ranker(
        query_tokens=query,
        candidate_tokens=candidates,
        history_tokens=history,
        history_mask=mask,
    )
    second = ranker(
        query_tokens=query,
        candidate_tokens=candidates[:, permutation],
        history_tokens=history,
        history_mask=mask,
    )
    assert torch.allclose(second.scores, first.scores[:, permutation], atol=3e-6, rtol=0)


def test_primary_preserves_candidate_event_token_matrix():
    ranker = model()
    query, candidates, history, mask = inputs()
    output = ranker(
        query_tokens=query,
        candidate_tokens=candidates,
        history_tokens=history,
        history_mask=mask,
    )
    assert output.gain_matrix.shape == (2, 3, 3, 3)
    assert output.event_innovations.shape == (2, 3, 3, 16)
    assert not torch.equal(output.gain_matrix[:, :, 0], output.gain_matrix[:, :, 1])


def test_eventwise_operator_does_not_reduce_to_pooled_gain():
    embedding = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    # Both histories have identical event-mean gains, but nonlinear per-event
    # innovations differ before the late integrator.
    first = torch.tensor([[[[3.0, 0.0], [-1.0, 0.0]]]])
    second = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    assert torch.allclose(first.mean(dim=2), second.mean(dim=2))
    innovation_a = eventwise_token_innovations(first, embedding)
    innovation_b = eventwise_token_innovations(second, embedding)
    assert not torch.allclose(innovation_a, innovation_b)


def test_hidden_write_is_bounded_and_zero_sum():
    write = bounded_zero_sum_write(torch.randn(4, 7, 11), 0.6)
    assert torch.allclose(write.sum(dim=1), torch.zeros(4, 11), atol=2e-6, rtol=0)
    assert float(write.norm(dim=-1).amax()) < 0.6


def test_exact_coordinate_is_monotone_inside_final_head():
    ranker = model()
    hidden = torch.randn(2, 3, ranker.d_model)
    low = torch.zeros(2, 3)
    high = torch.ones(2, 3)
    assert torch.all(ranker.ranking_head(hidden, high) > ranker.ranking_head(hidden, low))
    assert float(ranker.ranking_head.identity_weight.detach()) > 0


def test_exact_coordinate_counts_only_valid_item_tokens():
    ranker = model()
    _, candidates, history, mask = inputs()
    count, coordinate = ranker._identity_coordinate(candidates, history, mask)
    assert count[0, 0] == 1
    mask[0, 0] = False
    count2, coordinate2 = ranker._identity_coordinate(candidates, history, mask)
    assert count2[0, 0] == 0
    assert coordinate2[0, 0] < coordinate[0, 0]


def test_all_matched_controls_have_exact_parameter_count():
    matched = [
        "eventwise_predictive",
        "pooled_c10",
        "centered_attention",
        "eventwise_hidden",
    ]
    counts = [model(mode).trainable_parameter_count for mode in matched]
    assert len(set(counts)) == 1


def test_primary_load_bearing_parameters_receive_gradients():
    ranker = model().train()
    query, candidates, history, mask = inputs()
    output = ranker(
        query_tokens=query,
        candidate_tokens=candidates,
        history_tokens=history,
        history_mask=mask,
    )
    loss = output.scores.square().mean()
    loss.backward()
    missing = [
        name
        for name, parameter in ranker.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == ["scalar_log_scale"]


def test_unknown_mode_fails_closed():
    with pytest.raises(ValueError):
        model("invalid")
