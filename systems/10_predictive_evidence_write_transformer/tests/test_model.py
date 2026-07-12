from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from importlib import import_module

model_module = import_module("10_predictive_evidence_write_transformer.model")
PredictiveEvidenceWriteTransformer = model_module.PredictiveEvidenceWriteTransformer
bounded_zero_sum_write = model_module.bounded_zero_sum_write
predictive_token_evidence = model_module.predictive_token_evidence


def make_inputs():
    query = torch.tensor([[1, 4], [1, 5]])
    candidates = torch.tensor(
        [
            [[20, 4, 8], [21, 4, 9], [22, 5, 8]],
            [[23, 5, 9], [24, 5, 10], [25, 4, 9]],
        ]
    )
    history = torch.tensor(
        [
            [[20, 4, 8], [26, 5, 9]],
            [[27, 4, 8], [23, 5, 9]],
        ]
    )
    mask = torch.tensor([[True, True], [True, True]])
    return query, candidates, history, mask


def make_model(mode="predictive_gain"):
    torch.manual_seed(7)
    return PredictiveEvidenceWriteTransformer(
        vocab_size=64,
        candidate_token_count=3,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        max_sequence_length=16,
        dropout=0.0,
        max_write_norm=0.7,
        mode=mode,
    ).eval()


@pytest.mark.parametrize(
    "mode", ["predictive_gain", "paired_logit", "single_pass", "dual_stream", "centered_attention", "base"]
)
def test_no_history_is_bitwise_base_identity(mode):
    model = make_model(mode)
    query, candidates, history, mask = make_inputs()
    mask.zero_()
    out = model(query_tokens=query, candidate_tokens=candidates, history_tokens=history, history_mask=mask)
    assert torch.equal(out.scores, out.base_scores)


@pytest.mark.parametrize(
    "mode", ["predictive_gain", "paired_logit", "single_pass", "dual_stream", "centered_attention", "base"]
)
def test_candidate_permutation_equivariance(mode):
    model = make_model(mode)
    query, candidates, history, mask = make_inputs()
    permutation = torch.tensor([2, 0, 1])
    original = model(query_tokens=query, candidate_tokens=candidates, history_tokens=history, history_mask=mask)
    permuted = model(
        query_tokens=query,
        candidate_tokens=candidates[:, permutation],
        history_tokens=history,
        history_mask=mask,
    )
    assert torch.allclose(permuted.scores, original.scores[:, permutation], atol=2e-6, rtol=0)


def test_write_is_zero_sum_and_bounded():
    raw = torch.randn(5, 7, 13)
    write = bounded_zero_sum_write(raw, 0.6)
    assert torch.allclose(write.sum(dim=1), torch.zeros(5, 13), atol=2e-6, rtol=0)
    assert float(write.norm(dim=-1).amax()) < 0.6


def test_exact_identity_coordinate_has_strictly_monotone_final_logit():
    model = make_model()
    query, candidates, history, mask = make_inputs()
    count, coordinate = model._repeat_channel(candidates, history, mask)
    doubled = torch.cat((history, history[:, :1]), dim=1)
    doubled_mask = torch.ones((2, 3), dtype=torch.bool)
    count2, coordinate2 = model._repeat_channel(candidates, doubled, doubled_mask)
    assert torch.all(coordinate2 >= coordinate)
    hidden = torch.randn(2, 3, model.d_model)
    score = model.ranking_head(hidden, coordinate)
    score2 = model.ranking_head(hidden, coordinate2)
    assert torch.all(score2 >= score)
    assert torch.all((count2 > count) == (score2 > score))
    assert float(model.ranking_head.identity_weight.detach()) > 0


def test_tokenwise_write_does_not_reduce_to_summed_paired_logit():
    embeddings = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    first = torch.tensor([[[1.0, -1.0]]])
    second = torch.tensor([[[-1.0, 1.0]]])
    assert torch.equal(first.sum(dim=-1), second.sum(dim=-1))
    evidence_a = predictive_token_evidence(first, embeddings)
    evidence_b = predictive_token_evidence(second, embeddings)
    assert not torch.equal(evidence_a, evidence_b)


def test_capacity_controls_have_exact_parameter_count():
    modes = ["predictive_gain", "single_pass", "dual_stream", "centered_attention"]
    counts = [make_model(mode).trainable_parameter_count for mode in modes]
    assert len(set(counts)) == 1


def test_all_parameters_receive_gradient_in_primary():
    model = make_model().train()
    query, candidates, history, mask = make_inputs()
    output = model(query_tokens=query, candidate_tokens=candidates, history_tokens=history, history_mask=mask)
    loss = output.scores.square().mean() - 0.01 * output.log_probs_history.mean()
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    assert missing == ["paired_log_scale"]


def test_invalid_mode_fails_closed():
    with pytest.raises(ValueError):
        make_model("invalid")
