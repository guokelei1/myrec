from __future__ import annotations

import sys
from pathlib import Path

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.signal_probe import TargetAttentionSignalProbe
from train.losses import masked_listwise_loss


def _inputs(history_present: bool = True) -> dict[str, torch.Tensor]:
    torch.manual_seed(41)
    history_mask = torch.tensor(
        [[history_present, history_present], [history_present, False]],
        dtype=torch.bool,
    )
    return {
        "query": torch.randn(2, 8),
        "candidates": torch.randn(2, 3, 8),
        "history": torch.randn(2, 2, 8),
        "candidate_mask": torch.tensor(
            [[True, True, True], [True, True, False]], dtype=torch.bool
        ),
        "history_mask": history_mask,
        "history_event_weights": torch.tensor([[0.5, 1.0], [1.0, 0.0]]),
        "base_scores": torch.randn(2, 3, dtype=torch.float64),
    }


def _model() -> TargetAttentionSignalProbe:
    torch.manual_seed(43)
    return TargetAttentionSignalProbe(
        input_dim=8,
        evidence_dim=4,
        score_delta_max=1.0,
        dropout=0.0,
    )


def test_no_history_is_bit_exact_and_base_is_detached() -> None:
    model = _model()
    values = _inputs(False)
    values["base_scores"].requires_grad_(True)
    output = model(**values)
    assert torch.equal(output.scores, values["base_scores"].detach())
    assert not output.base_scores.requires_grad
    assert torch.count_nonzero(output.attention_weights) == 0


def test_candidate_permutation_is_equivariant() -> None:
    model = _model().eval()
    values = _inputs()
    output = model(**values)
    permutation = torch.tensor([2, 0, 1])
    permuted = dict(values)
    for key in ("candidates", "candidate_mask", "base_scores"):
        permuted[key] = values[key][:, permutation]
    permuted_output = model(**permuted)
    assert torch.allclose(
        permuted_output.scores,
        output.scores[:, permutation],
        atol=1e-7,
        rtol=1e-7,
    )


def test_history_present_initialization_is_bit_exact_base() -> None:
    model = _model().eval()
    values = _inputs()
    output = model(**values)
    assert torch.equal(output.scores, values["base_scores"])
    assert torch.count_nonzero(output.score_delta) == 0


def test_padded_nan_is_removed_before_projection() -> None:
    model = _model().eval()
    values = _inputs()
    values["candidates"][1, 2] = torch.nan
    values["history"][1, 1] = torch.nan
    output = model(**values)
    assert torch.isfinite(output.scores[values["candidate_mask"]]).all()
    assert torch.isfinite(
        output.attention_weights[
            values["candidate_mask"][:, :, None]
            & values["history_mask"][:, None, :]
        ]
    ).all()


def test_real_initialization_gets_ranking_gradients_on_two_steps() -> None:
    model = _model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    values = _inputs()
    relevance = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(**values)
        loss = masked_listwise_loss(
            output.scores, relevance, values["candidate_mask"]
        )
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        optimizer.step()
    for parameter in (
        model.query_projection.weight,
        model.candidate_projection.weight,
        model.history_key_projection.weight,
        model.history_value_projection.weight,
        model.output_projection.weight,
        model.score_head.weight,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0.0


def test_event_identity_shuffle_with_fixed_position_weights_is_observable() -> None:
    model = _model().eval()
    values = _inputs()
    original = model(**values)
    shuffled = dict(values)
    shuffled["history"] = values["history"].flip(1)
    shuffled_output = model(**shuffled)
    assert not torch.allclose(
        original.attention_weights, shuffled_output.attention_weights
    )
