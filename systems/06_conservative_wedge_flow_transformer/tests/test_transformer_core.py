from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.transformer_core import (  # noqa: E402
    BlockSparseWedgeFlowTransformerRanker,
)


def _model(*, num_layers: int = 2) -> BlockSparseWedgeFlowTransformerRanker:
    torch.manual_seed(211)
    return BlockSparseWedgeFlowTransformerRanker(
        input_dim=6,
        model_dim=8,
        num_heads=2,
        num_layers=num_layers,
        flow_dim=4,
        max_history_positions=5,
        ffn_dim=16,
        dropout=0.0,
        score_delta_max=0.6,
    )


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(223)
    return {
        "query_tokens": torch.randn(2, 3, 6),
        "candidate_tokens": torch.randn(2, 4, 2, 6),
        "history_tokens": torch.randn(2, 3, 6),
        "query_mask": torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        ),
        "candidate_token_mask": torch.tensor(
            [
                [[True, True], [True, False], [True, True], [True, True]],
                [[True, True], [True, True], [True, False], [False, False]],
            ],
            dtype=torch.bool,
        ),
        "history_mask": torch.tensor(
            [[True, True, True], [True, False, False]], dtype=torch.bool
        ),
        "history_prior": torch.tensor(
            [[1.0, 0.8, 0.6], [1.0, 0.0, 0.0]], dtype=torch.float32
        ),
    }


def _open_flow(
    model: BlockSparseWedgeFlowTransformerRanker,
    fraction: float = 0.8,
) -> None:
    with torch.no_grad():
        model.wedge_flow.raw_residual_scale.fill_(math.atanh(fraction))


def test_joint_fp32_base_is_trainable_but_strictly_history_blind() -> None:
    model = _model().train()
    values = _inputs()
    history = values["history_tokens"].clone().requires_grad_(True)
    values["history_tokens"] = history
    output = model(**values)
    assert output.base_scores.dtype == torch.float32
    history_gradient = torch.autograd.grad(
        output.base_scores.sum(),
        history,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert history_gradient is None or torch.count_nonzero(history_gradient) == 0

    model.zero_grad(set_to_none=True)
    output.base_scores.sum().backward()
    assert model.base_head.weight.grad is not None
    assert torch.count_nonzero(model.base_head.weight.grad) > 0


def test_open_wedge_is_the_only_nonzero_history_to_final_score_path() -> None:
    model = _model().eval()
    _open_flow(model)
    values = _inputs()
    history = values["history_tokens"].clone().requires_grad_(True)
    values["history_tokens"] = history
    output = model(**values)
    contrast = torch.tensor(
        [[1.0, -0.5, 0.25, -0.75], [-1.0, 1.0, 0.5, 0.0]]
    )
    final_gradient = torch.autograd.grad(
        (output.scores * contrast).sum(), history
    )[0]
    assert torch.isfinite(final_gradient).all()
    assert float(final_gradient.abs().sum()) > 0.0

    disabled_history = values["history_tokens"].detach().clone().requires_grad_(True)
    values["history_tokens"] = disabled_history
    disabled = model(**values, flow_enabled=False)
    disabled_gradient = torch.autograd.grad(
        (disabled.scores * contrast).sum(),
        disabled_history,
        allow_unused=True,
    )[0]
    assert disabled_gradient is None or torch.count_nonzero(disabled_gradient) == 0


def test_no_history_and_disabled_flow_are_exact_base_fallbacks() -> None:
    model = _model().eval()
    _open_flow(model)
    values = _inputs()
    disabled = model(**values, flow_enabled=False)
    assert torch.equal(disabled.scores, disabled.base_scores)
    assert torch.count_nonzero(disabled.applied_score_delta) == 0

    values["history_mask"] = torch.zeros_like(values["history_mask"])
    values["history_prior"] = torch.zeros_like(values["history_prior"])
    values["history_tokens"][:] = torch.nan
    empty = model(**values)
    assert torch.equal(empty.scores, empty.base_scores)
    assert torch.count_nonzero(empty.applied_score_delta) == 0


def test_whole_core_is_candidate_permutation_equivariant() -> None:
    model = _model().eval()
    _open_flow(model)
    values = _inputs()
    original = model(**values)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_values = dict(values)
    permuted_values["candidate_tokens"] = values["candidate_tokens"][
        :, permutation
    ]
    permuted_values["candidate_token_mask"] = values[
        "candidate_token_mask"
    ][:, permutation]
    permuted = model(**permuted_values)
    assert torch.allclose(
        permuted.base_scores,
        original.base_scores[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        permuted.scores,
        original.scores[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        permuted.candidate_states,
        original.candidate_states[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )


def test_padding_with_nonfinite_values_is_safe_even_for_all_masked_rows() -> None:
    model = _model().eval()
    _open_flow(model)
    values = _inputs()
    values["query_tokens"][0, 2] = torch.nan
    values["candidate_tokens"][0, 1, 1] = torch.inf
    values["candidate_tokens"][1, 3] = torch.nan
    values["history_tokens"][1, 1:] = torch.nan
    output = model(**values)
    assert torch.isfinite(output.token_states).all()
    assert torch.isfinite(output.base_scores).all()
    assert torch.isfinite(output.scores).all()
    pad_rows = output.token_roles == 0
    assert torch.count_nonzero(output.attention_mask[pad_rows]) == 0


def test_two_layer_barrier_has_no_history_or_cross_candidate_bypass() -> None:
    model = _model(num_layers=2).eval()
    values = _inputs()
    reference = model(**values)

    changed_history = dict(values)
    changed_history["history_tokens"] = (
        values["history_tokens"] * -17.0 + 31.0
    )
    history_variant = model(**changed_history)
    assert torch.equal(history_variant.base_scores, reference.base_scores)

    changed_candidate = dict(values)
    changed_candidate["candidate_tokens"] = values[
        "candidate_tokens"
    ].clone()
    changed_candidate["candidate_tokens"][:, 1] = (
        changed_candidate["candidate_tokens"][:, 1] * 19.0 - 7.0
    )
    candidate_variant = model(**changed_candidate)
    # Candidate 0 cannot receive candidate-1 information, directly or through
    # query tokens, even after multiple Transformer layers.
    assert torch.equal(
        candidate_variant.base_scores[:, 0], reference.base_scores[:, 0]
    )


def test_real_two_step_autograd_reaches_base_and_wedge_parameters() -> None:
    model = _model().train()
    _open_flow(model, 0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    values = _inputs()
    labels = torch.tensor(
        [[2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 1.0, 0.0]]
    )
    valid_candidates = values["candidate_token_mask"].any(dim=-1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(**values)
        masked_scores = output.scores.masked_fill(~valid_candidates, -torch.inf)
        targets = labels / labels.sum(dim=-1, keepdim=True)
        log_probability = torch.where(
            valid_candidates,
            torch.log_softmax(masked_scores, dim=-1),
            torch.zeros_like(masked_scores),
        )
        loss = -(targets * log_probability).sum()
        assert torch.isfinite(loss)
        loss.backward()
        assert model.base_head.weight.grad is not None
        assert torch.count_nonzero(model.base_head.weight.grad) > 0
        assert model.wedge_flow.factor_a_projection.weight.grad is not None
        assert torch.count_nonzero(
            model.wedge_flow.factor_a_projection.weight.grad
        ) > 0
        optimizer.step()
