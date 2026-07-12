from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import torch
from torch import nn


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from model.counterfactual_residual import (  # noqa: E402
    MODES,
    CounterfactualResidualStateTransformer,
    counterfactual_training_loss,
)


class FakeBackbone(nn.Module):
    def __init__(self, width: int = 16, layers: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width)
        self.embeddings = nn.Embedding(64, width)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        value = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            value = value + torch.tanh(layer(value))
        return SimpleNamespace(last_hidden_state=value * attention_mask[..., None])


def make_model(mode: str, *, zero: bool = False) -> CounterfactualResidualStateTransformer:
    torch.manual_seed(21)
    return CounterfactualResidualStateTransformer(
        backbone=FakeBackbone(),
        mode=mode,
        trainable_last_lm_layers=2,
        input_dim=16,
        hidden_dim=24,
        heads=4,
        layers=1,
        ffn_dim=48,
        dropout=0.0,
        max_history=5,
        zero_initial_output=zero,
        null_reference_stop_gradient=True,
    )


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(22)
    b, c, h, length = 2, 5, 3, 6
    query = torch.randint(1, 63, (b, length))
    candidate = torch.randint(1, 63, (b, c, length))
    history = torch.randint(1, 63, (b, h, length))
    wrong = torch.randint(1, 63, (b, h, length))
    query_mask = torch.ones_like(query, dtype=torch.bool)
    candidate_attention = torch.ones_like(candidate, dtype=torch.bool)
    history_attention = torch.ones_like(history, dtype=torch.bool)
    wrong_attention = torch.ones_like(wrong, dtype=torch.bool)
    return {
        "query_input_ids": query,
        "query_attention_mask": query_mask,
        "query_content_mask": query_mask,
        "candidate_input_ids": candidate,
        "candidate_attention_mask": candidate_attention,
        "candidate_content_mask": candidate_attention,
        "history_input_ids": history,
        "history_attention_mask": history_attention,
        "history_content_mask": history_attention,
        "history_event_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "wrong_history_input_ids": wrong,
        "wrong_history_attention_mask": wrong_attention,
        "wrong_history_content_mask": wrong_attention,
        "wrong_history_event_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "candidate_mask": torch.tensor([[True] * 5, [True, True, True, False, False]]),
        "base_scores": torch.randn(b, c),
        "item_only_scores": torch.randn(b, c),
        "repeat_request": torch.zeros(b, dtype=torch.bool),
        "query_present": torch.ones(b, dtype=torch.bool),
    }


def test_nohistory_and_repeat_are_exact() -> None:
    model = make_model("hidden_residual_wrong_neutral").eval()
    values = batch()
    values["history_event_mask"] = torch.zeros_like(values["history_event_mask"])
    output = model(**values)
    expected = values["base_scores"].masked_fill(~values["candidate_mask"], 0.0)
    assert torch.equal(output.scores, expected)
    assert torch.count_nonzero(output.correction) == 0

    values = batch()
    values["repeat_request"] = torch.ones(2, dtype=torch.bool)
    output = model(**values)
    expected = values["item_only_scores"].masked_fill(~values["candidate_mask"], 0.0)
    assert torch.equal(output.scores, expected)


def test_hidden_residual_is_history_specific_and_wrong_specific() -> None:
    model = make_model("hidden_residual_wrong_neutral").eval()
    output = model(**batch())
    assert bool(output.state_residual.ne(0).any())
    assert bool(output.wrong_state_residual.ne(0).any())
    assert not torch.equal(output.state_residual, output.wrong_state_residual)
    assert bool(output.correction.ne(0).any())


def test_modes_have_equal_parameters_and_distinct_readouts() -> None:
    values = batch()
    outputs = {}
    counts = []
    for mode in MODES:
        model = make_model(mode).eval()
        counts.append(model.parameter_count())
        outputs[mode] = model(**values).scores
    assert len(set(counts)) == 1
    assert not torch.equal(
        outputs["hidden_residual_wrong_neutral"],
        outputs["ordinary_factual_wrong_neutral"],
    )
    assert not torch.equal(
        outputs["hidden_residual_wrong_neutral"],
        outputs["logit_difference_wrong_neutral"],
    )


def test_candidate_permutation_equivariance() -> None:
    model = make_model("hidden_residual_wrong_neutral").eval()
    values = batch()
    output = model(**values)
    permutation = torch.tensor([4, 3, 2, 1, 0])
    reversed_values = dict(values)
    for name in (
        "candidate_input_ids",
        "candidate_attention_mask",
        "candidate_content_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        reversed_values[name] = values[name][:, permutation]
    reversed_output = model(**reversed_values)
    assert torch.allclose(
        output.scores,
        reversed_output.scores[:, permutation],
        atol=2e-6,
        rtol=0.0,
    )


def test_counterfactual_loss_reaches_adaptive_lm_joint_and_head() -> None:
    model = make_model("hidden_residual_wrong_neutral", zero=True).train()
    model.backbone.eval()
    values = batch()
    labels = torch.zeros_like(values["base_scores"])
    labels[:, 0] = 1.0
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad], lr=1e-2
    )
    active: set[str] = set()
    for _ in range(3):
        output = model(**values)
        loss, components = counterfactual_training_loss(
            output,
            labels,
            values["candidate_mask"],
            correction_l2_weight=0.01,
            wrong_neutrality_weight=0.25,
        )
        assert all(torch.isfinite(value) for value in components.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and bool(parameter.grad.ne(0).any()):
                active.add(name)
        optimizer.step()
    assert any(name.startswith("core.backbone.encoder.layer.2.") for name in active)
    assert any(name.startswith("core.backbone.encoder.layer.3.") for name in active)
    assert any(name.startswith("core.joint_transformer.") for name in active)
    assert any(name.startswith("core.output_head.") for name in active)
    assert any(name.startswith("residual_norm.") for name in active)
