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

from model.adaptive_joint_ranker import (  # noqa: E402
    MODES,
    AdaptiveJointLMRanker,
    listwise_loss,
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


def make_model(mode: str, *, zero: bool = False) -> AdaptiveJointLMRanker:
    torch.manual_seed(9)
    return AdaptiveJointLMRanker(
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
    )


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(13)
    b, c, h, length = 2, 5, 3, 6
    query_ids = torch.randint(1, 63, (b, length))
    candidate_ids = torch.randint(1, 63, (b, c, length))
    history_ids = torch.randint(1, 63, (b, h, length))
    query_mask = torch.ones_like(query_ids, dtype=torch.bool)
    candidate_attention = torch.ones_like(candidate_ids, dtype=torch.bool)
    history_attention = torch.ones_like(history_ids, dtype=torch.bool)
    return {
        "query_input_ids": query_ids,
        "query_attention_mask": query_mask,
        "query_content_mask": query_mask,
        "candidate_input_ids": candidate_ids,
        "candidate_attention_mask": candidate_attention,
        "candidate_content_mask": candidate_attention,
        "history_input_ids": history_ids,
        "history_attention_mask": history_attention,
        "history_content_mask": history_attention,
        "history_event_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "candidate_mask": torch.tensor([[True] * 5, [True, True, True, False, False]]),
        "base_scores": torch.randn(b, c),
        "item_only_scores": torch.randn(b, c),
        "repeat_request": torch.zeros(b, dtype=torch.bool),
        "query_present": torch.ones(b, dtype=torch.bool),
    }


def test_only_last_two_backbone_layers_are_trainable() -> None:
    adaptive = make_model("adaptive_history_lm")
    layers = adaptive.backbone.encoder.layer
    assert not any(parameter.requires_grad for layer in layers[:2] for parameter in layer.parameters())
    assert all(parameter.requires_grad for layer in layers[-2:] for parameter in layer.parameters())
    assert not adaptive.backbone.embeddings.weight.requires_grad
    frozen = make_model("frozen_history_lm")
    assert not any(parameter.requires_grad for parameter in frozen.backbone.parameters())


def test_primary_nohistory_and_repeat_fallbacks_are_exact() -> None:
    model = make_model("adaptive_history_lm").eval()
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


def test_query_candidate_control_is_active_without_history() -> None:
    model = make_model("adaptive_query_candidate_lm").eval()
    values = batch()
    values["history_event_mask"] = torch.zeros_like(values["history_event_mask"])
    output = model(**values)
    assert bool(output.active_request.all())
    assert bool(output.correction.ne(0).any())


def test_candidate_permutation_equivariance_and_mode_parameter_parity() -> None:
    values = batch()
    counts = []
    for mode in MODES:
        model = make_model(mode).eval()
        counts.append(model.parameter_count())
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
    assert len(set(counts)) == 1


def test_ranking_gradient_reaches_last_lm_layers_and_joint_ranker() -> None:
    model = make_model("adaptive_history_lm", zero=True).train()
    values = batch()
    labels = torch.zeros_like(values["base_scores"])
    labels[:, 0] = 1.0
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad], lr=1e-2
    )
    active: set[str] = set()
    for _ in range(3):
        output = model(**values)
        loss, _ = listwise_loss(
            output, labels, values["candidate_mask"], correction_l2_weight=0.01
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and bool(parameter.grad.ne(0).any()):
                active.add(name)
        optimizer.step()
    assert any(name.startswith("backbone.encoder.layer.2.") for name in active)
    assert any(name.startswith("backbone.encoder.layer.3.") for name in active)
    assert not any(name.startswith("backbone.encoder.layer.0.") for name in active)
    assert any(name.startswith("joint_transformer.") for name in active)
    assert any(name.startswith("output_head.") for name in active)
