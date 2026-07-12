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

from model.canonical_residual import (  # noqa: E402
    CanonicalCounterfactualResidualStateTransformer,
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


def make_model() -> CanonicalCounterfactualResidualStateTransformer:
    torch.manual_seed(31)
    return CanonicalCounterfactualResidualStateTransformer(
        backbone=FakeBackbone(),
        mode="hidden_residual_wrong_neutral",
        trainable_last_lm_layers=2,
        input_dim=16,
        hidden_dim=24,
        heads=4,
        layers=1,
        ffn_dim=48,
        dropout=0.0,
        max_history=5,
        zero_initial_output=False,
        null_reference_stop_gradient=True,
    ).eval()


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(32)
    b, c, h, length = 2, 5, 3, 6
    query = torch.randint(1, 63, (b, length))
    candidate = torch.randint(1, 63, (b, c, length))
    history = torch.randint(1, 63, (b, h, length))
    wrong = torch.randint(1, 63, (b, h, length))
    query_mask = torch.ones_like(query, dtype=torch.bool)
    candidate_attention = torch.ones_like(candidate, dtype=torch.bool)
    history_attention = torch.ones_like(history, dtype=torch.bool)
    return {
        "candidate_keys": torch.tensor([[50, 10, 40, 20, 30], [9, 4, 7, 0, 0]], dtype=torch.int64),
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
        "wrong_history_attention_mask": history_attention,
        "wrong_history_content_mask": history_attention,
        "wrong_history_event_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "candidate_mask": torch.tensor([[True] * 5, [True, True, True, False, False]]),
        "base_scores": torch.randn(b, c),
        "item_only_scores": torch.randn(b, c),
        "repeat_request": torch.zeros(b, dtype=torch.bool),
        "query_present": torch.ones(b, dtype=torch.bool),
    }


def test_canonical_wrapper_is_exact_under_caller_permutation() -> None:
    model = make_model()
    values = batch()
    output = model(**values)
    permutation = torch.tensor([4, 3, 2, 1, 0])
    reversed_values = dict(values)
    for name in (
        "candidate_keys",
        "candidate_input_ids",
        "candidate_attention_mask",
        "candidate_content_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        reversed_values[name] = values[name][:, permutation]
    reversed_output = model(**reversed_values)
    for name in (
        "scores",
        "correction",
        "wrong_scores",
        "wrong_correction",
        "state_residual",
        "wrong_state_residual",
    ):
        assert torch.equal(getattr(output, name), getattr(reversed_output, name)[:, permutation])


def test_wrapper_adds_no_parameters() -> None:
    model = make_model()
    assert model.parameter_count() == model.inner.parameter_count()
    assert model.trainable_parameter_count() == model.inner.trainable_parameter_count()


def test_duplicate_valid_keys_are_rejected() -> None:
    model = make_model()
    values = batch()
    values["candidate_keys"][0, 1] = values["candidate_keys"][0, 0]
    try:
        model(**values)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("C66 duplicate key was accepted")
