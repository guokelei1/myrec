from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

import model.authenticated_mediation as architecture  # noqa: E402


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, model_type="fake")
        self.embedding = nn.Embedding(20, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        state = self.dropout(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=state)


def test_neutral_initialization_and_exact_null_cancellation(monkeypatch) -> None:
    monkeypatch.setattr(
        architecture.AutoModel, "from_pretrained", lambda *args, **kwargs: FakeEncoder()
    )
    model = architecture.AuthenticatedMediationTransformer(
        "unused", mode=architecture.PRIMARY, correction_cap=2.0
    )
    identity = model.identity()
    assert identity["head_initialized_exact_zero"] is True
    assert identity["dropout_disabled"] is True
    ids = torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]])
    attention = torch.ones_like(ids)
    correction = model.correction_from_paired_logits(model(ids, attention))
    assert torch.equal(correction, torch.zeros_like(correction))


def test_primary_and_unauthenticated_control_have_matched_capacity(monkeypatch) -> None:
    monkeypatch.setattr(
        architecture.AutoModel, "from_pretrained", lambda *args, **kwargs: FakeEncoder()
    )
    primary = architecture.AuthenticatedMediationTransformer(
        "unused", mode="authenticated_mediation", correction_cap=2.0
    )
    control = architecture.AuthenticatedMediationTransformer(
        "unused", mode="unauthenticated_mediation", correction_cap=2.0
    )
    assert primary.parameter_count() == control.parameter_count()
    assert primary.uses_authentication is True
    assert control.uses_authentication is False
