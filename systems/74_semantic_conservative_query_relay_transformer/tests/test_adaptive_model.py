from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.adaptive_semantic_relay import (  # noqa: E402
    MODES,
    AdaptiveSemanticRelayLMRanker,
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


def model(mode: str) -> AdaptiveSemanticRelayLMRanker:
    torch.manual_seed(7)
    return AdaptiveSemanticRelayLMRanker(
        backbone=FakeBackbone(), mode=mode, trainable_last_lm_layers=2,
        input_dim=16, route_rank=4, max_history=3, temperature=0.1,
        profile_scale=1.0, correction_scale=3.0, route_init_std=0.02,
    )


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    b, c, h, length = 2, 5, 3, 6
    q = torch.randint(1, 63, (b, length))
    ci = torch.randint(1, 63, (b, c, length))
    hi = torch.randint(1, 63, (b, h, length))
    qm = torch.ones_like(q, dtype=torch.bool)
    cm = torch.ones_like(ci, dtype=torch.bool)
    hm = torch.ones_like(hi, dtype=torch.bool)
    return {
        "query_input_ids": q, "query_attention_mask": qm, "query_content_mask": qm,
        "candidate_input_ids": ci, "candidate_attention_mask": cm, "candidate_content_mask": cm,
        "history_input_ids": hi, "history_attention_mask": hm, "history_content_mask": hm,
        "history_event_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "candidate_mask": torch.tensor([[True] * 5, [True, True, True, False, False]]),
        "base_scores": torch.randn(b, c), "item_only_scores": torch.randn(b, c),
        "repeat_request": torch.zeros(b, dtype=torch.bool),
        "query_present": torch.ones(b, dtype=torch.bool),
    }


def test_last_two_lm_layers_and_mode_capacity() -> None:
    counts = set()
    for mode in MODES:
        value = model(mode)
        counts.add((value.parameter_count(), value.trainable_parameter_count()))
        assert not any(p.requires_grad for layer in value.backbone.encoder.layer[:2] for p in layer.parameters())
        assert all(p.requires_grad for layer in value.backbone.encoder.layer[-2:] for p in layer.parameters())
    assert len(counts) == 1


def test_exact_fallbacks_and_query_mask() -> None:
    value = model(MODES[0]).eval()
    rows = batch()
    nohistory = dict(rows); nohistory["history_event_mask"] = torch.zeros_like(rows["history_event_mask"])
    out = value(**nohistory)
    assert torch.equal(out.scores, rows["base_scores"].masked_fill(~rows["candidate_mask"], 0.0))
    masked = dict(rows); masked["query_present"] = torch.zeros_like(rows["query_present"])
    out = value(**masked)
    assert torch.equal(out.scores, rows["base_scores"].masked_fill(~rows["candidate_mask"], 0.0))
    repeat = dict(rows); repeat["repeat_request"] = torch.ones_like(rows["repeat_request"])
    out = value(**repeat)
    assert torch.equal(out.scores, rows["item_only_scores"].masked_fill(~rows["candidate_mask"], 0.0))


def test_candidate_permutation_and_gradients() -> None:
    rows = batch()
    value = model(MODES[0]).train()
    labels = torch.zeros_like(rows["base_scores"]); labels[:, 0] = 1.0
    optimizer = torch.optim.AdamW([p for p in value.parameters() if p.requires_grad], lr=1e-2)
    active = set()
    for _ in range(3):
        output = value(**rows); loss = listwise_loss(output, labels, rows["candidate_mask"])
        optimizer.zero_grad(); loss.backward()
        active |= {n for n, p in value.named_parameters() if p.grad is not None and bool(p.grad.ne(0).any())}
        optimizer.step()
    assert any(name.startswith("backbone.encoder.layer.2.") for name in active)
    assert "history_route.down.weight" in active and "candidate_route.down.weight" in active
    value.eval(); original = value(**rows).scores
    reverse = torch.arange(4, -1, -1); changed = dict(rows)
    for name in ("candidate_input_ids", "candidate_attention_mask", "candidate_content_mask", "candidate_mask", "base_scores", "item_only_scores"):
        changed[name] = rows[name][:, reverse]
    assert torch.allclose(original, value(**changed).scores[:, reverse], atol=2e-6, rtol=0.0)
