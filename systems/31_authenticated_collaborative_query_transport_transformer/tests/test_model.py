from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from types import SimpleNamespace


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

import model.query_transport as architecture  # noqa: E402
from model.query_transport import FrozenBGETransportRanker, LowRankQueryTransport  # noqa: E402


def make_model(seed: int = 7) -> LowRankQueryTransport:
    return LowRankQueryTransport(
        dim=8,
        rank=2,
        temperature=0.1,
        profile_scale=1.0,
        correction_scale=2.0,
        seed=seed,
    )


def test_no_history_is_exact_null_and_capacity_is_only_shared_adapter() -> None:
    model = make_model()
    query = F.normalize(torch.randn(8), dim=0)
    candidates = F.normalize(torch.randn(4, 8), dim=1)
    correction = model(query, torch.empty(0, 8), candidates)
    assert torch.equal(correction, torch.zeros_like(correction))
    assert model.trainable_parameter_count() == 2 * 8 * 2
    assert set(dict(model.named_parameters())) == {"down.weight", "up.weight"}
    assert torch.equal(model.up.weight, torch.zeros_like(model.up.weight))


def test_one_transported_query_is_candidate_permutation_equivariant() -> None:
    model = make_model()
    query = F.normalize(torch.randn(8), dim=0)
    history = F.normalize(torch.randn(3, 8), dim=1)
    candidates = F.normalize(torch.randn(5, 8), dim=1)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    first = model(query, history, candidates)
    second = model(query, history, candidates[permutation])
    # BLAS may change the final bit when the row layout changes.  The formal
    # scorer canonicalizes rows before this operation and requires exact output.
    assert torch.allclose(second, first[permutation], atol=1e-6, rtol=0)


def test_all_candidates_share_adapter_gradients() -> None:
    model = make_model()
    query = F.normalize(torch.randn(8), dim=0)
    history = F.normalize(torch.randn(3, 8), dim=1)
    candidates = F.normalize(torch.randn(5, 8), dim=1)
    loss = model(query, history, candidates).square().sum()
    loss.backward()
    assert model.up.weight.grad is not None
    assert bool(model.up.weight.grad.ne(0).any())


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_reference_ranker_keeps_transformer_in_the_scoring_path(monkeypatch) -> None:
    monkeypatch.setattr(
        architecture.AutoModel, "from_pretrained", lambda *args, **kwargs: FakeEncoder()
    )
    ranker = FrozenBGETransportRanker("unused", make_model())
    query = torch.tensor([[1, 2]])
    history = torch.tensor([[3, 4], [5, 6]])
    candidates = torch.tensor([[7, 8], [9, 10], [11, 12]])
    scores = ranker(
        query,
        torch.ones_like(query),
        history,
        torch.ones_like(history),
        candidates,
        torch.ones_like(candidates),
    )
    assert scores.shape == (3,)
    assert not any(parameter.requires_grad for parameter in ranker.encoder.parameters())
    assert all(parameter.requires_grad for parameter in ranker.transport.parameters())
