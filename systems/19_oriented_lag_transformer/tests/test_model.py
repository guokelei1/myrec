from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.olt import OLTRanker


def make_model(mode: str = "oriented") -> OLTRanker:
    torch.manual_seed(1901)
    return OLTRanker(
        input_dim=16,
        d_model=24,
        nhead=4,
        layers=2,
        ffn_dim=48,
        history_slots=5,
        dropout=0.0,
        affinity_dim=12,
        temperature=0.35,
        identity_bias=3.0,
        evidence_scale=2.0,
        lag_scale_max=1.0,
        mode=mode,
    )


def inputs(history_present: bool = True) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1902)
    return {
        "query": torch.randn(3, 16, generator=generator),
        "candidates": torch.randn(3, 6, 16, generator=generator),
        "history": torch.randn(3, 5, 16, generator=generator),
        "history_mask": torch.full((3, 5), history_present, dtype=torch.bool),
        "identity_relation": torch.zeros(3, 6, 5, dtype=torch.bool),
        "candidate_mask": torch.ones(3, 6, dtype=torch.bool),
    }


def state_hash(model: OLTRanker) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def test_no_history_is_bitwise_query_only_for_all_modes() -> None:
    values = inputs(False)
    for mode in sorted(OLTRanker.VALID_MODES):
        model = make_model(mode).eval()
        output = model(**values)
        base, _, _ = model.query_only(
            values["query"], values["candidates"], values["candidate_mask"]
        )
        assert torch.equal(output.scores, base)
        assert torch.equal(output.evidence, torch.zeros_like(output.evidence))


def test_modes_have_identical_parameters_and_initialization() -> None:
    models = {mode: make_model(mode) for mode in sorted(OLTRanker.VALID_MODES)}
    counts = {mode: sum(value.numel() for value in model.parameters()) for mode, model in models.items()}
    assert len(set(counts.values())) == 1
    names = {mode: tuple((name, tuple(value.shape)) for name, value in model.named_parameters()) for mode, model in models.items()}
    assert len(set(names.values())) == 1
    assert len({state_hash(model) for model in models.values()}) == 1


def test_candidate_permutation_equivariance() -> None:
    model = make_model().eval()
    values = inputs(True)
    values["identity_relation"][0, 2, 3] = True
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    original = model(**values).scores
    changed_values = dict(values)
    changed_values["candidates"] = values["candidates"][:, permutation]
    changed_values["candidate_mask"] = values["candidate_mask"][:, permutation]
    changed_values["identity_relation"] = values["identity_relation"][:, permutation]
    changed = model(**changed_values).scores
    assert torch.allclose(changed, original[:, permutation], atol=1e-5, rtol=0.0)


def test_identical_candidate_traces_have_zero_centered_evidence() -> None:
    model = make_model().eval()
    values = inputs(True)
    values["candidates"] = values["candidates"][:, :1].expand(-1, 6, -1).clone()
    output = model(**values)
    assert torch.allclose(output.candidate_trace, output.candidate_trace[:, :1].expand_as(output.candidate_trace))
    assert output.evidence.abs().max() <= 1e-6


def test_full_backward_is_finite_and_reaches_affinity_projections() -> None:
    model = make_model().train()
    values = inputs(True)
    output = model(**values)
    loss = F.cross_entropy(output.scores, torch.tensor([0, 2, 4]))
    loss.backward()
    for name in (
        "query_affinity.weight",
        "candidate_affinity.weight",
        "history_query_key.weight",
        "history_candidate_key.weight",
        "evidence_write.1.weight",
        "evidence_write.3.weight",
    ):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_evidence_can_only_change_scores_through_hidden_state_write() -> None:
    model = make_model().eval()
    values = inputs(True)
    with torch.no_grad():
        model.evidence_write[3].weight.zero_()
        model.evidence_write[3].bias.zero_()
    output = model(**values)
    assert output.evidence.abs().max() > 0
    assert torch.equal(output.scores, output.base_scores)
