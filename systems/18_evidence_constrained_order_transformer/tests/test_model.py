from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.ecot import ECOTRanker, protected_margin_violation, soft_constraint_penalty


def make_model(mode: str = "projection") -> ECOTRanker:
    torch.manual_seed(1801)
    return ECOTRanker(
        input_dim=16,
        d_model=24,
        nhead=4,
        layers=2,
        ffn_dim=48,
        history_slots=4,
        dropout=0.0,
        proposal_radius=1.25,
        repeat_bonus=0.6,
        projection_bisection_steps=48,
        mode=mode,
    )


def inputs(history: bool = True) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1802)
    return {
        "query": torch.randn(3, 16, generator=generator),
        "candidates": torch.randn(3, 5, 16, generator=generator),
        "history": torch.randn(3, 4, 16, generator=generator),
        "history_mask": torch.full((3, 4), history, dtype=torch.bool),
        "repeat_mask": torch.tensor(
            [[True, False, False, False, False], [False] * 5, [False, True, False, False, False]]
        ),
        "candidate_mask": torch.ones(3, 5, dtype=torch.bool),
    }


def state_hash(model: ECOTRanker) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def test_no_history_is_bitwise_query_only_for_every_mode() -> None:
    values = inputs(history=False)
    for mode in sorted(ECOTRanker.VALID_MODES):
        model = make_model(mode).eval()
        output = model(**values)
        base, _ = model.query_only(values["query"], values["candidates"])
        assert torch.equal(output.scores, base)
        assert torch.equal(output.anchor_scores, base)
        assert torch.equal(output.proposal_scores, base)


def test_trainable_modes_have_identical_parameterization_and_initialization() -> None:
    models = {mode: make_model(mode) for mode in sorted(ECOTRanker.VALID_MODES)}
    counts = {mode: sum(value.numel() for value in model.parameters()) for mode, model in models.items()}
    assert len(set(counts.values())) == 1
    names = {mode: [(name, tuple(value.shape)) for name, value in model.named_parameters()] for mode, model in models.items()}
    assert len({tuple(value) for value in names.values()}) == 1
    assert len({state_hash(model) for model in models.values()}) == 1


def test_projection_satisfies_anchor_margins_and_soft_penalty_detects_direct_violation() -> None:
    model = make_model("projection").eval()
    values = inputs(history=True)
    projected = model(**values)
    violation = protected_margin_violation(
        projected.scores, projected.anchor_scores, values["repeat_mask"]
    )
    assert violation.max() <= 1e-5
    direct = model(**values, mode="direct")
    penalty = soft_constraint_penalty(
        direct.scores, direct.anchor_scores, values["repeat_mask"]
    )
    assert torch.isfinite(penalty)
    assert penalty >= 0


def test_candidate_permutation_equivariance() -> None:
    model = make_model().eval()
    values = inputs(history=True)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    original = model(**values).scores
    changed_values = dict(values)
    changed_values["candidates"] = values["candidates"][:, permutation]
    changed_values["repeat_mask"] = values["repeat_mask"][:, permutation]
    changed_values["candidate_mask"] = values["candidate_mask"][:, permutation]
    changed = model(**changed_values).scores
    assert torch.allclose(changed, original[:, permutation], atol=1e-5, rtol=0.0)


def test_full_backward_has_finite_active_gradients() -> None:
    model = make_model().train()
    values = inputs(history=True)
    output = model(**values)
    targets = torch.tensor([0, 2, 1])
    loss = F.cross_entropy(output.scores, targets)
    loss.backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert sum(float(value.abs().sum()) for value in gradients) > 0
