from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.htct import HTCTRanker


def make_model(mode: str = "cone") -> HTCTRanker:
    torch.manual_seed(2001)
    return HTCTRanker(
        input_dim=16,
        d_model=24,
        nhead=4,
        ffn_dim=48,
        lower_layers=1,
        upper_layers=1,
        relation_dim=8,
        history_slots=7,
        dropout=0.0,
        solver_steps=12,
        ridge=0.02,
        evidence_scale_max=1.5,
        mode=mode,
    )


def inputs(history_events: int = 7) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2002)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    mask[:, :history_events] = True
    return {
        "query": torch.randn(3, 16, generator=generator),
        "candidates": torch.randn(3, 8, 16, generator=generator),
        "history": torch.randn(3, 7, 16, generator=generator),
        "history_mask": mask,
        "candidate_mask": torch.ones(3, 8, dtype=torch.bool),
    }


def state_hash(model: HTCTRanker) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def test_no_or_single_event_history_is_bitwise_base_for_all_modes() -> None:
    for events in (0, 1):
        values = inputs(events)
        for mode in sorted(HTCTRanker.VALID_MODES):
            output = make_model(mode).eval()(**values)
            assert torch.equal(output.scores, output.base_scores)
            assert torch.equal(output.reconstruction, torch.zeros_like(output.reconstruction))


def test_query_present_mask_is_exact_fallback() -> None:
    values = inputs(7)
    values["query_present"] = torch.zeros(3, dtype=torch.bool)
    output = make_model().eval()(**values)
    assert torch.equal(output.scores, output.base_scores)


def test_modes_have_identical_parameters_and_initialization() -> None:
    models = {mode: make_model(mode) for mode in sorted(HTCTRanker.VALID_MODES)}
    counts = {mode: sum(value.numel() for value in model.parameters()) for mode, model in models.items()}
    names = {mode: tuple((name, tuple(value.shape)) for name, value in model.named_parameters()) for mode, model in models.items()}
    assert len(set(counts.values())) == 1
    assert len(set(names.values())) == 1
    assert len({state_hash(model) for model in models.values()}) == 1


def test_candidate_permutation_equivariance() -> None:
    model = make_model().eval()
    values = inputs(7)
    permutation = torch.tensor([7, 2, 0, 5, 1, 6, 3, 4])
    original = model(**values).scores
    changed_values = dict(values)
    changed_values["candidates"] = values["candidates"][:, permutation]
    changed_values["candidate_mask"] = values["candidate_mask"][:, permutation]
    changed = model(**changed_values).scores
    torch.testing.assert_close(changed, original[:, permutation], atol=1e-5, rtol=0.0)


def test_zero_hidden_write_removes_score_effect_despite_nonzero_coefficients() -> None:
    model = make_model().eval()
    with torch.no_grad():
        model.relation_write[3].weight.zero_()
        model.relation_write[3].bias.zero_()
    output = model(**inputs(7))
    assert output.coefficients.abs().sum() > 0
    assert torch.equal(output.scores, output.base_scores)


def test_full_backward_is_finite_and_reaches_load_bearing_modules() -> None:
    model = make_model().train()
    values = inputs(7)
    values["query"].requires_grad_()
    values["candidates"].requires_grad_()
    values["history"].requires_grad_()
    output = model(**values)
    loss = F.cross_entropy(output.scores, torch.tensor([0, 3, 6]))
    loss.backward()
    parameters = dict(model.named_parameters())
    for name in (
        "input_projection.weight",
        "shared_token_encoder.1.weight",
        "lower_transformer.layers.0.self_attn.in_proj_weight",
        "upper_transformer.layers.0.self_attn.in_proj_weight",
        "relation_projection.weight",
        "relation_write.1.weight",
        "relation_write.3.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name
    for name in ("query", "candidates", "history"):
        gradient = values[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name
