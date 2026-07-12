from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.semantic_relay import MODES, SemanticConservativeQueryRelayTransformer  # noqa: E402
from probe.synthetic import make_dataset, query_masked, shuffled_history  # noqa: E402


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/design_gate.yaml").read_text())


def model(mode: str) -> SemanticConservativeQueryRelayTransformer:
    row = config()
    return SemanticConservativeQueryRelayTransformer(
        dim=row["data"]["dimension"],
        route_rank=row["model"]["route_rank"],
        max_history=row["data"]["history_events"],
        mode=mode,
        temperature=row["model"]["temperature"],
        profile_scale=row["model"]["profile_scale"],
        correction_scale=row["model"]["correction_scale"],
        route_init_std=row["model"]["route_init_std"],
    )


def test_modes_match_capacity_and_are_finite() -> None:
    row = config()
    data = make_dataset(row, examples=64, seed=41, split="train")
    counts = set()
    for mode in MODES:
        value = model(mode)
        output = value(**data.forward_kwargs())
        output.scores.sum().backward()
        counts.add(value.parameter_count())
        assert torch.isfinite(output.scores).all()
    assert len(counts) == 1


def test_exact_fallbacks_and_query_mask() -> None:
    row = config()
    data = make_dataset(row, examples=128, seed=43, split="validation")
    value = model("semantic_conservative_relay").eval()
    output = value(**data.forward_kwargs())
    assert torch.equal(output.scores[data.no_history_request], data.base_scores[data.no_history_request])
    assert torch.equal(output.scores[data.repeat_request], data.item_only_scores[data.repeat_request])
    masked = value(**query_masked(data).forward_kwargs())
    nonrepeat = ~data.repeat_request
    assert torch.equal(masked.scores[nonrepeat], data.base_scores[nonrepeat])


def test_chronology_bias_makes_event_order_load_bearing() -> None:
    row = config()
    data = make_dataset(row, examples=64, seed=47, split="validation")
    value = model("semantic_conservative_relay").eval()
    with torch.no_grad():
        value.chronology_bias.copy_(torch.linspace(-2.0, 2.0, data.history_tokens.shape[1]))
    clean = value(**data.forward_kwargs()).scores
    shuffled = value(**shuffled_history(data).forward_kwargs()).scores
    assert not torch.equal(clean[data.supported_request], shuffled[data.supported_request])


def test_candidate_permutation_equivariance() -> None:
    row = config()
    data = make_dataset(row, examples=32, seed=53, split="validation")
    value = model("semantic_conservative_relay").eval()
    original = value(**data.forward_kwargs()).scores
    reverse = torch.arange(data.candidate_tokens.shape[1] - 1, -1, -1)
    kwargs = data.forward_kwargs()
    for name in ("candidate_tokens", "candidate_mask", "base_scores", "item_only_scores"):
        kwargs[name] = kwargs[name][:, reverse]
    changed = value(**kwargs).scores[:, reverse]
    assert torch.allclose(original, changed, atol=1e-6, rtol=0.0)
