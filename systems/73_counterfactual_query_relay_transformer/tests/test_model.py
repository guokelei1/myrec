from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.query_relay import MODES, CounterfactualQueryRelayTransformer  # noqa: E402
from probe.synthetic import make_dataset, query_masked  # noqa: E402


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/design_gate.yaml").read_text())


def model(mode: str) -> CounterfactualQueryRelayTransformer:
    row = config()
    return CounterfactualQueryRelayTransformer(
        input_dim=row["data"]["dimension"],
        hidden_dim=row["model"]["hidden_dimension"],
        heads=row["model"]["heads"],
        ffn_dim=row["model"]["ffn_dimension"],
        max_history=row["data"]["history_events"],
        mode=mode,
        dropout=0.0,
        correction_cap=row["model"]["correction_cap"],
    )


def test_modes_have_equal_parameters_and_finite_gradients() -> None:
    row = config()
    data = make_dataset(row, examples=64, seed=17, split="train")
    counts = set()
    for mode in MODES:
        value = model(mode)
        output = value(**data.forward_kwargs())
        output.scores.sum().backward()
        counts.add(value.parameter_count())
        assert torch.isfinite(output.scores).all()
        assert value.output_head.weight.grad is not None
    assert len(counts) == 1


def test_exact_fallback_and_query_mask() -> None:
    row = config()
    data = make_dataset(row, examples=128, seed=19, split="validation")
    value = model("counterfactual_query_relay").eval()
    output = value(**data.forward_kwargs())
    assert torch.equal(
        output.scores[data.no_history_request],
        data.base_scores[data.no_history_request],
    )
    assert torch.equal(
        output.scores[data.repeat_request],
        data.item_only_scores[data.repeat_request],
    )
    masked = query_masked(data)
    masked_output = value(**masked.forward_kwargs())
    nonrepeat = ~data.repeat_request
    assert torch.equal(masked_output.scores[nonrepeat], data.base_scores[nonrepeat])


def test_primary_is_candidate_permutation_equivariant() -> None:
    row = config()
    data = make_dataset(row, examples=32, seed=23, split="validation")
    value = model("counterfactual_query_relay").eval()
    original = value(**data.forward_kwargs()).scores
    reverse = torch.arange(data.candidate_tokens.shape[1] - 1, -1, -1)
    kwargs = data.forward_kwargs()
    for name in (
        "candidate_tokens",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        kwargs[name] = kwargs[name][:, reverse]
    changed = value(**kwargs).scores[:, reverse]
    assert torch.allclose(original, changed, atol=1e-6, rtol=0.0)
