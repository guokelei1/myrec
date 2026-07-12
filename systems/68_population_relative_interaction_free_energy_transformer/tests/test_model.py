from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import (  # noqa: E402
    MODES,
    PopulationRelativeFreeEnergyRanker,
    interaction_from_energies,
)


def _model(mode: str) -> PopulationRelativeFreeEnergyRanker:
    torch.manual_seed(7)
    return PopulationRelativeFreeEnergyRanker(
        input_dim=8,
        hidden_dim=16,
        heads=4,
        layers=1,
        ffn_dim=32,
        temperature=0.4,
        mode=mode,
    ).double()


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    return {
        "query": torch.randn(3, 8, dtype=torch.float64),
        "candidates": torch.randn(3, 5, 8, dtype=torch.float64),
        "history": torch.randn(3, 4, 8, dtype=torch.float64),
        "reference": torch.randn(3, 7, 8, dtype=torch.float64),
        "base_scores": torch.randn(3, 5, dtype=torch.float64),
    }


def test_candidate_only_additive_energy_cancels() -> None:
    torch.manual_seed(1)
    u = torch.randn(2, 4, 3, dtype=torch.float64)
    r = torch.randn(2, 4, 5, dtype=torch.float64)
    u0 = torch.randn(2, 1, 3, dtype=torch.float64)
    r0 = torch.randn(2, 1, 5, dtype=torch.float64)
    offset = torch.randn(2, 4, 1, dtype=torch.float64)
    base = interaction_from_energies(
        u, r, u0, r0, mode="interaction_free_energy", temperature=0.3
    )
    shifted = interaction_from_energies(
        u + offset,
        r + offset,
        u0,
        r0,
        mode="interaction_free_energy",
        temperature=0.3,
    )
    torch.testing.assert_close(base, shifted, atol=1e-12, rtol=1e-12)


def test_separable_energy_is_candidate_constant_and_centers_to_zero() -> None:
    torch.manual_seed(2)
    a = torch.randn(2, 5, 1, dtype=torch.float64)
    b_user = torch.randn(2, 1, 4, dtype=torch.float64)
    b_ref = torch.randn(2, 1, 7, dtype=torch.float64)
    a0 = torch.randn(2, 1, 1, dtype=torch.float64)
    raw = interaction_from_energies(
        a + b_user,
        a + b_ref,
        a0 + b_user,
        a0 + b_ref,
        mode="interaction_free_energy",
        temperature=0.5,
    )
    centered = raw - raw.mean(dim=1, keepdim=True)
    torch.testing.assert_close(centered, torch.zeros_like(centered), atol=1e-12, rtol=0)


def test_equal_event_sets_cancel_exactly() -> None:
    torch.manual_seed(3)
    u = torch.randn(2, 5, 6, dtype=torch.float64)
    u0 = torch.randn(2, 1, 6, dtype=torch.float64)
    raw = interaction_from_energies(
        u, u.clone(), u0, u0.clone(),
        mode="interaction_free_energy", temperature=0.25,
    )
    torch.testing.assert_close(raw, torch.zeros_like(raw), atol=0, rtol=0)


def test_nohistory_querymask_and_repeat_are_exact_fallbacks() -> None:
    model = _model("interaction_free_energy")
    batch = _batch()
    off = torch.zeros(3, dtype=torch.bool)
    out = model(**batch, history_present=off)
    torch.testing.assert_close(out.scores, batch["base_scores"], atol=0, rtol=0)
    torch.testing.assert_close(out.correction, torch.zeros_like(out.correction), atol=0, rtol=0)

    out = model(**batch, query_present=off)
    torch.testing.assert_close(out.scores, batch["base_scores"], atol=0, rtol=0)

    repeat_scores = torch.randn_like(batch["base_scores"])
    out = model(
        **batch,
        repeat_mask=torch.ones(3, dtype=torch.bool),
        repeat_scores=repeat_scores,
    )
    torch.testing.assert_close(out.scores, repeat_scores, atol=0, rtol=0)


def test_candidate_permutation_equivariance_all_modes() -> None:
    batch = _batch()
    perm = torch.tensor([4, 2, 0, 3, 1])
    inv = torch.argsort(perm)
    for mode in MODES:
        model = _model(mode)
        direct = model(**batch).scores
        shuffled = model(
            query=batch["query"],
            candidates=batch["candidates"][:, perm],
            history=batch["history"],
            reference=batch["reference"],
            base_scores=batch["base_scores"][:, perm],
        ).scores[:, inv]
        torch.testing.assert_close(direct, shuffled, atol=2e-12, rtol=1e-12)


def test_all_parameter_groups_receive_gradient_all_modes() -> None:
    batch = _batch()
    for mode in MODES:
        model = _model(mode)
        out = model(**batch)
        loss = out.scores.square().mean()
        loss.backward()
        groups = {
            "input_projection": False,
            "token_type": False,
            "null_candidate": False,
            "triplet_transformer": False,
            "output_norm": False,
            "energy_head": False,
        }
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                continue
            active = bool(parameter.grad.abs().sum() > 0)
            for group in groups:
                if name == group or name.startswith(group + "."):
                    groups[group] |= active
        assert all(groups.values()), (mode, groups)


def test_parameter_count_is_equal_across_modes() -> None:
    counts = {_model(mode).parameter_count for mode in MODES}
    assert len(counts) == 1
