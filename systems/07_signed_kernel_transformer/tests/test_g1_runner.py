"""Pre-outcome structural tests for the executable C07 G1 protocol."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import g1_runner as g1  # noqa: E402


def fixed_hand_world(history_present: bool = True) -> g1.World:
    query = torch.zeros((2, g1.D))
    query[:, 0] = 1.0
    query[:, 5] = 1.0
    candidates = torch.zeros((2, g1.C, g1.D))
    candidates[:, :, : g1.C] = torch.eye(g1.C)[None]
    candidates[:, :, 5] = 1.0
    history_key = torch.zeros((2, g1.H, g1.D))
    history_key[:, 0, 0] = 2.0
    history_value = torch.zeros((2, g1.H, g1.D))
    history_value[:, 0, 0] = 1.0
    history_mask = torch.full((2, g1.H), history_present, dtype=torch.bool)
    exact = torch.zeros((2, g1.C, g1.H))
    exact[:, 0, 0] = 2.0
    target = torch.zeros(2, dtype=torch.long)
    foil = torch.ones(2, dtype=torch.long)
    subthreshold = torch.zeros((2, g1.H), dtype=torch.bool)
    return g1.World(
        query,
        candidates,
        history_key,
        history_value,
        history_mask,
        exact,
        target,
        foil,
        subthreshold,
    )


def test_old_lock_digest_constant_is_frozen() -> None:
    assert g1.LOCKED_MANIFEST == "66308db14f00e20de860a2060d147329fb93fa07b806951c227a5499746c2edd"
    assert g1.SEEDS == (20260711, 20260712, 20260713)
    assert (g1.C, g1.H, g1.D, g1.TAU, g1.KAPPA) == (5, 8, 16, 0.5, 1.0)
    verified = g1.verify_old_normative_lock(ROOT)
    assert verified["combined_manifest_sha256"] == g1.LOCKED_MANIFEST


def test_generator_is_deterministic_and_world_contracts_hold() -> None:
    first = g1.make_world(64, g1.cpu_generator(20260711, 101), "R")
    second = g1.make_world(64, g1.cpu_generator(20260711, 101), "R")
    assert g1.world_sha256(first) == g1.world_sha256(second)
    ranges = g1.oracle_ranges(first)
    assert torch.all(ranges[first.subthreshold_mask] <= g1.TAU)
    assert torch.all((ranges > g1.TAU).any(dim=1))

    supported = g1.make_world(64, g1.cpu_generator(20260711, 102), "S")
    supported_ranges = g1.oracle_ranges(supported)
    assert torch.all(supported.subthreshold_mask.sum(dim=1) == 5)
    assert torch.all(supported_ranges[supported.subthreshold_mask] <= g1.TAU)
    assert torch.all((supported_ranges > g1.TAU).sum(dim=1) >= 3)


def test_u_corruptions_preserve_declared_marginals() -> None:
    generator = g1.cpu_generator(20260711, 103)
    base = g1.make_world(64, generator, "U_BASE")
    corruptions = g1.make_u_corruptions(base, generator)
    assert set(corruptions) == set(g1.CORRUPTIONS)
    shuffled = corruptions["shuffled_event"]
    torch.testing.assert_close(
        shuffled.history_value.norm(dim=2).sort(dim=1).values,
        base.history_value.norm(dim=2).sort(dim=1).values,
    )
    torch.testing.assert_close(corruptions["query_masked"].query.norm(dim=1), base.query.norm(dim=1))
    assert torch.equal(shuffled.history_key, base.history_key)
    assert not torch.equal(shuffled.history_value, base.history_value)
    for world in corruptions.values():
        ranges = g1.oracle_ranges(world)
        assert torch.all(ranges[world.subthreshold_mask] <= g1.TAU)


def test_parameter_count_state_and_every_parameter_is_active_for_every_method() -> None:
    world = fixed_hand_world()
    counts = []
    hashes = []
    for method in g1.METHODS:
        torch.manual_seed(20260711 * 1000 + 401)
        model = g1.SyntheticRanker(method)
        counts.append(g1.parameter_count(model))
        hashes.append(g1.state_hash(model))
        output = model(world)
        loss = F.cross_entropy(output.logits, world.target)
        gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
        for (name, _), gradient in zip(model.named_parameters(), gradients, strict=True):
            assert gradient is not None, (method, name)
            assert torch.isfinite(gradient).all(), (method, name)
            assert torch.count_nonzero(gradient).item() > 0, (method, name)
    assert len(set(counts)) == 1
    assert len(set(hashes)) == 1


def test_every_method_has_exact_no_history_fallback() -> None:
    world = fixed_hand_world(history_present=False)
    for method in g1.METHODS:
        torch.manual_seed(7)
        model = g1.SyntheticRanker(method)
        model.eval()
        output = model(world)
        torch.testing.assert_close(output.logits, output.base_logits, atol=0.0, rtol=0.0)
        assert torch.isfinite(output.logits).all()


def test_training_schedule_has_exact_composition_and_is_reproducible() -> None:
    first = g1.build_schedule(20260711)
    second = g1.build_schedule(20260711)
    assert torch.equal(first.r, second.r)
    assert torch.equal(first.s, second.s)
    assert torch.equal(first.n, second.n)
    assert torch.equal(first.within, second.within)
    assert first.r.shape == (512, 16)
    assert torch.equal(first.r.flatten().bincount(minlength=4096), torch.full((4096,), 2))
    rotations = ((6, 5, 5), (5, 6, 5), (5, 5, 6))
    for update in range(9):
        assert tuple(first.u[update][name].numel() for name in g1.CORRUPTIONS) == rotations[
            update % 3
        ]
        assert sum(first.u[update][name].numel() for name in g1.CORRUPTIONS) == 16


def test_stable_tie_break_prefers_smaller_candidate_index() -> None:
    logits = torch.tensor([[1.0, 1.0, 0.0, 1.0, -1.0]])
    order = g1.stable_order(logits)
    assert order.tolist() == [[0, 1, 3, 2, 4]]


def test_hand_normalizers_have_expected_conservation_and_control_shapes() -> None:
    world = fixed_hand_world()
    for method in g1.METHODS:
        torch.manual_seed(19)
        model = g1.SyntheticRanker(method)
        output = model(world)
        assert output.logits.shape == (2, 5)
        assert output.evidence_logits.shape == (2, 5, 8)
        assert output.signed_weights.shape == (2, 5, 8)
        assert torch.isfinite(output.logits).all()
        if method in {"PDSK", "CENTER0", "GATED_CENTER", "DIFF_ATTN"}:
            torch.testing.assert_close(
                output.signed_weights.sum(dim=1),
                torch.zeros((2, 8)),
                atol=1e-6,
                rtol=0.0,
            )


def test_lock_verifier_fails_closed_on_missing_lock(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        g1.verify_execution_lock(tmp_path / "missing.json", tmp_path / "result.json")
