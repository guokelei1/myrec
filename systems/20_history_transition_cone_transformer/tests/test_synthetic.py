from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.synthetic import (
    EXACT_REPEAT,
    NO_HISTORY,
    SUPPORTED_COMPOSITION,
    batch_schedule,
    corrupt_supported,
    generate_split,
    permute_candidates,
)
from model.htct import solve_transition_coefficients


def generate():
    return generate_split(
        seed=20260724,
        split="eval",
        requests=128,
        candidates=8,
        history_slots=7,
        input_dim=16,
        relation_raw_dim=8,
        strata_weights=[0.25, 0.375, 0.375],
    )


def test_generator_is_deterministic_and_has_frozen_strata() -> None:
    first, second = generate(), generate()
    for name in first.__dataclass_fields__:
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert first.stratum.bincount(minlength=3).tolist() == [32, 48, 48]
    assert not first.history_mask[first.stratum.eq(NO_HISTORY)].any()


def test_supported_target_is_positive_composition_and_reverse_is_equal_norm() -> None:
    batch = generate()
    rows = torch.nonzero(batch.stratum.eq(SUPPORTED_COMPOSITION)).view(-1)
    for row in rows:
        target = batch.candidates[row, batch.target_index[row], :8] - batch.query[row, :8]
        reverse = batch.candidates[row, batch.reverse_index[row], :8] - batch.query[row, :8]
        transitions = batch.history[row, 1:, :8] - batch.history[row, :-1, :8]
        positive_reconstruction = batch.true_coefficients[row] @ transitions
        torch.testing.assert_close(target, positive_reconstruction, atol=1e-5, rtol=0.0)
        assert batch.true_coefficients[row].gt(0).sum() == 3
        torch.testing.assert_close(reverse, -target, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(target.norm(), reverse.norm(), atol=1e-6, rtol=0.0)


def test_supported_candidate_directions_form_sign_symmetric_pairs() -> None:
    batch = generate()
    rows = torch.nonzero(batch.stratum.eq(SUPPORTED_COMPOSITION)).view(-1)
    for row in rows[:8]:
        canonical = torch.empty_like(batch.candidates[row, :, :8])
        canonical[batch.canonical_ids[row]] = batch.candidates[row, :, :8]
        displacement = canonical - batch.query[row, :8]
        for start in range(0, 8, 2):
            torch.testing.assert_close(displacement[start], -displacement[start + 1], atol=1e-6, rtol=0.0)


def test_raw_geometry_requires_iterative_cone_and_has_no_fixed_tie_advantage() -> None:
    batch = generate()
    mask = batch.stratum.eq(SUPPORTED_COMPOSITION)
    relation = batch.candidates[mask, :, :8] - batch.query[mask][:, None, :8]
    transitions = batch.history[mask, 1:, :8] - batch.history[mask, :-1, :8]
    transition_mask = torch.ones(transitions.shape[:2], dtype=torch.bool)
    accuracies = {}
    for mode in ("cone", "span", "relu1", "simplex"):
        output = solve_transition_coefficients(
            relation, transitions, transition_mask, mode=mode, steps=16, ridge=0.02
        )
        reduction = 1.0 - (relation - output.reconstruction).square().sum(dim=-1)
        prediction = (
            reduction.double() - batch.candidate_ids[mask].double() * 1e-12
        ).argmax(dim=1)
        accuracies[mode] = float(prediction.eq(batch.target_index[mask]).double().mean())
    assert accuracies["cone"] >= 0.95
    assert accuracies["span"] <= 0.65
    assert accuracies["cone"] - accuracies["relu1"] >= 0.02
    assert accuracies["cone"] - accuracies["simplex"] >= 0.02

    target_ids = batch.candidate_ids[mask].gather(1, batch.target_index[mask][:, None]).squeeze(1)
    reverse_ids = batch.candidate_ids[mask].gather(1, batch.reverse_index[mask][:, None]).squeeze(1)
    assert target_ids.lt(reverse_ids).any()
    assert target_ids.gt(reverse_ids).any()


def test_repeat_target_is_query_identical() -> None:
    batch = generate()
    mask = batch.stratum.eq(EXACT_REPEAT)
    target = batch.candidates[mask].gather(
        1, batch.target_index[mask][:, None, None].expand(-1, 1, 16)
    ).squeeze(1)
    torch.testing.assert_close(target[:, :8], batch.query[mask, :8], atol=0.0, rtol=0.0)


def test_no_history_target_is_unique_quality_max() -> None:
    batch = generate()
    mask = batch.stratum.eq(NO_HISTORY)
    assert torch.equal(batch.candidates[mask, :, 8].argmax(dim=1), batch.target_index[mask])


def test_corruptions_and_candidate_permutation_preserve_labels() -> None:
    batch = generate()
    supported = batch.subset(torch.nonzero(batch.stratum.eq(SUPPORTED_COMPOSITION)).view(-1))
    for name in ("wrong_history", "shuffled_event", "query_mask", "coarse_only", "reversed_event"):
        changed = corrupt_supported(supported, seed=20260724, corruption=name)
        assert torch.equal(changed.candidates, supported.candidates)
        assert torch.equal(changed.target_index, supported.target_index)
        assert torch.equal(changed.reverse_index, supported.reverse_index)
    reversed_batch = corrupt_supported(supported, seed=20260724, corruption="reversed_event")
    assert torch.equal(reversed_batch.history, supported.history.flip(1))

    permutation = torch.arange(7, -1, -1)
    changed = permute_candidates(batch, permutation)
    target_ids = changed.candidate_ids.gather(1, changed.target_index[:, None]).squeeze(1)
    original_ids = batch.candidate_ids.gather(1, batch.target_index[:, None]).squeeze(1)
    assert torch.equal(target_ids, original_ids)


def test_batch_schedule_is_repeatable() -> None:
    first = batch_schedule(seed=20260724, requests=128, steps=12, batch_size=16)
    second = batch_schedule(seed=20260724, requests=128, steps=12, batch_size=16)
    assert torch.equal(first, second)
