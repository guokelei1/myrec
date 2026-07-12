from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.synthetic import (
    EXACT_REPEAT,
    NO_HISTORY,
    SUPPORTED_SUCCESSOR,
    batch_schedule,
    corrupt_supported,
    generate_split,
    permute_candidates,
)


def generate():
    return generate_split(
        seed=20260721,
        split="eval",
        requests=128,
        candidates=8,
        history_slots=8,
        input_dim=16,
        semantic_items=8,
        strata_weights=[0.25, 0.375, 0.375],
    )


def test_generator_is_deterministic_and_balanced() -> None:
    first, second = generate(), generate()
    for name in first.__dataclass_fields__:
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert first.stratum.bincount(minlength=3).tolist() == [32, 48, 48]
    assert not first.history_mask[first.stratum.eq(NO_HISTORY)].any()


def test_repeat_target_is_unique_exact_identity() -> None:
    batch = generate()
    mask = batch.stratum.eq(EXACT_REPEAT)
    relation = batch.identity_relation[mask]
    targets = batch.target_index[mask]
    assert relation.sum(dim=(1, 2)).eq(1).all()
    assert relation.gather(
        1, targets[:, None, None].expand(-1, 1, relation.shape[2])
    ).any(dim=2).all()


def test_supported_target_is_successor_and_predecessor_has_quality_bias() -> None:
    batch = generate()
    supported = batch.subset(torch.nonzero(batch.stratum.eq(SUPPORTED_SUCCESSOR)).view(-1))
    assert not supported.identity_relation.any()
    for row in range(len(supported)):
        query_semantic = int(supported.query[row, :8].argmax())
        history_semantics = supported.history[row, :, :8].argmax(dim=1)
        pivot = int(torch.nonzero(history_semantics.eq(query_semantic))[0])
        assert 0 < pivot < history_semantics.numel() - 1
        target_semantic = int(history_semantics[pivot + 1])
        predecessor = int(history_semantics[pivot - 1])
        target_id = int(supported.candidate_ids[row, supported.target_index[row]])
        assert target_id == target_semantic
        predecessor_position = int(torch.nonzero(supported.candidate_ids[row].eq(predecessor))[0])
        assert supported.candidates[row, predecessor_position, 8] > supported.candidates[row, supported.target_index[row], 8]


def test_corruptions_and_permutation_preserve_registered_fields() -> None:
    batch = generate()
    supported = batch.subset(torch.nonzero(batch.stratum.eq(SUPPORTED_SUCCESSOR)).view(-1))
    for name in ("wrong_history", "shuffled_event", "query_mask", "coarse_only", "reversed_event"):
        changed = corrupt_supported(supported, seed=20260721, corruption=name)
        assert torch.equal(changed.candidates, supported.candidates)
        assert torch.equal(changed.target_index, supported.target_index)
    reversed_batch = corrupt_supported(supported, seed=20260721, corruption="reversed_event")
    assert torch.equal(reversed_batch.history, supported.history.flip(1))

    permutation = torch.arange(7, -1, -1)
    changed = permute_candidates(batch, permutation)
    target_ids = changed.candidate_ids.gather(1, changed.target_index[:, None]).squeeze(1)
    original_ids = batch.candidate_ids.gather(1, batch.target_index[:, None]).squeeze(1)
    assert torch.equal(target_ids, original_ids)


def test_batch_schedule_is_repeatable() -> None:
    first = batch_schedule(seed=20260721, requests=128, steps=12, batch_size=16)
    second = batch_schedule(seed=20260721, requests=128, steps=12, batch_size=16)
    assert torch.equal(first, second)
