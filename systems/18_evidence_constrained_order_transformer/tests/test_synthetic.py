from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.synthetic import (
    NO_HISTORY,
    REPEAT_CONFLICT,
    SUPPORTED_NONREPEAT,
    batch_schedule,
    corrupt_supported,
    generate_split,
    permute_candidates,
)


def generate():
    return generate_split(
        seed=20260718,
        split="eval",
        requests=128,
        candidates=8,
        history_slots=8,
        input_dim=16,
        topics=4,
        strata_weights=[0.25, 0.375, 0.375],
    )


def test_generator_is_deterministic_balanced_and_disjoint() -> None:
    first = generate()
    second = generate()
    for name in first.__dataclass_fields__:
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert first.stratum.bincount(minlength=3).tolist() == [32, 48, 48]
    assert not first.repeat_mask[first.stratum.eq(SUPPORTED_NONREPEAT)].any()
    assert first.repeat_mask[first.stratum.eq(REPEAT_CONFLICT)].sum(1).eq(1).all()
    assert not first.history_mask[first.stratum.eq(NO_HISTORY)].any()


def test_repeat_target_is_unique_repeat_and_semantic_latest_conflicts() -> None:
    batch = generate()
    mask = batch.stratum.eq(REPEAT_CONFLICT)
    repeated = batch.repeat_mask[mask]
    targets = batch.target_index[mask]
    assert repeated.gather(1, targets.unsqueeze(1)).all()
    # The latest query-topic semantic event has the opposite style.
    queries = batch.query[mask, :4].argmax(1)
    histories = batch.history[mask]
    for row in range(histories.shape[0]):
        topic = int(queries[row])
        topic_positions = torch.nonzero(histories[row, :, topic].gt(0.8)).view(-1)
        latest = int(topic_positions.max())
        latest_style = histories[row, latest, 4 + topic].sign()
        target_candidate = batch.candidates[mask][row, targets[row]]
        target_style = target_candidate[4 + topic].sign()
        assert latest_style == -target_style


def test_supported_target_is_latest_query_topic_style_without_repeat() -> None:
    batch = generate()
    mask = batch.stratum.eq(SUPPORTED_NONREPEAT)
    supported = batch.subset(torch.nonzero(mask).view(-1))
    for row in range(len(supported)):
        topic = int(supported.query[row, :4].argmax())
        positions = torch.nonzero(supported.history[row, :, topic].gt(0.8)).view(-1)
        latest = int(positions.max())
        style = supported.history[row, latest, 4 + topic].sign()
        target = supported.candidates[row, supported.target_index[row]]
        assert target[topic] > 0.8
        assert target[4 + topic].sign() == style


def test_corruptions_preserve_candidates_and_targets() -> None:
    batch = generate()
    supported = batch.subset(torch.nonzero(batch.stratum.eq(SUPPORTED_NONREPEAT)).view(-1))
    for name in ("wrong_history", "shuffled_event", "query_mask", "coarse_only"):
        changed = corrupt_supported(supported, seed=20260718, corruption=name)
        assert torch.equal(changed.candidates, supported.candidates)
        assert torch.equal(changed.candidate_ids, supported.candidate_ids)
        assert torch.equal(changed.target_index, supported.target_index)
    assert not torch.equal(
        corrupt_supported(supported, seed=20260718, corruption="wrong_history").history,
        supported.history,
    )
    coarse = corrupt_supported(supported, seed=20260718, corruption="coarse_only")
    assert torch.equal(coarse.history[:, :, 4:8], torch.zeros_like(coarse.history[:, :, 4:8]))


def test_schedule_and_candidate_permutation_are_deterministic() -> None:
    schedule = batch_schedule(seed=20260718, requests=128, steps=10, batch_size=16)
    assert torch.equal(
        schedule,
        batch_schedule(seed=20260718, requests=128, steps=10, batch_size=16),
    )
    batch = generate()
    permutation = torch.arange(7, -1, -1)
    changed = permute_candidates(batch, permutation)
    assert torch.equal(changed.candidates, batch.candidates[:, permutation])
    restored = changed.candidate_ids.gather(1, changed.target_index.unsqueeze(1)).squeeze(1)
    original = batch.candidate_ids.gather(1, batch.target_index.unsqueeze(1)).squeeze(1)
    assert torch.equal(restored, original)
