from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.synthetic import (
    EXACT_RECURRENCE,
    NO_HISTORY,
    SUPPORTED_TRANSFER,
    corrupt_supported,
    generate_split,
    permute_candidates,
    remove_identity,
)


def generate() -> object:
    return generate_split(
        seed=20260730,
        split="audit",
        requests=512,
        candidates=8,
        history_slots=6,
        input_dim=24,
        strata_weights=[0.2, 0.4, 0.4],
    )


def test_generator_has_all_strata_and_exact_fallback_base() -> None:
    batch = generate()
    assert set(batch.stratum.tolist()) == {NO_HISTORY, EXACT_RECURRENCE, SUPPORTED_TRANSFER}
    nohistory = batch.stratum.eq(NO_HISTORY)
    assert batch.history_mask[nohistory].eq(0).all()
    assert batch.base_scores[nohistory].argmax(dim=-1).eq(batch.target_index[nohistory]).all()
    repeated = batch.stratum.eq(EXACT_RECURRENCE)
    rows = torch.nonzero(repeated).flatten()
    assert batch.identity[rows, batch.target_index[rows]].any(dim=-1).all()
    supported = batch.stratum.eq(SUPPORTED_TRANSFER)
    assert not batch.identity[supported].any()


def test_corruptions_change_the_intended_evidence() -> None:
    batch = generate()
    supported = batch.subset(torch.nonzero(batch.stratum.eq(SUPPORTED_TRANSFER)).flatten())
    wrong = corrupt_supported(supported, seed=4, corruption="wrong_history")
    shuffled = corrupt_supported(supported, seed=4, corruption="shuffled_event")
    coarse = corrupt_supported(supported, seed=4, corruption="coarse_only")
    assert not torch.equal(wrong.history, supported.history)
    assert not torch.equal(shuffled.history, supported.history)
    assert coarse.history[..., 3:].eq(0).all()
    repeated = batch.subset(torch.nonzero(batch.stratum.eq(EXACT_RECURRENCE)).flatten())
    assert repeated.identity.any()
    assert not remove_identity(repeated).identity.any()


def test_candidate_permutation_updates_target_and_relations() -> None:
    batch = generate()
    permutation = torch.tensor([3, 1, 6, 0, 7, 2, 4, 5])
    changed = permute_candidates(batch, permutation)
    torch.testing.assert_close(changed.candidates, batch.candidates[:, permutation])
    original_target_state = batch.candidates[
        torch.arange(len(batch)), batch.target_index
    ]
    changed_target_state = changed.candidates[
        torch.arange(len(batch)), changed.target_index
    ]
    torch.testing.assert_close(original_target_state, changed_target_state)
