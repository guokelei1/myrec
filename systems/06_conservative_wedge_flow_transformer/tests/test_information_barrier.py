import pytest
import torch

from model.information_barrier import (
    CANDIDATE,
    HISTORY,
    PAD,
    QUERY,
    build_information_barrier_mask,
)


def test_attention_barrier_has_no_history_or_cross_candidate_bypass() -> None:
    # q, candidate-0 x2, candidate-1 x2, history x2, pad
    roles = torch.tensor(
        [[QUERY, CANDIDATE, CANDIDATE, CANDIDATE, CANDIDATE, HISTORY, HISTORY, PAD]]
    )
    groups = torch.tensor([[-1, 0, 0, 1, 1, -1, -1, -1]])
    mask = build_information_barrier_mask(roles, groups)[0]
    assert mask[0].tolist() == [True, False, False, False, False, False, False, False]
    assert mask[1].tolist() == [True, True, True, False, False, False, False, False]
    assert mask[3].tolist() == [True, False, False, True, True, False, False, False]
    assert mask[5].tolist() == [True, False, False, False, False, True, True, False]
    assert not bool(mask[7].any())
    assert not bool(mask[:, 7].any())


def test_candidate_requires_nonnegative_group_id() -> None:
    with pytest.raises(ValueError):
        build_information_barrier_mask(
            torch.tensor([[QUERY, CANDIDATE]]), torch.tensor([[-1, -1]])
        )
