import pytest
import torch

from myrec.mechanism.mask_softmax_scoring import build_visibility_masks


def _path():
    return {
        "ids": torch.zeros(1, 8, dtype=torch.long),
        "mask": torch.ones(1, 8, dtype=torch.bool),
        "positions": torch.tensor([[6, 7]]),
        "starts": torch.tensor([2]),
        "ends": torch.tensor([4]),
        "candidate_starts": torch.tensor([4]),
        "candidate_ends": torch.tensor([6]),
    }


def test_visibility_masks_are_position_matched_and_future_safe():
    masks = build_visibility_masks(_path())
    assert set(masks) == {"native", "prefix_history_swap", "candidate_visibility_swap"}
    for name, mask in masks.items():
        assert mask.shape == (1, 2, 8), name
        assert not bool(mask[0, 0, 7])
        # The second readout is at position 7, so self-visibility is valid;
        # only genuinely future keys must be blocked.
        assert bool(mask[0, 1, 7])
        assert bool(mask[0, 0, 6])


def test_history_and_candidate_alternatives_remove_only_registered_spans():
    masks = build_visibility_masks(_path())
    native = masks["native"]
    history = masks["prefix_history_swap"]
    candidate = masks["candidate_visibility_swap"]
    assert bool(native[0, 0, 2]) and bool(native[0, 0, 3])
    assert not bool(history[0, 0, 2]) and not bool(history[0, 0, 3])
    assert bool(native[0, 0, 4]) and bool(native[0, 0, 5])
    assert not bool(candidate[0, 0, 4]) and not bool(candidate[0, 0, 5])
    assert bool(history[0, 0, 6]) and bool(candidate[0, 0, 6])


def test_empty_candidate_span_is_rejected():
    path = _path()
    path["candidate_ends"] = torch.tensor([4])
    with pytest.raises(ValueError, match="candidate span"):
        build_visibility_masks(path)
