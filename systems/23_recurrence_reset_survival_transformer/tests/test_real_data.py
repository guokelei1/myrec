from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.real_data import CompactLabels, masked_zscore_numpy, to_device


def test_compact_labels_require_selected_requests() -> None:
    labels = CompactLabels(
        request_indices=np.asarray([7, 9]),
        offsets=np.asarray([0, 2, 5]),
        values=np.asarray([1, 0, 0, 1, 0], dtype=np.float32),
    )
    mask = np.asarray([[1, 1, 0], [1, 1, 1]], dtype=bool)
    assert labels.padded([7, 9], mask).tolist() == [[1, 0, 0], [0, 1, 0]]


def test_registered_item_only_composition() -> None:
    candidate_mask = np.asarray([[1, 1, 1]], dtype=bool)
    batch = {
        "query_numpy": np.zeros((1, 4), dtype=np.float32),
        "candidate_numpy": np.zeros((1, 3, 4), dtype=np.float32),
        "candidate_mask_numpy": candidate_mask,
        "history_numpy": np.zeros((1, 2, 4), dtype=np.float32),
        "history_mask_numpy": np.asarray([[1, 1]], dtype=bool),
        "event_weights_numpy": np.ones((1, 2), dtype=np.float32),
        "repeat_mask_numpy": np.zeros((1, 3, 2), dtype=bool),
        "base_scores_numpy": np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32),
        "item_component_numpy": np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32),
    }
    tensors = to_device(batch, torch.device("cpu"), beta=0.3)
    expected = 0.3 * masked_zscore_numpy(
        batch["base_scores_numpy"], candidate_mask
    ) + 0.7 * masked_zscore_numpy(batch["item_component_numpy"], candidate_mask)
    assert np.array_equal(tensors["item_only_scores"].numpy(), expected)


def test_numpy_zscore_constant_is_zero() -> None:
    values = np.asarray([[4.0, 4.0, 99.0]], dtype=np.float32)
    mask = np.asarray([[1, 1, 0]], dtype=bool)
    assert np.array_equal(masked_zscore_numpy(values, mask), np.zeros_like(values))
