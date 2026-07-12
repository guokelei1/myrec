from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.gate_metrics import bootstrap, retention
from train.losses import masked_listwise_loss
from train.real_data import iter_batches, zscore_numpy
from train.run_train_gate import FORWARD_NAMES, forward_kwargs


def test_loss_and_metrics() -> None:
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    assert masked_listwise_loss(torch.tensor([[2.0, 0.0, -1.0]]), labels, mask) < masked_listwise_loss(
        torch.tensor([[-1.0, 0.0, 2.0]]), labels, mask
    )
    values = np.asarray([[1.0, 2.0, 3.0, 99.0]], dtype=np.float32)
    selected = np.asarray([[1, 1, 1, 0]], dtype=bool)
    result = zscore_numpy(values, selected)
    assert abs(float(result[0, :3].mean())) < 1e-7 and result[0, 3] == 0.0
    assert bootstrap(np.asarray([1.0, 2.0]), samples=100, seed=3)["mean"] == 1.5
    assert retention(np.asarray([-1.0, 0.0]), np.ones(2), samples=100, seed=3)["applicable"] is False


def test_batch_budget_counts_query_special_tokens() -> None:
    class Data:
        candidate_offsets = np.asarray([0, 10, 20])
        history_offsets = np.asarray([0, 10, 20])

    class Store:
        data = Data()
        feature_position = {0: 0, 1: 1}
        query_attention = np.asarray(
            [[1, 1, 1, 0], [1, 1, 1, 0]], dtype=bool
        )
        query_content = np.asarray(
            [[0, 1, 0, 0], [0, 1, 0, 0]], dtype=bool
        )

    rows = list(
        iter_batches(
            Store(),
            [0, 1],
            seed=0,
            shuffle=False,
            max_requests=8,
            max_bridge_cells=500,
        )
    )
    assert [row.tolist() for row in rows] == [[0], [1]]


def test_runner_maps_registered_query_tokens_to_model_argument() -> None:
    tensors = {name: torch.tensor(1) for name in FORWARD_NAMES}
    values = forward_kwargs(tensors)
    assert "query_token_ids" not in values
    assert values["query_ids"] is tensors["query_token_ids"]
