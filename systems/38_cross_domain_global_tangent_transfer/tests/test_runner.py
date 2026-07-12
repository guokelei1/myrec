from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.global_tangent import PRIMARY, LowRankGlobalTangentTransfer  # noqa: E402
from train.run_train_gate import score_mode, train_mode  # noqa: E402
from train.store import CompactLabels  # noqa: E402


class _TinyStore:
    def __init__(self) -> None:
        self.embedding = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def role_indices(self, role: str) -> list[int]:
        assert role == "fit"
        return [0, 1]

    def has_repeat(self, index: int) -> bool:
        return False

    def candidate_count(self, index: int) -> int:
        return 3

    def query(self, index: int) -> np.ndarray:
        return np.asarray([1.0, 0.2 + 0.1 * index, 0.1, 0.0], dtype=np.float32)

    def history_positions(self, index: int, source: str) -> np.ndarray:
        if source == "none":
            return np.empty(0, dtype=np.int64)
        return np.asarray([2, 3] if source == "true" else [0, 1], dtype=np.int64)

    def candidate_positions(self, index: int) -> np.ndarray:
        return np.asarray([0, 1, 2], dtype=np.int64)

    def items(self, positions: np.ndarray) -> np.ndarray:
        return self.embedding[positions]

    def base_row(self, index: int) -> np.ndarray:
        return np.asarray([0.2, 0.1, 0.0], dtype=np.float32)


def test_cpu_training_and_scoring_smoke() -> None:
    store = _TinyStore()
    labels = CompactLabels(
        request_indices=np.asarray([0, 1], dtype=np.int64),
        offsets=np.asarray([0, 3, 6], dtype=np.int64),
        values=np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32),
    )
    config = {
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 1,
            "max_requests_per_batch": 2,
            "listwise_loss_weight": 1.0,
            "direction_loss_weight": 1.0,
            "gradient_clip_norm": 1.0,
        }
    }
    model = LowRankGlobalTangentTransfer(
        dim=4,
        rank=2,
        temperature=0.1,
        profile_scale=1.0,
        correction_scale=2.0,
        seed=38,
        mode=PRIMARY,
    )
    report = train_mode(model, store, labels, config, seed=38, device=torch.device("cpu"))
    assert report["finite"] is True
    assert report["steps"] == 1
    assert report["nonzero_gradient_parameter_count"] > 0
    scores, corrections = score_mode(
        model,
        store,
        [0, 1],
        "true",
        torch.device("cpu"),
    )
    assert len(scores) == len(corrections) == 2
    assert all(np.isfinite(row).all() for row in scores)
