"""Frozen C43 states, history rows, and staged train-label access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from train.structure import PackedStructure, candidate_key_sha256, read_json, sha256_file


@dataclass(frozen=True)
class CompactLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.request_indices) + 1:
            raise ValueError("C43 compact-label offsets differ")
        if int(self.offsets[-1]) != len(self.values):
            raise ValueError("C43 compact-label values differ")

    @property
    def positions(self) -> dict[int, int]:
        return {int(value): row for row, value in enumerate(self.request_indices)}

    def row(self, index: int, count: int) -> np.ndarray:
        position = self.positions.get(int(index))
        if position is None:
            raise PermissionError(f"C43 label unavailable: {index}")
        start, stop = int(self.offsets[position]), int(self.offsets[position + 1])
        if stop - start != int(count):
            raise ValueError("C43 label/candidate count differs")
        return np.asarray(self.values[start:stop], dtype=np.float32).copy()

    def rows(self, indices: Sequence[int], counts: Sequence[int]) -> list[np.ndarray]:
        return [self.row(int(index), int(count)) for index, count in zip(indices, counts)]


def open_original_labels(
    *,
    data: PackedStructure,
    indices: Sequence[int],
    path: str | Path,
    expected_sha256: str,
    selection_path: str | Path,
    selection_sha256: str,
) -> CompactLabels:
    if sha256_file(selection_path) != selection_sha256:
        raise RuntimeError("C43 selection changed before label access")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("C43 train candidate labels changed")
    source = np.load(path, mmap_mode="r")
    rows: list[np.ndarray] = []
    offsets = [0]
    for raw in indices:
        index = int(raw)
        start, stop = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
        row = np.asarray(source[start:stop], dtype=np.float32).copy()
        rows.append(row)
        offsets.append(offsets[-1] + len(row))
    values = np.concatenate(rows).astype(np.float32, copy=False) if rows else np.empty(0, np.float32)
    return CompactLabels(
        np.asarray(indices, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        values,
    )


def zscore_row(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not len(array):
        return array.copy()
    mean = float(np.asarray(array, dtype=np.float64).mean())
    scale = float(np.asarray(array, dtype=np.float64).std())
    if scale <= 1e-8:
        return np.zeros_like(array)
    return ((array - mean) / scale).astype(np.float32)


class FrozenTransferStore:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.data = PackedStructure(config["paths"]["packed_train_root"])
        self.selection = read_json(config["paths"]["selection"])
        root = Path(config["paths"]["artifact_root"])
        self.feature_indices = np.load(root / "feature_request_indices.npy", mmap_mode="r")
        self.score_offsets = np.load(root / "feature_candidate_offsets.npy", mmap_mode="r")
        self.base_scores = np.load(root / "base_scores.npy", mmap_mode="r")
        self.query_embeddings = np.load(root / "query_embeddings.npy", mmap_mode="r")
        self.history_request_indices = np.load(root / "history_request_indices.npy", mmap_mode="r")
        self.true_history_offsets = np.load(root / "true_history_offsets.npy", mmap_mode="r")
        self.true_history_items = np.load(root / "true_history_items.npy", mmap_mode="r")
        self.wrong_history_offsets = np.load(root / "wrong_history_offsets.npy", mmap_mode="r")
        self.wrong_history_items = np.load(root / "wrong_history_items.npy", mmap_mode="r")
        self.raw_items = np.load(config["paths"]["raw_item_embeddings"], mmap_mode="r")
        self.feature_position = {int(value): row for row, value in enumerate(self.feature_indices)}
        self.history_position = {
            int(value): row for row, value in enumerate(self.history_request_indices)
        }
        if len(self.feature_position) != len(self.feature_indices):
            raise ValueError("C43 feature requests overlap")
        if not np.array_equal(self.feature_indices, self.history_request_indices):
            raise ValueError("C43 feature/history request rows differ")
        if len(self.score_offsets) != len(self.feature_indices) + 1:
            raise ValueError("C43 score offsets differ")
        if int(self.score_offsets[-1]) != len(self.base_scores):
            raise ValueError("C43 score values differ")
        if len(self.true_history_offsets) != len(self.feature_indices) + 1:
            raise ValueError("C43 true history offsets differ")
        if len(self.wrong_history_offsets) != len(self.feature_indices) + 1:
            raise ValueError("C43 wrong history offsets differ")
        if int(self.true_history_offsets[-1]) != len(self.true_history_items):
            raise ValueError("C43 true history values differ")
        if int(self.wrong_history_offsets[-1]) != len(self.wrong_history_items):
            raise ValueError("C43 wrong history values differ")
        dim = int(config["model"]["embedding_dim"])
        if self.query_embeddings.shape != (len(self.feature_indices), dim):
            raise ValueError("C43 query embedding shape differs")
        if self.raw_items.ndim != 2 or self.raw_items.shape[1] != dim:
            raise ValueError("C43 item embedding shape differs")

    def role_indices(self, role: str) -> list[int]:
        return [int(value) for value in self.selection["roles"][role]["indices"]]

    def candidate_hash(self, indices: Sequence[int]) -> str:
        return candidate_key_sha256(self.data, indices)

    def candidate_count(self, index: int) -> int:
        return int(self.data.candidate_offsets[index + 1] - self.data.candidate_offsets[index])

    def candidate_embedding_indices(self, index: int) -> np.ndarray:
        return self.data.candidate_indices(index).astype(np.int64, copy=False)

    def candidate_item_ids(self, index: int) -> np.ndarray:
        start, stop = int(self.data.candidate_offsets[index]), int(self.data.candidate_offsets[index + 1])
        return np.asarray(self.data.candidate_item_ids[start:stop]).copy()

    def request_id(self, index: int) -> str:
        return self.data.request_ids[int(index)]

    def base_row(self, index: int) -> np.ndarray:
        row = self.feature_position[int(index)]
        start, stop = int(self.score_offsets[row]), int(self.score_offsets[row + 1])
        if stop - start != self.candidate_count(int(index)):
            raise ValueError("C43 base/candidate count differs")
        return zscore_row(np.asarray(self.base_scores[start:stop], dtype=np.float32))

    def query(self, index: int) -> np.ndarray:
        return np.asarray(self.query_embeddings[self.feature_position[int(index)]], dtype=np.float32)

    def item_embeddings(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.raw_items[np.asarray(indices, dtype=np.int64)], dtype=np.float32)

    def history(self, index: int, source: str) -> np.ndarray:
        row = self.history_position[int(index)]
        if source == "true":
            start, stop = int(self.true_history_offsets[row]), int(self.true_history_offsets[row + 1])
            return np.asarray(self.true_history_items[start:stop], dtype=np.int64)
        if source == "wrong":
            start, stop = int(self.wrong_history_offsets[row]), int(self.wrong_history_offsets[row + 1])
            return np.asarray(self.wrong_history_items[start:stop], dtype=np.int64)
        if source == "none":
            return np.empty(0, dtype=np.int64)
        raise ValueError(f"unknown C43 history source: {source}")

    def has_repeat(self, index: int) -> bool:
        return bool(
            set(int(value) for value in self.candidate_embedding_indices(index))
            & set(int(value) for value in self.data.history_indices(index))
        )

    def item_only_row(self, index: int) -> np.ndarray:
        candidates = self.candidate_embedding_indices(index)
        history = self.data.history_indices(index).astype(np.int64, copy=False)
        start, stop = int(self.data.history_offsets[index]), int(self.data.history_offsets[index + 1])
        weights = np.asarray(self.data.history_event_weights[start:stop], dtype=np.float32)
        if len(history):
            reverse = np.maximum(len(history) - np.arange(len(history)), 1).astype(np.float32)
            component = 3.0 * (
                (candidates[:, None] == history[None, :]).astype(np.float32)
                * (weights / np.sqrt(reverse))[None, :]
            ).sum(axis=1)
        else:
            component = np.zeros(len(candidates), dtype=np.float32)
        beta = float(self.config["base"]["item_only_beta"])
        return beta * self.base_row(index) + (1.0 - beta) * zscore_row(component)


def to_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)
