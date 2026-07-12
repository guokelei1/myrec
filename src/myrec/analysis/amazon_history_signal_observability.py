"""Adapter from frozen Amazon-C4 C38 features to the HSO diagnostic core."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _position_values(path: Path, field: str) -> list[str]:
    output: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for expected, line in enumerate(handle):
            row = json.loads(line)
            if int(row["position"]) != expected:
                raise ValueError(f"Amazon HSO {path.name} position differs")
            output.append(str(row[field]))
    return output


class AmazonObservabilityData:
    """Label-free C38 feature surface with pseudo rows for stored wrong history."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.original_indices = np.load(
            self.root / "feature_request_indices.npy", mmap_mode="r"
        )
        self.candidate_offsets = np.load(
            self.root / "candidate_offsets.npy", mmap_mode="r"
        )
        self.candidate_indices = np.load(
            self.root / "candidate_item_positions.npy", mmap_mode="r"
        )
        self.history_offsets = np.load(
            self.root / "true_history_offsets.npy", mmap_mode="r"
        )
        self.history_indices = np.load(
            self.root / "true_history_item_positions.npy", mmap_mode="r"
        )
        self.wrong_history_offsets = np.load(
            self.root / "wrong_history_offsets.npy", mmap_mode="r"
        )
        self.wrong_history_indices = np.load(
            self.root / "wrong_history_item_positions.npy", mmap_mode="r"
        )
        self.request_ids = _position_values(self.root / "requests.jsonl", "request_id")
        self.item_ids = _position_values(self.root / "items.jsonl", "item_id")
        count = len(self.request_ids)
        if not (
            len(self.original_indices) == count
            and len(self.candidate_offsets) == count + 1
            and len(self.history_offsets) == count + 1
            and len(self.wrong_history_offsets) == count + 1
        ):
            raise ValueError("Amazon HSO feature cardinality differs")
        self._labels_opened = False

    @property
    def labels_opened(self) -> bool:
        return self._labels_opened

    def candidates(self, index: int) -> np.ndarray:
        if not 0 <= int(index) < len(self.request_ids):
            raise IndexError("Amazon HSO candidate index is pseudo history row")
        start, stop = int(self.candidate_offsets[index]), int(
            self.candidate_offsets[index + 1]
        )
        return np.asarray(self.candidate_indices[start:stop], dtype=np.int64)

    def candidate_ids(self, index: int) -> np.ndarray:
        return np.asarray(
            [self.item_ids[int(value)] for value in self.candidates(index)], dtype=object
        )

    @staticmethod
    def _recency_weights(length: int) -> np.ndarray:
        if not length:
            return np.empty(0, dtype=np.float32)
        reverse = np.arange(length, 0, -1, dtype=np.float32)
        return (1.0 / np.sqrt(reverse)).astype(np.float32)

    def history(
        self, index: int, max_history: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        count = len(self.request_ids)
        if 0 <= int(index) < count:
            row = int(index)
            offsets, values = self.history_offsets, self.history_indices
        elif count <= int(index) < 2 * count:
            row = int(index) - count
            offsets, values = self.wrong_history_offsets, self.wrong_history_indices
        else:
            raise IndexError("Amazon HSO history row differs")
        start, stop = int(offsets[row]), int(offsets[row + 1])
        if max_history is not None:
            start = max(start, stop - int(max_history))
        items = np.asarray(values[start:stop], dtype=np.int64)
        return items, self._recency_weights(len(items))

    def wrong_mapping(self, indices: Sequence[int]) -> dict[int, int]:
        count = len(self.request_ids)
        return {int(index): count + int(index) for index in indices}

    def strict_nonrepeat_indices(self) -> np.ndarray:
        selected = []
        for index in range(len(self.request_ids)):
            history, _ = self.history(index)
            if len(history) and not np.intersect1d(
                self.candidates(index), history, assume_unique=False
            ).size:
                selected.append(index)
        return np.asarray(selected, dtype=np.int64)

    def candidate_hash(self, indices: Sequence[int]) -> str:
        digest = hashlib.sha256()
        for index_value in indices:
            index = int(index_value)
            payload = json.dumps(
                [self.request_ids[index], *self.candidate_ids(index).tolist()],
                separators=(",", ":"),
            ).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()


class AmazonFrozenSemanticStore:
    def __init__(self, data: AmazonObservabilityData) -> None:
        self.data = data
        self.queries = np.load(data.root / "query_embeddings.npy", mmap_mode="r")
        self.items = np.load(data.root / "item_embeddings.npy", mmap_mode="r")
        if self.queries.shape[0] != len(data.request_ids):
            raise ValueError("Amazon HSO query table differs")
        if self.items.shape[0] != len(data.item_ids):
            raise ValueError("Amazon HSO item table differs")
        if self.queries.shape[1] != self.items.shape[1]:
            raise ValueError("Amazon HSO embedding width differs")

    @property
    def width(self) -> int:
        return int(self.items.shape[1])

    def query(self, request_indices: Sequence[int]) -> np.ndarray:
        return np.asarray(
            self.queries[np.asarray(request_indices, dtype=np.int64)], dtype=np.float32
        )

    def item(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.items[np.asarray(indices, dtype=np.int64)], dtype=np.float32)


def load_feature_users(
    records_blind_path: str | Path, data: AmazonObservabilityData
) -> list[str]:
    wanted = {int(value): position for position, value in enumerate(data.original_indices)}
    users: list[str | None] = [None] * len(data.request_ids)
    with Path(records_blind_path).open("r", encoding="utf-8") as handle:
        for original_index, line in enumerate(handle):
            position = wanted.get(original_index)
            if position is None:
                continue
            row = json.loads(line)
            if str(row["request_id"]) != data.request_ids[position]:
                raise ValueError("Amazon HSO request alignment differs")
            users[position] = str(row["user_id"])
    if any(value is None for value in users):
        raise ValueError("Amazon HSO user coverage differs")
    return [str(value) for value in users]


def load_all_feature_labels(
    records_train_path: str | Path, data: AmazonObservabilityData
) -> list[np.ndarray]:
    wanted = {int(value): position for position, value in enumerate(data.original_indices)}
    labels: list[np.ndarray | None] = [None] * len(data.request_ids)
    with Path(records_train_path).open("r", encoding="utf-8") as handle:
        for original_index, line in enumerate(handle):
            position = wanted.get(original_index)
            if position is None:
                continue
            row = json.loads(line)
            if str(row["request_id"]) != data.request_ids[position]:
                raise ValueError("Amazon HSO label request differs")
            candidate_ids = [str(value["item_id"]) for value in row["candidates"]]
            if candidate_ids != data.candidate_ids(position).tolist():
                raise ValueError("Amazon HSO label candidate order differs")
            values = np.asarray(
                [float(value.get("clicked", 0) or 0) for value in row["candidates"]],
                dtype=np.float32,
            )
            if int((values > 0).sum()) != 1:
                raise ValueError("Amazon HSO expects one positive per request")
            labels[position] = values
    if any(value is None for value in labels):
        raise ValueError("Amazon HSO label coverage differs")
    data._labels_opened = True
    return [np.asarray(value, dtype=np.float32) for value in labels]
