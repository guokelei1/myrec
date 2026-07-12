"""Frozen C39 embedding store and role-scoped train-label opener."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from train.selection import read_json, sha256_file


@dataclass(frozen=True)
class CompactLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.request_indices) + 1:
            raise ValueError("C39 compact label offsets differ")
        if int(self.offsets[-1]) != len(self.values):
            raise ValueError("C39 compact label values differ")

    @property
    def positions(self) -> dict[int, int]:
        return {int(value): position for position, value in enumerate(self.request_indices)}

    def row(self, index: int, expected_count: int) -> np.ndarray:
        position = self.positions.get(int(index))
        if position is None:
            raise PermissionError(f"C39 labels unavailable for record {index}")
        start, stop = int(self.offsets[position]), int(self.offsets[position + 1])
        if stop - start != expected_count:
            raise ValueError("C39 compact label candidate count differs")
        return np.asarray(self.values[start:stop], dtype=np.float32).copy()


class FrozenTransferStore:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        paths = config["paths"]
        self.selection = read_json(paths["selection"])
        root = Path(paths["feature_root"])
        embedding_manifest = read_json(root / "embedding_manifest.json")
        if not embedding_manifest.get("finite", False):
            raise RuntimeError("C39 frozen embeddings are not finite")
        self.feature_indices = np.load(root / "feature_request_indices.npy", mmap_mode="r")
        self.feature_position = {
            int(index): position for position, index in enumerate(self.feature_indices)
        }
        if len(self.feature_position) != len(self.feature_indices):
            raise ValueError("C39 feature request indices overlap")
        self.candidate_offsets = np.load(root / "candidate_offsets.npy", mmap_mode="r")
        self.candidate_item_positions = np.load(
            root / "candidate_item_positions.npy", mmap_mode="r"
        )
        self.true_history_offsets = np.load(root / "true_history_offsets.npy", mmap_mode="r")
        self.true_history_item_positions = np.load(
            root / "true_history_item_positions.npy", mmap_mode="r"
        )
        self.wrong_history_offsets = np.load(root / "wrong_history_offsets.npy", mmap_mode="r")
        self.wrong_history_item_positions = np.load(
            root / "wrong_history_item_positions.npy", mmap_mode="r"
        )
        self.item_embeddings = np.load(root / "item_embeddings.npy", mmap_mode="r")
        self.query_embeddings = np.load(root / "query_embeddings.npy", mmap_mode="r")
        self.base_scores = np.load(root / "base_scores.npy", mmap_mode="r")
        self.item_ids = _load_position_values(root / "items.jsonl", "item_id")
        self.request_ids = _load_position_values(root / "requests.jsonl", "request_id")
        request_count = len(self.feature_indices)
        if len(self.candidate_offsets) != request_count + 1:
            raise ValueError("C39 candidate offsets differ")
        if len(self.true_history_offsets) != request_count + 1:
            raise ValueError("C39 true-history offsets differ")
        if len(self.wrong_history_offsets) != request_count + 1:
            raise ValueError("C39 wrong-history offsets differ")
        if int(self.candidate_offsets[-1]) != len(self.candidate_item_positions):
            raise ValueError("C39 candidate positions differ")
        if int(self.true_history_offsets[-1]) != len(self.true_history_item_positions):
            raise ValueError("C39 true-history positions differ")
        if int(self.wrong_history_offsets[-1]) != len(self.wrong_history_item_positions):
            raise ValueError("C39 wrong-history positions differ")
        dimension = int(config["model"]["embedding_dim"])
        if self.item_embeddings.shape != (len(self.item_ids), dimension):
            raise ValueError("C39 item embedding shape differs")
        if self.query_embeddings.shape != (request_count, dimension):
            raise ValueError("C39 query embedding shape differs")
        if len(self.base_scores) != len(self.candidate_item_positions):
            raise ValueError("C39 base score shape differs")
        if len(self.request_ids) != request_count:
            raise ValueError("C39 request id count differs")

    def role_indices(self, role: str) -> list[int]:
        indices = [int(value) for value in self.selection["roles"][role]["indices"]]
        if any(index not in self.feature_position for index in indices):
            raise PermissionError(f"C39 role {role} has no opened features")
        return indices

    def row_position(self, index: int) -> int:
        try:
            return self.feature_position[int(index)]
        except KeyError as exc:
            raise PermissionError(f"C39 feature unavailable for record {index}") from exc

    def candidate_positions(self, index: int) -> np.ndarray:
        row = self.row_position(index)
        start, stop = int(self.candidate_offsets[row]), int(self.candidate_offsets[row + 1])
        return np.asarray(self.candidate_item_positions[start:stop], dtype=np.int64)

    def candidate_ids(self, index: int) -> list[str]:
        return [self.item_ids[position] for position in self.candidate_positions(index)]

    def candidate_count(self, index: int) -> int:
        return len(self.candidate_positions(index))

    def history_positions(self, index: int, source: str) -> np.ndarray:
        row = self.row_position(index)
        if source == "true":
            offsets, values = self.true_history_offsets, self.true_history_item_positions
        elif source == "wrong":
            offsets, values = self.wrong_history_offsets, self.wrong_history_item_positions
        elif source == "none":
            return np.empty(0, dtype=np.int64)
        else:
            raise ValueError(f"unknown C39 history source: {source}")
        start, stop = int(offsets[row]), int(offsets[row + 1])
        return np.asarray(values[start:stop], dtype=np.int64)

    def has_repeat(self, index: int) -> bool:
        return bool(
            set(int(value) for value in self.candidate_positions(index))
            & set(int(value) for value in self.history_positions(index, "true"))
        )

    def query(self, index: int) -> np.ndarray:
        return np.asarray(
            self.query_embeddings[self.row_position(index)],
            dtype=np.float32,
        )

    def items(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(self.item_embeddings[positions], dtype=np.float32)

    def base_row(self, index: int) -> np.ndarray:
        row = self.row_position(index)
        start, stop = int(self.candidate_offsets[row]), int(self.candidate_offsets[row + 1])
        return np.asarray(self.base_scores[start:stop], dtype=np.float32).copy()

    def request_id(self, index: int) -> str:
        return self.request_ids[self.row_position(index)]


def open_role_labels(
    *,
    records_train_path: str | Path,
    records_train_sha256: str,
    selection_path: str | Path,
    selection_sha256: str,
    store: FrozenTransferStore,
    role: str,
) -> CompactLabels:
    """Parse only the authorized role's label-bearing JSON rows."""

    records_train_path = Path(records_train_path)
    if sha256_file(records_train_path) != records_train_sha256:
        raise RuntimeError("C39 label-bearing train records changed")
    if sha256_file(selection_path) != selection_sha256:
        raise RuntimeError("C39 selection changed before label access")
    indices = store.role_indices(role)
    offsets_by_line = _binary_line_offsets(records_train_path)
    label_offsets = [0]
    rows = []
    with records_train_path.open("rb") as handle:
        for index in indices:
            handle.seek(offsets_by_line[index])
            record = json.loads(handle.readline())
            if str(record["request_id"]) != store.request_id(index):
                raise ValueError("C39 label row request differs")
            candidates = record["candidates"]
            if [str(item["item_id"]) for item in candidates] != store.candidate_ids(index):
                raise ValueError("C39 label row candidates differ")
            labels = np.asarray([float(item["clicked"]) for item in candidates], dtype=np.float32)
            if int((labels > 0).sum()) != 1:
                raise ValueError("C39 expects exactly one positive per request")
            rows.append(labels)
            label_offsets.append(label_offsets[-1] + len(labels))
    return CompactLabels(
        request_indices=np.asarray(indices, dtype=np.int64),
        offsets=np.asarray(label_offsets, dtype=np.int64),
        values=np.concatenate(rows).astype(np.float32, copy=False),
    )


def _binary_line_offsets(path: Path) -> list[int]:
    offsets = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            offsets.append(offset)
    return offsets


def _load_position_values(path: Path, field: str) -> list[str]:
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for expected, line in enumerate(handle):
            row = json.loads(line)
            if int(row["position"]) != expected:
                raise ValueError(f"C39 {path.name} positions differ")
            output.append(str(row[field]))
    return output
