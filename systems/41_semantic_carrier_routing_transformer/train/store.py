"""C41 frozen embedding store and role-scoped label opener."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


@dataclass(frozen=True)
class CompactLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.request_indices) + 1:
            raise ValueError("C41 label offsets differ")
        if int(self.offsets[-1]) != len(self.values):
            raise ValueError("C41 label values differ")

    @property
    def positions(self) -> dict[int, int]:
        return {
            int(value): position
            for position, value in enumerate(self.request_indices)
        }

    def row(self, index: int, expected_count: int) -> np.ndarray:
        position = self.positions.get(int(index))
        if position is None:
            raise PermissionError(f"C41 labels unavailable for {index}")
        start, stop = int(self.offsets[position]), int(self.offsets[position + 1])
        if stop - start != expected_count:
            raise ValueError("C41 label candidate count differs")
        return np.asarray(self.values[start:stop], dtype=np.float32).copy()


class FrozenStore:
    def __init__(self, config: Mapping[str, Any]) -> None:
        paths = config["paths"]
        self.selection = read_json(paths["selection"])
        root = Path(paths["feature_root"])
        manifest = read_json(root / "embedding_manifest.json")
        if not manifest.get("finite", False):
            raise RuntimeError("C41 embeddings are not finite")
        self.feature_indices = np.load(
            root / "feature_request_indices.npy", mmap_mode="r"
        )
        self.feature_position = {
            int(index): position
            for position, index in enumerate(self.feature_indices)
        }
        if len(self.feature_position) != len(self.feature_indices):
            raise ValueError("C41 feature indices overlap")
        self.candidate_offsets = np.load(root / "candidate_offsets.npy", mmap_mode="r")
        self.candidate_item_positions = np.load(
            root / "candidate_item_positions.npy", mmap_mode="r"
        )
        self.true_history_offsets = np.load(
            root / "true_history_offsets.npy", mmap_mode="r"
        )
        self.true_history_item_positions = np.load(
            root / "true_history_item_positions.npy", mmap_mode="r"
        )
        self.wrong_history_offsets = np.load(
            root / "wrong_history_offsets.npy", mmap_mode="r"
        )
        self.wrong_history_item_positions = np.load(
            root / "wrong_history_item_positions.npy", mmap_mode="r"
        )
        self.item_embeddings = np.load(root / "item_embeddings.npy", mmap_mode="r")
        self.query_embeddings = np.load(root / "query_embeddings.npy", mmap_mode="r")
        self.base_scores = np.load(root / "base_scores.npy", mmap_mode="r")
        self.item_ids = _load_position_values(root / "items.jsonl", "item_id")
        self.request_ids = _load_position_values(root / "requests.jsonl", "request_id")
        count = len(self.feature_indices)
        for offsets in (
            self.candidate_offsets,
            self.true_history_offsets,
            self.wrong_history_offsets,
        ):
            if len(offsets) != count + 1:
                raise ValueError("C41 offsets differ")
        if int(self.candidate_offsets[-1]) != len(self.candidate_item_positions):
            raise ValueError("C41 candidate positions differ")
        if int(self.true_history_offsets[-1]) != len(self.true_history_item_positions):
            raise ValueError("C41 true history positions differ")
        if int(self.wrong_history_offsets[-1]) != len(self.wrong_history_item_positions):
            raise ValueError("C41 wrong history positions differ")
        dim = int(config["model"]["embedding_dim"])
        if self.item_embeddings.shape != (len(self.item_ids), dim):
            raise ValueError("C41 item embedding shape differs")
        if self.query_embeddings.shape != (count, dim):
            raise ValueError("C41 query embedding shape differs")
        if len(self.request_ids) != count:
            raise ValueError("C41 request IDs differ")

    def role_indices(self, role: str) -> list[int]:
        indices = [int(value) for value in self.selection["roles"][role]["indices"]]
        if any(index not in self.feature_position for index in indices):
            raise PermissionError(f"C41 role {role} has no features")
        return indices

    def row_position(self, index: int) -> int:
        try:
            return self.feature_position[int(index)]
        except KeyError as exc:
            raise PermissionError(f"C41 feature unavailable for {index}") from exc

    def _positions(self, index: int, offsets: np.ndarray, values: np.ndarray) -> np.ndarray:
        row = self.row_position(index)
        start, stop = int(offsets[row]), int(offsets[row + 1])
        return np.asarray(values[start:stop], dtype=np.int64)

    def candidate_positions(self, index: int) -> np.ndarray:
        return self._positions(index, self.candidate_offsets, self.candidate_item_positions)

    def history_positions(self, index: int, source: str) -> np.ndarray:
        if source == "true":
            return self._positions(
                index, self.true_history_offsets, self.true_history_item_positions
            )
        if source == "wrong":
            return self._positions(
                index, self.wrong_history_offsets, self.wrong_history_item_positions
            )
        if source == "none":
            return np.empty(0, dtype=np.int64)
        raise ValueError(f"unknown C41 history source: {source}")

    def candidate_ids(self, index: int) -> list[str]:
        return [self.item_ids[position] for position in self.candidate_positions(index)]

    def candidate_count(self, index: int) -> int:
        return len(self.candidate_positions(index))

    def has_repeat(self, index: int) -> bool:
        return bool(
            set(self.candidate_positions(index).tolist())
            & set(self.history_positions(index, "true").tolist())
        )

    def query(self, index: int) -> np.ndarray:
        return np.asarray(
            self.query_embeddings[self.row_position(index)], dtype=np.float32
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
    store: FrozenStore,
    role: str,
) -> CompactLabels:
    records_train_path = Path(records_train_path)
    if sha256_file(records_train_path) != records_train_sha256:
        raise RuntimeError("C41 label-bearing records changed")
    if sha256_file(selection_path) != selection_sha256:
        raise RuntimeError("C41 selection changed")
    indices = store.role_indices(role)
    offsets_by_line = _binary_line_offsets(records_train_path)
    offsets = [0]
    rows = []
    with records_train_path.open("rb") as handle:
        for index in indices:
            handle.seek(offsets_by_line[index])
            record = json.loads(handle.readline())
            if str(record["request_id"]) != store.request_id(index):
                raise ValueError("C41 label request differs")
            candidates = record["candidates"]
            if [str(item["item_id"]) for item in candidates] != store.candidate_ids(index):
                raise ValueError("C41 label candidates differ")
            labels = np.asarray(
                [float(item["clicked"]) for item in candidates], dtype=np.float32
            )
            if int((labels > 0).sum()) != 1:
                raise ValueError("C41 expects one positive")
            rows.append(labels)
            offsets.append(offsets[-1] + len(labels))
    return CompactLabels(
        request_indices=np.asarray(indices, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
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
                raise ValueError(f"C41 {path.name} positions differ")
            output.append(str(row[field]))
    return output
