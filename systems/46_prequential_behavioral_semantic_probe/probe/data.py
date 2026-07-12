from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


class PackedTrain:
    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        self.root = root
        self.request_ids = [json.loads(line)["request_id"] for line in (root / "request_ids.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.timestamps = np.load(root / "timestamps.npy", mmap_mode="r")
        self.candidate_offsets = np.load(root / "candidate_offsets.npy", mmap_mode="r")
        self.candidate_indices = np.load(root / "candidate_embedding_indices.npy", mmap_mode="r")
        self.candidate_item_ids = np.load(root / "candidate_item_ids.npy", mmap_mode="r")
        self.history_offsets = np.load(root / "history_offsets.npy", mmap_mode="r")
        self.history_indices = np.load(root / "history_embedding_indices.npy", mmap_mode="r")
        count = len(self.request_ids)
        if len(self.timestamps) != count or len(self.candidate_offsets) != count + 1 or len(self.history_offsets) != count + 1:
            raise ValueError("C46 packed train cardinality differs")

    def candidates(self, index: int) -> np.ndarray:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return np.asarray(self.candidate_indices[start:stop], dtype=np.int64)

    def candidate_ids(self, index: int) -> np.ndarray:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return np.asarray(self.candidate_item_ids[start:stop]).copy()

    def history(self, index: int) -> np.ndarray:
        start, stop = int(self.history_offsets[index]), int(self.history_offsets[index + 1])
        return np.asarray(self.history_indices[start:stop], dtype=np.int64)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_key_sha256(data: PackedTrain, indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        row = [data.request_ids[int(index)], *[str(x) for x in data.candidate_ids(int(index))]]
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, request_id: str) -> bytes:
    return hashlib.sha256(f"c46:{seed}:{namespace}:{request_id}".encode()).digest()


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
