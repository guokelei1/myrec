"""C26 frozen selection and file utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
ROLE_COUNTS = {
    "fit": 6000,
    "internal_A": 1200,
    "delayed_B": 1200,
    "escrow": 1200,
    "structural_repeat": 512,
    "structural_nohistory": 512,
}
FEATURE_ROLES = (
    "fit",
    "internal_A",
    "delayed_B",
    "structural_repeat",
    "structural_nohistory",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"immutable C26 output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(target)


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(target)


def load_config(path: str | Path, *, require_selection: bool = False) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("candidate_id") != "c26":
        raise ValueError("unexpected C26 config")
    if config.get("gate_id") != "c26_query_pivot_token_bridge_v1":
        raise ValueError("unexpected C26 gate")
    actual = {
        "fit": int(config["selection"]["fit_requests"]),
        "internal_A": int(config["selection"]["internal_A_requests"]),
        "delayed_B": int(config["selection"]["delayed_B_requests"]),
        "escrow": int(config["selection"]["escrow_requests"]),
        "structural_repeat": int(config["selection"]["structural_repeat_requests"]),
        "structural_nohistory": int(config["selection"]["structural_nohistory_requests"]),
    }
    if actual != ROLE_COUNTS or int(config["resources"]["physical_gpu"]) != 2:
        raise ValueError("C26 frozen role/GPU registration differs")
    forbidden = ("qrels", "records_train", "records_dev", "records_test", "metrics.json")
    for name, raw in config.get("paths", {}).items():
        if any(token in str(raw).lower() for token in forbidden):
            raise ValueError(f"forbidden C26 path {name}: {raw}")
    if require_selection:
        expected = str(config["paths"].get("selection_sha256", ""))
        if len(expected) != 64 or expected == "TO_BE_FROZEN":
            raise ValueError("C26 selection hash not frozen")
        if sha256_file(config["paths"]["selection"]) != expected:
            raise ValueError("C26 selection changed")
    return config


class PackedStructure:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.request_ids = [
            str(json.loads(line)["request_id"])
            for line in (self.root / "request_ids.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.candidate_offsets = np.load(self.root / "candidate_offsets.npy", mmap_mode="r")
        self.candidate_embedding_indices = np.load(
            self.root / "candidate_embedding_indices.npy", mmap_mode="r"
        )
        self.candidate_item_ids = np.load(self.root / "candidate_item_ids.npy", mmap_mode="r")
        self.history_offsets = np.load(self.root / "history_offsets.npy", mmap_mode="r")
        self.history_embedding_indices = np.load(
            self.root / "history_embedding_indices.npy", mmap_mode="r"
        )
        self.history_event_weights = np.load(
            self.root / "history_event_weights.npy", mmap_mode="r"
        )
        if len(self.request_ids) + 1 != len(self.candidate_offsets):
            raise ValueError("C26 packed request/candidate mismatch")

    def candidate_indices(self, index: int) -> np.ndarray:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return np.asarray(self.candidate_embedding_indices[start:stop])

    def history_indices(self, index: int) -> np.ndarray:
        start, stop = int(self.history_offsets[index]), int(self.history_offsets[index + 1])
        return np.asarray(self.history_embedding_indices[start:stop])


def candidate_key_sha256(data: PackedStructure, indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for raw_index in indices:
        index = int(raw_index)
        start, stop = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
        payload = json.dumps(
            [data.request_ids[index], [str(value) for value in data.candidate_item_ids[start:stop]]],
            separators=(",", ":"),
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
