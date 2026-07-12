"""Label-free C24 selection and shared file utilities."""

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
    "internal_A": 600,
    "escrow": 340,
    "structural_single_repeat": 512,
    "structural_nohistory": 512,
    "structural_nonrepeat": 512,
}


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
        raise FileExistsError(f"immutable C24 output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")
    temporary.replace(target)


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")
    temporary.replace(target)


def load_config(path: str | Path, *, require_selection: bool = False) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("candidate_id") != "c24":
        raise ValueError("unexpected C24 config")
    if config.get("gate_id") != "c24_multi_recurrence_competition_v1":
        raise ValueError("unexpected C24 gate")
    actual = {
        "fit": int(config["selection"]["fit_multi_repeat_requests"]),
        "internal_A": int(config["selection"]["internal_A_multi_repeat_requests"]),
        "escrow": int(config["selection"]["escrow_multi_repeat_requests"]),
        "structural_single_repeat": int(
            config["selection"]["structural_single_repeat_requests"]
        ),
        "structural_nohistory": int(config["selection"]["structural_nohistory_requests"]),
        "structural_nonrepeat": int(config["selection"]["structural_nonrepeat_requests"]),
    }
    if actual != ROLE_COUNTS:
        raise ValueError(f"C24 role counts changed: {actual}")
    if int(config["resources"]["physical_gpu"]) != 0:
        raise ValueError("C24 physical GPU changed")
    forbidden = ("qrels", "records_dev", "records_test", "metrics.json")
    for name, raw in config.get("paths", {}).items():
        lowered = str(raw).lower()
        if any(token in lowered for token in forbidden):
            raise ValueError(f"forbidden C24 path {name}: {raw}")
    if require_selection:
        expected = str(config["paths"].get("selection_sha256", ""))
        if len(expected) != 64 or expected == "TO_BE_FROZEN":
            raise ValueError("C24 selection hash not frozen")
        if sha256_file(config["paths"]["selection"]) != expected:
            raise ValueError("C24 selection changed")
    return config


class PackedStructure:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.request_ids = [
            str(json.loads(line)["request_id"])
            for line in (self.root / "request_ids.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.query_indices = np.load(self.root / "query_indices.npy", mmap_mode="r")
        self.timestamps = np.load(self.root / "timestamps.npy", mmap_mode="r")
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
            raise ValueError("C24 packed request/candidate mismatch")
        if len(self.query_indices) != len(self.request_ids) or len(self.timestamps) != len(
            self.request_ids
        ):
            raise ValueError("C24 packed request arrays mismatch")

    def candidate_indices(self, index: int) -> np.ndarray:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return np.asarray(self.candidate_embedding_indices[start:stop])

    def history_indices(self, index: int) -> np.ndarray:
        start, stop = int(self.history_offsets[index]), int(self.history_offsets[index + 1])
        return np.asarray(self.history_embedding_indices[start:stop])

    def repeat_candidate_count(self, index: int) -> int:
        history = self.history_indices(index)
        if len(history) == 0:
            return 0
        return int(np.isin(self.candidate_indices(index), history).sum())


def stable_key(seed: int, role: str, request_id: str) -> tuple[bytes, str]:
    payload = (
        b"c24-multi-recurrence-v1\0"
        + str(seed).encode("ascii")
        + b"\0"
        + role.encode("utf-8")
        + b"\0"
        + request_id.encode("utf-8")
    )
    return hashlib.sha256(payload).digest(), request_id


def candidate_key_sha256(data: PackedStructure, indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for raw_index in indices:
        index = int(raw_index)
        start, stop = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
        payload = json.dumps(
            [
                data.request_ids[index],
                [str(value) for value in data.candidate_item_ids[start:stop]],
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_selection(
    data: PackedStructure,
    c23: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    c23_fit = [int(value) for value in c23["roles"]["fit"]["indices"]]
    delayed_source = [
        int(value)
        for role in ("delayed_B", "escrow")
        for value in c23["roles"][role]["indices"]
    ]
    fit_multi = [index for index in c23_fit if data.repeat_candidate_count(index) >= 2]
    delayed_multi = [
        index for index in delayed_source if data.repeat_candidate_count(index) >= 2
    ]
    fit_single = [index for index in c23_fit if data.repeat_candidate_count(index) == 1]
    if len(fit_multi) != 6785 or len(delayed_multi) != 940:
        raise ValueError(
            f"C24 structural pool changed: fit={len(fit_multi)} delayed={len(delayed_multi)}"
        )

    def ranked(pool: Sequence[int], role: str) -> list[int]:
        return sorted(
            pool,
            key=lambda index: (stable_key(seed, role, data.request_ids[index]), index),
        )

    fit = sorted(ranked(fit_multi, "fit")[: ROLE_COUNTS["fit"]])
    delayed_order = ranked(delayed_multi, "delayed_partition")
    internal = sorted(delayed_order[: ROLE_COUNTS["internal_A"]])
    escrow = sorted(delayed_order[ROLE_COUNTS["internal_A"] :])
    single = sorted(ranked(fit_single, "structural_single_repeat")[:512])
    nohistory = [
        int(value) for value in c23["roles"]["structural_nohistory"]["indices"]
    ]
    nonrepeat = [
        int(value) for value in c23["roles"]["structural_nonrepeat"]["indices"]
    ]
    roles = {
        "fit": fit,
        "internal_A": internal,
        "escrow": escrow,
        "structural_single_repeat": single,
        "structural_nohistory": nohistory,
        "structural_nonrepeat": nonrepeat,
    }
    if {role: len(values) for role, values in roles.items()} != ROLE_COUNTS:
        raise AssertionError("C24 selected role counts differ")
    if set(internal) & set(escrow) or set(internal) | set(escrow) != set(delayed_multi):
        raise AssertionError("C24 delayed partition differs")
    flat = [value for values in roles.values() for value in values]
    if len(flat) != len(set(flat)):
        raise AssertionError("C24 roles overlap")
    return {
        "candidate_id": "c24",
        "selection_id": "c24_multi_recurrence_selection_v1",
        "status": "frozen_before_any_c24_delayed_label_or_outcome",
        "seed": int(seed),
        "pool_counts": {
            "c23_fit_multi_repeat": len(fit_multi),
            "c23_delayed_plus_escrow_multi_repeat_unopened": len(delayed_multi),
            "c23_fit_single_repeat": len(fit_single),
        },
        "roles": {
            role: {
                "indices": values,
                "request_ids": [data.request_ids[index] for index in values],
                "candidate_key_sha256": candidate_key_sha256(data, values),
            }
            for role, values in roles.items()
        },
        "checks": {
            "c23_fit_labels_previously_opened": True,
            "c23_delayed_escrow_labels_opened": False,
            "c24_internal_A_labels_opened": False,
            "roles_pairwise_disjoint": True,
            "delayed_multi_pool_exhaustively_partitioned": True,
            "dev_test_qrels_metrics_read": False,
        },
    }
