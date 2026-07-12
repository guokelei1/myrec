"""Label-free packed structure and immutable C23 selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
FORBIDDEN_PATH_PARTS = ("qrels", "records_dev", "records_test", "metrics.json")
ROLE_COUNTS = {
    "fit": 12_000,
    "internal_A": 1_200,
    "delayed_B": 600,
    "escrow": 958,
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
        raise FileExistsError(f"immutable output already exists: {target}")
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


def load_config(path: str | Path, *, require_frozen_selection: bool = False) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("C23 config must be a mapping")
    if config.get("candidate_id") != "c23" or config.get("gate_id") != (
        "c23_recurrence_reset_train_gate_v1"
    ):
        raise ValueError("unexpected C23 config identity")
    actual_counts = {
        "fit": int(config["selection"]["fit_repeat_requests"]),
        "internal_A": int(config["selection"]["internal_A_repeat_requests"]),
        "delayed_B": int(config["selection"]["delayed_B_repeat_requests"]),
        "escrow": int(config["selection"]["escrow_repeat_requests"]),
        "structural_nohistory": int(
            config["selection"]["structural_nohistory_requests"]
        ),
        "structural_nonrepeat": int(
            config["selection"]["structural_nonrepeat_requests"]
        ),
    }
    if actual_counts != ROLE_COUNTS:
        raise ValueError(f"C23 role counts changed: {actual_counts}")
    if config["selection"].get("freeze_before_any_label_shaped_array") is not True:
        raise ValueError("C23 label barrier disabled")
    if int(config["resources"].get("physical_gpu", -1)) != 3:
        raise ValueError("C23 physical GPU assignment changed")
    for name, raw in config.get("paths", {}).items():
        lowered = str(raw).lower()
        if any(part in lowered for part in FORBIDDEN_PATH_PARTS):
            raise ValueError(f"forbidden C23 path {name}: {raw}")
        if "_final_" in lowered and name == "calibration_checkpoint":
            raise ValueError("C23 may not use a final D2 checkpoint")
        if "full_train" in lowered and name == "internal_train_popularity":
            raise ValueError("C23 may not use full-train popularity")
    if require_frozen_selection:
        expected = str(config["paths"].get("selection_sha256", ""))
        if len(expected) != 64 or expected == "TO_BE_FROZEN":
            raise ValueError("C23 selection hash is not frozen in config")
        if sha256_file(config["paths"]["selection"]) != expected:
            raise ValueError("C23 selection changed after freeze")
    return config


@dataclass(frozen=True)
class PackedStructure:
    root: Path
    request_ids: list[str]
    query_indices: np.ndarray
    timestamps: np.ndarray
    candidate_offsets: np.ndarray
    candidate_embedding_indices: np.ndarray
    candidate_item_ids: np.ndarray
    history_offsets: np.ndarray
    history_embedding_indices: np.ndarray
    history_event_weights: np.ndarray

    @classmethod
    def load(cls, root: str | Path) -> "PackedStructure":
        root = Path(root)
        request_ids: list[str] = []
        with (root / "request_ids.jsonl").open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                request_id = row.get("request_id") if isinstance(row, dict) else None
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError(f"invalid request ID at line {line_number}")
                request_ids.append(request_id)
        value = cls(
            root=root,
            request_ids=request_ids,
            query_indices=np.load(root / "query_indices.npy", mmap_mode="r"),
            timestamps=np.load(root / "timestamps.npy", mmap_mode="r"),
            candidate_offsets=np.load(root / "candidate_offsets.npy", mmap_mode="r"),
            candidate_embedding_indices=np.load(
                root / "candidate_embedding_indices.npy", mmap_mode="r"
            ),
            candidate_item_ids=np.load(root / "candidate_item_ids.npy", mmap_mode="r"),
            history_offsets=np.load(root / "history_offsets.npy", mmap_mode="r"),
            history_embedding_indices=np.load(
                root / "history_embedding_indices.npy", mmap_mode="r"
            ),
            history_event_weights=np.load(
                root / "history_event_weights.npy", mmap_mode="r"
            ),
        )
        value.validate()
        return value

    def validate(self) -> None:
        requests = len(self.request_ids)
        if requests != len(set(self.request_ids)):
            raise ValueError("packed request IDs are not unique")
        if len(self.query_indices) != requests or len(self.timestamps) != requests:
            raise ValueError("request-shaped structural arrays differ")
        if len(self.candidate_offsets) != requests + 1:
            raise ValueError("candidate offsets differ")
        if len(self.history_offsets) != requests + 1:
            raise ValueError("history offsets differ")
        if int(self.candidate_offsets[-1]) != len(self.candidate_embedding_indices):
            raise ValueError("candidate embedding rows differ")
        if int(self.candidate_offsets[-1]) != len(self.candidate_item_ids):
            raise ValueError("candidate item rows differ")
        if int(self.history_offsets[-1]) != len(self.history_embedding_indices):
            raise ValueError("history item rows differ")
        if int(self.history_offsets[-1]) != len(self.history_event_weights):
            raise ValueError("history event rows differ")

    def __len__(self) -> int:
        return len(self.request_ids)

    def candidate_indices(self, index: int) -> np.ndarray:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return np.asarray(self.candidate_embedding_indices[start:stop])

    def history_indices(self, index: int) -> np.ndarray:
        start, stop = int(self.history_offsets[index]), int(self.history_offsets[index + 1])
        return np.asarray(self.history_embedding_indices[start:stop])

    def stratum(self, index: int) -> str:
        history = self.history_indices(index)
        if len(history) == 0:
            return "nohistory"
        candidate = self.candidate_indices(index)
        if np.isin(candidate, history).any():
            return "repeat"
        return "nonrepeat"


def stable_key(seed: int, role: str, request_id: str) -> tuple[bytes, str]:
    payload = (
        b"c23-recurrence-reset-v1\0"
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


def identity_witness_sha256(data: PackedStructure, indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for raw_index in indices:
        index = int(raw_index)
        payload = json.dumps(
            [
                data.request_ids[index],
                [int(value) for value in data.candidate_indices(index)],
                [int(value) for value in data.history_indices(index)],
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def selected_ids_from_prior(path: str | Path) -> set[str]:
    value = read_json(path)
    found: set[str] = set()
    for row in value.values():
        if isinstance(row, dict) and isinstance(row.get("request_ids"), list):
            found.update(str(item) for item in row["request_ids"])
    roles = value.get("roles")
    if isinstance(roles, dict):
        for row in roles.values():
            if isinstance(row, dict) and isinstance(row.get("request_ids"), list):
                found.update(str(item) for item in row["request_ids"])
    if not found:
        raise ValueError(f"prior selection has no request IDs: {path}")
    return found


def build_selection(
    data: PackedStructure,
    *,
    cut: int,
    seed: int,
    blacklist: set[str],
) -> dict[str, Any]:
    if not 0 < cut < len(data):
        raise ValueError("invalid train cut")
    pools = {
        "pre_repeat": [i for i in range(cut) if data.stratum(i) == "repeat"],
        "post_repeat": [i for i in range(cut, len(data)) if data.stratum(i) == "repeat"],
        "post_nohistory": [
            i
            for i in range(cut, len(data))
            if data.stratum(i) == "nohistory" and data.request_ids[i] not in blacklist
        ],
        "pre_nonrepeat": [
            i
            for i in range(cut)
            if data.stratum(i) == "nonrepeat" and data.request_ids[i] not in blacklist
        ],
    }
    used: set[int] = set()

    def take(pool_name: str, role: str) -> list[int]:
        ranked = sorted(
            (i for i in pools[pool_name] if i not in used),
            key=lambda i: (stable_key(seed, role, data.request_ids[i]), i),
        )
        count = ROLE_COUNTS[role]
        if len(ranked) < count:
            raise ValueError(f"insufficient {pool_name} rows for {role}: {len(ranked)}")
        chosen = sorted(ranked[:count])
        used.update(chosen)
        return chosen

    roles = {
        "fit": take("pre_repeat", "fit"),
        "internal_A": take("post_repeat", "internal_A"),
        "delayed_B": take("post_repeat", "delayed_B"),
        "escrow": take("post_repeat", "escrow"),
        "structural_nohistory": take("post_nohistory", "structural_nohistory"),
        "structural_nonrepeat": take("pre_nonrepeat", "structural_nonrepeat"),
    }
    flat = [index for indices in roles.values() for index in indices]
    if len(flat) != len(set(flat)):
        raise AssertionError("C23 selection roles overlap")
    repeat_roles = ("fit", "internal_A", "delayed_B", "escrow")
    if any(data.stratum(i) != "repeat" for role in repeat_roles for i in roles[role]):
        raise AssertionError("C23 repeat role contains a non-repeat request")
    if any(data.request_ids[i] in blacklist for role in repeat_roles for i in roles[role]):
        raise AssertionError("C23 repeat role overlaps registered prior selection")
    if len(pools["post_repeat"]) != sum(ROLE_COUNTS[role] for role in repeat_roles[1:]):
        raise AssertionError("C23 delayed post-cut repeat partition is not exhaustive")

    role_rows: dict[str, Any] = {}
    for role, indices in roles.items():
        role_rows[role] = {
            "indices": indices,
            "request_ids": [data.request_ids[i] for i in indices],
            "candidate_key_sha256": candidate_key_sha256(data, indices),
            "identity_witness_sha256": identity_witness_sha256(data, indices),
        }
    return {
        "candidate_id": "c23",
        "selection_id": "c23_recurrence_reset_selection_v1",
        "status": "frozen_before_any_c23_label_or_outcome",
        "seed": int(seed),
        "cut": int(cut),
        "hash_rule": (
            "ascending sha256(c23-recurrence-reset-v1\\0seed\\0role\\0request_id), "
            "then packed index"
        ),
        "pool_counts": {name: len(values) for name, values in pools.items()},
        "roles": role_rows,
        "checks": {
            "labels_opened_before_selection": False,
            "repeat_roles_pairwise_disjoint": True,
            "post_repeat_partition_exhaustive": True,
            "registered_prior_selection_overlap": 0,
            "dev_test_qrels_metrics_read": False,
        },
    }
