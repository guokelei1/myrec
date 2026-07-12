"""Label-free C25 selection and shared file utilities."""

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
        raise FileExistsError(f"immutable C25 output exists: {target}")
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
    if not isinstance(config, dict) or config.get("candidate_id") != "c25":
        raise ValueError("unexpected C25 config")
    if config.get("gate_id") != "c25_anchored_mobius_interaction_v1":
        raise ValueError("unexpected C25 gate")
    actual = {
        "fit": int(config["selection"]["fit_nonrepeat_requests"]),
        "internal_A": int(config["selection"]["internal_A_nonrepeat_requests"]),
        "delayed_B": int(config["selection"]["delayed_B_nonrepeat_requests"]),
        "escrow": int(config["selection"]["escrow_nonrepeat_requests"]),
        "structural_repeat": int(config["selection"]["structural_repeat_requests"]),
        "structural_nohistory": int(config["selection"]["structural_nohistory_requests"]),
    }
    if actual != ROLE_COUNTS:
        raise ValueError(f"C25 role counts changed: {actual}")
    if int(config["resources"]["physical_gpu"]) != 1:
        raise ValueError("C25 physical GPU changed")
    forbidden = ("qrels", "records_dev", "records_test", "metrics.json")
    for name, raw in config.get("paths", {}).items():
        lowered = str(raw).lower()
        if any(token in lowered for token in forbidden):
            raise ValueError(f"forbidden C25 path {name}: {raw}")
    if require_selection:
        expected = str(config["paths"].get("selection_sha256", ""))
        if len(expected) != 64 or expected == "TO_BE_FROZEN":
            raise ValueError("C25 selection hash not frozen")
        if sha256_file(config["paths"]["selection"]) != expected:
            raise ValueError("C25 selection changed")
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
            raise ValueError("C25 packed request/candidate mismatch")
        if len(self.query_indices) != len(self.request_ids) or len(self.timestamps) != len(
            self.request_ids
        ):
            raise ValueError("C25 packed request arrays mismatch")

    def candidate_indices(self, index: int) -> np.ndarray:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return np.asarray(self.candidate_embedding_indices[start:stop])

    def history_indices(self, index: int) -> np.ndarray:
        start, stop = int(self.history_offsets[index]), int(self.history_offsets[index + 1])
        return np.asarray(self.history_embedding_indices[start:stop])

    def history_count(self, index: int) -> int:
        return int(self.history_offsets[index + 1] - self.history_offsets[index])

    def repeat_candidate_count(self, index: int) -> int:
        history = self.history_indices(index)
        return int(np.isin(self.candidate_indices(index), history).sum()) if len(history) else 0


def stable_key(seed: int, role: str, request_id: str) -> tuple[bytes, str]:
    payload = (
        b"c25-anchored-mobius-v1\0"
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
            [data.request_ids[index], [str(value) for value in data.candidate_item_ids[start:stop]]],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def donor_key_sha256(data: PackedStructure, recipients: Sequence[int], donors: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for recipient, donor in zip(recipients, donors):
        payload = json.dumps(
            [data.request_ids[int(recipient)], data.request_ids[int(donor)]],
            separators=(",", ":"),
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _length_bin(length: int, edges: Sequence[int]) -> int:
    return next((position for position, edge in enumerate(edges) if length <= edge), len(edges))


def build_selection(data: PackedStructure, config: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(config["selection_seed"])
    nonrepeat: list[int] = []
    repeat: list[int] = []
    nohistory: list[int] = []
    for index in range(len(data.request_ids)):
        history_count = data.history_count(index)
        if history_count == 0:
            nohistory.append(index)
        elif data.repeat_candidate_count(index) == 0:
            nonrepeat.append(index)
        else:
            repeat.append(index)
    if (len(nonrepeat), len(repeat), len(nohistory)) != (29277, 25122, 42540):
        raise ValueError(
            f"C25 structural pools changed: {len(nonrepeat)}/{len(repeat)}/{len(nohistory)}"
        )

    ordered = sorted(
        nonrepeat,
        key=lambda index: (stable_key(seed, "outcome_partition", data.request_ids[index]), index),
    )
    offset = 0
    roles: dict[str, list[int]] = {}
    for role in ("fit", "internal_A", "delayed_B", "escrow"):
        count = ROLE_COUNTS[role]
        roles[role] = sorted(ordered[offset : offset + count])
        offset += count
    roles["structural_repeat"] = sorted(
        sorted(
            repeat,
            key=lambda index: (stable_key(seed, "structural_repeat", data.request_ids[index]), index),
        )[: ROLE_COUNTS["structural_repeat"]]
    )
    roles["structural_nohistory"] = sorted(
        sorted(
            nohistory,
            key=lambda index: (
                stable_key(seed, "structural_nohistory", data.request_ids[index]),
                index,
            ),
        )[: ROLE_COUNTS["structural_nohistory"]]
    )
    flat = [index for values in roles.values() for index in values]
    if len(flat) != len(set(flat)) or {name: len(values) for name, values in roles.items()} != ROLE_COUNTS:
        raise AssertionError("C25 roles overlap or counts differ")

    outcome_set = set(flat)
    reserve = [index for index in nonrepeat if index not in outcome_set]
    edges = [int(value) for value in config["selection"]["donor_length_bins"]]
    quantiles = int(config["selection"]["donor_time_quantiles"])
    boundaries = np.quantile(
        np.asarray([data.timestamps[index] for index in nonrepeat], dtype=np.float64),
        np.linspace(0.0, 1.0, quantiles + 1)[1:-1],
    )

    def bucket(index: int) -> tuple[int, int]:
        return (
            _length_bin(data.history_count(index), edges),
            int(np.searchsorted(boundaries, float(data.timestamps[index]), side="right")),
        )

    grouped: dict[tuple[int, int], list[int]] = {}
    length_grouped: dict[int, list[int]] = {}
    for donor in reserve:
        grouped.setdefault(bucket(donor), []).append(donor)
        length_grouped.setdefault(bucket(donor)[0], []).append(donor)
    for values in (*grouped.values(), *length_grouped.values()):
        values.sort(key=lambda index: (stable_key(seed, "donor_pool", data.request_ids[index]), index))
    reserve.sort(key=lambda index: (stable_key(seed, "donor_fallback", data.request_ids[index]), index))

    def donor_for(recipient: int) -> int:
        candidates = grouped.get(bucket(recipient), [])
        if len(candidates) < 2:
            candidates = length_grouped.get(bucket(recipient)[0], [])
        if not candidates:
            candidates = reserve
        start = int.from_bytes(
            stable_key(seed, "donor_start", data.request_ids[recipient])[0][:8], "big"
        ) % len(candidates)
        recipient_candidates = set(int(value) for value in data.candidate_indices(recipient))
        for step in range(len(candidates)):
            donor = int(candidates[(start + step) % len(candidates)])
            if donor != recipient and recipient_candidates.isdisjoint(
                int(value) for value in data.history_indices(donor)
            ):
                return donor
        raise RuntimeError(f"C25 donor unavailable: {recipient}")

    donors = {
        role: [donor_for(index) for index in roles[role]]
        for role in ("fit", "internal_A", "delayed_B")
    }
    for role, values in donors.items():
        recipients = roles[role]
        if any(donor in outcome_set for donor in values):
            raise AssertionError("C25 donor intersects outcome roles")
        if any(
            not set(int(value) for value in data.candidate_indices(recipient)).isdisjoint(
                int(value) for value in data.history_indices(donor)
            )
            for recipient, donor in zip(recipients, values)
        ):
            raise AssertionError("C25 donor history repeats recipient candidate")

    return {
        "candidate_id": "c25",
        "selection_id": "c25_anchored_mobius_selection_v1",
        "status": "frozen_before_any_c25_label_or_outcome",
        "seed": seed,
        "pool_counts": {
            "strict_nonrepeat_history_present": len(nonrepeat),
            "repeat_present": len(repeat),
            "nohistory": len(nohistory),
            "donor_reserve": len(reserve),
        },
        "roles": {
            role: {
                "indices": values,
                "request_ids": [data.request_ids[index] for index in values],
                "candidate_key_sha256": candidate_key_sha256(data, values),
            }
            for role, values in roles.items()
        },
        "wrong_history_donors": {
            role: {
                "indices": values,
                "request_ids": [data.request_ids[index] for index in values],
                "mapping_sha256": donor_key_sha256(data, roles[role], values),
            }
            for role, values in donors.items()
        },
        "donor_matching": {
            "history_length_edges": edges,
            "timestamp_quantiles": quantiles,
            "recipient_candidate_overlap_forbidden": True,
            "outcome_role_donors_forbidden": True,
        },
        "checks": {
            "labels_opened": False,
            "roles_pairwise_disjoint": True,
            "strict_nonrepeat_fit_A_B_escrow": True,
            "donors_outside_outcome_roles": True,
            "donor_candidate_overlap_zero": True,
            "dev_test_qrels_metrics_read": False,
        },
    }
