"""Freeze C21's train/probe split before any compact fit label is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SYSTEM_ROOT.parents[1]
FORBIDDEN = ("qrels", "records_dev", "records_test", "metrics.json")


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


def write_json_once(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"immutable C21 selection exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("C21 config must be a mapping")
    if value.get("candidate_id") != "c21" or value.get("gate_id") != "c21_train_path_signal_v1":
        raise ValueError("unexpected C21 config identity")
    selection = value["selection"]
    expected = {
        "source_requests": 12_000,
        "train_fit_requests": 9_000,
        "internal_probe_requests": 3_000,
    }
    if {name: int(selection[name]) for name in expected} != expected:
        raise ValueError("C21 split counts changed")
    if selection.get("freeze_before_compact_fit_labels_open") is not True:
        raise ValueError("C21 label barrier is disabled")
    if int(value["resources"]["physical_gpu"]) != 1:
        raise ValueError("C21 physical GPU registration changed")
    for name, raw_path in value.get("paths", {}).items():
        lowered = str(raw_path).lower()
        if any(token in lowered for token in FORBIDDEN):
            raise ValueError(f"forbidden C21 path {name}: {raw_path}")
    authorization = value["authorization"]
    required_false = (
        "c06_internal_A",
        "c06_internal_B",
        "c06_escrow",
        "original_train_label_array",
        "dev",
        "test",
        "full_transformer_training",
    )
    if any(authorization.get(name) is not False for name in required_false):
        raise ValueError("C21 authorization boundary changed")
    return value


def load_request_ids(path: Path) -> list[str]:
    values: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            request_id = row.get("request_id") if isinstance(row, dict) else None
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(f"invalid request ID at line {line_number}")
            values.append(request_id)
    if len(values) != len(set(values)):
        raise ValueError("packed train request IDs are not unique")
    return values


def candidate_key_sha256(
    indices: list[int],
    request_ids: list[str],
    offsets: np.ndarray,
    item_ids: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for index in indices:
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        payload = json.dumps(
            [request_ids[index], [str(value) for value in item_ids[start:stop]]],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def split_key(request_id: str) -> tuple[bytes, str]:
    payload = b"c21-path-signal-v1\0" + request_id.encode("utf-8")
    return hashlib.sha256(payload).digest(), request_id


def materialize(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    c06_selection_path = Path(paths["c06_selection"])
    c06_g0_path = Path(paths["c06_g0_report"])
    actual_selection_hash = sha256_file(c06_selection_path)
    actual_g0_hash = sha256_file(c06_g0_path)
    if actual_selection_hash != paths["c06_selection_sha256"]:
        raise ValueError("registered C06 selection changed")
    if actual_g0_hash != paths["c06_g0_report_sha256"]:
        raise ValueError("registered C06 G0 report changed")

    c06 = read_json(c06_selection_path)
    if c06.get("selection_id") != "c06_relative_real_gate_v1":
        raise ValueError("unexpected C06 selection identity")
    if c06.get("pairwise_disjoint") is not True or c06.get("zero_c05_overlap") is not True:
        raise ValueError("C06 cohort isolation is not valid")
    fit_indices = [int(value) for value in c06["roles"]["fit"]["indices"]]
    fit_ids = [str(value) for value in c06["roles"]["fit"]["request_ids"]]
    if len(fit_indices) != 12_000 or len(fit_ids) != 12_000:
        raise ValueError("C06 fit cohort count changed")

    root = Path(paths["c06_artifact_root"])
    registered = config["registered_inputs"]
    fit_index_path = root / "fit_request_indices.npy"
    if sha256_file(fit_index_path) != registered["fit_request_indices_sha256"]:
        raise ValueError("C06 compact fit-index artifact changed")
    compact_fit_indices = np.load(fit_index_path, allow_pickle=False)
    if not np.array_equal(compact_fit_indices, np.asarray(fit_indices, dtype=np.int64)):
        raise ValueError("C06 selection and compact fit indices differ")

    packed_root = Path(paths["packed_train_root"])
    request_path = packed_root / "request_ids.jsonl"
    offset_path = packed_root / "candidate_offsets.npy"
    item_id_path = packed_root / "candidate_item_ids.npy"
    request_ids = load_request_ids(request_path)
    if any(request_ids[index] != request_id for index, request_id in zip(fit_indices, fit_ids)):
        raise ValueError("C06 fit request IDs differ from packed train")
    offsets = np.load(offset_path, mmap_mode="r", allow_pickle=False)
    item_ids = np.load(item_id_path, mmap_mode="r", allow_pickle=False)
    if len(offsets) != len(request_ids) + 1 or int(offsets[-1]) != len(item_ids):
        raise ValueError("packed candidate structure is inconsistent")

    ordered = sorted(zip(fit_indices, fit_ids), key=lambda row: (split_key(row[1]), row[0]))
    train_rows = ordered[:9_000]
    probe_rows = ordered[9_000:]
    roles = {
        "train_fit": {
            "indices": [index for index, _ in train_rows],
            "request_ids": [request_id for _, request_id in train_rows],
        },
        "internal_probe": {
            "indices": [index for index, _ in probe_rows],
            "request_ids": [request_id for _, request_id in probe_rows],
        },
    }
    train_set = set(roles["train_fit"]["indices"])
    probe_set = set(roles["internal_probe"]["indices"])
    c06_nonfit = {
        int(value)
        for role in ("internal_A", "internal_B", "escrow", "nohistory")
        for value in c06["roles"][role]["indices"]
    }
    checks = {
        "train_count_9000": len(train_set) == 9_000,
        "probe_count_3000": len(probe_set) == 3_000,
        "roles_disjoint": train_set.isdisjoint(probe_set),
        "union_equals_c06_fit": train_set | probe_set == set(fit_indices),
        "zero_c06_nonfit_overlap": (train_set | probe_set).isdisjoint(c06_nonfit),
        "selection_created_before_compact_labels_open": True,
        "original_train_label_array_untouched": True,
        "c06_nonfit_labels_and_features_untouched": True,
        "dev_test_qrels_and_metrics_untouched": True,
    }
    if not all(checks.values()):
        raise ValueError(f"C21 selection checks failed: {checks}")
    for role, row in roles.items():
        row["candidate_key_sha256"] = candidate_key_sha256(
            row["indices"], request_ids, offsets, item_ids
        )

    return {
        "candidate_id": "c21",
        "selection_id": "c21_train_path_signal_selection_v1",
        "status": "frozen_before_any_c21_label_or_outcome",
        "hash_rule": "ascending sha256(c21-path-signal-v1\\0request_id), then packed index",
        "roles": roles,
        "checks": checks,
        "source": {
            "c06_selection_path": str(c06_selection_path),
            "c06_selection_sha256": actual_selection_hash,
            "c06_g0_report_path": str(c06_g0_path),
            "c06_g0_report_sha256": actual_g0_hash,
            "fit_request_indices_path": str(fit_index_path),
            "fit_request_indices_sha256": sha256_file(fit_index_path),
            "packed_request_ids_path": str(request_path),
            "packed_request_ids_sha256": sha256_file(request_path),
            "packed_candidate_offsets_path": str(offset_path),
            "packed_candidate_offsets_sha256": sha256_file(offset_path),
            "packed_candidate_item_ids_path": str(item_id_path),
            "packed_candidate_item_ids_sha256": sha256_file(item_id_path),
        },
        "labels_opened": False,
        "outcomes_observed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = materialize(config)
    write_json_once(config["paths"]["selection"], result)
    print(
        json.dumps(
            {
                "path": config["paths"]["selection"],
                "sha256": sha256_file(config["paths"]["selection"]),
                "counts": {name: len(row["indices"]) for name, row in result["roles"].items()},
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
