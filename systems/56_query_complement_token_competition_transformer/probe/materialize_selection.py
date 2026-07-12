"""Freeze a label-blind split inside C26's already exposed fit role."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from probe.locking import (  # noqa: E402
    load_config,
    read_json,
    sha256_file,
    verify_execution,
    write_once,
)


class PackedStructure:
    def __init__(self, root: Path) -> None:
        self.request_ids = [
            str(json.loads(line)["request_id"])
            for line in (root / "request_ids.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.candidate_offsets = np.load(root / "candidate_offsets.npy", mmap_mode="r")
        self.candidate_item_ids = np.load(root / "candidate_item_ids.npy", mmap_mode="r")


def candidate_hash(data: PackedStructure, indices: Sequence[int]) -> str:
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


def stable_order(indices: Sequence[int], request_ids: Sequence[str], seed: int, role: str) -> list[int]:
    return sorted(
        (int(value) for value in indices),
        key=lambda index: (
            hashlib.sha256(f"c56:{seed}:{role}:{request_ids[index]}".encode()).digest(),
            index,
        ),
    )


def materialize(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    paths = config["paths"]
    integrity = config["integrity"]
    for name, expected_name in (
        ("c26_config", "c26_config_sha256"),
        ("c26_selection", "c26_selection_sha256"),
        ("c26_g0_report", "c26_g0_report_sha256"),
    ):
        if sha256_file(REPO_ROOT / paths[name]) != integrity[expected_name]:
            raise RuntimeError(f"C56 registered source changed: {name}")
    source = read_json(REPO_ROOT / paths["c26_selection"])
    data = PackedStructure(REPO_ROOT / paths["packed_train_root"])
    settings = config["selection"]
    seed = int(settings["split_seed"])
    fit = [int(value) for value in source["roles"]["fit"]["indices"]]
    ordered = stable_order(fit, data.request_ids, seed, "fit")
    holdout_count = int(settings["holdout_requests"])
    train = ordered[:-holdout_count]
    holdout = ordered[-holdout_count:]
    nohistory = stable_order(
        source["roles"]["structural_nohistory"]["indices"],
        data.request_ids,
        seed,
        "nohistory",
    )[: int(settings["structural_nohistory_requests"])]
    repeat = stable_order(
        source["roles"]["structural_repeat"]["indices"],
        data.request_ids,
        seed,
        "repeat",
    )[: int(settings["structural_repeat_requests"])]
    donor_by_fit = dict(
        zip(
            map(int, source["roles"]["fit"]["indices"]),
            map(int, source["wrong_history_donors"]["fit"]["indices"]),
        )
    )
    wrong = {
        "train": [donor_by_fit[index] for index in train],
        "holdout": [donor_by_fit[index] for index in holdout],
    }
    roles = {
        "train": train,
        "holdout": holdout,
        "structural_nohistory": nohistory,
        "structural_repeat": repeat,
    }
    hashes = {name: candidate_hash(data, indices) for name, indices in roles.items()}
    checks = {
        "fit_coverage_exact": set(train + holdout) == set(fit),
        "train_holdout_disjoint": not (set(train) & set(holdout)),
        "exact_train_count": len(train) == int(settings["train_requests"]),
        "exact_holdout_count": len(holdout) == holdout_count,
        "wrong_donors_complete": all(len(wrong[name]) == len(roles[name]) for name in wrong),
        "wrong_donor_candidate_overlap_frozen_by_c26": bool(
            source["checks"]["donor_candidate_overlap_zero"]
        ),
        "fit_labels_closed": True,
        "C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    value = {
        "candidate_id": "c56",
        "selection_id": "c56_c26_fit_hash_split_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "label_blind_split_frozen" if all(checks.values()) else "failed",
        "execution_lock_sha256": execution_hash,
        "split_seed": seed,
        "roles": roles,
        "wrong_history_donors": wrong,
        "candidate_key_sha256": hashes,
        "checks": checks,
        "fit_labels_read": False,
        "C26_internal_A_delayed_B_escrow_opened": False,
        "dev_test_qrels_opened": False,
    }
    write_once(REPO_ROOT / paths["selection"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
