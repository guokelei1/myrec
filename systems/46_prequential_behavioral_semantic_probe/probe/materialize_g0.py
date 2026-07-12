"""Open only early-source labels and materialize C46 prefix/next examples."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from probe.data import PackedTrain, atomic_json, sha256_file, stable_key  # noqa: E402
from probe.protocol import load_config, verify_proposal_lock  # noqa: E402


def save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path), "shape": list(value.shape), "dtype": str(value.dtype)}


def load_users(path: Path, data: PackedTrain) -> list[str]:
    positions = {request_id: index for index, request_id in enumerate(data.request_ids)}
    users: list[str | None] = [None] * len(data.request_ids)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = positions.get(str(row["request_id"]))
            if index is not None:
                users[index] = str(row["user_id"])
    if any(value is None for value in users):
        raise ValueError("C46 source user metadata incomplete")
    return [str(value) for value in users]


def build_examples(
    data: PackedTrain,
    users: list[str],
    labels: np.ndarray,
    *,
    start: int,
    stop: int,
    max_history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    memories: dict[str, list[int]] = defaultdict(list)
    offsets = [0]
    values: list[np.ndarray] = []
    targets: list[int] = []
    repeated_skipped = 0
    requests_with_positive = 0
    index = start
    timestamp_groups = 0
    while index < stop:
        timestamp = int(data.timestamps[index])
        end = index + 1
        while end < stop and int(data.timestamps[end]) == timestamp:
            end += 1
        timestamp_groups += 1
        updates: list[tuple[str, list[int]]] = []
        for request in range(index, end):
            c_start, c_stop = int(data.candidate_offsets[request]), int(data.candidate_offsets[request + 1])
            row_labels = np.asarray(labels[c_start:c_stop], dtype=np.float32)
            candidates = data.candidates(request)
            positives = [int(value) for value in candidates[row_labels > 0]]
            requests_with_positive += int(bool(positives))
            memory = memories[users[request]]
            prefix = np.asarray(memory[-max_history:], dtype=np.int64)
            for target in positives:
                if target in memory:
                    repeated_skipped += 1
                    continue
                if len(prefix):
                    values.append(prefix.copy())
                    offsets.append(offsets[-1] + len(prefix))
                    targets.append(target)
            updates.append((users[request], positives))
        for user, positives in updates:
            memories[user].extend(positives)
        index = end
    flat = np.concatenate(values).astype(np.int64, copy=False) if values else np.empty(0, dtype=np.int64)
    return (
        np.asarray(offsets, dtype=np.int64),
        flat,
        np.asarray(targets, dtype=np.int64),
        {
            "source_requests": stop - start,
            "timestamp_groups": timestamp_groups,
            "requests_with_positive": requests_with_positive,
            "examples": len(targets),
            "exact_next_recurrence_skipped": repeated_skipped,
            "users_with_memory": len(memories),
        },
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    _, lock_hash = verify_proposal_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    report_path = root / "g0_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    root.mkdir(parents=True, exist_ok=True)
    for key, expected in [
        ("packed_manifest", "packed_manifest_sha256"),
        ("label_free_request_metadata", "label_free_request_metadata_sha256"),
        ("raw_item_embeddings", "raw_item_embeddings_sha256"),
        ("train_candidate_labels", "train_candidate_labels_sha256"),
        ("candidate_manifest", "candidate_manifest_sha256"),
    ]:
        if sha256_file(REPO_ROOT / config["paths"][key]) != config["integrity"][expected]:
            raise RuntimeError(f"C46 input changed: {key}")
    if sha256_file(REPO_ROOT / config["paths"]["selection"]) != config["paths"]["selection_sha256"]:
        raise RuntimeError("C46 selection changed")
    selection = json.loads((REPO_ROOT / config["paths"]["selection"]).read_text(encoding="utf-8"))
    data = PackedTrain(REPO_ROOT / config["paths"]["packed_train_root"])
    users = load_users(REPO_ROOT / config["paths"]["label_free_request_metadata"], data)
    label_path = REPO_ROOT / config["paths"]["train_candidate_labels"]
    source_labels = np.load(label_path, mmap_mode="r")
    source = config["source"]
    start, stop = int(source["request_start"]), int(source["request_stop_exclusive"])
    offsets, items, targets, audit = build_examples(
        data,
        users,
        source_labels,
        start=start,
        stop=stop,
        max_history=int(source["max_history"]),
    )
    unique_targets = np.unique(targets)
    order = sorted(
        range(len(targets)),
        key=lambda position: (
            stable_key(20263000, "target-shuffle", f"{position}:{int(targets[position])}"),
            position,
        ),
    )
    shuffled = np.empty_like(targets)
    if len(order):
        for position, source_position in zip(order, order[1:] + order[:1]):
            shuffled[position] = targets[source_position]
    same_fraction = float(np.mean(shuffled == targets)) if len(targets) else 1.0
    outputs = {
        "source_offsets.npy": save_array(root, "source_offsets.npy", offsets),
        "source_items.npy": save_array(root, "source_items.npy", items),
        "source_targets.npy": save_array(root, "source_targets.npy", targets),
        "source_shuffled_targets.npy": save_array(root, "source_shuffled_targets.npy", shuffled),
        "negative_pool.npy": save_array(root, "negative_pool.npy", unique_targets),
    }
    target_indices = [int(value) for value in selection["roles"]["internal_A"]["indices"]]
    checks = {
        "source_strictly_before_A": int(np.asarray(data.timestamps[start:stop]).max()) < int(np.asarray(data.timestamps[target_indices]).min()),
        "minimum_examples": len(targets) >= int(source["minimum_examples"]),
        "minimum_unique_targets": len(unique_targets) >= int(source["minimum_unique_targets"]),
        "shuffled_pair_identity_fraction_below_005": same_fraction < 0.05,
        "source_items_in_embedding_table": not len(items) or int(items.max()) < np.load(REPO_ROOT / config["paths"]["raw_item_embeddings"], mmap_mode="r").shape[0],
        "targets_in_embedding_table": not len(targets) or int(targets.max()) < np.load(REPO_ROOT / config["paths"]["raw_item_embeddings"], mmap_mode="r").shape[0],
        "A_candidate_hash": selection["roles"]["internal_A"]["candidate_key_sha256"] == config["integrity"]["A_candidate_key_sha256"],
    }
    report = {
        "candidate_id": "c46",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed_G0_terminal",
        "proposal_lock_sha256": lock_hash,
        "source_label_scope": [start, stop],
        "source_labels_opened": True,
        "A_features_scores_opened": False,
        "A_labels_opened": False,
        "dev_test_qrels_read": False,
        "audit": audit,
        "unique_targets": len(unique_targets),
        "shuffled_target_identity_fraction": same_fraction,
        "checks": checks,
        "outputs": outputs,
    }
    atomic_json(report_path, report)
    print(json.dumps({"candidate_id": "c46", "stage": "g0", "status": report["status"], "examples": len(targets), "unique_targets": len(unique_targets)}, sort_keys=True))


if __name__ == "__main__":
    main()
