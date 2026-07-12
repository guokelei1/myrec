"""Create the label-free C46 outcome role from unopened delayed/escrow roles."""

from __future__ import annotations

from collections import defaultdict
import glob
import json
from pathlib import Path
import re
import sys

import numpy as np

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from probe.data import (  # noqa: E402
    PackedTrain,
    atomic_json,
    candidate_key_sha256,
    sha256_file,
    stable_key,
)


SEED = 20263000
SOURCE_STOP = 40_000
OUTCOME_MIN_INDEX = 50_000
OUTCOME_REQUESTS = 600
KUAI_CANDIDATES = set(range(23, 38)) | {21, 43}
MATERIALIZED_INTERNAL_A = {28, 29, 31, 32, 33, 34, 35, 36, 37, 43}


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
        raise ValueError("C46 user metadata incomplete")
    return [str(value) for value in users]


def main() -> None:
    output = REPO_ROOT / "artifacts/c46_prequential_behavioral_semantic_probe/signal_gate_v1/selection.json"
    if output.exists():
        raise FileExistsError(output)
    packed_root = REPO_ROOT / "artifacts/analysis/supervised_motivation_diagnostics/data/train"
    metadata_path = REPO_ROOT / "data/interim/kuaisearch/v0_lite/time_window_seed20260708/requests.jsonl"
    data = PackedTrain(packed_root)
    users = load_users(metadata_path, data)

    selections: list[tuple[int, Path, dict]] = []
    source_hashes: dict[str, str] = {}
    for raw in glob.glob(str(REPO_ROOT / "artifacts/c*_*/**/selection.json"), recursive=True):
        match = re.search(r"/artifacts/c(\d+)_", raw)
        if not match or int(match.group(1)) not in KUAI_CANDIDATES:
            continue
        path = Path(raw)
        candidate = int(match.group(1))
        selection = json.loads(path.read_text(encoding="utf-8"))
        selections.append((candidate, path, selection))
        source_hashes[str(path.relative_to(REPO_ROOT))] = sha256_file(path)

    # Any fit/probe role had labels opened. Any structural role or registered
    # materialized A had features/scores opened. They cannot become C46-A.
    excluded: set[int] = set()
    for candidate, _, selection in selections:
        for role, row in selection.get("roles", {}).items():
            if not isinstance(row, dict):
                continue
            is_open = (
                role in {"fit", "train_fit", "internal_probe"}
                or role.startswith("structural")
                or (candidate in MATERIALIZED_INTERNAL_A and role == "internal_A")
            )
            if is_open:
                excluded.update(int(value) for value in row.get("indices", []))

    inherited: set[int] = set()
    inherited_sources: dict[str, int] = {}
    for candidate, path, selection in selections:
        if candidate not in range(23, 38):
            continue
        for role in ("delayed_B", "escrow"):
            row = selection.get("roles", {}).get(role)
            if not isinstance(row, dict):
                continue
            values = {int(value) for value in row.get("indices", [])}
            inherited.update(values)
            inherited_sources[f"c{candidate}:{role}:{path.relative_to(REPO_ROOT)}"] = len(values)

    eligible = []
    for index in inherited - excluded:
        history = data.history(index)
        if index < OUTCOME_MIN_INDEX or not len(history):
            continue
        if set(int(value) for value in history) & set(int(value) for value in data.candidates(index)):
            continue
        eligible.append(index)
    eligible.sort(key=lambda index: (stable_key(SEED, "outcome", data.request_ids[index]), index))
    if len(eligible) < OUTCOME_REQUESTS:
        raise RuntimeError(f"C46 eligible outcome requests insufficient: {len(eligible)}")
    targets = eligible[:OUTCOME_REQUESTS]

    # Donors are label-free records matched by history-length bin and timestamp
    # decile. They need not be outcome-untouched because only their prior
    # history is used, never their candidate outcome.
    edges = [1, 2, 3, 5, 10, 20, 50]
    timestamps = np.asarray(data.timestamps, dtype=np.float64)
    quantile_edges = np.quantile(timestamps[OUTCOME_MIN_INDEX:], np.linspace(0, 1, 11))

    def bucket(index: int) -> tuple[int, int]:
        length = len(data.history(index))
        length_bin = int(np.searchsorted(edges, length, side="left"))
        time_bin = min(9, int(np.searchsorted(quantile_edges[1:-1], timestamps[index], side="right")))
        return length_bin, time_bin

    reserve = [
        index
        for index in range(OUTCOME_MIN_INDEX, len(data.request_ids))
        if index not in set(targets) and len(data.history(index))
    ]
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_length: dict[int, list[int]] = defaultdict(list)
    for index in reserve:
        grouped[bucket(index)].append(index)
        by_length[bucket(index)[0]].append(index)
    for key, values in grouped.items():
        values.sort(key=lambda index: (stable_key(SEED, f"donor:{key}", data.request_ids[index]), index))
    for key, values in by_length.items():
        values.sort(key=lambda index: (stable_key(SEED, f"donor-len:{key}", data.request_ids[index]), index))
    reserve.sort(key=lambda index: (stable_key(SEED, "donor-fallback", data.request_ids[index]), index))

    donors: list[int] = []
    exact_bucket = []
    exact_length = []
    for recipient in targets:
        choices = grouped.get(bucket(recipient), []) or by_length.get(bucket(recipient)[0], []) or reserve
        start = int.from_bytes(stable_key(SEED, "donor-start", data.request_ids[recipient])[:8], "big") % len(choices)
        recipient_candidates = set(int(value) for value in data.candidates(recipient))
        chosen = None
        for offset in range(len(choices)):
            donor = int(choices[(start + offset) % len(choices)])
            if users[donor] == users[recipient]:
                continue
            if recipient_candidates & set(int(value) for value in data.history(donor)):
                continue
            chosen = donor
            break
        if chosen is None:
            raise RuntimeError("C46 wrong-history donor unavailable")
        donors.append(chosen)
        exact_bucket.append(bucket(recipient) == bucket(chosen))
        exact_length.append(bucket(recipient)[0] == bucket(chosen)[0])

    result = {
        "candidate_id": "c46",
        "selection_id": "c46_prequential_behavioral_signal_v1",
        "status": "frozen_label_free_before_c46_proposal_or_outcome",
        "seed": SEED,
        "source_training_boundary": {
            "request_index_start": 0,
            "request_index_stop_exclusive": SOURCE_STOP,
            "max_timestamp": int(np.asarray(data.timestamps[:SOURCE_STOP]).max()),
        },
        "roles": {
            "internal_A": {
                "indices": targets,
                "request_ids": [data.request_ids[index] for index in targets],
                "candidate_key_sha256": candidate_key_sha256(data, targets),
            }
        },
        "wrong_history_donors": {
            "indices": donors,
            "request_ids": [data.request_ids[index] for index in donors],
        },
        "provenance": {
            "eligible_roles": "KuaiSearch C23-C37 delayed_B/escrow only",
            "eligible_role_counts": inherited_sources,
            "selection_file_sha256": source_hashes,
            "excluded_roles": "all fit/train_fit/internal_probe/structural and every previously materialized internal_A",
            "materialized_internal_A_candidates": sorted(MATERIALIZED_INTERNAL_A),
            "outcome_min_index": OUTCOME_MIN_INDEX,
            "eligible_after_exclusion_and_contract": len(eligible),
        },
        "checks": {
            "source_strictly_before_outcome": int(np.asarray(data.timestamps[:SOURCE_STOP]).max()) < int(np.asarray(data.timestamps[targets]).min()),
            "target_count": len(targets) == OUTCOME_REQUESTS,
            "target_unique": len(set(targets)) == len(targets),
            "target_excluded_overlap_zero": not bool(set(targets) & excluded),
            "strict_nonrepeat": all(not (set(map(int, data.history(index))) & set(map(int, data.candidates(index)))) for index in targets),
            "history_present": all(len(data.history(index)) > 0 for index in targets),
            "wrong_different_user": all(users[a] != users[b] for a, b in zip(targets, donors)),
            "wrong_candidate_overlap_zero": all(not (set(map(int, data.candidates(a))) & set(map(int, data.history(b)))) for a, b in zip(targets, donors)),
            "donor_exact_bucket_fraction": float(np.mean(exact_bucket)),
            "donor_exact_length_bin_fraction": float(np.mean(exact_length)),
            "labels_read": False,
            "dev_test_qrels_read": False,
        },
        "integrity": {
            "packed_manifest_sha256": sha256_file(packed_root.parent / "manifest.json"),
            "label_free_metadata_sha256": sha256_file(metadata_path),
        },
    }
    negative_declarations = {"labels_read", "dev_test_qrels_read"}
    failed = [
        key
        for key, value in result["checks"].items()
        if isinstance(value, bool)
        and (
            (key not in negative_declarations and not value)
            or (key in negative_declarations and value)
        )
    ]
    if failed:
        raise RuntimeError(f"C46 label-free selection checks failed: {failed}")
    atomic_json(output, result)
    print(json.dumps({"candidate_id": "c46", "selection": str(output), "targets": len(targets), "eligible": len(eligible)}, sort_keys=True))


if __name__ == "__main__":
    main()
