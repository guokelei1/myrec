"""Label-free C39 cohort selection from the untouched C38 reserve."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROLE_ORDER = ("fit", "internal_A")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_blind_records(path: str | Path) -> list[dict[str, Any]]:
    """Load label-free records while mechanically rejecting candidate labels."""

    records = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            record = json.loads(line)
            request_id = str(record["request_id"])
            if request_id in seen:
                raise ValueError(f"duplicate blind request_id at line {line_no}")
            seen.add(request_id)
            candidates = record.get("candidates")
            history = record.get("history")
            if not isinstance(candidates, list) or len(candidates) < 2:
                raise ValueError(f"invalid candidates at line {line_no}")
            if not isinstance(history, list) or not history:
                raise ValueError(f"empty history at line {line_no}")
            if any(
                "clicked" in candidate or "purchased" in candidate
                for candidate in candidates
            ):
                raise PermissionError("C39 selection received candidate labels")
            request_ts = int(record["ts"])
            if any(int(event["ts"]) >= request_ts for event in history):
                raise ValueError(f"non-causal history at line {line_no}")
            records.append(record)
    return records


def request_key(record: Mapping[str, Any]) -> str:
    return "\x1f".join(
        (
            str(record["request_id"]),
            str(record["user_id"]),
            str(record["session_id"]),
        )
    )


def candidate_key_sha256(records: list[dict[str, Any]], indices: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        record = records[int(index)]
        digest.update(request_key(record).encode("utf-8"))
        digest.update(b"\x1e")
        for candidate in record["candidates"]:
            digest.update(str(candidate["item_id"]).encode("utf-8"))
            digest.update(b"\x1f")
        digest.update(b"\n")
    return digest.hexdigest()


def materialize_selection(
    *,
    records_path: str | Path,
    standardized_manifest_path: str | Path,
    candidate_manifest_path: str | Path,
    c0_report_path: str | Path,
    predecessor_selection_path: str | Path,
    predecessor_selection_sha256: str,
    output_path: str | Path,
    seed: int,
    internal_a_requests: int,
    length_bins: list[int],
) -> dict[str, Any]:
    records = load_blind_records(records_path)
    c0 = read_json(c0_report_path)
    if c0.get("overall_status") != "passed":
        raise PermissionError("Amazon-C4 C0 report has not passed")
    if not c0.get("checks", {}).get("dev_test_records_label_free", False):
        raise PermissionError("Amazon-C4 label-isolation check has not passed")
    if sha256_file(predecessor_selection_path) != predecessor_selection_sha256:
        raise RuntimeError("C38 selection changed before C39 selection")
    predecessor = read_json(predecessor_selection_path)
    if predecessor.get("candidate_id") != "c38":
        raise ValueError("C39 predecessor is not C38")
    if predecessor.get("train_requests") != len(records):
        raise ValueError("C39 record count differs from C38")
    if internal_a_requests <= 0:
        raise ValueError("C39 internal-A count must be positive")
    if sorted(set(length_bins)) != length_bins or not length_bins:
        raise ValueError("C39 history length bins must be sorted and unique")

    fit_indices = [int(value) for value in predecessor["roles"]["fit"]["indices"]]
    predecessor_target_indices = {
        int(value)
        for role in ("internal_A", "delayed_B", "escrow")
        for value in predecessor["roles"][role]["indices"]
    }
    unused = [int(value) for value in predecessor["unused_indices"]]
    if set(unused) & (set(fit_indices) | predecessor_target_indices):
        raise ValueError("C38 unused reserve overlaps a predecessor role")
    ordered_unused = sorted(
        unused,
        key=lambda index: (
            hashlib.sha256(
                f"{seed}\x1f{request_key(records[index])}".encode("utf-8")
            ).hexdigest(),
            request_key(records[index]),
        ),
    )
    if internal_a_requests > len(ordered_unused):
        raise ValueError("C39 A exceeds the C38 unused reserve")
    internal_a_indices = ordered_unused[:internal_a_requests]
    reserve_indices = ordered_unused[internal_a_requests:]
    roles = {}
    for role, indices in (
        ("fit", fit_indices),
        ("internal_A", internal_a_indices),
    ):
        roles[role] = {
            "indices": indices,
            "request_ids": [str(records[index]["request_id"]) for index in indices],
            "candidate_key_sha256": candidate_key_sha256(records, indices),
        }

    selected_indices = [*fit_indices, *internal_a_indices]
    bins: defaultdict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        bins[_length_bin(len(record["history"]), length_bins)].append(index)
    wrong_donors: dict[str, dict[str, Any]] = {}
    same_bin_count = 0
    same_user_count = 0
    for index in selected_indices:
        target = records[index]
        target_bin = _length_bin(len(target["history"]), length_bins)
        donor_candidates = [
            donor
            for donor in bins[target_bin]
            if str(records[donor]["user_id"]) != str(target["user_id"])
        ]
        same_bin = bool(donor_candidates)
        if not donor_candidates:
            donor_candidates = [
                donor
                for donor, record in enumerate(records)
                if str(record["user_id"]) != str(target["user_id"])
                and record["history"]
            ]
        if not donor_candidates:
            raise ValueError("no valid C39 wrong-history donor")
        donor = min(
            donor_candidates,
            key=lambda raw: (
                abs(
                    _length_bin(len(records[raw]["history"]), length_bins)
                    - target_bin
                ),
                hashlib.sha256(
                    f"{seed}\x1fwrong\x1f{request_key(target)}\x1f{request_key(records[raw])}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            ),
        )
        same_bin_count += int(same_bin)
        same_user_count += int(
            str(records[donor]["user_id"]) == str(target["user_id"])
        )
        wrong_donors[str(index)] = {
            "donor_index": donor,
            "target_history_length": len(target["history"]),
            "donor_history_length": len(records[donor]["history"]),
            "target_length_bin": target_bin,
            "donor_length_bin": _length_bin(
                len(records[donor]["history"]), length_bins
            ),
            "same_bin": same_bin,
        }

    selected_count = len(selected_indices)
    selection = {
        "candidate_id": "c39",
        "seed": int(seed),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "standardized_manifest_path": str(standardized_manifest_path),
        "standardized_manifest_sha256": sha256_file(standardized_manifest_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "c0_report_path": str(c0_report_path),
        "c0_report_sha256": sha256_file(c0_report_path),
        "predecessor_selection_path": str(predecessor_selection_path),
        "predecessor_selection_sha256": predecessor_selection_sha256,
        "train_requests": len(records),
        "roles": roles,
        "reserve_indices": reserve_indices,
        "history_length_bins": length_bins,
        "wrong_donors": wrong_donors,
        "wrong_donor_audit": {
            "requests": selected_count,
            "coverage_fraction": len(wrong_donors) / selected_count,
            "same_length_bin_fraction": same_bin_count / selected_count,
            "same_user_assignments": same_user_count,
        },
        "outcome_isolation": {
            "c38_fit_reused_as_training_only": len(fit_indices),
            "c39_internal_A_from_c38_unused": len(internal_a_indices),
            "c39_reserve_from_c38_unused": len(reserve_indices),
            "overlap_with_c38_internal_A_delayed_B_escrow": len(
                set(internal_a_indices) & predecessor_target_indices
            ),
            "c39_internal_A_previously_featured_scored_labeled": False,
        },
        "label_access": {
            "records_train_blind_opened": True,
            "records_train_labels_opened": False,
            "dev_test_records_labels_qrels_opened": False,
        },
    }
    write_json(output_path, selection)
    return selection


def _length_bin(length: int, boundaries: list[int]) -> int:
    for boundary in boundaries:
        if length <= boundary:
            return boundary
    return boundaries[-1]
