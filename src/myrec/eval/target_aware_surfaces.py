"""Evaluator-side target-aware request surfaces.

This module receives gains only after the shared evaluator has opened qrels.
Training and scoring code must never import it to construct model inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


OBSERVED_POSITIVE_PARTITION = (
    "target_repeat",
    "target_nonrepeat_other_candidate_overlap",
    "target_nonrepeat_no_candidate_overlap",
    "target_nonrepeat_no_history",
)

ALL_REQUEST_PARTITION = (*OBSERVED_POSITIVE_PARTITION, "no_observed_positive")


def build_target_aware_surface_memberships(
    records_path: str | Path,
    candidates: dict[str, list[str]],
    gains: dict[str, dict[str, float]],
) -> dict[str, set[str]]:
    """Join label-free records with evaluator gains and assign disjoint surfaces."""

    records_path = Path(records_path)
    records = {str(row["request_id"]): row for row in iter_jsonl(records_path)}
    if not records:
        raise ValueError(f"empty records file: {records_path}")
    if set(records) != set(candidates):
        raise ValueError("records and candidate manifest have different request coverage")
    if set(gains) != set(candidates):
        raise ValueError("qrels gains and candidate manifest have different request coverage")

    members = {
        "all": set(),
        "observed_positive": set(),
        "no_observed_positive": set(),
        "history_present": set(),
        "candidate_overlap": set(),
        "no_candidate_overlap_history_present": set(),
        "no_history": set(),
        **{name: set() for name in OBSERVED_POSITIVE_PARTITION},
    }

    for request_id in sorted(candidates):
        record = records[request_id]
        manifest_candidates = set(candidates[request_id])
        record_candidates = {
            str(candidate["item_id"]) for candidate in record.get("candidates", [])
        }
        if manifest_candidates != record_candidates:
            raise ValueError(
                f"record/candidate identity mismatch for request_id={request_id}"
            )
        gain_items = {
            str(item_id) for item_id, gain in gains[request_id].items() if float(gain) > 0
        }
        unknown_gain_items = gain_items - manifest_candidates
        if unknown_gain_items:
            raise ValueError(
                f"positive gains contain non-candidates for request_id={request_id}: "
                f"{sorted(unknown_gain_items)[:5]}"
            )

        history_items = {
            str(event["item_id"]) for event in record.get("history", [])
        }
        history_present = bool(history_items)
        candidate_overlap = bool(history_items & manifest_candidates)

        members["all"].add(request_id)
        if history_present:
            members["history_present"].add(request_id)
            members[
                "candidate_overlap"
                if candidate_overlap
                else "no_candidate_overlap_history_present"
            ].add(request_id)
        else:
            members["no_history"].add(request_id)

        if not gain_items:
            members["no_observed_positive"].add(request_id)
            continue

        members["observed_positive"].add(request_id)
        if gain_items & history_items:
            target_surface = "target_repeat"
        elif not history_present:
            target_surface = "target_nonrepeat_no_history"
        elif candidate_overlap:
            target_surface = "target_nonrepeat_other_candidate_overlap"
        else:
            target_surface = "target_nonrepeat_no_candidate_overlap"
        members[target_surface].add(request_id)

    _assert_partition(
        members["observed_positive"],
        [members[name] for name in OBSERVED_POSITIVE_PARTITION],
        "observed-positive target-aware",
    )
    _assert_partition(
        members["all"],
        [members[name] for name in ALL_REQUEST_PARTITION],
        "all-request target-aware",
    )
    _assert_partition(
        members["all"],
        [
            members["candidate_overlap"],
            members["no_candidate_overlap_history_present"],
            members["no_history"],
        ],
        "label-free candidate-overlap",
    )
    return members


def materialize_target_aware_surfaces(
    records_path: str | Path,
    candidates: dict[str, list[str]],
    gains: dict[str, dict[str, float]],
    output_dir: str | Path,
    *,
    label_mode: str,
    candidate_manifest_path: str | Path,
    qrels_path: str | Path,
) -> dict[str, Any]:
    """Write auditable evaluator-only request-ID surfaces and their hashes."""

    records_path = Path(records_path)
    candidate_manifest_path = Path(candidate_manifest_path)
    qrels_path = Path(qrels_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    members = build_target_aware_surface_memberships(records_path, candidates, gains)

    files: dict[str, Any] = {}
    for name, request_ids in sorted(members.items()):
        path = output_dir / f"{name}.txt"
        with path.open("w", encoding="utf-8") as handle:
            for request_id in sorted(request_ids):
                handle.write(request_id + "\n")
        files[name] = {
            "path": str(path),
            "requests": len(request_ids),
            "sha256": sha256_file(path),
        }

    manifest = {
        "analysis_type": "evaluator_side_target_aware_request_surfaces",
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "definitions": {
            "candidate_overlap": (
                "history present and at least one slate candidate occurs in history; "
                "label-free diagnostic only"
            ),
            "no_candidate_overlap_history_present": (
                "history present and the history/candidate item-id sets are disjoint; "
                "label-free diagnostic only"
            ),
            "no_history": "history item-id set is empty; label-free diagnostic",
            "no_observed_positive": (
                "the registered endpoint has no positive-gain candidate; target "
                "recurrence is undefined"
            ),
            "observed_positive": (
                "at least one candidate has positive gain under the registered endpoint"
            ),
            "target_repeat": (
                "at least one positive-gain candidate occurs in history"
            ),
            "target_nonrepeat_other_candidate_overlap": (
                "positive eligible; no positive candidate occurs in history, but "
                "another slate candidate does"
            ),
            "target_nonrepeat_no_candidate_overlap": (
                "positive eligible; history present and disjoint from the complete slate"
            ),
            "target_nonrepeat_no_history": (
                "positive eligible and no history is available"
            ),
        },
        "files": files,
        "label_aware": True,
        "label_mode": label_mode,
        "partitions": {
            "all_requests": list(ALL_REQUEST_PARTITION),
            "candidate_overlap_diagnostics": [
                "candidate_overlap",
                "no_candidate_overlap_history_present",
                "no_history",
            ],
            "observed_positive": list(OBSERVED_POSITIVE_PARTITION),
        },
        "qrels_path": str(qrels_path),
        "qrels_sha256": sha256_file(qrels_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _assert_partition(
    population: set[str], parts: Iterable[set[str]], name: str
) -> None:
    union: set[str] = set()
    for part in parts:
        overlap = union & part
        if overlap:
            raise AssertionError(f"{name} partition overlaps: {sorted(overlap)[:5]}")
        union.update(part)
    if union != population:
        raise AssertionError(
            f"{name} partition coverage mismatch: "
            f"missing={sorted(population - union)[:5]} extra={sorted(union - population)[:5]}"
        )

