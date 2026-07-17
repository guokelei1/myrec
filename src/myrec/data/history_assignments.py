"""Build label-free true/null/matched-wrong history assignments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json, write_jsonl


MOTIVATION_V12_ASSIGNMENT_IMPLEMENTATION_FILES = (
    "scripts/materialize_history_assignments.py",
    "src/myrec/data/history_assignments.py",
)
MOTIVATION_V12_ASSIGNMENT_SEED = 20_260_714
MOTIVATION_V12_GLOBAL_DONOR_SHORTLIST_SIZE = 512
MOTIVATION_V12_PUBLIC_CONDITIONS = ("full", "null", "wrong")
MOTIVATION_V12_CONDITION_ASSIGNMENTS = {
    "full": "true",
    "null": "null",
    "wrong": "wrong",
}


def motivation_v12_assignment_recipe() -> dict[str, Any]:
    """Return the exact post-release assignment recipe for the new holdout."""

    return {
        "schema_version": 1,
        "conditions": list(MOTIVATION_V12_PUBLIC_CONDITIONS),
        "condition_assignment_names": dict(MOTIVATION_V12_CONDITION_ASSIGNMENTS),
        "seed": MOTIVATION_V12_ASSIGNMENT_SEED,
        "wrong_user_external_donor_records_role": "development_records_train",
        "wrong_user_donor_pool_construction": (
            "external_records_train_then_append_target_requests_absent_by_request_id"
        ),
        "target_population_included_in_wrong_user_donor_pool": True,
        "global_donor_shortlist_size": (
            MOTIVATION_V12_GLOBAL_DONOR_SHORTLIST_SIZE
        ),
        "global_shortlist_scope": (
            "global_other_user_fallback_only_exact_query_donors_never_shortened"
        ),
        "wrong_user_constraints": [
            "different_user",
            "donor_events_strictly_before_target_ts",
            "donor_history_excludes_every_target_candidate_item",
        ],
        "qrels_read": False,
        "model_scores_read": False,
    }


def motivation_v12_assignment_implementation_paths() -> dict[str, Path]:
    """Return the two generator files whose bytes are frozen by release."""

    root = Path(__file__).resolve().parents[3]
    return {
        relative_path: root / relative_path
        for relative_path in MOTIVATION_V12_ASSIGNMENT_IMPLEMENTATION_FILES
    }


def current_motivation_v12_assignment_implementation_identity() -> dict[str, Any]:
    """Hash the assignment module and its only production CLI entry point."""

    entries = []
    for relative_path, path in sorted(
        motivation_v12_assignment_implementation_paths().items()
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"missing Motivation V1.2 assignment implementation: {path}"
            )
        entries.append({"path": relative_path, "sha256": sha256_file(path)})
    return {
        "schema_version": 1,
        "digest": sha256_text(
            json.dumps(entries, sort_keys=True, separators=(",", ":"))
        ),
        "files": entries,
    }


def materialize_history_assignments(
    records_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    donor_records_path: str | Path | None = None,
    seed: int = 20260714,
    global_donor_shortlist_size: int | None = 512,
    motivation_v12_release_lock_path: str | Path | None = None,
    motivation_v12_enforce_registered_recipe: bool = True,
) -> dict[str, Any]:
    """Write true/null/wrong assignments without reading qrels or model scores."""

    records_path = Path(records_path)
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    if global_donor_shortlist_size is not None and global_donor_shortlist_size <= 0:
        raise ValueError("global_donor_shortlist_size must be positive or None")
    _require_release_for_registered_v12_holdout(
        records_path,
        motivation_v12_release_lock_path=motivation_v12_release_lock_path,
    )
    release_binding = None
    if motivation_v12_release_lock_path is not None:
        if donor_records_path is None:
            raise ValueError(
                "Motivation V1.2 assignment materialization requires "
                "--donor-records"
            )
        records_path = records_path.resolve()
        donor_records_path = Path(donor_records_path).resolve()
        output_dir = output_dir.resolve()
        report_path = report_path.resolve()
        release_binding = _motivation_v12_release_binding(
            records_path=records_path,
            donor_records_path=donor_records_path,
            release_lock_path=Path(motivation_v12_release_lock_path).resolve(),
            seed=seed,
            global_donor_shortlist_size=global_donor_shortlist_size,
            enforce_registered_v12_recipe=(
                motivation_v12_enforce_registered_recipe
            ),
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"history assignment directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(records_path)
    if donor_records_path is None:
        donor_records = records
    else:
        donor_records_path = Path(donor_records_path)
        donor_records = _load_records(donor_records_path)
        donor_ids = {record["request_id"] for record in donor_records}
        donor_records.extend(
            record for record in records if record["request_id"] not in donor_ids
        )
    rows, assignment_audit = _build_assignment_rows(
        records,
        donor_records,
        seed=seed,
        global_donor_shortlist_size=global_donor_shortlist_size,
    )

    paths = {
        "true": output_dir / "true.jsonl",
        "null": output_dir / "null.jsonl",
        "wrong": output_dir / "wrong.jsonl",
    }
    for condition, condition_rows in rows.items():
        write_jsonl(paths[condition], condition_rows)
    report = _build_assignment_report(
        records_path=records_path,
        donor_records_path=(
            Path(donor_records_path) if donor_records_path is not None else None
        ),
        donor_pool_requests=len(donor_records),
        request_count=len(records),
        seed=seed,
        global_donor_shortlist_size=global_donor_shortlist_size,
        assignment_audit=assignment_audit,
        paths=paths,
        release_binding=release_binding,
    )
    write_json(output_dir / "manifest.json", report)
    write_json(report_path, report)
    if release_binding is not None:
        verify_motivation_v12_history_assignments(
            output_dir / "manifest.json",
            standardized_dir=records_path.parent,
            release_lock_path=Path(motivation_v12_release_lock_path).resolve(),
            enforce_registered_v12_recipe=(
                motivation_v12_enforce_registered_recipe
            ),
        )
    return report


def verify_motivation_v12_history_assignments(
    manifest_path: str | Path,
    *,
    standardized_dir: str | Path,
    release_lock_path: str | Path,
    enforce_registered_v12_recipe: bool = True,
) -> dict[str, Any]:
    """Deterministically reproduce and verify the released holdout assignments."""

    manifest_path = Path(manifest_path).resolve()
    standardized_dir = Path(standardized_dir).resolve()
    release_lock_path = Path(release_lock_path).resolve()
    if manifest_path != (manifest_path.parent / "manifest.json"):
        raise ValueError("history assignment manifest path is not canonical")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing history assignment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("history assignment manifest must be a JSON object")
    records_path = standardized_dir / "records_confirmation.jsonl"
    donor_value = manifest.get("donor_records_path")
    if not isinstance(donor_value, str) or not donor_value.strip():
        raise ValueError("released history assignment manifest has no donor path")
    donor_records_path = Path(donor_value).resolve()
    recipe = motivation_v12_assignment_recipe()
    release_binding = _motivation_v12_release_binding(
        records_path=records_path,
        donor_records_path=donor_records_path,
        release_lock_path=release_lock_path,
        seed=recipe["seed"],
        global_donor_shortlist_size=recipe["global_donor_shortlist_size"],
        enforce_registered_v12_recipe=enforce_registered_v12_recipe,
    )
    records = _load_records(records_path)
    donor_records = _load_records(donor_records_path)
    donor_ids = {record["request_id"] for record in donor_records}
    donor_records.extend(
        record for record in records if record["request_id"] not in donor_ids
    )
    expected_rows, assignment_audit = _build_assignment_rows(
        records,
        donor_records,
        seed=recipe["seed"],
        global_donor_shortlist_size=recipe["global_donor_shortlist_size"],
    )
    paths = {
        condition: manifest_path.parent / f"{condition}.jsonl"
        for condition in ("true", "null", "wrong")
    }
    expected_report = _build_assignment_report(
        records_path=records_path,
        donor_records_path=donor_records_path,
        donor_pool_requests=len(donor_records),
        request_count=len(records),
        seed=recipe["seed"],
        global_donor_shortlist_size=recipe["global_donor_shortlist_size"],
        assignment_audit=assignment_audit,
        paths=paths,
        release_binding=release_binding,
    )
    if manifest != expected_report:
        raise ValueError(
            "released history assignment manifest differs from the frozen recipe"
        )
    verified_files = {}
    for condition, expected in expected_rows.items():
        path = paths[condition]
        observed = list(iter_jsonl(path))
        if observed != expected:
            raise ValueError(
                "history assignment rows differ from deterministic frozen "
                f"generation: {condition}"
            )
        verified_files[condition] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "requests": len(observed),
        }
    return {
        "schema_version": 1,
        "passed": True,
        "qrels_read": False,
        "model_scores_read": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "release_binding": release_binding,
        "recipe": recipe,
        "files": verified_files,
        "request_count": len(records),
        "deterministically_regenerated": True,
    }


def _build_assignment_report(
    *,
    records_path: Path,
    donor_records_path: Path | None,
    donor_pool_requests: int,
    request_count: int,
    seed: int,
    global_donor_shortlist_size: int | None,
    assignment_audit: Mapping[str, Any],
    paths: Mapping[str, Path],
    release_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "evidence_mode": (
            "first_round_pilot" if release_binding is not None else "exploratory"
        ),
        "conditions": list(MOTIVATION_V12_PUBLIC_CONDITIONS),
        "condition_assignment_names": dict(MOTIVATION_V12_CONDITION_ASSIGNMENTS),
        "source_records_path": str(records_path),
        "source_records_sha256": sha256_file(records_path),
        "donor_records_path": (
            str(donor_records_path) if donor_records_path is not None else str(records_path)
        ),
        "donor_records_sha256": (
            sha256_file(donor_records_path)
            if donor_records_path is not None
            else sha256_file(records_path)
        ),
        "donor_pool_requests": donor_pool_requests,
        "donor_pool_construction": (
            "external_records_then_append_target_requests_absent_by_request_id"
            if donor_records_path is not None
            else "target_records_only"
        ),
        "qrels_read": False,
        "model_scores_read": False,
        "seed": seed,
        "requests": request_count,
        "match_counts": dict(assignment_audit["match_counts"]),
        "matched_history_length_absolute_difference": assignment_audit[
            "matched_history_length_absolute_difference"
        ],
        "target_candidate_leakage_violations": assignment_audit[
            "target_candidate_leakage_violations"
        ],
        "history_not_strictly_before_target_violations": assignment_audit[
            "history_not_strictly_before_target_violations"
        ],
        "matching": {
            "priority": ["exact_query_other_user", "global_other_user"],
            "global_donor_shortlist_size": global_donor_shortlist_size,
            "global_shortlist": (
                "deterministic cyclic window over source-order donors, with start "
                "derived from seed and target request id; all exact-query donors "
                "remain eligible"
            ),
            "distance": [
                "history_length_absolute_difference",
                "click_purchase_composition_l1",
                "history_time_span_absolute_difference",
                "seeded_hash_tie_break",
            ],
            "constraints": [
                "different_user",
                "donor events truncated to ts < target ts",
                "donor history excludes every target candidate item",
            ],
            "caveat": (
                "Global fallback is a mechanical pilot control, not a final "
                "provenance-matched wrong-user design."
            ),
        },
        "files": {
            condition: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for condition, path in paths.items()
        },
    }
    if release_binding is not None:
        report["motivation_v12_release_binding"] = dict(release_binding)
    return report


def _build_assignment_rows(
    records: list[dict[str, Any]],
    donor_records: list[dict[str, Any]],
    *,
    seed: int,
    global_donor_shortlist_size: int | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    exact_query_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    donors = []
    for record in donor_records:
        if record["history"]:
            exact_query_groups[record["normalized_query"]].append(record)
            donors.append(record)

    true_rows = []
    null_rows = []
    wrong_rows = []
    match_counts: Counter[str] = Counter()
    length_differences: list[int] = []
    target_leakage_violations = 0
    future_violations = 0
    for target in records:
        true_rows.append(
            {
                "request_id": target["request_id"],
                "history": target["history"],
                "assignment": "true",
                "donor_request_id": target["request_id"],
            }
        )
        null_rows.append(
            {
                "request_id": target["request_id"],
                "history": [],
                "assignment": "null",
                "donor_request_id": None,
            }
        )
        if not target["history"]:
            wrong = {
                "request_id": target["request_id"],
                "history": [],
                "assignment": "wrong",
                "donor_request_id": None,
                "match_type": "target_no_history",
            }
        else:
            wrong = _match_wrong_history(
                target,
                exact_query_groups[target["normalized_query"]],
                donors,
                seed=seed,
                global_donor_shortlist_size=global_donor_shortlist_size,
            )
        wrong_rows.append(wrong)
        match_counts[str(wrong["match_type"])] += 1
        wrong_history = wrong["history"]
        if target["history"] and wrong_history:
            length_differences.append(
                abs(len(target["history"]) - len(wrong_history))
            )
        wrong_ids = {str(event["item_id"]) for event in wrong_history}
        target_leakage_violations += int(bool(wrong_ids & target["candidate_ids"]))
        future_violations += sum(
            int(int(event["ts"]) >= target["ts"]) for event in wrong_history
        )
    return {
        "true": true_rows,
        "null": null_rows,
        "wrong": wrong_rows,
    }, {
        "match_counts": dict(match_counts),
        "matched_history_length_absolute_difference": _summary(length_differences),
        "target_candidate_leakage_violations": target_leakage_violations,
        "history_not_strictly_before_target_violations": future_violations,
    }


def _require_release_for_registered_v12_holdout(
    records_path: Path,
    *,
    motivation_v12_release_lock_path: str | Path | None,
) -> None:
    manifest_path = records_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(manifest, dict):
        return
    from myrec.data.kuaisearch_holdout import V12_DATASET_VERSION

    if (
        manifest.get("dataset_version") == V12_DATASET_VERSION
        and records_path.name == "records_confirmation.jsonl"
        and motivation_v12_release_lock_path is None
    ):
        raise ValueError(
            "registered Motivation V1.2 holdout assignments require the "
            "post-selection release lock"
        )


def _motivation_v12_release_binding(
    *,
    records_path: Path,
    donor_records_path: Path,
    release_lock_path: Path,
    seed: int,
    global_donor_shortlist_size: int | None,
    enforce_registered_v12_recipe: bool,
) -> dict[str, Any]:
    """Validate the published holdout and bind the exact released recipe."""

    recipe = motivation_v12_assignment_recipe()
    if seed != recipe["seed"]:
        raise ValueError(
            "Motivation V1.2 history assignment seed differs from release recipe"
        )
    if global_donor_shortlist_size != recipe["global_donor_shortlist_size"]:
        raise ValueError(
            "Motivation V1.2 global donor shortlist differs from release recipe"
        )
    standardized_dir = records_path.parent.resolve()
    expected_records_path = standardized_dir / "records_confirmation.jsonl"
    if records_path.resolve() != expected_records_path:
        raise ValueError(
            "Motivation V1.2 target records must be records_confirmation.jsonl"
        )
    holdout_manifest_path = standardized_dir / "manifest.json"
    if not holdout_manifest_path.is_file():
        raise FileNotFoundError(
            f"missing published holdout manifest: {holdout_manifest_path}"
        )
    holdout_manifest = json.loads(
        holdout_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(holdout_manifest, dict):
        raise ValueError("published holdout manifest must be a JSON object")
    freeze_gate = holdout_manifest.get("freeze_gate")
    release_gate = (
        freeze_gate.get("post_selection_recipe_checkpoint_lock")
        if isinstance(freeze_gate, dict)
        else None
    )
    if not isinstance(release_gate, dict):
        raise ValueError("published holdout has no post-selection release gate")
    declared_release_path = release_gate.get("path")
    if (
        not isinstance(declared_release_path, str)
        or Path(declared_release_path).resolve() != release_lock_path.resolve()
    ):
        raise ValueError(
            "history assignment release lock path differs from published holdout"
        )
    inputs = holdout_manifest.get("inputs")
    donor_input = (
        inputs.get("development_records_train")
        if isinstance(inputs, dict)
        else None
    )
    if not isinstance(donor_input, dict):
        raise ValueError(
            "published holdout does not bind development_records_train"
        )
    declared_donor_path = donor_input.get("path")
    if (
        not isinstance(declared_donor_path, str)
        or Path(declared_donor_path).resolve() != donor_records_path.resolve()
    ):
        raise ValueError(
            "Motivation V1.2 external donor must be development records_train"
        )

    from myrec.data.kuaisearch_holdout import verify_published_holdout

    holdout_audit = verify_published_holdout(
        standardized_dir,
        recipe_checkpoint_lock_path=release_lock_path,
        open_qrels=False,
        enforce_registered_v12_recipe=enforce_registered_v12_recipe,
    )
    if holdout_audit.get("qrels_opened") is not False:
        raise ValueError("assignment release validation must remain qrels-free")
    assignment_release = holdout_audit.get("history_assignment_release")
    expected_release = {
        "schema_version": 1,
        "recipe": recipe,
        "implementation": (
            current_motivation_v12_assignment_implementation_identity()
        ),
    }
    if assignment_release != expected_release:
        raise ValueError(
            "published holdout history assignment release differs from current "
            "frozen generator"
        )
    release_inputs = holdout_audit.get("release_input_sha256")
    expected_donor_sha256 = (
        release_inputs.get("development_records_train")
        if isinstance(release_inputs, dict)
        else None
    )
    donor_sha256 = sha256_file(donor_records_path)
    if donor_sha256 != expected_donor_sha256 or donor_input.get(
        "sha256"
    ) != expected_donor_sha256:
        raise ValueError(
            "Motivation V1.2 external donor hash differs from release input"
        )
    confirmation_file = holdout_audit.get("verified_files", {}).get(
        "records_confirmation"
    )
    if not isinstance(confirmation_file, dict) or confirmation_file.get(
        "sha256"
    ) != sha256_file(records_path):
        raise ValueError(
            "Motivation V1.2 target records differ from published holdout"
        )
    return {
        "schema_version": 1,
        "dataset_id": "kuaisearch",
        "dataset_version": holdout_audit["dataset_version"],
        "conditions": list(recipe["conditions"]),
        "condition_assignment_names": dict(
            recipe["condition_assignment_names"]
        ),
        "source_records_role": "registered_new_holdout_confirmation",
        "source_records_path": str(records_path.resolve()),
        "source_records_sha256": confirmation_file["sha256"],
        "external_donor_records_role": "development_records_train",
        "external_donor_records_path": str(donor_records_path.resolve()),
        "external_donor_records_sha256": donor_sha256,
        "target_population_included_in_wrong_user_donor_pool": True,
        "protocol_sha256": holdout_audit["protocol_sha256"],
        "holdout_manifest_sha256": holdout_audit["manifest_sha256"],
        "holdout_integrity_lock_sha256": holdout_audit[
            "integrity_lock_sha256"
        ],
        "post_selection_release_lock_path": str(release_lock_path.resolve()),
        "post_selection_release_lock_sha256": holdout_audit[
            "post_selection_recipe_checkpoint_lock_sha256"
        ],
        "recipe": recipe,
        "implementation": expected_release["implementation"],
        "qrels_read": False,
        "model_scores_read": False,
        "verified_before_materialization": True,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    result = []
    request_ids: set[str] = set()
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in request_ids:
            raise ValueError(f"duplicate request_id={request_id}")
        request_ids.add(request_id)
        history = [dict(event) for event in row.get("history", [])]
        request_ts = int(row["ts"])
        for event in history:
            if int(event["ts"]) >= request_ts:
                raise ValueError(f"noncausal true history for request_id={request_id}")
        result.append(
            {
                "request_id": request_id,
                "user_id": str(row["user_id"]),
                "ts": request_ts,
                "normalized_query": " ".join(str(row["query"]).casefold().split()),
                "history": history,
                "candidate_ids": {
                    str(candidate["item_id"]) for candidate in row["candidates"]
                },
            }
        )
    if not result:
        raise ValueError(f"empty records file: {path}")
    return result


def _match_wrong_history(
    target: dict[str, Any],
    exact_donors: list[dict[str, Any]],
    global_donors: list[dict[str, Any]],
    *,
    seed: int,
    global_donor_shortlist_size: int | None,
) -> dict[str, Any]:
    exact = _eligible_donors(target, exact_donors, seed=seed)
    if exact:
        match_type = "exact_query_other_user"
        donor, history = min(exact, key=lambda row: row[0])[-2:]
    else:
        shortlisted = _cyclic_shortlist(
            global_donors,
            target_id=target["request_id"],
            seed=seed,
            limit=global_donor_shortlist_size,
        )
        global_matches = _eligible_donors(target, shortlisted, seed=seed)
        if not global_matches:
            return {
                "request_id": target["request_id"],
                "history": [],
                "assignment": "wrong",
                "donor_request_id": None,
                "match_type": "unmatched",
            }
        match_type = "global_other_user"
        donor, history = min(global_matches, key=lambda row: row[0])[-2:]
    return {
        "request_id": target["request_id"],
        "history": history,
        "assignment": "wrong",
        "donor_request_id": donor["request_id"],
        "donor_user_id": donor["user_id"],
        "match_type": match_type,
    }


def _cyclic_shortlist(
    donors: list[dict[str, Any]],
    *,
    target_id: str,
    seed: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Select a deterministic bounded global pool without reading outcomes."""

    if limit is None or len(donors) <= limit:
        return donors
    payload = f"{seed}|global-shortlist|{target_id}"
    start = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    start %= len(donors)
    stop = start + limit
    if stop <= len(donors):
        return donors[start:stop]
    return [*donors[start:], *donors[: stop - len(donors)]]


def _eligible_donors(
    target: dict[str, Any],
    donors: list[dict[str, Any]],
    *,
    seed: int,
) -> list[tuple[tuple[int, int, int, str], dict[str, Any], list[dict[str, Any]]]]:
    result = []
    target_history = target["history"]
    target_composition = _event_composition(target_history)
    target_span = _history_span(target_history)
    for donor in donors:
        if donor["user_id"] == target["user_id"]:
            continue
        history = [
            event
            for event in donor["history"]
            if int(event["ts"]) < target["ts"]
            and str(event["item_id"]) not in target["candidate_ids"]
        ]
        if not history:
            continue
        composition = _event_composition(history)
        distance = (
            abs(len(target_history) - len(history)),
            abs(target_composition[0] - composition[0])
            + abs(target_composition[1] - composition[1]),
            abs(target_span - _history_span(history)),
            _tie_hash(seed, target["request_id"], donor["request_id"]),
        )
        result.append((distance, donor, history))
    return result


def _event_composition(history: list[dict[str, Any]]) -> tuple[int, int]:
    purchases = sum(event.get("event") == "purchase" for event in history)
    return len(history) - purchases, purchases


def _history_span(history: list[dict[str, Any]]) -> int:
    if len(history) < 2:
        return 0
    times = [int(event["ts"]) for event in history]
    return max(times) - min(times)


def _tie_hash(seed: int, target_id: str, donor_id: str) -> str:
    payload = f"{seed}|{target_id}|{donor_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }
