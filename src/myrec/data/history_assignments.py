"""Build label-free true/null/matched-wrong history assignments."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json, write_jsonl


def materialize_history_assignments(
    records_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    donor_records_path: str | Path | None = None,
    seed: int = 20260714,
    global_donor_shortlist_size: int | None = 512,
) -> dict[str, Any]:
    """Write true/null/wrong assignments without reading qrels or model scores."""

    records_path = Path(records_path)
    output_dir = Path(output_dir)
    if global_donor_shortlist_size is not None and global_donor_shortlist_size <= 0:
        raise ValueError("global_donor_shortlist_size must be positive or None")
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
            length_differences.append(abs(len(target["history"]) - len(wrong_history)))
        wrong_ids = {str(event["item_id"]) for event in wrong_history}
        target_leakage_violations += int(bool(wrong_ids & target["candidate_ids"]))
        future_violations += sum(
            int(int(event["ts"]) >= target["ts"]) for event in wrong_history
        )

    paths = {
        "true": output_dir / "true.jsonl",
        "null": output_dir / "null.jsonl",
        "wrong": output_dir / "wrong.jsonl",
    }
    for condition, rows in (
        ("true", true_rows),
        ("null", null_rows),
        ("wrong", wrong_rows),
    ):
        write_jsonl(paths[condition], rows)
    report = {
        "evidence_mode": "exploratory",
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
        "donor_pool_requests": len(donor_records),
        "qrels_read": False,
        "model_scores_read": False,
        "seed": seed,
        "requests": len(records),
        "match_counts": dict(match_counts),
        "matched_history_length_absolute_difference": _summary(length_differences),
        "target_candidate_leakage_violations": target_leakage_violations,
        "history_not_strictly_before_target_violations": future_violations,
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
    write_json(output_dir / "manifest.json", report)
    write_json(report_path, report)
    return report


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
