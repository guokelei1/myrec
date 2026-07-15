"""Materialize label-free request-surface memberships for exploratory analysis."""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Any

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


def materialize_request_surfaces(
    records_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    records_path = Path(records_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    members: dict[str, set[str]] = {
        "all": set(),
        "history_present": set(),
        "strict_nonrepeat": set(),
        "repeat": set(),
        "no_history": set(),
        "repeated_query": set(),
        "repeated_query_history_present": set(),
        "repeated_query_repeat": set(),
        "repeated_query_strict_nonrepeat": set(),
        "singleton_query": set(),
    }
    normalized_query_by_request: dict[str, str] = {}
    query_counts: Counter[str] = Counter()
    for record in iter_jsonl(records_path):
        request_id = str(record["request_id"])
        if request_id in members["all"]:
            raise ValueError(f"duplicate request_id={request_id}")
        members["all"].add(request_id)
        normalized_query = "".join(str(record.get("query", "")).casefold().split())
        normalized_query_by_request[request_id] = normalized_query
        query_counts[normalized_query] += 1
        history_ids = {str(event["item_id"]) for event in record.get("history", [])}
        candidate_ids = {
            str(candidate["item_id"]) for candidate in record.get("candidates", [])
        }
        history_present = bool(history_ids)
        strict_nonrepeat = history_present and history_ids.isdisjoint(candidate_ids)
        recorded_history_present = record.get("masks", {}).get("history_present")
        recorded_strict = record.get("masks", {}).get("strict_nonrepeat")
        if recorded_history_present is not history_present:
            raise ValueError(f"history_present mask mismatch for request_id={request_id}")
        if recorded_strict is not None and bool(recorded_strict) != strict_nonrepeat:
            raise ValueError(f"strict_nonrepeat mask mismatch for request_id={request_id}")
        if not history_present:
            members["no_history"].add(request_id)
        else:
            members["history_present"].add(request_id)
            members["strict_nonrepeat" if strict_nonrepeat else "repeat"].add(request_id)

    for request_id, normalized_query in normalized_query_by_request.items():
        target = "repeated_query" if query_counts[normalized_query] > 1 else "singleton_query"
        members[target].add(request_id)
        if target == "repeated_query" and request_id in members["history_present"]:
            members["repeated_query_history_present"].add(request_id)
            if request_id in members["strict_nonrepeat"]:
                members["repeated_query_strict_nonrepeat"].add(request_id)
            else:
                members["repeated_query_repeat"].add(request_id)

    files = {}
    for name, request_ids in members.items():
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
        "source_records_path": str(records_path),
        "source_records_sha256": sha256_file(records_path),
        "label_free": True,
        "definitions": {
            "history_present": "history item-id set is non-empty",
            "strict_nonrepeat": "history present and history/candidate item-id sets are disjoint",
            "repeat": "history present and at least one candidate item-id occurs in history",
            "no_history": "history item-id set is empty",
            "repeated_query": "normalized exact query occurs in at least two requests",
            "repeated_query_history_present": (
                "repeated_query intersection history_present"
            ),
            "repeated_query_repeat": "repeated_query intersection repeat",
            "repeated_query_strict_nonrepeat": (
                "repeated_query intersection strict_nonrepeat"
            ),
            "singleton_query": "normalized exact query occurs in exactly one request",
        },
        "files": files,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
