"""Validation for the label-isolated standardized PPS record interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl

ALLOWED_SPLITS = {"train", "dev", "confirmation", "test"}
EVALUATION_LABEL_KEYS = {
    "clicked",
    "is_clicked",
    "is_purchased",
    "label",
    "labels",
    "purchased",
    "relevance",
    "target",
}


def validate_standardized_record(record: dict[str, Any], split: str) -> None:
    """Validate one method-visible standardized record.

    The check is deliberately dataset-independent. Source-specific adapters may
    map raw fields into this interface, but model code must only see records that
    pass this contract.
    """

    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    for key in ("request_id", "user_id", "session_id", "ts", "query", "history", "candidates", "masks"):
        if key not in record:
            raise ValueError(f"missing required field: {key}")

    _require_identifier(record["request_id"], "request_id")
    _require_identifier(record["user_id"], "user_id")
    _require_identifier(record["session_id"], "session_id")
    request_ts = _require_integer(record["ts"], "ts")
    if not isinstance(record["query"], str) or not record["query"].strip():
        raise ValueError("query must be a non-empty string")

    history = record["history"]
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    for index, event in enumerate(history):
        if not isinstance(event, dict):
            raise ValueError(f"history[{index}] must be an object")
        _require_identifier(event.get("item_id"), f"history[{index}].item_id")
        event_ts = _require_integer(event.get("ts"), f"history[{index}].ts")
        if event_ts >= request_ts:
            raise ValueError(
                f"history[{index}].ts must be strictly before request ts: "
                f"{event_ts} >= {request_ts}"
            )
        if not isinstance(event.get("event"), str) or not event["event"].strip():
            raise ValueError(f"history[{index}].event must be a non-empty string")

    candidates = record["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("candidates must be a list with at least two entries")
    candidate_ids: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidates[{index}] must be an object")
        candidate_id = _require_identifier(
            candidate.get("item_id"), f"candidates[{index}].item_id"
        )
        candidate_ids.append(candidate_id)
        if split != "train":
            leaked = sorted(EVALUATION_LABEL_KEYS & set(candidate))
            if leaked:
                raise ValueError(
                    f"evaluation labels leaked into {split} candidates[{index}]: {leaked}"
                )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate item_id values must be unique within a request")

    masks = record["masks"]
    if not isinstance(masks, dict):
        raise ValueError("masks must be an object")
    history_present = masks.get("history_present")
    if not isinstance(history_present, bool):
        raise ValueError("masks.history_present must be boolean")
    if history_present != bool(history):
        raise ValueError("masks.history_present does not match history emptiness")
    text_coverage = masks.get("text_coverage")
    if isinstance(text_coverage, bool) or not isinstance(text_coverage, (int, float)):
        raise ValueError("masks.text_coverage must be numeric")
    if not 0.0 <= float(text_coverage) <= 1.0:
        raise ValueError("masks.text_coverage must be in [0, 1]")

    if split != "train":
        leaked_top_level = sorted(EVALUATION_LABEL_KEYS & set(record))
        if leaked_top_level:
            raise ValueError(f"evaluation labels leaked into {split} record: {leaked_top_level}")


def audit_standardized_file(path: str | Path, split: str) -> dict[str, Any]:
    """Validate a JSONL record file and return label-free structural counts."""

    path = Path(path)
    request_ids: set[str] = set()
    candidate_counts: list[int] = []
    history_lengths: list[int] = []
    history_present_requests = 0
    strict_nonrepeat_requests = 0
    query_counts: dict[str, int] = {}

    for line_no, record in enumerate(iter_jsonl(path), start=1):
        try:
            validate_standardized_record(record, split)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        request_id = str(record["request_id"])
        if request_id in request_ids:
            raise ValueError(f"{path}:{line_no}: duplicate request_id={request_id}")
        request_ids.add(request_id)

        candidates = record["candidates"]
        history = record["history"]
        candidate_counts.append(len(candidates))
        history_lengths.append(len(history))
        history_present_requests += int(bool(history))
        history_ids = {str(event["item_id"]) for event in history}
        candidate_ids = {str(candidate["item_id"]) for candidate in candidates}
        strict_nonrepeat_requests += int(bool(history) and history_ids.isdisjoint(candidate_ids))
        normalized_query = " ".join(record["query"].casefold().split())
        query_counts[normalized_query] = query_counts.get(normalized_query, 0) + 1

    if not request_ids:
        raise ValueError(f"empty standardized file: {path}")
    repeated_query_requests = sum(count for count in query_counts.values() if count > 1)
    return {
        "candidate_count": _int_summary(candidate_counts),
        "history_length": _int_summary(history_lengths),
        "history_present_requests": history_present_requests,
        "path": str(path),
        "repeated_query_requests": repeated_query_requests,
        "request_count": len(request_ids),
        "sha256": sha256_file(path),
        "split": split,
        "strict_nonrepeat_requests": strict_nonrepeat_requests,
        "unique_normalized_queries": len(query_counts),
    }


def _require_identifier(value: Any, field: str) -> str:
    if value is None or isinstance(value, bool) or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty identifier")
    return str(value)


def _require_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _int_summary(values: list[int]) -> dict[str, float | int]:
    return {
        "max": max(values),
        "mean": sum(values) / len(values),
        "min": min(values),
    }
