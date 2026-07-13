"""Label-safe helpers for the doc/31 R0-A information-object audit."""

from __future__ import annotations

from collections.abc import Iterable
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _item_has_text(item: dict[str, Any]) -> bool:
    if _has_text(item.get("title")) or _has_text(item.get("brand")):
        return True
    categories = item.get("cat")
    return isinstance(categories, list) and any(_has_text(value) for value in categories)


def summarize_records(paths: Iterable[str | Path]) -> tuple[dict[str, Any], set[str]]:
    """Summarize information objects without consulting any qrels file.

    Candidate label keys in training records are deliberately ignored.  The returned
    summary describes only fields available to a label-free scorer.
    """

    requests = 0
    users: set[str] = set()
    candidate_lengths: list[int] = []
    history_lengths: list[int] = []
    query_text = 0
    candidate_text_rows = 0
    candidate_rows = 0
    history_text_rows = 0
    history_rows = 0
    history_present = 0
    repeat_present = 0
    event_type_rows = 0
    event_timestamp_rows = 0
    strictly_prior_violations = 0
    request_timestamp_rows = 0

    for raw_path in paths:
        path = Path(raw_path)
        if "qrels" in path.name.lower():
            raise PermissionError(f"R0-A may not read qrels: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                requests += 1
                users.add(str(row["user_id"]))
                query_text += int(_has_text(row.get("query")))
                request_ts = row.get("ts")
                request_timestamp_rows += int(request_ts is not None)
                candidates = row.get("candidates") or []
                history = row.get("history") or []
                candidate_lengths.append(len(candidates))
                history_lengths.append(len(history))
                history_present += int(bool(history))
                candidate_ids = {str(item["item_id"]) for item in candidates}
                history_ids = {str(item["item_id"]) for item in history}
                repeat_present += int(bool(candidate_ids & history_ids))
                candidate_rows += len(candidates)
                candidate_text_rows += sum(_item_has_text(item) for item in candidates)
                history_rows += len(history)
                history_text_rows += sum(_item_has_text(item) for item in history)
                for event in history:
                    event_type_rows += int(_has_text(event.get("event")))
                    event_ts = event.get("ts")
                    event_timestamp_rows += int(event_ts is not None)
                    if request_ts is not None and event_ts is not None:
                        strictly_prior_violations += int(event_ts >= request_ts)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    summary = {
        "requests": requests,
        "users": len(users),
        "query_plaintext_rate": ratio(query_text, requests),
        "candidate_count_mean": ratio(sum(candidate_lengths), requests),
        "candidate_count_median": median(candidate_lengths) if candidate_lengths else None,
        "history_count_mean": ratio(sum(history_lengths), requests),
        "history_count_median": median(history_lengths) if history_lengths else None,
        "history_present_rate": ratio(history_present, requests),
        "repeat_present_rate": ratio(repeat_present, requests),
        "candidate_text_coverage": ratio(candidate_text_rows, candidate_rows),
        "history_text_coverage": ratio(history_text_rows, history_rows),
        "request_timestamp_rate": ratio(request_timestamp_rows, requests),
        "history_event_type_rate": ratio(event_type_rows, history_rows),
        "history_event_timestamp_rate": ratio(event_timestamp_rows, history_rows),
        "strictly_prior_violations": strictly_prior_violations,
    }
    return summary, users


def split_user_overlap(train_users: set[str], dev_users: set[str]) -> dict[str, Any]:
    overlap = train_users & dev_users
    return {
        "train_users": len(train_users),
        "dev_users": len(dev_users),
        "overlap_users": len(overlap),
        "dev_user_overlap_rate": len(overlap) / len(dev_users) if dev_users else None,
        "user_disjoint": not overlap,
    }


def mde_from_reference_ci(
    ci95: tuple[float, float] | list[float],
    reference_units: int,
    target_units: int,
    *,
    alpha_z: float = 1.959963984540054,
    power_z: float = 0.8416212335729143,
) -> float:
    """Approximate two-sided 80%-power MDE by scaling a paired CI standard error."""

    if reference_units <= 0 or target_units <= 0:
        raise ValueError("power units must be positive")
    low, high = (float(ci95[0]), float(ci95[1]))
    if not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise ValueError("invalid reference confidence interval")
    reference_se = (high - low) / (2.0 * alpha_z)
    target_se = reference_se * math.sqrt(reference_units / target_units)
    return (alpha_z + power_z) * target_se


def required_units_for_mde(
    ci95: tuple[float, float] | list[float],
    reference_units: int,
    target_mde: float,
    *,
    alpha_z: float = 1.959963984540054,
    power_z: float = 0.8416212335729143,
) -> int:
    if target_mde <= 0:
        raise ValueError("target MDE must be positive")
    low, high = (float(ci95[0]), float(ci95[1]))
    reference_se = (high - low) / (2.0 * alpha_z)
    required = reference_units * ((alpha_z + power_z) * reference_se / target_mde) ** 2
    return max(1, math.ceil(required))

