"""Shared preparation helpers for the ordinary R0 full-token baseline."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import hashlib
import math
from typing import Any


def item_text(item: dict[str, Any]) -> str:
    categories = item.get("cat")
    category = (
        " > ".join(str(value) for value in categories if value)
        if isinstance(categories, list)
        else ""
    )
    return " ".join(
        value.strip()
        for value in (
            str(item.get("title") or ""),
            str(item.get("brand") or ""),
            str(item.get("seller") or ""),
            category,
        )
        if value.strip()
    )


def history_length_bin(length: int, boundaries: list[int]) -> int:
    if length <= 0:
        return 0
    for boundary in boundaries:
        if length <= boundary:
            return int(boundary)
    return int(boundaries[-1])


def build_donor_index(
    records: list[dict[str, Any]], boundaries: list[int]
) -> dict[int, tuple[list[int], list[tuple[int, int]]]]:
    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for index, record in enumerate(records):
        for prefix_len, event in enumerate(record.get("history") or [], start=1):
            key = history_length_bin(prefix_len, boundaries)
            grouped.setdefault(key, []).append((int(event["ts"]), index, prefix_len))
    result = {}
    for key, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                row[0],
                str(records[row[1]]["request_id"]),
                row[2],
            )
        )
        result[key] = (
            [row[0] for row in rows],
            [(row[1], row[2]) for row in rows],
        )
    return result


def choose_fresh_wrong_donor(
    records: list[dict[str, Any]],
    donor_index: dict[int, tuple[list[int], list[tuple[int, int]]]],
    target_index: int,
    boundaries: list[int],
    *,
    seed: int,
    freshness_ratio_max: float,
    search_back: int,
) -> tuple[int | None, int | None, int | None, float | None]:
    target = records[target_index]
    if not target.get("history"):
        return None, None, None, None
    target_length = len(target["history"])
    target_key = history_length_bin(target_length, boundaries)
    # A donor request may occur later than the target while its entire history is
    # already available at target time.  Index on the latest donor event, not the
    # donor request timestamp, to implement the rolling request-time boundary.
    target_latest = max(int(event["ts"]) for event in target["history"])
    target_age = max(1, int(target["ts"]) - target_latest)
    minimum_age = max(1, math.ceil(target_age / float(freshness_ratio_max)))
    maximum_age = max(minimum_age, math.floor(target_age * float(freshness_ratio_max)))
    earliest_latest_event = int(target["ts"]) - maximum_age
    latest_latest_event = int(target["ts"]) - minimum_age
    compatible_keys = sorted(key for key in donor_index if key >= target_key)
    key_offset = int.from_bytes(
        hashlib.sha256(f"{seed}:r0-full-token-key:{target['request_id']}".encode()).digest()[:8],
        "big",
    ) % max(1, len(compatible_keys))
    for key_step in range(len(compatible_keys)):
        key = compatible_keys[(key_offset + key_step) % len(compatible_keys)]
        timestamps, indices = donor_index[key]
        start = bisect_left(timestamps, earliest_latest_event)
        stop = bisect_right(timestamps, latest_latest_event)
        span = stop - start
        if span <= 0:
            continue
        offset = int.from_bytes(
            hashlib.sha256(
                f"{seed}:r0-full-token-window:{key}:{target['request_id']}".encode()
            ).digest()[:8],
            "big",
        ) % span
        for step in range(min(span, int(search_back))):
            index, prefix_len = indices[start + ((offset + step) % span)]
            donor = records[index]
            if index == target_index or str(donor["user_id"]) == str(target["user_id"]):
                continue
            if prefix_len < target_length:
                continue
            slice_start = prefix_len - target_length
            donor_history = donor["history"][slice_start:prefix_len]
            donor_latest = int(donor_history[-1]["ts"])
            if donor_latest >= int(target["ts"]):
                continue
            donor_age = max(1, int(target["ts"]) - donor_latest)
            ratio = max(target_age, donor_age) / min(target_age, donor_age)
            if ratio > float(freshness_ratio_max):
                continue
            return index, slice_start, prefix_len, ratio
    return None, None, None, None


def clicked_labels(record: dict[str, Any]) -> list[float]:
    return [float(candidate.get("clicked", 0)) for candidate in record["candidates"]]
