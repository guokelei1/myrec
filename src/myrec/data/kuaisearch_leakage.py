"""Recall-log cross-check for KuaiSearch recent-history leakage."""

from __future__ import annotations

import bisect
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.data.kuaisearch_audit import (
    KuaiSearchRawPaths,
    RANK_HISTORY_FIELDS,
    request_key,
    request_id_from_key,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json

SESSION_ID_RE = re.compile(r'"session_id"\s*:\s*"?([^",}\s]+)"?')


def check_history_leakage(
    raw_dir: str | Path,
    seed: int = 20260708,
    sample_size: int = 1000,
    max_examples: int = 10,
) -> dict[str, Any]:
    """Check ranking recent-history fields against recall user-item events.

    This is a log-internal cross-reference check for datasets where the ranking
    table exposes ordered recent item IDs but not per-history event timestamps.
    """

    paths = KuaiSearchRawPaths.from_raw_dir(raw_dir)
    _assert_files(paths)
    recall_state = _scan_recall(paths.recall)
    sampled = _sample_recall_requests(
        recall_state["requests"],
        median_time_index=recall_state["median_time_index"],
        sample_size=sample_size,
        seed=seed,
    )
    rank_hits = _scan_rank_for_history(paths.rank, sampled["sampled_keys"])
    classification = _classify_histories(
        sampled_keys=sampled["sampled_keys"],
        rank_hits=rank_hits,
        events=recall_state["events"],
        max_examples=max_examples,
    )
    status = _status_from_classification(classification, sample_size)
    report = {
        "check_id": "kuaisearch_c0_history_leakage_recall_cross_reference",
        "dataset_id": "kuaisearch",
        "dataset_version": "v0_lite",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(Path(raw_dir)),
        "seed": seed,
        "sample_size": sample_size,
        "status": status,
        "rule": {
            "sample_source": "recall_lite requests with time_index >= global median",
            "pass_if": {
                "checked_requests": sample_size,
                "future_only_observed_rate_max": 0.001,
                "past_supported_total_history_rate_min": 0.20,
            },
            "manual_review_if": {
                "same_time_only_observed_rate_gt": 0.05,
            },
            "caveat": "This is log-internal cross-reference evidence, not an official per-event timestamp guarantee. Unobserved history items are neutral and cannot be falsified within the recall window.",
        },
        "recall_event_index": {
            "path": str(paths.recall),
            "rows": recall_state["rows"],
            "total_events": recall_state["total_events"],
            "user_item_pairs": len(recall_state["events"]),
            "users": recall_state["users"],
            "time_index_min": recall_state["time_index_min"],
            "time_index_max": recall_state["time_index_max"],
            "median_time_index": recall_state["median_time_index"],
        },
        "sampling": {
            "eligible_requests": sampled["eligible_count"],
            "sampled_requests": len(sampled["sampled_keys"]),
            "sample_request_ids_sha256": sampled["sample_request_ids_sha256"],
            "time_index_min": sampled["time_index_min"],
            "time_index_max": sampled["time_index_max"],
        },
        "rank_history_lookup": {
            "path": str(paths.rank),
            "checked_requests": classification["checked_requests"],
            "missing_sampled_requests": classification["missing_sampled_requests"],
            "history_fields": list(RANK_HISTORY_FIELDS),
        },
        "classification": classification,
    }
    return report


def merge_history_leakage_into_c0(
    c0_report_path: str | Path,
    leakage_report_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    c0_report_path = Path(c0_report_path)
    leakage_report_path = Path(leakage_report_path)
    with c0_report_path.open("r", encoding="utf-8") as handle:
        c0_report = json.load(handle)
    with leakage_report_path.open("r", encoding="utf-8") as handle:
        leakage_report = json.load(handle)

    classification = leakage_report["classification"]
    c0_report["checks"]["history_future_leakage"] = {
        "status": leakage_report["status"],
        "evidence": {
            "method": "recall_log_cross_reference",
            "report_path": str(leakage_report_path),
            "report_sha256": sha256_file(leakage_report_path),
            "checked_requests": classification["checked_requests"],
            "total_history_items": classification["total_history_items"],
            "observed_history_items": classification["observed_history_items"],
            "past_supported_total_history_rate": classification["rates"][
                "past_supported_over_total_history"
            ],
            "future_only_observed_rate": classification["rates"]["future_only_over_observed"],
            "same_time_only_observed_rate": classification["rates"][
                "same_time_only_over_observed"
            ],
            "future_only_examples": classification["future_only_examples"],
            "caveat": leakage_report["rule"]["caveat"],
        },
        "rule": (
            "Recall-log cross-reference: sample 1000 recall requests with "
            "time_index >= global median; pass iff checked_requests == 1000, "
            "future_only/(past_supported+same_time_only+future_only) <= 0.001, "
            "and past_supported/total_history_items >= 0.20."
        ),
    }
    c0_report["external_history_leakage_check"] = {
        "report_path": str(leakage_report_path),
        "report_sha256": sha256_file(leakage_report_path),
        "status": leakage_report["status"],
        "generated_at": leakage_report["generated_at"],
        "caveat": leakage_report["rule"]["caveat"],
    }
    c0_report["overall_status"] = (
        "passed"
        if all(check["status"] == "passed" for check in c0_report["checks"].values())
        else "failed"
    )
    write_json(output_path or c0_report_path, c0_report)
    return c0_report


def _assert_files(paths: KuaiSearchRawPaths) -> None:
    missing = [str(path) for path in (paths.recall, paths.rank) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required KuaiSearch files: " + ", ".join(missing))


def _scan_recall(path: Path) -> dict[str, Any]:
    events: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    requests: list[tuple[str, str, str, int]] = []
    time_values: list[int] = []
    users: set[str] = set()
    total_events = 0
    rows = 0

    for row in iter_jsonl(path):
        rows += 1
        key = request_key(row)
        user_id = key[0]
        time_index = key[3]
        users.add(user_id)
        requests.append(key)
        time_values.append(time_index)
        event_items = set(int(item) for item in row.get("clicked_item_ids", []))
        event_items.update(int(item) for item in row.get("purchased_item_ids", []))
        for item_id in event_items:
            events[(user_id, item_id)].append(time_index)
            total_events += 1

    for times in events.values():
        times.sort()
    ordered_times = sorted(time_values)
    median_time_index = ordered_times[len(ordered_times) // 2] if ordered_times else None
    return {
        "events": dict(events),
        "requests": requests,
        "rows": rows,
        "total_events": total_events,
        "users": len(users),
        "time_index_min": ordered_times[0] if ordered_times else None,
        "time_index_max": ordered_times[-1] if ordered_times else None,
        "median_time_index": median_time_index,
    }


def _sample_recall_requests(
    requests: list[tuple[str, str, str, int]],
    median_time_index: int,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    eligible = [key for key in requests if key[3] >= median_time_index]
    if len(eligible) < sample_size:
        sampled = eligible
    else:
        sampled = random.Random(seed).sample(eligible, sample_size)
    sampled.sort(key=lambda key: (key[3], key[0], key[1], key[2]))
    request_ids = [request_id_from_key(key) for key in sampled]
    digest_payload = "\n".join(request_ids)
    return {
        "eligible_count": len(eligible),
        "sampled_keys": sampled,
        "sample_request_ids_sha256": _sha256_string(digest_payload),
        "time_index_min": min((key[3] for key in sampled), default=None),
        "time_index_max": max((key[3] for key in sampled), default=None),
    }


def _scan_rank_for_history(
    path: Path,
    sampled_keys: list[tuple[str, str, str, int]],
) -> dict[tuple[str, str, str, int], list[int]]:
    sampled_set = set(sampled_keys)
    session_ids = {key[1] for key in sampled_keys}
    hits: dict[tuple[str, str, str, int], list[int]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = SESSION_ID_RE.search(line)
            if not match or match.group(1) not in session_ids:
                continue
            row = json.loads(line)
            key = request_key(row)
            if key not in sampled_set or key in hits:
                continue
            history: list[int] = []
            for field in RANK_HISTORY_FIELDS:
                history.extend(int(item) for item in row.get(field, []) if int(item) != 0)
            hits[key] = history
            if len(hits) == len(sampled_set):
                break
    return hits


def _classify_histories(
    sampled_keys: list[tuple[str, str, str, int]],
    rank_hits: dict[tuple[str, str, str, int], list[int]],
    events: dict[tuple[str, int], list[int]],
    max_examples: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total_history_items = 0
    checked_requests = 0
    future_examples: list[dict[str, Any]] = []
    same_time_examples: list[dict[str, Any]] = []
    future_example_keys: set[tuple[tuple[str, str, str, int], int]] = set()
    same_time_example_keys: set[tuple[tuple[str, str, str, int], int]] = set()
    missing_requests: list[dict[str, Any]] = []

    for key in sampled_keys:
        history = rank_hits.get(key)
        if history is None:
            if len(missing_requests) < max_examples:
                missing_requests.append(_request_example(key))
            continue
        checked_requests += 1
        user_id = key[0]
        request_time = key[3]
        total_history_items += len(history)
        for item_id in history:
            item_events = events.get((user_id, item_id), [])
            label = _classify_item_events(item_events, request_time)
            counts[label] += 1
            example_key = (key, item_id)
            if (
                label == "future_only"
                and len(future_examples) < max_examples
                and example_key not in future_example_keys
            ):
                future_example_keys.add(example_key)
                future_examples.append(
                    {
                        **_request_example(key),
                        "history_item_id": item_id,
                        "event_times": item_events[:20],
                    }
                )
            elif (
                label == "same_time_only"
                and len(same_time_examples) < max_examples
                and example_key not in same_time_example_keys
            ):
                same_time_example_keys.add(example_key)
                same_time_examples.append(
                    {
                        **_request_example(key),
                        "history_item_id": item_id,
                        "event_times": item_events[:20],
                    }
                )

    observed = counts["past_supported"] + counts["same_time_only"] + counts["future_only"]
    rates = {
        "past_supported_over_total_history": _safe_rate(counts["past_supported"], total_history_items),
        "observed_over_total_history": _safe_rate(observed, total_history_items),
        "future_only_over_observed": _safe_rate(counts["future_only"], observed),
        "same_time_only_over_observed": _safe_rate(counts["same_time_only"], observed),
        "same_time_only_over_total_history": _safe_rate(counts["same_time_only"], total_history_items),
        "unobserved_over_total_history": _safe_rate(counts["unobserved"], total_history_items),
    }
    return {
        "checked_requests": checked_requests,
        "missing_sampled_requests": len(sampled_keys) - checked_requests,
        "missing_request_examples": missing_requests,
        "total_history_items": total_history_items,
        "observed_history_items": observed,
        "counts": {
            "past_supported": counts["past_supported"],
            "same_time_only": counts["same_time_only"],
            "future_only": counts["future_only"],
            "unobserved": counts["unobserved"],
        },
        "rates": rates,
        "manual_review_required": rates["same_time_only_over_observed"] > 0.05,
        "future_only_examples": future_examples,
        "same_time_only_examples": same_time_examples,
    }


def _classify_item_events(event_times: list[int], request_time: int) -> str:
    if not event_times:
        return "unobserved"
    index = bisect.bisect_left(event_times, request_time)
    if index > 0:
        return "past_supported"
    if index < len(event_times) and event_times[index] == request_time:
        return "same_time_only"
    return "future_only"


def _status_from_classification(classification: dict[str, Any], sample_size: int) -> str:
    return (
        "passed"
        if classification["checked_requests"] == sample_size
        and classification["rates"]["future_only_over_observed"] <= 0.001
        and classification["rates"]["past_supported_over_total_history"] >= 0.20
        else "failed"
    )


def _request_example(key: tuple[str, str, str, int]) -> dict[str, Any]:
    return {
        "request_id": request_id_from_key(key),
        "user_id": key[0],
        "session_id": key[1],
        "query": key[2],
        "time_index": key[3],
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sha256_string(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
