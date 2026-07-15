"""Exploratory, outcome-free source audit for KuaiSearch public JSONL files."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json

RECALL_REQUIRED_FIELDS = (
    "user_id",
    "session_id",
    "query",
    "time_index",
    "impressed_item_ids",
    "clicked_item_ids",
    "purchased_item_ids",
)
RANK_HISTORY_FIELDS = (
    "recently_clicked_item_ids",
    "recently_purchased_item_ids",
)
SESSION_ID_RE = re.compile(r'"session_id"\s*:\s*"?([^",}\s]+)"?')

RequestKey = tuple[str, str, str, int]


def audit_kuaisearch_source(
    raw_dir: str | Path,
    report_path: str | Path,
    *,
    seed: int = 20260714,
    collision_query_limit: int = 500,
    collision_requests_per_query: int = 50,
    rank_history_sample_size: int = 1000,
    max_recall_rows: int | None = None,
    max_rank_scan_rows: int | None = None,
    included_source_splits: tuple[str, ...] = ("train",),
) -> dict[str, Any]:
    """Audit source populations without reading model outputs or evaluator qrels.

    Collision enumeration is an exploratory power scout over a deterministic
    reservoir from the most repeated exact queries. It is not a frozen cohort.
    """

    if collision_query_limit < 1:
        raise ValueError("collision_query_limit must be positive")
    if collision_requests_per_query < 2:
        raise ValueError("collision_requests_per_query must be at least two")
    if rank_history_sample_size < 1:
        raise ValueError("rank_history_sample_size must be positive")
    if set(included_source_splits) != {"train"}:
        raise ValueError(
            "exploratory source audit is locked to split=train; evaluation-source labels stay closed"
        )

    raw_dir = Path(raw_dir)
    recall_path, source_variant = _resolve_source_path(raw_dir, "recall")
    rank_path, rank_variant = _resolve_source_path(raw_dir, "rank")
    if rank_variant != source_variant:
        raise ValueError("KuaiSearch recall and rank source variants differ")
    for path in (recall_path, rank_path):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing KuaiSearch source file: {path}")

    first_pass = _scan_recall_first_pass(
        recall_path,
        included_source_splits=set(included_source_splits),
        max_rows=max_recall_rows,
    )
    repeated_queries = {
        query
        for query, count in first_pass["query_counts"].items()
        if count > 1
    }
    selected_queries = {
        query
        for query, _ in sorted(
            first_pass["query_counts"].items(),
            key=lambda pair: (-pair[1], pair[0]),
        )[:collision_query_limit]
        if query in repeated_queries
    }
    second_pass = _scan_recall_second_pass(
        recall_path,
        events=first_pass["events"],
        user_first_event=first_pass["user_first_event"],
        selected_queries=selected_queries,
        collision_requests_per_query=collision_requests_per_query,
        rank_history_sample_size=rank_history_sample_size,
        seed=seed,
        included_source_splits=set(included_source_splits),
        max_rows=max_recall_rows,
    )
    collision = _collision_opportunity(second_pass["collision_records"])
    rank_history = _scan_rank_histories(
        rank_path,
        sample_records=second_pass["rank_sample_records"],
        events=first_pass["events"],
        max_rows=max_rank_scan_rows,
    )

    query_counts: Counter[str] = first_pass["query_counts"]
    rows = first_pass["rows"]
    report = {
        "audit_id": "history_response_gap_kuaisearch_lite_source_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_mode": "exploratory",
        "dataset_id": f"kuaisearch_{source_variant}",
        "scope_warning": (
            f"All observations have {source_variant}-source exploratory scope. "
            "A bounded scan or time-window scout is not a frozen confirmation."
        ),
        "parameters": {
            "seed": seed,
            "collision_query_limit": collision_query_limit,
            "collision_requests_per_query": collision_requests_per_query,
            "rank_history_sample_size": rank_history_sample_size,
            "max_recall_rows": max_recall_rows,
            "max_rank_scan_rows": max_rank_scan_rows,
            "included_source_splits": list(included_source_splits),
            "source_variant": source_variant,
        },
        "source_files": {
            "recall": _file_info(recall_path, hash_limit_bytes=512 * 1024 * 1024),
            "rank": _file_info(rank_path, hash_limit_bytes=512 * 1024 * 1024),
        },
        "recall": {
            "rows": rows,
            "source_rows_seen": first_pass["source_rows_seen"],
            "source_split_counts_seen": dict(first_pass["source_split_counts_seen"]),
            "excluded_source_rows": first_pass["source_rows_seen"] - rows,
            "complete_scan": max_recall_rows is None,
            "first_row_keys": first_pass["first_row_keys"],
            "missing_required_fields": dict(first_pass["missing_required_fields"]),
            "duplicate_request_keys": first_pass["duplicate_request_keys"],
            "unique_users": len(first_pass["users"]),
            "unique_normalized_queries": len(query_counts),
            "repeated_exact_query_groups": len(repeated_queries),
            "repeated_exact_query_requests": sum(
                count for count in query_counts.values() if count > 1
            ),
            "candidate_count": _summary(first_pass["candidate_counts"]),
            "clicked_count": _summary(first_pass["clicked_counts"]),
            "purchased_count": _summary(first_pass["purchased_counts"]),
            "requests_with_click": first_pass["requests_with_click"],
            "requests_with_purchase": first_pass["requests_with_purchase"],
            "duplicate_candidate_requests": first_pass["duplicate_candidate_requests"],
            "click_outside_slate": first_pass["click_outside_slate"],
            "purchase_outside_slate": first_pass["purchase_outside_slate"],
            "split_counts": dict(first_pass["split_counts"]),
            "time_index": _summary(first_pass["time_values"]),
        },
        "reconstructed_prior_history": {
            "definition": (
                "same-user clicked/purchased items observed in recall rows with "
                "event time strictly before the target request"
            ),
            "history_present_requests": second_pass["history_present_requests"],
            "history_present_rate": _rate(second_pass["history_present_requests"], rows),
            "strict_nonrepeat_requests": second_pass["strict_nonrepeat_requests"],
            "strict_nonrepeat_rate_over_history_present": _rate(
                second_pass["strict_nonrepeat_requests"],
                second_pass["history_present_requests"],
            ),
            "requests_with_recurrent_candidate": second_pass[
                "requests_with_recurrent_candidate"
            ],
            "recurrent_candidate_count": _summary(
                second_pass["recurrent_candidate_counts"]
            ),
            "caveat": (
                "This history is causally valid inside the observed recall log, "
                "but interactions outside that log are unobserved."
            ),
        },
        "exact_query_collision_scout": {
            **collision,
            "selection": {
                "query_groups_considered": len(selected_queries),
                "strategy": (
                    "top exact normalized queries by source frequency, then a "
                    "seeded per-query request reservoir"
                ),
                "requests_per_query_cap": collision_requests_per_query,
                "sampled_requests": sum(
                    len(rows) for rows in second_pass["collision_records"].values()
                ),
                "source_requests_in_considered_groups": sum(
                    query_counts[query] for query in selected_queries
                ),
            },
            "scope": (
                "Power/opportunity scout only; no labels or model scores were "
                "used and this is not a frozen confirmation cohort."
            ),
        },
        "raw_rank_recent_history_sample": rank_history,
        "integrity": {
            "evaluation_source_behavior_fields_accessed": False,
            "included_source_splits": list(included_source_splits),
            "note": (
                "Rows outside the included split are counted by split name and "
                "discarded before click, purchase, candidate, query, user, or "
                "history fields are accessed."
            ),
        },
        "direct_observations": _direct_observations(
            rows=rows,
            repeated_query_requests=sum(
                count for count in query_counts.values() if count > 1
            ),
            history_present_requests=second_pass["history_present_requests"],
            strict_nonrepeat_requests=second_pass["strict_nonrepeat_requests"],
            collision=collision,
            rank_history=rank_history,
        ),
    }
    write_json(report_path, report)
    return report


def _resolve_source_path(raw_dir: Path, stem: str) -> tuple[Path, str]:
    candidates = (
        (raw_dir / f"{stem}_lite" / "train.jsonl", "lite"),
        (raw_dir / stem / "train.jsonl", "full"),
    )
    present = [(path, variant) for path, variant in candidates if path.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(
            f"expected exactly one KuaiSearch {stem} source variant under {raw_dir}; "
            f"found {[str(path) for path, _ in present]}"
        )
    return present[0]


def _scan_recall_first_pass(
    path: Path,
    *,
    included_source_splits: set[str],
    max_rows: int | None,
) -> dict[str, Any]:
    source_rows_seen = 0
    source_split_counts_seen: Counter[str] = Counter()
    rows = 0
    first_row_keys: list[str] = []
    missing_required_fields: Counter[str] = Counter()
    request_keys: set[RequestKey] = set()
    duplicate_request_keys = 0
    users: set[str] = set()
    query_counts: Counter[str] = Counter()
    candidate_counts: list[int] = []
    clicked_counts: list[int] = []
    purchased_counts: list[int] = []
    requests_with_click = 0
    requests_with_purchase = 0
    duplicate_candidate_requests = 0
    click_outside_slate = 0
    purchase_outside_slate = 0
    split_counts: Counter[str] = Counter()
    time_values: list[int] = []
    events: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    user_first_event: dict[str, int] = {}

    for row in iter_jsonl(path):
        source_rows_seen += 1
        source_split = str(row.get("split", ""))
        source_split_counts_seen[source_split] += 1
        if source_split not in included_source_splits:
            if max_rows is not None and source_rows_seen >= max_rows:
                break
            continue
        rows += 1
        if not first_row_keys:
            first_row_keys = sorted(row)
        for field in RECALL_REQUIRED_FIELDS:
            if field not in row:
                missing_required_fields[field] += 1
        key = _request_key(row)
        if key in request_keys:
            duplicate_request_keys += 1
        request_keys.add(key)
        user, _, query, time_index = key
        users.add(user)
        query_counts[_normalize_query(query)] += 1
        time_values.append(time_index)
        split_counts[source_split] += 1

        candidates = [int(value) for value in row.get("impressed_item_ids", [])]
        clicked = {int(value) for value in row.get("clicked_item_ids", [])}
        purchased = {int(value) for value in row.get("purchased_item_ids", [])}
        candidate_set = set(candidates)
        candidate_counts.append(len(candidates))
        clicked_counts.append(len(clicked))
        purchased_counts.append(len(purchased))
        requests_with_click += int(bool(clicked))
        requests_with_purchase += int(bool(purchased))
        duplicate_candidate_requests += int(len(candidate_set) != len(candidates))
        click_outside_slate += len(clicked - candidate_set)
        purchase_outside_slate += len(purchased - candidate_set)
        for item_id in clicked | purchased:
            events[(user, item_id)].append(time_index)
            previous = user_first_event.get(user)
            if previous is None or time_index < previous:
                user_first_event[user] = time_index
        if max_rows is not None and source_rows_seen >= max_rows:
            break

    for event_times in events.values():
        event_times.sort()
    return {
        "source_rows_seen": source_rows_seen,
        "source_split_counts_seen": source_split_counts_seen,
        "rows": rows,
        "first_row_keys": first_row_keys,
        "missing_required_fields": missing_required_fields,
        "duplicate_request_keys": duplicate_request_keys,
        "users": users,
        "query_counts": query_counts,
        "candidate_counts": candidate_counts,
        "clicked_counts": clicked_counts,
        "purchased_counts": purchased_counts,
        "requests_with_click": requests_with_click,
        "requests_with_purchase": requests_with_purchase,
        "duplicate_candidate_requests": duplicate_candidate_requests,
        "click_outside_slate": click_outside_slate,
        "purchase_outside_slate": purchase_outside_slate,
        "split_counts": split_counts,
        "time_values": time_values,
        "events": dict(events),
        "user_first_event": user_first_event,
    }


def _scan_recall_second_pass(
    path: Path,
    *,
    events: dict[tuple[str, int], list[int]],
    user_first_event: dict[str, int],
    selected_queries: set[str],
    collision_requests_per_query: int,
    rank_history_sample_size: int,
    seed: int,
    included_source_splits: set[str],
    max_rows: int | None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    source_rows_seen = 0
    rows = 0
    history_present_requests = 0
    strict_nonrepeat_requests = 0
    requests_with_recurrent_candidate = 0
    recurrent_candidate_counts: list[int] = []
    collision_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    collision_seen: Counter[str] = Counter()
    rank_sample_records: dict[RequestKey, dict[str, Any]] = {}
    rank_sample_seen = 0

    for row in iter_jsonl(path):
        source_rows_seen += 1
        if str(row.get("split", "")) not in included_source_splits:
            if max_rows is not None and source_rows_seen >= max_rows:
                break
            continue
        rows += 1
        key = _request_key(row)
        user, _, query, time_index = key
        normalized_query = _normalize_query(query)
        candidates = {int(value) for value in row.get("impressed_item_ids", [])}
        has_prior = user_first_event.get(user, time_index) < time_index
        recurrent = {
            item_id
            for item_id in candidates
            if _has_event_before(events.get((user, item_id), []), time_index)
        }
        if has_prior:
            history_present_requests += 1
            if not recurrent:
                strict_nonrepeat_requests += 1
            rank_sample_seen += 1
            sample_record = {
                "candidate_ids": candidates,
                "request_time": time_index,
                "user_id": user,
            }
            if len(rank_sample_records) < rank_history_sample_size:
                rank_sample_records[key] = sample_record
            else:
                replacement = rng.randrange(rank_sample_seen)
                if replacement < rank_history_sample_size:
                    replace_key = next(
                        candidate_key
                        for index, candidate_key in enumerate(rank_sample_records)
                        if index == replacement
                    )
                    del rank_sample_records[replace_key]
                    rank_sample_records[key] = sample_record
        if recurrent:
            requests_with_recurrent_candidate += 1
        recurrent_candidate_counts.append(len(recurrent))

        if has_prior and normalized_query in selected_queries:
            strict_candidates = candidates - recurrent
            record = {
                "request_key": key,
                "user_id": user,
                "strict_candidate_ids": strict_candidates,
            }
            collision_seen[normalized_query] += 1
            target = collision_records[normalized_query]
            if len(target) < collision_requests_per_query:
                target.append(record)
            else:
                replacement = rng.randrange(collision_seen[normalized_query])
                if replacement < collision_requests_per_query:
                    target[replacement] = record
        if max_rows is not None and source_rows_seen >= max_rows:
            break

    return {
        "history_present_requests": history_present_requests,
        "strict_nonrepeat_requests": strict_nonrepeat_requests,
        "requests_with_recurrent_candidate": requests_with_recurrent_candidate,
        "recurrent_candidate_counts": recurrent_candidate_counts,
        "collision_records": dict(collision_records),
        "rank_sample_records": rank_sample_records,
    }


def _collision_opportunity(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    eligible_pairs = 0
    eligible_queries: set[str] = set()
    eligible_requests: set[RequestKey] = set()
    eligible_users: set[str] = set()
    shared_candidate_counts: list[int] = []
    query_pair_counts: list[int] = []
    same_user_pairs_skipped = 0

    for query, rows in groups.items():
        query_eligible_pairs = 0
        overlap: Counter[tuple[int, int]] = Counter()
        inverted: defaultdict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            for item_id in row["strict_candidate_ids"]:
                inverted[item_id].append(index)
        for indices in inverted.values():
            for left, right in combinations(indices, 2):
                overlap[(left, right)] += 1
        for (left_index, right_index), count in overlap.items():
            if count < 2:
                continue
            left = rows[left_index]
            right = rows[right_index]
            if left["user_id"] == right["user_id"]:
                same_user_pairs_skipped += 1
                continue
            eligible_pairs += 1
            query_eligible_pairs += 1
            eligible_queries.add(query)
            eligible_requests.update((left["request_key"], right["request_key"]))
            eligible_users.update((left["user_id"], right["user_id"]))
            shared_candidate_counts.append(count)
        if query_eligible_pairs:
            query_pair_counts.append(query_eligible_pairs)
    ordered_query_pair_counts = sorted(query_pair_counts, reverse=True)
    return {
        "cross_user_pairs_with_at_least_two_shared_strict_candidates": eligible_pairs,
        "eligible_query_groups": len(eligible_queries),
        "eligible_requests": len(eligible_requests),
        "eligible_users": len(eligible_users),
        "shared_strict_candidate_count": _summary(shared_candidate_counts),
        "eligible_pairs_per_query": _summary(query_pair_counts),
        "pair_concentration": {
            "largest_query_share": _rate(
                ordered_query_pair_counts[0] if ordered_query_pair_counts else 0,
                eligible_pairs,
            ),
            "top_10_query_share": _rate(
                sum(ordered_query_pair_counts[:10]), eligible_pairs
            ),
            "warning": (
                "Request-pair count is not an effective sample size; inference "
                "must cluster by query and account for reused users/requests."
            ),
        },
        "same_user_pairs_skipped": same_user_pairs_skipped,
    }


def _scan_rank_histories(
    path: Path,
    *,
    sample_records: dict[RequestKey, dict[str, Any]],
    events: dict[tuple[str, int], list[int]],
    max_rows: int | None,
) -> dict[str, Any]:
    sampled_keys = set(sample_records)
    sampled_sessions = {key[1] for key in sampled_keys}
    found: dict[RequestKey, list[int]] = {}
    rows_scanned = 0
    first_row_keys: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows_scanned += 1
            match = SESSION_ID_RE.search(line)
            if match and match.group(1) in sampled_sessions:
                row = json.loads(line)
                key = _request_key(row)
                if key in sampled_keys and key not in found:
                    if not first_row_keys:
                        first_row_keys = sorted(row)
                    history: list[int] = []
                    for field in RANK_HISTORY_FIELDS:
                        history.extend(
                            int(value)
                            for value in (row.get(field, []) or [])
                            if int(value) != 0
                        )
                    found[key] = history
                    if len(found) == len(sampled_keys):
                        break
            if max_rows is not None and rows_scanned >= max_rows:
                break

    classifications: Counter[str] = Counter()
    total_history_items = 0
    requests_with_candidate_overlap = 0
    raw_history_lengths: list[int] = []
    unique_history_lengths: list[int] = []
    for key, history in found.items():
        user, _, _, request_time = key
        total_history_items += len(history)
        raw_history_lengths.append(len(history))
        unique_history = set(history)
        unique_history_lengths.append(len(unique_history))
        candidates = sample_records[key]["candidate_ids"]
        requests_with_candidate_overlap += int(bool(unique_history & candidates))
        for item_id in history:
            classifications[
                _classify_event_times(events.get((user, item_id), []), request_time)
            ] += 1
    observed = (
        classifications["past_supported"]
        + classifications["same_time_only"]
        + classifications["future_only"]
    )
    timestamp_fields = sorted(
        key
        for key in first_row_keys
        if "recently" in key and ("time" in key or "timestamp" in key or key.endswith("ts"))
    )
    return {
        "requested_requests": len(sampled_keys),
        "found_requests": len(found),
        "missing_requests": len(sampled_keys) - len(found),
        "rank_rows_scanned": rows_scanned,
        "complete_rank_scan": max_rows is None or len(found) == len(sampled_keys),
        "history_fields": list(RANK_HISTORY_FIELDS),
        "per_history_timestamp_fields": timestamp_fields,
        "raw_history_length": _summary(raw_history_lengths),
        "unique_history_length": _summary(unique_history_lengths),
        "requests_with_raw_history_candidate_overlap": requests_with_candidate_overlap,
        "classification": {
            "total_history_items": total_history_items,
            "past_supported": classifications["past_supported"],
            "same_time_only": classifications["same_time_only"],
            "future_only": classifications["future_only"],
            "unobserved": classifications["unobserved"],
            "observed_history_items": observed,
            "past_supported_over_total": _rate(
                classifications["past_supported"], total_history_items
            ),
            "future_only_over_observed": _rate(
                classifications["future_only"], observed
            ),
            "unobserved_over_total": _rate(
                classifications["unobserved"], total_history_items
            ),
        },
        "caveat": (
            "The rank file provides no per-event history timestamps. These "
            "classifications only cross-reference events visible in recall."
        ),
    }


def _direct_observations(
    *,
    rows: int,
    repeated_query_requests: int,
    history_present_requests: int,
    strict_nonrepeat_requests: int,
    collision: dict[str, Any],
    rank_history: dict[str, Any],
) -> list[str]:
    return [
        f"Scanned {rows} recall requests; {repeated_query_requests} belong to repeated exact-query groups.",
        f"Observed-log prior history exists for {history_present_requests} requests; {strict_nonrepeat_requests} of those have no recurrent candidate.",
        (
            "The label-free exact-query scout found "
            f"{collision['cross_user_pairs_with_at_least_two_shared_strict_candidates']} "
            "cross-user request pairs with at least two shared strict candidates."
        ),
        (
            "Raw recent-history lookup found "
            f"{rank_history['found_requests']}/{rank_history['requested_requests']} "
            "sampled requests; timestamp validity remains only partially observable."
        ),
    ]


def _request_key(row: dict[str, Any]) -> RequestKey:
    return (
        str(row["user_id"]),
        str(row["session_id"]),
        str(row["query"]),
        int(row["time_index"]),
    )


def _normalize_query(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _has_event_before(times: list[int], request_time: int) -> bool:
    return bool(times) and times[0] < request_time


def _classify_event_times(times: list[int], request_time: int) -> str:
    if not times:
        return "unobserved"
    if times[0] < request_time:
        return "past_supported"
    if times[0] == request_time:
        return "same_time_only"
    return "future_only"


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "mean": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def _percentile(values: list[int], quantile: float) -> float | int:
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _file_info(path: Path, *, hash_limit_bytes: int) -> dict[str, Any]:
    size = path.stat().st_size
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": sha256_file(path) if size <= hash_limit_bytes else None,
        "sha256_status": "computed" if size <= hash_limit_bytes else "skipped_large_file",
    }
