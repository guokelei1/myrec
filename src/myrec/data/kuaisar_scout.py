"""Build a label-isolated KuaiSAR Small natural-search scout."""

from __future__ import annotations

import ast
import bisect
import csv
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from myrec.data.contracts import audit_standardized_file, validate_standardized_record
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import write_json


@dataclass(frozen=True)
class SearchCandidate:
    item_id: str
    clicked: bool
    item_type: str
    source_position: int


@dataclass(frozen=True)
class SearchRequest:
    user_id: str
    session_id: str
    timestamp: int
    query_tokens: tuple[str, ...]
    search_source: str
    candidates: tuple[SearchCandidate, ...]
    raw_candidate_rows: int
    duplicate_candidate_rows: int
    conflicting_click_rows: int
    conflicting_item_type_rows: int


HistoryEvent = tuple[int, str, str]


def build_kuaisar_small_scout(
    raw_dir: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    dataset_version: str = "small_user_input_scout10k_v1",
    max_requests: int = 10_000,
    dev_fraction: float = 0.20,
    min_candidate_count: int = 2,
    max_candidate_count: int = 100,
    max_history_len: int = 20,
    search_sources: tuple[str, ...] = ("USER_INPUT",),
    dataset_id: str = "kuaisar_small",
    release_name: str = "KuaiSAR Small",
    archive_md5_expected: str = "daea8cbf605db6bd5841740f0e4a12d9",
    archive_path: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize a latest-window exploratory scout from a KuaiSAR release."""

    if max_requests < 10:
        raise ValueError("max_requests must be at least 10")
    if not 0.0 < dev_fraction < 0.5:
        raise ValueError("dev_fraction must be in (0, 0.5)")
    if min_candidate_count < 2 or max_candidate_count < min_candidate_count:
        raise ValueError("invalid candidate-count boundary")
    if max_history_len < 1:
        raise ValueError("max_history_len must be positive")
    if not search_sources:
        raise ValueError("search_sources must be non-empty")

    raw_dir = Path(raw_dir)
    source_dir = _resolve_source_dir(raw_dir)
    if archive_path is None:
        archive_name = "KuaiSAR_v2.zip" if dataset_id == "kuaisar_full" else "KuaiSAR.zip"
        archive = raw_dir / archive_name
    else:
        archive = Path(archive_path)
    archive_md5_actual = _md5_file(archive) if archive.is_file() else None
    src_path = source_dir / "src_inter.csv"
    rec_path = source_dir / "rec_inter.csv"
    item_path = source_dir / "item_features.csv"
    for path in (src_path, rec_path, item_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing KuaiSAR source file: {path}")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"scout output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected, source_audit = _select_latest_sessions(
        src_path,
        max_requests=max_requests,
        min_candidate_count=min_candidate_count,
        max_candidate_count=max_candidate_count,
        search_sources=frozenset(search_sources),
    )
    split_by_request, split_info = _time_split(selected, dev_fraction=dev_fraction)
    selected_users = {request.user_id for request in selected}
    max_target_time = max(request.timestamp for request in selected)
    events_by_user, history_source_audit = _collect_positive_history(
        src_path,
        rec_path,
        selected_users=selected_users,
        max_target_time=max_target_time,
    )
    history_by_request = _causal_histories(
        selected,
        events_by_user,
        max_history_len=max_history_len,
    )
    needed_item_ids = {
        candidate.item_id
        for request in selected
        for candidate in request.candidates
    }
    needed_item_ids.update(
        item_id for history in history_by_request.values() for _, item_id, _ in history
    )
    item_map = _load_item_map(item_path, needed_item_ids)
    outputs = _write_scout(
        output_dir,
        selected,
        split_by_request=split_by_request,
        history_by_request=history_by_request,
        item_map=item_map,
        dataset_version=dataset_version,
    )

    manifest = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "evidence_mode": "exploratory_independent_source_replication",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_warning": (
            "KuaiSAR uses anonymized query/caption word IDs. This scout can test "
            "functional query-conditioned history ranking, not pretrained natural-"
            "language semantic transfer or independent confirmation."
        ),
        "source": {
            "raw_dir": str(source_dir),
            "release": release_name,
            "archive_md5_expected": archive_md5_expected,
            "archive_path": str(archive),
            "archive_md5_actual": archive_md5_actual,
            "src_inter": _file_info(src_path),
            "rec_inter": _file_info(rec_path),
            "item_features": _file_info(item_path),
        },
        "selection": {
            "strategy": (
                "latest eligible user-input search sessions; eligibility uses only "
                "source schema, candidate count, and presence of both clicked and "
                "unclicked exposed candidates"
            ),
            "search_sources": list(search_sources),
            "max_requests": max_requests,
            "min_candidate_count": min_candidate_count,
            "max_candidate_count": max_candidate_count,
            "max_history_len": max_history_len,
            **source_audit,
            **split_info,
        },
        "history": {
            "sources": ["earlier positive search clicks", "earlier positive recommendation clicks"],
            "causal_rule": "event_timestamp < target_search_session_timestamp",
            **history_source_audit,
        },
        "item_join": {
            "needed_item_ids": len(needed_item_ids),
            "loaded_item_ids": len(item_map),
            "missing_item_ids": len(needed_item_ids - set(item_map)),
            "coverage": len(item_map) / len(needed_item_ids) if needed_item_ids else 0.0,
            "missing_item_examples": sorted(needed_item_ids - set(item_map))[:20],
        },
        "outputs": outputs,
        "admission_checks": {
            "archive_md5_matches": archive_md5_actual == archive_md5_expected,
            "reconstructable_mixed_feedback_slates": source_audit["eligible_sessions"] > 0,
            "strict_causal_history": outputs["history_not_strictly_before_target_violations"] == 0,
            "candidate_identity_unique": outputs["duplicate_candidate_id_violations"] == 0,
            "dev_records_label_free": True,
            "candidate_text_coverage_at_least_95pct": outputs["candidate_text_coverage"] >= 0.95,
            "history_present_dev_requests_at_least_500": outputs["counts"].get("dev_history_present", 0) >= 500,
            "strict_nonrepeat_dev_requests_at_least_500": outputs["counts"].get("dev_strict_nonrepeat", 0) >= 500,
        },
    }
    manifest["admission_passed"] = all(manifest["admission_checks"].values())
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(report_path, manifest)
    return manifest


def _resolve_source_dir(raw_dir: Path) -> Path:
    for dirname in ("KuaiSAR_v2", "KuaiSAR_final"):
        nested = raw_dir / dirname
        if nested.is_dir():
            return nested
    return raw_dir


def _select_latest_sessions(
    src_path: Path,
    *,
    max_requests: int,
    min_candidate_count: int,
    max_candidate_count: int,
    search_sources: frozenset[str],
) -> tuple[list[SearchRequest], dict[str, Any]]:
    heap: list[tuple[int, str, SearchRequest]] = []
    counts: Counter[str] = Counter()
    seen_source_identities: set[tuple[str, str]] = set()
    candidate_counts: list[int] = []
    for request in _iter_search_sessions(src_path):
        counts["source_sessions"] += 1
        source_identity = (request.user_id, request.session_id)
        if source_identity in seen_source_identities:
            counts["source_session_identity_reuse_variants"] += 1
        else:
            seen_source_identities.add(source_identity)
        counts["source_candidate_rows"] += request.raw_candidate_rows
        counts["source_duplicate_candidate_rows"] += request.duplicate_candidate_rows
        counts["source_conflicting_click_rows"] += request.conflicting_click_rows
        counts["source_conflicting_item_type_rows"] += request.conflicting_item_type_rows
        counts[f"source_{request.search_source}"] += 1
        if request.search_source not in search_sources:
            continue
        counts["selected_source_sessions"] += 1
        candidate_count = len(request.candidates)
        if not min_candidate_count <= candidate_count <= max_candidate_count:
            counts["excluded_candidate_count"] += 1
            continue
        item_ids = [candidate.item_id for candidate in request.candidates]
        if len(set(item_ids)) != len(item_ids):
            counts["excluded_duplicate_candidates"] += 1
            continue
        positives = sum(candidate.clicked for candidate in request.candidates)
        if positives == 0:
            counts["excluded_no_click"] += 1
            continue
        if positives == candidate_count:
            counts["excluded_no_negative"] += 1
            continue
        counts["eligible_sessions"] += 1
        candidate_counts.append(candidate_count)
        request_id = _request_id(request)
        entry = (request.timestamp, request_id, request)
        if len(heap) < max_requests:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    if len(heap) < 10:
        raise ValueError("not enough eligible KuaiSAR search sessions")
    selected = [entry[2] for entry in sorted(heap)]
    return selected, {
        **dict(counts),
        "selected_requests": len(selected),
        "eligible_candidate_count": _summary(candidate_counts),
        "selected_time_min": min(request.timestamp for request in selected),
        "selected_time_max": max(request.timestamp for request in selected),
    }


def _time_split(
    requests: list[SearchRequest], *, dev_fraction: float
) -> tuple[dict[str, str], dict[str, Any]]:
    ordered = sorted(requests, key=lambda request: (request.timestamp, _request_id(request)))
    split_index = max(1, int(len(ordered) * (1.0 - dev_fraction)))
    boundary = ordered[split_index].timestamp
    while split_index > 0 and ordered[split_index - 1].timestamp == boundary:
        split_index -= 1
    if split_index == 0:
        raise ValueError("timestamp-tie containment produced empty train split")
    result = {
        _request_id(request): "train" if index < split_index else "dev"
        for index, request in enumerate(ordered)
    }
    return result, {
        "train_requests": split_index,
        "dev_requests": len(ordered) - split_index,
        "dev_time_min": boundary,
        "timestamp_ties_kept_with_dev": max(0, int(len(ordered) * (1.0 - dev_fraction)) - split_index),
        "session_overlap": 0,
    }


def _collect_positive_history(
    src_path: Path,
    rec_path: Path,
    *,
    selected_users: set[str],
    max_target_time: int,
) -> tuple[dict[str, list[HistoryEvent]], dict[str, Any]]:
    events: defaultdict[str, list[HistoryEvent]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for request in _iter_search_sessions(src_path):
        if request.user_id not in selected_users or request.timestamp >= max_target_time:
            continue
        for candidate in request.candidates:
            if candidate.clicked:
                events[request.user_id].append(
                    (request.timestamp, candidate.item_id, "search_click")
                )
                counts["search_click_events"] += 1
    with rec_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = str(row["user_id"])
            if user_id not in selected_users or int(float(row["click"])) <= 0:
                continue
            timestamp = int(float(row["timestamp"]))
            if timestamp >= max_target_time:
                continue
            events[user_id].append((timestamp, str(row["item_id"]), "rec_click"))
            counts["recommendation_click_events"] += 1
    duplicates_removed = 0
    for user_id, user_events in events.items():
        ordered = sorted(set(user_events), key=lambda event: (event[0], event[1], event[2]))
        duplicates_removed += len(user_events) - len(ordered)
        events[user_id] = ordered
    return dict(events), {
        **dict(counts),
        "selected_users": len(selected_users),
        "selected_users_with_positive_history_source": len(events),
        "exact_duplicate_events_removed": duplicates_removed,
    }


def _causal_histories(
    requests: list[SearchRequest],
    events_by_user: dict[str, list[HistoryEvent]],
    *,
    max_history_len: int,
) -> dict[str, tuple[HistoryEvent, ...]]:
    times_by_user = {
        user_id: [event[0] for event in events]
        for user_id, events in events_by_user.items()
    }
    result = {}
    for request in requests:
        events = events_by_user.get(request.user_id, [])
        stop = bisect.bisect_left(times_by_user.get(request.user_id, []), request.timestamp)
        result[_request_id(request)] = tuple(events[max(0, stop - max_history_len) : stop])
    return result


def _load_item_map(
    item_path: Path, needed_item_ids: set[str], *, max_caption_tokens: int = 32
) -> dict[str, dict[str, Any]]:
    # Full contains a few very long quoted feature fields.  They are not used
    # as free text, but the CSV parser must cross them to reach later rows.
    csv.field_size_limit(256 * 1024 * 1024)
    result: dict[str, dict[str, Any]] = {}
    with item_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = str(row["item_id"])
            if item_id not in needed_item_ids:
                continue
            caption = _parse_token_sequence(row.get("caption", ""))[:max_caption_tokens]
            categories = [
                str(row.get(field, "")).strip()
                for field in (
                    "first_level_category_name_en",
                    "second_level_category_name_en",
                    "third_level_category_name_en",
                    "fourth_level_category_name_en",
                )
                if str(row.get(field, "")).strip().casefold() not in {"", "empty", "unknown", "nan"}
            ]
            token_text = " ".join(f"w{token}" for token in caption)
            category_text = " > ".join(categories)
            title = "; ".join(
                value for value in (token_text, f"category {category_text}" if category_text else "") if value
            )
            result[item_id] = {
                "item_id": item_id,
                "title": title,
                "brand": "",
                "cat": categories,
            }
            if len(result) == len(needed_item_ids):
                break
    return result


def _write_scout(
    output_dir: Path,
    requests: list[SearchRequest],
    *,
    split_by_request: dict[str, str],
    history_by_request: dict[str, tuple[HistoryEvent, ...]],
    item_map: dict[str, dict[str, Any]],
    dataset_version: str,
) -> dict[str, Any]:
    record_paths = {split: output_dir / f"records_{split}.jsonl" for split in ("train", "dev")}
    qrels_paths = {split: output_dir / f"qrels_{split}.jsonl" for split in ("train", "dev")}
    record_handles = {split: path.open("w", encoding="utf-8") for split, path in record_paths.items()}
    qrels_handles = {split: path.open("w", encoding="utf-8") for split, path in qrels_paths.items()}
    candidate_entries: list[dict[str, Any]] = []
    request_entries: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    candidate_text_hits = candidate_total = 0
    history_text_hits = history_total = 0
    candidate_counts: list[int] = []
    history_lengths: list[int] = []
    duplicate_candidate_violations = 0
    future_history_violations = 0
    try:
        for request in sorted(requests, key=lambda row: (row.timestamp, _request_id(row))):
            request_id = _request_id(request)
            split = split_by_request[request_id]
            candidates = []
            for candidate in request.candidates:
                payload = dict(item_map.get(candidate.item_id, _missing_item(candidate.item_id)))
                payload.update(
                    {
                        "item_type": candidate.item_type,
                        "source_position": candidate.source_position,
                    }
                )
                candidate_text_hits += int(bool(payload["title"].strip()))
                candidate_total += 1
                if split == "train":
                    payload.update(
                        {
                            "clicked": int(candidate.clicked),
                            "purchased": 0,
                            "relevance": int(candidate.clicked),
                        }
                    )
                candidates.append(payload)
            history = []
            for event_time, item_id, event_type in history_by_request[request_id]:
                payload = dict(item_map.get(item_id, _missing_item(item_id)))
                payload.update({"event": event_type, "ts": event_time})
                history_text_hits += int(bool(payload["title"].strip()))
                history_total += 1
                future_history_violations += int(event_time >= request.timestamp)
                history.append(payload)
            candidate_ids = [candidate.item_id for candidate in request.candidates]
            duplicate_candidate_violations += int(len(candidate_ids) != len(set(candidate_ids)))
            history_ids = {event[1] for event in history_by_request[request_id]}
            strict_nonrepeat = bool(history) and history_ids.isdisjoint(candidate_ids)
            query = " ".join(f"w{token}" for token in request.query_tokens)
            record = {
                "request_id": request_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "ts": request.timestamp,
                "query": query,
                "history": history,
                "candidates": candidates,
                "masks": {
                    "history_present": bool(history),
                    "strict_nonrepeat": strict_nonrepeat,
                    "text_coverage": sum(bool(candidate["title"].strip()) for candidate in candidates) / len(candidates),
                    "history_text_coverage": (
                        sum(bool(event["title"].strip()) for event in history) / len(history)
                        if history
                        else 1.0
                    ),
                },
            }
            validate_standardized_record(record, split)
            record_handles[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            clicked = sorted(candidate.item_id for candidate in request.candidates if candidate.clicked)
            qrels_handles[split].write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "clicked": clicked,
                        "purchased": [],
                        "relevance": {item_id: 1 for item_id in clicked},
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            candidate_entries.append(
                {"split": split, "request_id": request_id, "candidate_item_ids": candidate_ids}
            )
            request_entries.append(
                {
                    "split": split,
                    "request_id": request_id,
                    "query_sha256": sha256_text(query),
                    "candidate_item_ids_sha256": sha256_text(json.dumps(candidate_ids, separators=(",", ":"))),
                }
            )
            counts[f"{split}_requests"] += 1
            counts[f"{split}_history_present"] += int(bool(history))
            counts[f"{split}_strict_nonrepeat"] += int(strict_nonrepeat)
            counts[f"{split}_repeat"] += int(bool(history) and not strict_nonrepeat)
            candidate_counts.append(len(candidates))
            history_lengths.append(len(history))
    finally:
        for handle in (*record_handles.values(), *qrels_handles.values()):
            handle.close()

    candidate_manifest_path = output_dir / "candidate_manifest.json"
    request_manifest_path = output_dir / "request_manifest.json"
    write_json(candidate_manifest_path, {"dataset_version": dataset_version, "entries": candidate_entries})
    write_json(request_manifest_path, {"dataset_version": dataset_version, "entries": request_entries})
    audits = {split: audit_standardized_file(record_paths[split], split) for split in ("train", "dev")}
    return {
        "counts": dict(counts),
        "candidate_count": _summary(candidate_counts),
        "history_length": _summary(history_lengths),
        "candidate_text_coverage": candidate_text_hits / candidate_total if candidate_total else 0.0,
        "history_text_coverage": history_text_hits / history_total if history_total else 1.0,
        "duplicate_candidate_id_violations": duplicate_candidate_violations,
        "history_not_strictly_before_target_violations": future_history_violations,
        "structural_audits": audits,
        "files": {
            **{f"records_{split}": _file_info(record_paths[split]) for split in ("train", "dev")},
            **{f"qrels_{split}": _file_info(qrels_paths[split]) for split in ("train", "dev")},
            "candidate_manifest": _file_info(candidate_manifest_path),
            "request_manifest": _file_info(request_manifest_path),
        },
        "label_isolation": {
            "dev_records_label_free": True,
            "dev_labels_path": str(qrels_paths["dev"]),
            "scoring_code_may_read_dev_labels": False,
        },
    }


def _iter_search_sessions(path: Path) -> Iterator[SearchRequest]:
    """Yield source sessions after deterministic global consolidation.

    The official release contains a small number of session rows that recur
    after intervening sessions.  Contiguity is therefore not a source
    invariant.  Grouping by the complete session identity prevents an input
    ordering artifact from rejecting the dataset.  Repeated item rows are
    collapsed in first-observed order and their click flag is combined with
    logical OR; the raw/duplicate/conflict counts remain attached to the
    request for admission auditing.
    """

    SessionKey = tuple[str, str, int, tuple[str, ...], str]
    sessions: dict[SessionKey, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row["user_id"]),
                str(row["search_session_id"]),
                int(float(row["search_session_timestamp"])),
                _parse_token_sequence(row["keyword"]),
                str(row.get("search_source") or row.get("search_session_source") or ""),
            )
            if not key[3]:
                raise ValueError(f"empty KuaiSAR keyword for session={key[1]}")
            if not key[4]:
                raise ValueError(f"empty KuaiSAR search source for session={key[1]}")
            state = sessions.setdefault(
                key,
                {
                    "candidates": {},
                    "raw_candidate_rows": 0,
                    "duplicate_candidate_rows": 0,
                    "conflicting_click_rows": 0,
                    "conflicting_item_type_rows": 0,
                },
            )
            state["raw_candidate_rows"] += 1
            item_id = str(row["item_id"])
            clicked = int(float(row["click_cnt"])) > 0
            item_type = str(row["item_type"])
            existing = state["candidates"].get(item_id)
            if existing is None:
                state["candidates"][item_id] = SearchCandidate(
                    item_id=item_id,
                    clicked=clicked,
                    item_type=item_type,
                    source_position=len(state["candidates"]),
                )
                continue
            state["duplicate_candidate_rows"] += 1
            if existing.clicked != clicked:
                state["conflicting_click_rows"] += 1
            if existing.item_type != item_type:
                state["conflicting_item_type_rows"] += 1
            if clicked and not existing.clicked:
                state["candidates"][item_id] = SearchCandidate(
                    item_id=item_id,
                    clicked=True,
                    item_type=item_type,
                    source_position=existing.source_position,
                )
    for key, state in sorted(
        sessions.items(), key=lambda pair: (pair[0][2], pair[0][0], pair[0][1])
    ):
        yield _make_request(
            key,
            list(state["candidates"].values()),
            raw_candidate_rows=int(state["raw_candidate_rows"]),
            duplicate_candidate_rows=int(state["duplicate_candidate_rows"]),
            conflicting_click_rows=int(state["conflicting_click_rows"]),
            conflicting_item_type_rows=int(state["conflicting_item_type_rows"]),
        )


def _make_request(
    key: tuple[str, str, int, tuple[str, ...], str],
    candidates: list[SearchCandidate],
    *,
    raw_candidate_rows: int,
    duplicate_candidate_rows: int,
    conflicting_click_rows: int,
    conflicting_item_type_rows: int,
) -> SearchRequest:
    return SearchRequest(
        user_id=key[0],
        session_id=key[1],
        timestamp=key[2],
        query_tokens=key[3],
        search_source=key[4],
        candidates=tuple(candidates),
        raw_candidate_rows=raw_candidate_rows,
        duplicate_candidate_rows=duplicate_candidate_rows,
        conflicting_click_rows=conflicting_click_rows,
        conflicting_item_type_rows=conflicting_item_type_rows,
    )


def _parse_token_sequence(value: str) -> tuple[str, ...]:
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"expected token sequence, got {type(parsed).__name__}")
    return tuple(str(int(token)) for token in parsed)


def _request_id(request: SearchRequest) -> str:
    payload = json.dumps(
        [
            request.user_id,
            request.session_id,
            request.timestamp,
            request.query_tokens,
            request.search_source,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "ksar_" + sha256_text(payload)[:24]


def _missing_item(item_id: str) -> dict[str, Any]:
    return {"item_id": item_id, "title": "", "brand": "", "cat": []}


def _file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {"count": len(values), "min": min(values), "mean": sum(values) / len(values), "max": max(values)}
