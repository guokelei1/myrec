"""Build a label-isolated KuaiSearch Lite scout from source-train rows."""

from __future__ import annotations

import bisect
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.data.contracts import audit_standardized_file, validate_standardized_record
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json

RequestKey = tuple[str, str, str, int]
Event = tuple[int, int, str, str]


@dataclass(frozen=True)
class SourceRequest:
    key: RequestKey
    candidate_item_ids: tuple[int, ...]
    clicked_item_ids: frozenset[int]
    purchased_item_ids: frozenset[int]
    split: str
    history: tuple[Event, ...]


def build_kuaisearch_lite_scout(
    raw_dir: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    dataset_version: str = "lite_scout10k_v1",
    max_requests: int = 10_000,
    dev_fraction: float = 0.20,
    min_candidate_count: int = 2,
    max_candidate_count: int = 100,
    max_history_len: int = 20,
    include_history_query: bool = False,
) -> dict[str, Any]:
    """Materialize an exploratory time-window scout without source-test rows."""

    if max_requests < 10:
        raise ValueError("max_requests must be at least 10")
    if not 0.0 < dev_fraction < 0.5:
        raise ValueError("dev_fraction must be in (0, 0.5)")
    if min_candidate_count < 2 or max_candidate_count < min_candidate_count:
        raise ValueError("invalid candidate-count boundary")
    if max_history_len < 1:
        raise ValueError("max_history_len must be positive")

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    recall_path, source_variant = _resolve_source_path(raw_dir, "recall")
    items_path, items_variant = _resolve_source_path(raw_dir, "items")
    if items_variant != source_variant:
        raise ValueError("KuaiSearch recall and item source variants differ")
    for path in (recall_path, items_path):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing KuaiSearch source file: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"scout output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = _collect_source_state(
        recall_path,
        min_candidate_count=min_candidate_count,
        max_candidate_count=max_candidate_count,
    )
    selected_keys, split_by_key, split_info = _select_latest_time_window(
        source["eligible_keys"],
        max_requests=max_requests,
        dev_fraction=dev_fraction,
    )
    selected = _load_selected_requests(
        recall_path,
        selected_keys=selected_keys,
        split_by_key=split_by_key,
        events_by_user=source["events_by_user"],
        max_history_len=max_history_len,
    )
    needed_item_ids = {
        item_id
        for request in selected
        for item_id in (
            *request.candidate_item_ids,
            *(event[1] for event in request.history),
        )
    }
    item_map = _load_item_map(items_path, needed_item_ids)
    write_result = _write_scout(
        output_dir,
        selected,
        item_map=item_map,
        dataset_version=dataset_version,
        include_history_query=include_history_query,
    )

    manifest = {
        "dataset_id": "kuaisearch",
        "dataset_version": dataset_version,
        "evidence_mode": "exploratory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_warning": (
            "This is a source-train time-window scout for implementation and "
            "motivation exploration. It is not a frozen confirmation cohort "
            "and does not represent the entire source population."
        ),
        "source": {
            "raw_dir": str(raw_dir),
            "source_variant": source_variant,
            "recall_path": str(recall_path),
            "recall_sha256": sha256_file(recall_path),
            "items_path": str(items_path),
            "items_size_bytes": items_path.stat().st_size,
            "included_source_splits": ["train"],
            "excluded_source_split_counts": dict(source["excluded_split_counts"]),
            "evaluation_source_behavior_fields_accessed": False,
        },
        "selection": {
            "strategy": "latest source-train time window after label-free candidate-size filter",
            "max_requests": max_requests,
            "min_candidate_count": min_candidate_count,
            "max_candidate_count": max_candidate_count,
            "max_history_len": max_history_len,
            "eligible_source_requests": len(source["eligible_keys"]),
            **split_info,
        },
        "history": {
            "source": "same-user recall click/purchase events",
            "causal_rule": "event_time < request_time",
            "prior_query_included": include_history_query,
            "outside_log_events": "unobserved",
            "raw_rank_recent_fields_used": False,
        },
        "item_join": {
            "needed_item_ids": len(needed_item_ids),
            "loaded_item_ids": len(item_map),
            "missing_item_ids": len(needed_item_ids - set(item_map)),
            "missing_item_examples": [
                str(value) for value in sorted(needed_item_ids - set(item_map))[:20]
            ],
        },
        "outputs": write_result,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["outputs"]["manifest"] = {
        "path": str(manifest_path),
        "sha256_status": "self_reference_not_recorded",
    }
    write_json(manifest_path, manifest)
    write_json(report_path, manifest)
    return manifest


def _resolve_source_path(raw_dir: Path, stem: str) -> tuple[Path, str]:
    """Resolve the public Lite or Full directory without mixing variants."""

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


def _collect_source_state(
    path: Path,
    *,
    min_candidate_count: int,
    max_candidate_count: int,
) -> dict[str, Any]:
    eligible_keys: list[RequestKey] = []
    events_by_user: defaultdict[str, list[Event]] = defaultdict(list)
    excluded_split_counts: Counter[str] = Counter()
    for row in iter_jsonl(path):
        source_split = str(row.get("split", ""))
        if source_split != "train":
            excluded_split_counts[source_split] += 1
            continue
        key = _request_key(row)
        candidates = row.get("impressed_item_ids", [])
        if min_candidate_count <= len(candidates) <= max_candidate_count:
            eligible_keys.append(key)
        user_id = key[0]
        time_index = key[3]
        clicked = {int(value) for value in row.get("clicked_item_ids", [])}
        purchased = {int(value) for value in row.get("purchased_item_ids", [])}
        for item_id in clicked | purchased:
            event = "purchase" if item_id in purchased else "click"
            events_by_user[user_id].append((time_index, item_id, event, key[2]))
    for events in events_by_user.values():
        events.sort(key=lambda event: (event[0], event[1], event[2], event[3]))
    return {
        "eligible_keys": eligible_keys,
        "events_by_user": dict(events_by_user),
        "excluded_split_counts": excluded_split_counts,
    }


def _select_latest_time_window(
    eligible_keys: list[RequestKey],
    *,
    max_requests: int,
    dev_fraction: float,
) -> tuple[set[RequestKey], dict[RequestKey, str], dict[str, Any]]:
    if len(eligible_keys) < 10:
        raise ValueError("not enough eligible source-train requests")
    ordered = sorted(eligible_keys, key=lambda key: (key[3], _request_id(key)))
    selected = ordered[-min(max_requests, len(ordered)) :]
    tentative = max(1, int(len(selected) * (1.0 - dev_fraction)))
    boundary_time = selected[tentative][3]
    split_index = tentative
    while split_index > 0 and selected[split_index - 1][3] == boundary_time:
        split_index -= 1
    if split_index == 0 or split_index == len(selected):
        raise ValueError("time-tie containment produced an empty train or dev split")
    train = selected[:split_index]
    dev = selected[split_index:]
    train_sessions = {key[1] for key in train}
    dev_sessions = {key[1] for key in dev}
    overlap = train_sessions & dev_sessions
    if overlap:
        raise ValueError(
            f"session identifiers cross the scout split: {sorted(overlap)[:5]}"
        )
    split_by_key = {key: "train" for key in train}
    split_by_key.update({key: "dev" for key in dev})
    return set(selected), split_by_key, {
        "selected_requests": len(selected),
        "train_requests": len(train),
        "dev_requests": len(dev),
        "time_index_min": selected[0][3],
        "time_index_max": selected[-1][3],
        "dev_time_index_min": boundary_time,
        "time_ties_kept_with_dev": tentative - split_index,
        "session_overlap": 0,
    }


def _load_selected_requests(
    path: Path,
    *,
    selected_keys: set[RequestKey],
    split_by_key: dict[RequestKey, str],
    events_by_user: dict[str, list[Event]],
    max_history_len: int,
) -> list[SourceRequest]:
    result: list[SourceRequest] = []
    event_times_by_user = {
        user: [event[0] for event in events]
        for user, events in events_by_user.items()
    }
    for row in iter_jsonl(path):
        if str(row.get("split", "")) != "train":
            continue
        key = _request_key(row)
        if key not in selected_keys:
            continue
        user_id = key[0]
        request_time = key[3]
        events = events_by_user.get(user_id, [])
        event_times = event_times_by_user.get(user_id, [])
        stop = bisect.bisect_left(event_times, request_time)
        history = tuple(events[max(0, stop - max_history_len) : stop])
        result.append(
            SourceRequest(
                key=key,
                candidate_item_ids=tuple(
                    int(value) for value in row.get("impressed_item_ids", [])
                ),
                clicked_item_ids=frozenset(
                    int(value) for value in row.get("clicked_item_ids", [])
                ),
                purchased_item_ids=frozenset(
                    int(value) for value in row.get("purchased_item_ids", [])
                ),
                split=split_by_key[key],
                history=history,
            )
        )
    if len(result) != len(selected_keys):
        raise ValueError(
            f"selected request reload mismatch: {len(result)} != {len(selected_keys)}"
        )
    return sorted(result, key=lambda request: (request.key[3], _request_id(request.key)))


def _load_item_map(path: Path, needed_item_ids: set[int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        item_id = int(row["item_id"])
        if item_id not in needed_item_ids:
            continue
        result[item_id] = {
            "item_id": str(item_id),
            "title": str(row.get("item_title") or ""),
            "brand": str(row.get("brand_name") or ""),
            "cat": [
                str(value)
                for value in (
                    row.get("category_level1_name"),
                    row.get("category_level2_name"),
                    row.get("category_level3_name"),
                )
                if value not in (None, "", "UNKNOWN")
            ],
        }
        if len(result) == len(needed_item_ids):
            break
    return result


def _write_scout(
    output_dir: Path,
    requests: list[SourceRequest],
    *,
    item_map: dict[int, dict[str, Any]],
    dataset_version: str,
    include_history_query: bool,
) -> dict[str, Any]:
    record_paths = {
        split: output_dir / f"records_{split}.jsonl" for split in ("train", "dev")
    }
    qrels_paths = {
        split: output_dir / f"qrels_{split}.jsonl" for split in ("train", "dev")
    }
    record_handles = {
        split: path.open("w", encoding="utf-8")
        for split, path in record_paths.items()
    }
    qrels_handles = {
        split: path.open("w", encoding="utf-8")
        for split, path in qrels_paths.items()
    }
    candidate_entries: list[dict[str, Any]] = []
    request_entries: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    candidate_counts: list[int] = []
    history_lengths: list[int] = []
    repeated_queries: defaultdict[str, Counter[str]] = defaultdict(Counter)
    try:
        for request in requests:
            request_id = _request_id(request.key)
            user_id, session_id, query, request_time = request.key
            candidates = []
            text_hits = 0
            for item_id in request.candidate_item_ids:
                payload = dict(item_map.get(item_id, _missing_item(item_id)))
                text_hits += int(bool(payload["title"].strip()))
                if request.split == "train":
                    clicked = int(item_id in request.clicked_item_ids)
                    purchased = int(item_id in request.purchased_item_ids)
                    payload.update(
                        {
                            "clicked": clicked,
                            "purchased": purchased,
                            "relevance": 2 if purchased else clicked,
                        }
                    )
                candidates.append(payload)
            history = []
            history_text_hits = 0
            for event_time, item_id, event_type, event_query in request.history:
                payload = dict(item_map.get(item_id, _missing_item(item_id)))
                history_text_hits += int(bool(payload["title"].strip()))
                payload.update({"event": event_type, "ts": event_time})
                if include_history_query:
                    payload["query"] = event_query
                history.append(payload)
            history_ids = {int(event[1]) for event in request.history}
            strict_nonrepeat = bool(history) and history_ids.isdisjoint(
                request.candidate_item_ids
            )
            record = {
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
                "ts": request_time,
                "query": query,
                "history": history,
                "candidates": candidates,
                "masks": {
                    "history_present": bool(history),
                    "strict_nonrepeat": strict_nonrepeat,
                    "text_coverage": text_hits / len(candidates),
                    "history_text_coverage": (
                        history_text_hits / len(history) if history else 1.0
                    ),
                },
            }
            validate_standardized_record(record, request.split)
            record_handles[request.split].write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            relevance = {
                str(item_id): 2 if item_id in request.purchased_item_ids else 1
                for item_id in request.clicked_item_ids | request.purchased_item_ids
            }
            qrels_handles[request.split].write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "clicked": sorted(str(value) for value in request.clicked_item_ids),
                        "purchased": sorted(
                            str(value) for value in request.purchased_item_ids
                        ),
                        "relevance": relevance,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            candidate_ids = [str(value) for value in request.candidate_item_ids]
            candidate_entries.append(
                {
                    "split": request.split,
                    "request_id": request_id,
                    "candidate_item_ids": candidate_ids,
                }
            )
            request_entries.append(
                {
                    "split": request.split,
                    "request_id": request_id,
                    "query_sha256": sha256_text(query),
                    "candidate_item_ids_sha256": sha256_text(
                        json.dumps(candidate_ids, separators=(",", ":"))
                    ),
                }
            )
            counts[f"{request.split}_requests"] += 1
            counts[f"{request.split}_history_present"] += int(bool(history))
            counts[f"{request.split}_strict_nonrepeat"] += int(strict_nonrepeat)
            counts[f"{request.split}_with_click"] += int(bool(request.clicked_item_ids))
            counts[f"{request.split}_with_purchase"] += int(
                bool(request.purchased_item_ids)
            )
            candidate_counts.append(len(candidates))
            history_lengths.append(len(history))
            repeated_queries[request.split][_normalize_query(query)] += 1
    finally:
        for handle in (*record_handles.values(), *qrels_handles.values()):
            handle.close()

    candidate_manifest_path = output_dir / "candidate_manifest.json"
    request_manifest_path = output_dir / "request_manifest.json"
    write_json(
        candidate_manifest_path,
        {"dataset_version": dataset_version, "entries": candidate_entries},
    )
    write_json(
        request_manifest_path,
        {"dataset_version": dataset_version, "entries": request_entries},
    )
    audits = {
        split: audit_standardized_file(record_paths[split], split)
        for split in ("train", "dev")
    }
    return {
        "counts": dict(counts),
        "candidate_count": _summary(candidate_counts),
        "history_length": _summary(history_lengths),
        "repeated_query_requests": {
            split: sum(
                count for count in query_counts.values() if count > 1
            )
            for split, query_counts in repeated_queries.items()
        },
        "structural_audits": audits,
        "files": {
            **{
                f"records_{split}": _file_info(record_paths[split])
                for split in ("train", "dev")
            },
            **{
                f"qrels_{split}": _file_info(qrels_paths[split])
                for split in ("train", "dev")
            },
            "candidate_manifest": _file_info(candidate_manifest_path),
            "request_manifest": _file_info(request_manifest_path),
        },
        "label_isolation": {
            "dev_records_label_free": True,
            "dev_labels_path": str(qrels_paths["dev"]),
            "scoring_code_may_read_dev_labels": False,
        },
    }


def _request_key(row: dict[str, Any]) -> RequestKey:
    return (
        str(row["user_id"]),
        str(row["session_id"]),
        str(row["query"]),
        int(row["time_index"]),
    )


def _request_id(key: RequestKey) -> str:
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return "ks_" + sha256_text(payload)[:24]


def _normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _missing_item(item_id: int) -> dict[str, Any]:
    return {"item_id": str(item_id), "title": "", "brand": "", "cat": []}


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }
