"""Build KuaiSearch standardized JSONL records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from myrec.data.kuaisearch_audit import KuaiSearchRawPaths, request_key
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


@dataclass
class SelectedRequest:
    key: tuple[str, str, str, int]
    request_id: str
    user_id: str
    session_id: str
    query: str
    time_index: int
    position: int
    split: str
    candidate_item_ids: list[int]
    clicked_item_ids: set[int]
    purchased_item_ids: set[int]
    history_events: list[dict[str, Any]]


def build_standardized_kuaisearch(
    raw_dir: str | Path,
    window_requests_path: str | Path,
    output_dir: str | Path,
    c0_report_path: str | Path | None = "reports/pps_c0_data_audit.json",
    leakage_report_path: str | Path | None = "reports/pps_c0_history_leakage_check.json",
    max_history_len: int = 50,
) -> dict[str, Any]:
    paths = KuaiSearchRawPaths.from_raw_dir(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_manifest = _load_selected_manifest(window_requests_path)
    selected_requests = _load_selected_recall_requests(paths.recall, selected_manifest)
    _attach_fallback_histories(selected_requests, max_history_len=max_history_len)
    retained, filter_stats = _filter_requests(selected_requests)
    needed_item_ids = _needed_item_ids(retained)
    item_map = _load_item_map(paths.items, needed_item_ids)
    missing_item_ids = sorted(needed_item_ids - set(item_map))

    output_stats = _write_standardized_files(output_dir, retained, item_map, missing_item_ids)
    manifest = {
        "dataset_id": "kuaisearch",
        "dataset_version": "v0_lite",
        "raw_dir": str(Path(raw_dir)),
        "window_requests_path": str(window_requests_path),
        "window_requests_sha256": sha256_file(window_requests_path),
        "output_dir": str(output_dir),
        "history_source": {
            "type": "recall_prior_events_fallback",
            "reason": "raw ranking recently_* fields failed C0 leakage cross-reference",
            "construction": (
                "For each selected request, history is the same user's click/purchase "
                "events from selected recall-window requests with event_time < request_time; "
                "keep the most recent <= 50 events in ascending time order."
            ),
            "raw_recently_fields_used": False,
            "max_history_len": max_history_len,
        },
        "input_counts": {
            "selected_window_requests": len(selected_manifest),
            "selected_recall_requests_loaded": len(selected_requests),
        },
        "filter_stats": filter_stats,
        "item_join": {
            "needed_item_ids": len(needed_item_ids),
            "loaded_item_ids": len(item_map),
            "missing_item_ids": len(missing_item_ids),
            "missing_item_examples": [str(item_id) for item_id in missing_item_ids[:20]],
        },
        "outputs": output_stats,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["outputs"]["manifest"] = {
        "path": str(manifest_path),
        "sha256": "self_reference_not_recorded",
    }
    write_json(manifest_path, manifest)

    if c0_report_path:
        _merge_fallback_history_into_c0(
            c0_report_path=Path(c0_report_path),
            standardized_manifest_path=manifest_path,
            leakage_report_path=Path(leakage_report_path) if leakage_report_path else None,
            manifest=manifest,
        )
    return manifest


def _load_selected_manifest(path: str | Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    selected = {}
    for row in iter_jsonl(path):
        key = (str(row["user_id"]), str(row["session_id"]), str(row["query"]), int(row["time_index"]))
        selected[key] = row
    return selected


def _load_selected_recall_requests(
    recall_path: Path,
    selected_manifest: dict[tuple[str, str, str, int], dict[str, Any]],
) -> list[SelectedRequest]:
    selected = []
    selected_keys = set(selected_manifest)
    for row in iter_jsonl(recall_path):
        key = request_key(row)
        if key not in selected_keys:
            continue
        info = selected_manifest[key]
        selected.append(
            SelectedRequest(
                key=key,
                request_id=str(info["request_id"]),
                user_id=key[0],
                session_id=key[1],
                query=key[2],
                time_index=key[3],
                position=int(info["position"]),
                split=str(info["split"]),
                candidate_item_ids=[int(item) for item in row.get("impressed_item_ids", [])],
                clicked_item_ids=set(int(item) for item in row.get("clicked_item_ids", [])),
                purchased_item_ids=set(int(item) for item in row.get("purchased_item_ids", [])),
                history_events=[],
            )
        )
    selected.sort(key=lambda req: (req.time_index, req.position, req.request_id))
    return selected


def _attach_fallback_histories(requests: list[SelectedRequest], max_history_len: int) -> None:
    user_history: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    index = 0
    while index < len(requests):
        time_index = requests[index].time_index
        end = index + 1
        while end < len(requests) and requests[end].time_index == time_index:
            end += 1

        for request in requests[index:end]:
            request.history_events = [dict(event) for event in user_history[request.user_id][-max_history_len:]]

        for request in requests[index:end]:
            event_items = set(request.clicked_item_ids) | set(request.purchased_item_ids)
            for item_id in sorted(event_items):
                event = "purchase" if item_id in request.purchased_item_ids else "click"
                user_history[request.user_id].append(
                    {"item_id": item_id, "event": event, "ts": request.time_index}
                )
        index = end


def _filter_requests(requests: list[SelectedRequest]) -> tuple[list[SelectedRequest], dict[str, Any]]:
    retained = []
    stats = {
        "by_split_before": dict(Counter(request.split for request in requests)),
        "candidate_count_lt5": dict(Counter()),
        "dev_test_no_clicked_positive": dict(Counter()),
        "by_split_after": dict(Counter()),
    }
    candidate_lt5 = Counter()
    no_clicked = Counter()
    after = Counter()
    for request in requests:
        if len(request.candidate_item_ids) < 5:
            candidate_lt5[request.split] += 1
            continue
        if request.split in {"dev", "test"} and not request.clicked_item_ids:
            no_clicked[request.split] += 1
            continue
        retained.append(request)
        after[request.split] += 1
    stats["candidate_count_lt5"] = dict(candidate_lt5)
    stats["dev_test_no_clicked_positive"] = dict(no_clicked)
    stats["by_split_after"] = dict(after)
    return retained, stats


def _needed_item_ids(requests: list[SelectedRequest]) -> set[int]:
    item_ids: set[int] = set()
    for request in requests:
        item_ids.update(request.candidate_item_ids)
        item_ids.update(int(event["item_id"]) for event in request.history_events)
    return item_ids


def _load_item_map(items_path: Path, needed_item_ids: set[int]) -> dict[int, dict[str, Any]]:
    item_map: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(items_path):
        item_id = int(row["item_id"])
        if item_id not in needed_item_ids:
            continue
        item_map[item_id] = _normalize_item(row)
        if len(item_map) == len(needed_item_ids):
            break
    return item_map


def _normalize_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": str(row["item_id"]),
        "title": str(row.get("item_title") or ""),
        "brand": str(row.get("brand_name") or ""),
        "seller": str(row.get("seller_name") or ""),
        "cat": [
            str(row.get("category_level1_name") or ""),
            str(row.get("category_level2_name") or ""),
            str(row.get("category_level3_name") or ""),
        ],
    }


def _write_standardized_files(
    output_dir: Path,
    requests: list[SelectedRequest],
    item_map: dict[int, dict[str, Any]],
    missing_item_ids: list[int],
) -> dict[str, Any]:
    paths = {
        "records_train": output_dir / "records_train.jsonl",
        "records_dev": output_dir / "records_dev.jsonl",
        "records_test": output_dir / "records_test.jsonl",
        "qrels_dev": output_dir / "qrels_dev.jsonl",
        "qrels_test": output_dir / "qrels_test.jsonl",
        "item_catalog": output_dir / "item_catalog.jsonl",
        "candidate_manifest": output_dir / "candidate_manifest.json",
        "split_manifest": output_dir / "split_manifest.json",
    }
    handles = {
        "train": paths["records_train"].open("w", encoding="utf-8"),
        "dev": paths["records_dev"].open("w", encoding="utf-8"),
        "test": paths["records_test"].open("w", encoding="utf-8"),
    }
    qrel_handles = {
        "dev": paths["qrels_dev"].open("w", encoding="utf-8"),
        "test": paths["qrels_test"].open("w", encoding="utf-8"),
    }
    split_manifest: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    candidate_entries = []
    per_split_counts = Counter()
    history_lengths: list[int] = []
    text_coverage_values: list[float] = []

    try:
        for request in sorted(requests, key=lambda req: (req.position, req.request_id)):
            record, qrel, candidate_ids, text_coverage = _build_record(request, item_map)
            if request.split != "train":
                for candidate in record["candidates"]:
                    candidate.pop("clicked", None)
                    candidate.pop("purchased", None)
            handles[request.split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if request.split in qrel_handles:
                qrel_handles[request.split].write(json.dumps(qrel, ensure_ascii=False, sort_keys=True) + "\n")
            split_manifest[request.split].append(request.request_id)
            candidate_entries.append(
                {
                    "request_id": request.request_id,
                    "split": request.split,
                    "candidate_item_ids": candidate_ids,
                }
            )
            per_split_counts[request.split] += 1
            history_lengths.append(len(request.history_events))
            text_coverage_values.append(text_coverage)
    finally:
        for handle in handles.values():
            handle.close()
        for handle in qrel_handles.values():
            handle.close()

    with paths["item_catalog"].open("w", encoding="utf-8") as handle:
        for item_id in sorted(item_map):
            handle.write(json.dumps(item_map[item_id], ensure_ascii=False, sort_keys=True) + "\n")
    candidate_manifest = {
        "dataset_id": "kuaisearch",
        "dataset_version": "v0_lite",
        "entries": candidate_entries,
    }
    write_json(paths["candidate_manifest"], candidate_manifest)
    write_json(paths["split_manifest"], split_manifest)

    return {
        "counts_by_split": dict(per_split_counts),
        "history": _summarize_history_lengths(history_lengths),
        "record_text_coverage": _summarize_float_values(text_coverage_values),
        "files": {name: _file_info(path) for name, path in paths.items()},
        "missing_item_ids": len(missing_item_ids),
    }


def _build_record(
    request: SelectedRequest,
    item_map: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str], float]:
    text_total = 0
    text_hits = 0
    history = []
    for event in request.history_events:
        item = item_map.get(int(event["item_id"]))
        text_total += 1
        if item:
            text_hits += 1
        history.append(
            {
                **_item_or_missing(int(event["item_id"]), item),
                "event": event["event"],
                "ts": int(event["ts"]),
            }
        )

    candidates = []
    candidate_ids = []
    for item_id in request.candidate_item_ids:
        item = item_map.get(item_id)
        text_total += 1
        if item:
            text_hits += 1
        candidate_ids.append(str(item_id))
        candidates.append(
            {
                **_item_or_missing(item_id, item),
                "clicked": 1 if item_id in request.clicked_item_ids else 0,
                "purchased": 1 if item_id in request.purchased_item_ids else 0,
            }
        )
    text_coverage = text_hits / text_total if text_total else 1.0
    record = {
        "request_id": request.request_id,
        "user_id": request.user_id,
        "session_id": request.session_id,
        "ts": request.time_index,
        "query": request.query,
        "history": history,
        "candidates": candidates,
        "masks": {
            "history_present": bool(history),
            "text_coverage": text_coverage,
            "history_source": "recall_prior_events_fallback",
        },
    }
    qrel = {
        "request_id": request.request_id,
        "clicked": [str(item_id) for item_id in request.candidate_item_ids if item_id in request.clicked_item_ids],
        "purchased": [
            str(item_id) for item_id in request.candidate_item_ids if item_id in request.purchased_item_ids
        ],
    }
    return record, qrel, candidate_ids, text_coverage


def _item_or_missing(item_id: int, item: dict[str, Any] | None) -> dict[str, Any]:
    if item:
        return dict(item)
    return {
        "item_id": str(item_id),
        "title": "",
        "brand": "",
        "seller": "",
        "cat": ["", "", ""],
    }


def _summarize_history_lengths(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
        "history_present_rate": sum(1 for value in values if value > 0) / len(values),
    }


def _summarize_float_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def _file_info(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "rows": _line_count(path) if path.suffix == ".jsonl" else None,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _merge_fallback_history_into_c0(
    c0_report_path: Path,
    standardized_manifest_path: Path,
    leakage_report_path: Path | None,
    manifest: dict[str, Any],
) -> None:
    if not c0_report_path.exists():
        return
    with c0_report_path.open("r", encoding="utf-8") as handle:
        c0_report = json.load(handle)
    leakage_info: dict[str, Any] = {}
    if leakage_report_path and leakage_report_path.exists():
        leakage_info = {
            "raw_recently_leakage_report_path": str(leakage_report_path),
            "raw_recently_leakage_report_sha256": sha256_file(leakage_report_path),
        }
    c0_report["checks"]["history_future_leakage"] = {
        "status": "passed",
        "evidence": {
            "method": "fallback_history_construction",
            "raw_recently_fields_used": False,
            "fallback_history_source": "recall_prior_events",
            "standardized_manifest_path": str(standardized_manifest_path),
            "standardized_manifest_sha256": sha256_file(standardized_manifest_path),
            "history_present_rate": manifest["outputs"]["history"]["history_present_rate"],
            "history_length_distribution": manifest["outputs"]["history"],
            "construction": manifest["history_source"]["construction"],
            "caveat": (
                "Raw ranking recently_* history failed leakage cross-reference and is rejected. "
                "Standardized history is rebuilt from recall-window events with event_time < request_time; "
                "this guarantees no future leakage by construction but yields shorter histories and empty "
                "history for early-window requests."
            ),
            **leakage_info,
        },
        "rule": (
            "History must not contain future events. Because raw ranking recently_* failed the registered "
            "cross-reference check, standardized records use only recall prior events with event_time < request_time."
        ),
    }
    c0_report["history_source_decision"] = {
        "selected": "recall_prior_events_fallback",
        "raw_recently_fields_rejected": True,
        **leakage_info,
    }
    c0_report["overall_status"] = (
        "passed"
        if all(check["status"] == "passed" for check in c0_report["checks"].values())
        else "failed"
    )
    write_json(c0_report_path, c0_report)
