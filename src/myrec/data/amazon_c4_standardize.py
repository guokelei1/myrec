"""Build the unified PPS interface for Amazon-C4 plus temporal histories.

The converter deliberately keeps retrieval, history construction, and label
isolation in one audited boundary.  Downstream models only consume the
standardized JSONL files and never reopen the raw release.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sqlite3
import threading
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_FTS_INDEX_VERSION = 2
_MAX_BM25_QUERY_TERMS = 8


@dataclass
class AmazonC4Request:
    request_id: str
    split: str
    qid: int
    query: str
    user_id: str
    positive_item_id: str
    positive_category: str
    history_events: list[dict[str, Any]]
    candidate_rows: list[dict[str, str]]


def build_sampled_metadata_fts(
    sampled_metadata_path: str | Path,
    index_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build a deterministic SQLite FTS5/BM25 index over the official 1M pool."""

    sampled_metadata_path = Path(sampled_metadata_path)
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = index_path.with_suffix(index_path.suffix + ".manifest.json")
    source_sha256 = sha256_file(sampled_metadata_path)
    if not force and index_path.exists() and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("sampled_metadata_sha256") == source_sha256
            and manifest.get("index_version") == _FTS_INDEX_VERSION
        ):
            return manifest

    if index_path.exists():
        index_path.unlink()
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE TABLE items ("
            "item_id TEXT PRIMARY KEY, category TEXT NOT NULL, metadata_text TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE items_fts USING fts5("
            "metadata_text, content='items', content_rowid='rowid', "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        rows = 0
        batch: list[tuple[str, str, str]] = []
        with sampled_metadata_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                row = json.loads(line)
                item_id = str(row["item_id"])
                category = str(row.get("category") or "Unknown")
                metadata_text = str(row.get("metadata") or "")
                batch.append((item_id, category, metadata_text))
                if len(batch) >= 10_000:
                    connection.executemany("INSERT INTO items VALUES (?, ?, ?)", batch)
                    rows += len(batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO items VALUES (?, ?, ?)", batch)
                rows += len(batch)
        connection.execute("INSERT INTO items_fts(items_fts) VALUES ('rebuild')")
        connection.execute(
            "CREATE VIRTUAL TABLE items_vocab USING fts5vocab(items_fts, 'row')"
        )
        connection.execute(
            "CREATE TABLE term_stats ("
            "term TEXT PRIMARY KEY, document_frequency INTEGER NOT NULL)"
        )
        connection.execute(
            "INSERT INTO term_stats SELECT term, doc FROM items_vocab"
        )
        connection.execute("DROP TABLE items_vocab")
        connection.commit()
        duplicate_check = connection.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT item_id) FROM items"
        ).fetchone()[0]
        if duplicate_check != 0:
            raise ValueError("sampled metadata contains duplicate item ids")
    finally:
        connection.close()

    manifest = {
        "index_version": _FTS_INDEX_VERSION,
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "rows": rows,
        "sampled_metadata_path": str(sampled_metadata_path),
        "sampled_metadata_sha256": source_sha256,
        "retrieval": {
            "engine": "SQLite FTS5",
            "ranker": "BM25",
            "tokenizer": "unicode61 remove_diacritics=2",
            "query_term_selection": (
                "up to 8 lowest-document-frequency unique query terms; "
                "document frequency is frozen from the sampled catalog"
            ),
            "tie_break": "item_id ascending",
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def retrieve_bm25_candidates(
    connection: sqlite3.Connection,
    query: str,
    *,
    top_k: int,
) -> list[dict[str, str]]:
    """Return a fixed deterministic BM25 pool from the sampled catalog."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    tokens = _unique_query_tokens(query, maximum=128)
    if tokens:
        placeholders = ",".join("?" for _ in tokens)
        term_rows = connection.execute(
            f"SELECT term, document_frequency FROM term_stats "
            f"WHERE term IN ({placeholders})",
            tokens,
        ).fetchall()
        tokens = [
            str(term)
            for term, _ in sorted(
                term_rows,
                key=lambda row: (int(row[1]), str(row[0])),
            )[:_MAX_BM25_QUERY_TERMS]
        ]
    if not tokens:
        rows = connection.execute(
            "SELECT item_id, category, metadata_text FROM items "
            "ORDER BY item_id LIMIT ?",
            (top_k,),
        ).fetchall()
    else:
        expression = " OR ".join(f'"{token}"' for token in tokens)
        rows = connection.execute(
            "SELECT items.item_id, items.category, items.metadata_text "
            "FROM items_fts JOIN items ON items.rowid = items_fts.rowid "
            "WHERE items_fts MATCH ? "
            "ORDER BY bm25(items_fts), items.item_id LIMIT ?",
            (expression, top_k),
        ).fetchall()
    output = [
        {"item_id": str(item_id), "category": str(category), "metadata_text": str(text)}
        for item_id, category, text in rows
    ]
    if len(output) < top_k:
        # Documents with no matching term all have the same zero lexical score.
        # Complete that tied tail by item id so every request has a fixed-size
        # pool even for degenerate or tiny-catalog queries.
        seen = {row["item_id"] for row in output}
        fallback_rows = connection.execute(
            "SELECT item_id, category, metadata_text FROM items "
            "ORDER BY item_id LIMIT ?",
            (top_k + len(seen),),
        ).fetchall()
        for item_id, category, text in fallback_rows:
            if str(item_id) in seen:
                continue
            output.append(
                {
                    "item_id": str(item_id),
                    "category": str(category),
                    "metadata_text": str(text),
                }
            )
            seen.add(str(item_id))
            if len(output) == top_k:
                break
    return output


def build_standardized_amazon_c4(
    *,
    c4_csv_path: str | Path,
    history_root: str | Path,
    sampled_metadata_path: str | Path,
    reviews_metadata_dir: str | Path,
    fts_index_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path | None = "reports/pps_c0_amazon_c4_data_audit.json",
    max_history_len: int = 50,
    bm25_top_k: int = 100,
    retrieval_workers: int = 4,
    metadata_workers: int = 1,
    candidate_cache_path: str | Path | None = None,
    candidate_cache_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize Amazon-C4 train/dev/test records with physical label isolation."""

    if max_history_len <= 0:
        raise ValueError("max_history_len must be positive")
    c4_csv_path = Path(c4_csv_path)
    history_root = Path(history_root)
    sampled_metadata_path = Path(sampled_metadata_path)
    reviews_metadata_dir = Path(reviews_metadata_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_manifest = build_sampled_metadata_fts(sampled_metadata_path, fts_index_path)
    query_rows = _load_c4_queries(c4_csv_path)
    requests, source_audit = _load_history_requests(
        history_root,
        query_rows,
        max_history_len=max_history_len,
    )
    candidate_cache = None
    if candidate_cache_path is not None or candidate_cache_report_path is not None:
        if candidate_cache_path is None or candidate_cache_report_path is None:
            raise ValueError("candidate cache and report must be provided together")
        candidate_cache = _attach_cached_candidate_pools(
            requests=requests,
            cache_path=Path(candidate_cache_path),
            cache_report_path=Path(candidate_cache_report_path),
            index_manifest=index_manifest,
        )
    else:
        _attach_candidate_pools(
            Path(fts_index_path),
            requests,
            top_k=bm25_top_k,
            workers=retrieval_workers,
        )
    connection = sqlite3.connect(fts_index_path)
    try:
        sampled_fallback = _load_sampled_fallbacks(
            connection,
            _needed_item_ids(requests),
        )
    finally:
        connection.close()

    needed_item_ids = _needed_item_ids(requests)
    full_metadata, metadata_scan = _load_reviews_metadata(
        reviews_metadata_dir,
        _history_item_ids_by_category(requests),
        workers=metadata_workers,
    )
    item_map = _merge_item_metadata(needed_item_ids, full_metadata, sampled_fallback)
    output_stats = _write_standardized_files(
        output_dir,
        requests,
        item_map,
        bm25_top_k=bm25_top_k,
    )

    history_item_ids = {
        str(event["item_id"])
        for request in requests
        for event in request.history_events
    }
    history_metadata_hits = len(history_item_ids & set(item_map))
    history_id_coverage = history_metadata_hits / len(history_item_ids) if history_item_ids else 1.0
    source_history_text_coverage = output_stats["source_history_event_text_coverage"]
    consumed_history_text_coverage = output_stats["history_event_text_coverage"]
    retained_train_history = output_stats["retained_history_length_by_split"]["train"]
    checks = {
        "source_qid_user_positive_alignment": source_audit["alignment_failures"] == 0,
        "source_target_absent_from_history": source_audit["target_in_history_rows"] == 0,
        "source_history_nonempty": source_audit["empty_history_rows"] == 0,
        "candidate_count_at_least_protocol_minimum": (
            output_stats["candidate_count"]["min"] >= min(10, bm25_top_k)
        ),
        "positive_in_every_candidate_pool": output_stats["positive_missing"] == 0,
        "consumed_history_text_coverage_at_least_95pct": (
            consumed_history_text_coverage >= 0.95
        ),
        "missing_history_event_drop_fraction_at_most_10pct": (
            output_stats["missing_history_event_drop_fraction"] <= 0.10
        ),
        "train_history_nonempty_after_text_mask": (
            output_stats["history_empty_after_text_mask_by_split"]["train"] == 0
        ),
        "train_retained_history_median_at_least_protocol_minimum": (
            retained_train_history["median"] >= min(10, max_history_len)
        ),
        "dev_test_records_label_free": output_stats["dev_test_candidate_label_fields"] == 0,
        "train_records_have_labels": output_stats["train_candidate_label_fields_missing"] == 0,
    }
    manifest = {
        "dataset_id": "amazon_c4",
        "dataset_version": "v0_history_bm25_100",
        "output_dir": str(output_dir),
        "source": {
            "amazon_c4_csv": _file_info(c4_csv_path),
            "history_root": str(history_root),
            "sampled_metadata": _file_info(sampled_metadata_path),
            "reviews_metadata_dir": str(reviews_metadata_dir),
            "history_release_rule": (
                "The upstream release sorts interactions by timestamp and cuts history "
                "before the target product's first purchase; the target product is absent "
                "from every consumed history row."
            ),
        },
        "protocol": {
            "split_source": "upstream Amazon-C4 history release train/dev/test",
            "history_order": "timestamp ascending; most recent <= max_history_len",
            "max_history_len": max_history_len,
            "candidate_pool": "SQLite FTS5 BM25 top-k over official sampled 1M catalog plus positive",
            "bm25_top_k": bm25_top_k,
            "retrieval_workers": retrieval_workers,
            "metadata_workers": metadata_workers,
            "label_mapping": "the target purchase is clicked=1 and purchased=1; all pool negatives are 0",
            "request_ts": "one millisecond after the maximum released history timestamp (surrogate)",
            "missing_history_text_policy": (
                "Drop individual history events whose joined title is blank before any model can "
                "consume the record; retain the request and audit source coverage, drop fraction, "
                "post-mask emptiness, and retained history length by split."
            ),
            "candidate_cache": candidate_cache,
        },
        "index": index_manifest,
        "source_audit": source_audit,
        "metadata_scan": metadata_scan,
        "item_join": {
            "needed_item_ids": len(needed_item_ids),
            "history_item_ids": len(history_item_ids),
            "history_metadata_hits": history_metadata_hits,
            "history_id_coverage": history_id_coverage,
            "source_history_text_coverage": source_history_text_coverage,
            "consumed_history_text_coverage": consumed_history_text_coverage,
            "full_metadata_hits": len(full_metadata),
            "sampled_fallback_hits": len(set(item_map) - set(full_metadata)),
        },
        "outputs": output_stats,
        "checks": checks,
        "overall_status": "passed" if all(checks.values()) else "failed",
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["outputs"]["files"]["manifest"] = {
        "path": str(manifest_path),
        "sha256": "self_reference_not_recorded",
        "size_bytes": manifest_path.stat().st_size,
        "rows": None,
    }
    write_json(manifest_path, manifest)
    if report_path is not None:
        write_json(report_path, manifest)
    return manifest


def _unique_query_tokens(text: str, *, maximum: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
        if len(output) == maximum:
            break
    return output


def _load_c4_queries(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            qid = int(row["qid"])
            if qid in rows:
                raise ValueError(f"duplicate Amazon-C4 qid: {qid}")
            rows[qid] = {
                "query": str(row["query"]),
                "item_id": str(row["item_id"]),
                "user_id": str(row["user_id"]),
            }
    return rows


def _load_history_requests(
    history_root: Path,
    query_rows: dict[int, dict[str, str]],
    *,
    max_history_len: int,
) -> tuple[list[AmazonC4Request], dict[str, Any]]:
    requests: list[AmazonC4Request] = []
    seen: set[tuple[str, int, str, str]] = set()
    split_counts: Counter[str] = Counter()
    alignment_failures = 0
    target_in_history_rows = 0
    empty_history_rows = 0
    raw_history_lengths: list[int] = []
    retained_history_lengths: list[int] = []
    for split in ("train", "dev", "test"):
        paths = sorted(history_root.glob(f"*/{split}.jsonl"))
        if not paths:
            raise FileNotFoundError(f"no Amazon-C4 history files for {split}")
        for path in paths:
            for row in iter_jsonl(path):
                user_id = str(row["user_id"])
                qid = int(row["query"])
                positive_item_id = str(row["pos_product"])
                positive_category = str(row["pos_product_category"])
                key = (user_id, qid, positive_item_id, positive_category)
                if key in seen:
                    raise ValueError(f"duplicate history request key for qid {qid}")
                seen.add(key)
                source = query_rows.get(qid)
                if source is None:
                    raise ValueError(f"history qid missing from Amazon-C4: {qid}")
                if source["user_id"] != user_id or source["item_id"] != positive_item_id:
                    alignment_failures += 1

                events = _flatten_history(row["grouped_purchase_history"])
                raw_history_lengths.append(len(events))
                if any(str(event["item_id"]) == positive_item_id for event in events):
                    target_in_history_rows += 1
                if not events:
                    empty_history_rows += 1
                events = events[-max_history_len:]
                retained_history_lengths.append(len(events))
                requests.append(
                    AmazonC4Request(
                        request_id=f"amazon_c4_{split}_{qid}",
                        split=split,
                        qid=qid,
                        query=source["query"],
                        user_id=user_id,
                        positive_item_id=positive_item_id,
                        positive_category=positive_category,
                        history_events=events,
                        candidate_rows=[],
                    )
                )
                split_counts[split] += 1
    requests.sort(key=lambda request: (request.split, request.qid, request.request_id))
    return requests, {
        "requests": len(requests),
        "counts_by_split": dict(split_counts),
        "unique_request_keys": len(seen),
        "alignment_failures": alignment_failures,
        "target_in_history_rows": target_in_history_rows,
        "empty_history_rows": empty_history_rows,
        "raw_history_length": _summarize_numbers(raw_history_lengths),
        "retained_history_length": _summarize_numbers(retained_history_lengths),
    }


def _flatten_history(grouped: Any) -> list[dict[str, Any]]:
    if not isinstance(grouped, dict):
        raise ValueError("grouped_purchase_history must be an object")
    events: list[dict[str, Any]] = []
    for category, values in grouped.items():
        if not isinstance(values, list):
            raise ValueError("history category value must be a list")
        for value in values:
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError("history event must be [item_id, timestamp_ms, verified_purchase]")
            events.append(
                {
                    "item_id": str(value[0]),
                    "ts": int(value[1]),
                    "verified_purchase": bool(value[2]),
                    "source_category": str(category),
                }
            )
    events.sort(
        key=lambda event: (
            int(event["ts"]),
            str(event["item_id"]),
            str(event["source_category"]),
            bool(event["verified_purchase"]),
        )
    )
    return events


def _attach_candidate_pools(
    index_path: Path,
    requests: list[AmazonC4Request],
    *,
    top_k: int,
    workers: int,
) -> None:
    if workers <= 0:
        raise ValueError("retrieval_workers must be positive")
    local = threading.local()

    def retrieve(request: AmazonC4Request) -> list[dict[str, str]]:
        if not hasattr(local, "connection"):
            local.connection = sqlite3.connect(
                f"file:{index_path}?mode=ro",
                uri=True,
            )
        connection = local.connection
        rows = retrieve_bm25_candidates(connection, request.query, top_k=top_k)
        seen = {row["item_id"] for row in rows}
        if request.positive_item_id not in seen:
            fallback = connection.execute(
                "SELECT item_id, category, metadata_text FROM items WHERE item_id = ?",
                (request.positive_item_id,),
            ).fetchone()
            if fallback is None:
                raise ValueError(
                    f"positive item {request.positive_item_id} is absent from sampled metadata"
                )
            rows.append(
                {
                    "item_id": str(fallback[0]),
                    "category": str(fallback[1]),
                    "metadata_text": str(fallback[2]),
                }
            )
        return [{"item_id": row["item_id"]} for row in rows]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        candidate_rows = list(executor.map(retrieve, requests))
    for request, rows in zip(requests, candidate_rows):
        request.candidate_rows = rows


def _attach_cached_candidate_pools(
    *,
    requests: list[AmazonC4Request],
    cache_path: Path,
    cache_report_path: Path,
    index_manifest: dict[str, Any],
) -> dict[str, Any]:
    report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    expected_cache_sha = report["outputs"]["files"]["candidate_manifest"]["sha256"]
    if sha256_file(cache_path) != expected_cache_sha:
        raise RuntimeError("Amazon-C4 candidate cache changed")
    if report["index"]["index_sha256"] != index_manifest["index_sha256"]:
        raise RuntimeError("Amazon-C4 candidate cache uses another FTS index")
    cached_top_k = int(report["protocol"]["bm25_top_k"])
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    by_request = {
        str(entry["request_id"]): [str(value) for value in entry["candidate_item_ids"]]
        for entry in cache["entries"]
    }
    if len(by_request) != len(cache["entries"]):
        raise ValueError("Amazon-C4 candidate cache has duplicate requests")
    expected_ids = {request.request_id for request in requests}
    if set(by_request) != expected_ids:
        raise ValueError("Amazon-C4 candidate cache request set differs")
    for request in requests:
        item_ids = by_request[request.request_id]
        if len(item_ids) not in {cached_top_k, cached_top_k + 1} or len(item_ids) != len(
            set(item_ids)
        ):
            raise ValueError("Amazon-C4 cached candidate row differs")
        if request.positive_item_id not in item_ids:
            raise ValueError("Amazon-C4 cached candidate row lost its positive")
        request.candidate_rows = [{"item_id": item_id} for item_id in item_ids]
    return {
        "path": str(cache_path),
        "sha256": expected_cache_sha,
        "source_report_path": str(cache_report_path),
        "source_report_sha256": sha256_file(cache_report_path),
        "index_sha256": index_manifest["index_sha256"],
        "validated_requests": len(requests),
    }


def _needed_item_ids(requests: Iterable[AmazonC4Request]) -> set[str]:
    needed: set[str] = set()
    for request in requests:
        needed.update(str(event["item_id"]) for event in request.history_events)
        needed.update(row["item_id"] for row in request.candidate_rows)
    return needed


def _history_item_ids_by_category(
    requests: Iterable[AmazonC4Request],
) -> dict[str, set[str]]:
    output: defaultdict[str, set[str]] = defaultdict(set)
    for request in requests:
        for event in request.history_events:
            output[str(event["source_category"])].add(str(event["item_id"]))
    return dict(output)


def _load_sampled_fallbacks(
    connection: sqlite3.Connection,
    item_ids: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    ordered = sorted(item_ids)
    for start in range(0, len(ordered), 800):
        chunk = ordered[start : start + 800]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"SELECT item_id, category, metadata_text FROM items WHERE item_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for item_id, category, text in rows:
            result[str(item_id)] = {
                "item_id": str(item_id),
                "title": str(text)[:2048],
                "brand": "",
                "seller": "",
                "cat": [str(category), "", ""],
            }
    return result


def _load_reviews_metadata(
    metadata_dir: Path,
    needed_item_ids_by_category: dict[str, set[str]],
    *,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if workers <= 0:
        raise ValueError("metadata_workers must be positive")
    jobs = []
    for category, item_ids in sorted(needed_item_ids_by_category.items()):
        path = metadata_dir / f"meta_{category}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        jobs.append((path, category, item_ids))
    if not jobs:
        raise ValueError("Amazon-C4 retained history has no metadata categories")
    if workers == 1:
        scanned = [_scan_reviews_metadata_file(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            scanned = list(executor.map(_scan_reviews_metadata_file, jobs))
    result: dict[str, dict[str, Any]] = {}
    duplicate_hits = 0
    scanned_rows = 0
    files = []
    for row in scanned:
        scanned_rows += int(row["rows"])
        files.append(row["file"])
        for item_id, normalized in row["matches"].items():
            previous = result.get(item_id)
            if previous is not None:
                duplicate_hits += 1
                if _metadata_quality(normalized) <= _metadata_quality(previous):
                    continue
            result[item_id] = normalized
    needed_items = set().union(*needed_item_ids_by_category.values())
    missing_items = needed_items - set(result)
    fallback_scan: dict[str, Any] = {
        "triggered": False,
        "missing_before": len(missing_items),
        "matched_items": 0,
        "files": [],
    }
    if missing_items:
        # The upstream history release uses `Unknown` when its category mapper
        # fails.  Such parent ASINs can still have metadata in another official
        # category archive.  Search the same frozen archives without changing
        # the history event or its source-category field.
        fallback_jobs = [
            (
                path,
                path.name.removeprefix("meta_").removesuffix(".jsonl.gz"),
                missing_items,
            )
            for path in sorted(metadata_dir.glob("meta_*.jsonl.gz"))
            if path.name != "meta_Unknown.jsonl.gz"
        ]
        if workers == 1:
            fallback_rows = [_scan_reviews_metadata_file(job) for job in fallback_jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                fallback_rows = list(executor.map(_scan_reviews_metadata_file, fallback_jobs))
        fallback_matches: dict[str, dict[str, Any]] = {}
        for row in fallback_rows:
            fallback_scan["files"].append(row["file"])
            for item_id, normalized in row["matches"].items():
                previous = fallback_matches.get(item_id)
                if previous is None or _metadata_quality(normalized) > _metadata_quality(previous):
                    fallback_matches[item_id] = normalized
        result.update(fallback_matches)
        fallback_scan.update(
            {
                "triggered": True,
                "matched_items": len(fallback_matches),
                "missing_after": len(needed_items - set(result)),
            }
        )
    return result, {
        "files": files,
        "scanned_rows": scanned_rows,
        "needed_items": len(needed_items),
        "matched_items": len(result),
        "duplicate_item_hits": duplicate_hits,
        "workers": workers,
        "cross_category_fallback": fallback_scan,
    }


def _scan_reviews_metadata_file(
    job: tuple[Path, str, set[str]],
) -> dict[str, Any]:
    path, category, needed_item_ids = job
    matches: dict[str, dict[str, Any]] = {}
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows += 1
            row = json.loads(line)
            item_id = str(row["parent_asin"])
            if item_id not in needed_item_ids:
                continue
            normalized = _normalize_reviews_item(row, source_category=category)
            previous = matches.get(item_id)
            if previous is None or _metadata_quality(normalized) > _metadata_quality(previous):
                matches[item_id] = normalized
    return {
        "matches": matches,
        "rows": rows,
        "file": {
            "path": str(path),
            "rows": rows,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "needed_items": len(needed_item_ids),
            "matched_items": len(matches),
        },
    }


def _normalize_reviews_item(row: dict[str, Any], *, source_category: str) -> dict[str, Any]:
    raw_categories = row.get("categories")
    categories = [str(value) for value in raw_categories] if isinstance(raw_categories, list) else []
    main_category = str(row.get("main_category") or source_category)
    category_values = []
    for value in [main_category, *categories, source_category]:
        if value and value not in category_values:
            category_values.append(value)
    category_values = (category_values + ["", ""])[:3]
    return {
        "item_id": str(row["parent_asin"]),
        "title": str(row.get("title") or ""),
        "brand": str(row.get("store") or ""),
        "seller": str(row.get("store") or ""),
        "cat": category_values,
    }


def _metadata_quality(item: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(bool(item.get("title"))),
        len(str(item.get("title") or "")),
        int(bool(item.get("brand"))),
    )


def _merge_item_metadata(
    needed_item_ids: set[str],
    full_metadata: dict[str, dict[str, Any]],
    sampled_fallback: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item_id in needed_item_ids:
        full = full_metadata.get(item_id)
        fallback = sampled_fallback.get(item_id)
        if full is not None and full.get("title"):
            output[item_id] = dict(full)
        elif fallback is not None:
            output[item_id] = dict(fallback)
        elif full is not None:
            output[item_id] = dict(full)
    return output


def _write_standardized_files(
    output_dir: Path,
    requests: list[AmazonC4Request],
    item_map: dict[str, dict[str, Any]],
    *,
    bm25_top_k: int,
) -> dict[str, Any]:
    paths = {
        "records_train": output_dir / "records_train.jsonl",
        "records_train_blind": output_dir / "records_train_blind.jsonl",
        "records_dev": output_dir / "records_dev.jsonl",
        "records_test": output_dir / "records_test.jsonl",
        "qrels_dev": output_dir / "qrels_dev.jsonl",
        "qrels_test": output_dir / "qrels_test.jsonl",
        "item_catalog": output_dir / "item_catalog.jsonl",
        "candidate_manifest": output_dir / "candidate_manifest.json",
        "split_manifest": output_dir / "split_manifest.json",
    }
    record_handles = {
        split: paths[f"records_{split}"].open("w", encoding="utf-8")
        for split in ("train", "dev", "test")
    }
    train_blind_handle = paths["records_train_blind"].open("w", encoding="utf-8")
    qrel_handles = {
        split: paths[f"qrels_{split}"].open("w", encoding="utf-8")
        for split in ("dev", "test")
    }
    split_manifest: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    candidate_entries = []
    candidate_counts: list[int] = []
    positive_missing = 0
    source_history_text_hits = 0
    source_history_rows = 0
    history_text_hits = 0
    history_rows = 0
    missing_history_events_dropped = 0
    history_empty_after_text_mask_by_split: Counter[str] = Counter()
    retained_history_lengths_by_split: dict[str, list[int]] = defaultdict(list)
    dev_test_candidate_label_fields = 0
    train_candidate_label_fields_missing = 0
    try:
        for request in requests:
            history = []
            for event in request.history_events:
                item = _item_or_missing(str(event["item_id"]), item_map)
                source_history_rows += 1
                has_text = bool(str(item.get("title", "")).strip())
                source_history_text_hits += int(has_text)
                if not has_text:
                    missing_history_events_dropped += 1
                    continue
                history_rows += 1
                history_text_hits += 1
                history.append(
                    {
                        **item,
                        "event": "purchase",
                        "ts": int(event["ts"]),
                        "verified_purchase": bool(event["verified_purchase"]),
                    }
                )
            retained_history_lengths_by_split[request.split].append(len(history))
            if not history:
                history_empty_after_text_mask_by_split[request.split] += 1
            candidates = []
            candidate_ids = []
            for candidate_row in request.candidate_rows:
                item_id = candidate_row["item_id"]
                candidate_ids.append(item_id)
                candidate = {
                    **_item_or_missing(item_id, item_map),
                    "clicked": int(item_id == request.positive_item_id),
                    "purchased": int(item_id == request.positive_item_id),
                }
                candidates.append(candidate)
            if request.positive_item_id not in candidate_ids:
                positive_missing += 1
            candidate_counts.append(len(candidate_ids))
            request_ts = max((int(event["ts"]) for event in request.history_events), default=-1) + 1
            text_total = len(history) + len(candidates)
            text_hits = sum(bool(item["title"]) for item in history) + sum(
                bool(item["title"]) for item in candidates
            )
            record = {
                "request_id": request.request_id,
                "user_id": request.user_id,
                "session_id": f"amazon_c4_{request.qid}",
                "ts": request_ts,
                "query": request.query,
                "history": history,
                "candidates": candidates,
                "masks": {
                    "history_present": bool(history),
                    "history_source": "amazon_reviews_2023_temporal_cutoff_release",
                    "history_source_events": len(request.history_events),
                    "history_missing_text_events_dropped": len(request.history_events) - len(history),
                    "text_coverage": text_hits / text_total if text_total else 1.0,
                },
            }
            if request.split != "train":
                for candidate in record["candidates"]:
                    candidate.pop("clicked")
                    candidate.pop("purchased")
                dev_test_candidate_label_fields += sum(
                    int("clicked" in candidate or "purchased" in candidate)
                    for candidate in record["candidates"]
                )
            else:
                train_candidate_label_fields_missing += sum(
                    int("clicked" not in candidate or "purchased" not in candidate)
                    for candidate in record["candidates"]
                )
                blind_record = json.loads(json.dumps(record))
                for candidate in blind_record["candidates"]:
                    candidate.pop("clicked")
                    candidate.pop("purchased")
                train_blind_handle.write(
                    json.dumps(blind_record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            record_handles[request.split].write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            if request.split in qrel_handles:
                qrel_handles[request.split].write(
                    json.dumps(
                        {
                            "request_id": request.request_id,
                            "clicked": [request.positive_item_id],
                            "purchased": [request.positive_item_id],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            split_manifest[request.split].append(request.request_id)
            candidate_entries.append(
                {
                    "request_id": request.request_id,
                    "split": request.split,
                    "candidate_item_ids": candidate_ids,
                }
            )
    finally:
        for handle in record_handles.values():
            handle.close()
        train_blind_handle.close()
        for handle in qrel_handles.values():
            handle.close()

    with paths["item_catalog"].open("w", encoding="utf-8") as handle:
        for item_id in sorted(item_map):
            handle.write(json.dumps(item_map[item_id], ensure_ascii=False, sort_keys=True) + "\n")
    write_json(
        paths["candidate_manifest"],
        {
            "dataset_id": "amazon_c4",
            "dataset_version": "v0_history_bm25_100",
            "candidate_construction": f"BM25 top-{bm25_top_k} plus positive",
            "entries": candidate_entries,
        },
    )
    write_json(paths["split_manifest"], split_manifest)
    ordered_counts = sorted(candidate_counts)
    return {
        "counts_by_split": {split: len(values) for split, values in split_manifest.items()},
        "candidate_count": _summarize_numbers(ordered_counts),
        "positive_missing": positive_missing,
        "source_history_event_rows": source_history_rows,
        "source_history_text_hits": source_history_text_hits,
        "source_history_event_text_coverage": (
            source_history_text_hits / source_history_rows if source_history_rows else 1.0
        ),
        "history_event_rows": history_rows,
        "history_text_hits": history_text_hits,
        "history_event_text_coverage": history_text_hits / history_rows if history_rows else 1.0,
        "missing_history_events_dropped": missing_history_events_dropped,
        "missing_history_event_drop_fraction": (
            missing_history_events_dropped / source_history_rows if source_history_rows else 0.0
        ),
        "history_empty_after_text_mask_by_split": {
            split: history_empty_after_text_mask_by_split[split]
            for split in ("train", "dev", "test")
        },
        "retained_history_length_by_split": {
            split: _summarize_numbers(retained_history_lengths_by_split[split])
            for split in ("train", "dev", "test")
        },
        "dev_test_candidate_label_fields": dev_test_candidate_label_fields,
        "train_candidate_label_fields_missing": train_candidate_label_fields_missing,
        "files": {name: _file_info(path) for name, path in paths.items()},
    }


def _item_or_missing(item_id: str, item_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item = item_map.get(item_id)
    if item is not None:
        return dict(item)
    return {
        "item_id": item_id,
        "title": "",
        "brand": "",
        "seller": "",
        "cat": ["", "", ""],
    }


def _summarize_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
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
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": _line_count(path) if path.suffix == ".jsonl" else None,
    }


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)
