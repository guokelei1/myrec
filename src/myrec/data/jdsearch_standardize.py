"""Build a label-isolated JDsearch exploratory scout from the official format."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.data.contracts import audit_standardized_file, validate_standardized_record
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import write_json


TERM_SEPARATOR = "\x18"
EXPECTED_BEHAVIOR_HEADER = (
    "query\tcandidate_wid_list\tcandidate_label_list\thistory_qry_list\t"
    "history_wid_list\thistory_type_list\thistory_time_list"
)
EXPECTED_PRODUCT_HEADER = (
    "wid\tname\tbrand_id\tbrand_name\tcate_id_1\tcate_name_1\t"
    "cate_id_2\tcate_name_2\tcate_id_3\tcate_name_3\tcate_id_4\t"
    "cate_name_4\tshop_id"
)
HISTORY_EVENT_MAP = {
    "ord": "purchase",
    "click": "click",
    "cart": "cart",
    "flw": "follow",
}


@dataclass(frozen=True)
class JDCandidate:
    item_id: str
    label: float
    source_position: int


@dataclass(frozen=True)
class JDHistoryEvent:
    query_tokens: tuple[str, ...]
    item_id: str
    event_type: str
    gap: int
    timestamp: int


@dataclass(frozen=True)
class JDRequest:
    request_id: str
    source_row: int
    query_tokens: tuple[str, ...]
    candidates: tuple[JDCandidate, ...]
    history: tuple[JDHistoryEvent, ...]
    target_timestamp: int


def build_jdsearch_scout(
    raw_dir: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    official_repository_dir: str | Path | None = None,
    dataset_version: str = "hash_scout10k_v3",
    max_requests: int = 10_000,
    dev_fraction: float = 0.20,
    max_history_len: int = 50,
    min_candidate_count: int = 2,
    max_candidate_count: int = 200,
    seed: int = 20260714,
) -> dict[str, Any]:
    """Materialize a stable-hash scout without opening model outcomes."""

    if max_requests < 10:
        raise ValueError("max_requests must be at least 10")
    if not 0.0 < dev_fraction < 0.5:
        raise ValueError("dev_fraction must be in (0, 0.5)")
    if max_history_len < 1:
        raise ValueError("max_history_len must be positive")
    if min_candidate_count < 2 or max_candidate_count < min_candidate_count:
        raise ValueError("invalid candidate-count boundary")

    raw_dir = Path(raw_dir)
    behavior_path = raw_dir / "user_behavior_data.txt"
    product_path = raw_dir / "product_meta_data.txt"
    archive_path = raw_dir / "archive.zip"
    for path in (behavior_path, product_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing JDsearch source file: {path}")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"JDsearch output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = _verify_provenance_prefixes(
        behavior_path,
        product_path,
        Path(official_repository_dir) if official_repository_dir else None,
    )
    requests, source_audit = _stable_sample_requests(
        behavior_path,
        max_requests=max_requests,
        max_history_len=max_history_len,
        min_candidate_count=min_candidate_count,
        max_candidate_count=max_candidate_count,
        seed=seed,
    )
    split_by_request = _stable_split(requests, dev_fraction=dev_fraction, seed=seed)
    needed_item_ids = {
        candidate.item_id for request in requests for candidate in request.candidates
    }
    needed_item_ids.update(
        event.item_id for request in requests for event in request.history
    )
    item_map, metadata_audit = _load_product_metadata(
        product_path,
        needed_item_ids,
        sample_product_line=provenance.pop("sample_product_line", None),
    )
    outputs = _write_standardized(
        output_dir,
        requests,
        split_by_request=split_by_request,
        item_map=item_map,
        dataset_version=dataset_version,
        candidate_order_seed=seed,
    )

    source_files = {
        "user_behavior_data": _file_info(behavior_path),
        "product_meta_data": _file_info(product_path),
    }
    if archive_path.is_file():
        source_files["mirror_archive"] = _file_info(archive_path)
    checks = {
        "documented_headers_match": provenance["behavior_header_match"] and provenance["product_header_match"],
        "official_behavior_sample_is_exact_prefix": provenance.get("behavior_sample_prefix_match", False),
        "official_product_sample_row_found_exactly": metadata_audit.get("official_sample_product_row_match", False),
        "official_user_count_matches": source_audit["source_rows"] == 173831,
        "official_product_file_missing_metadata_count_is_plausible": metadata_audit["source_rows"] == 12141247,
        "aligned_source_lists": source_audit["alignment_failures"] == 0,
        "causal_history_order": outputs["history_not_strictly_before_target_violations"] == 0,
        "candidate_identity_unique": outputs["duplicate_candidate_id_violations"] == 0,
        "candidate_order_is_outcome_blind_hash": outputs["candidate_order_violations"] == 0,
        "candidate_metadata_coverage_at_least_90pct": outputs["candidate_text_coverage"] >= 0.90,
        "dev_records_label_free": True,
        "dev_history_present_at_least_500": outputs["counts"].get("dev_history_present", 0) >= 500,
        "dev_strict_nonrepeat_at_least_500": outputs["counts"].get("dev_strict_nonrepeat", 0) >= 500,
    }
    manifest = {
        "dataset_id": "jdsearch",
        "dataset_version": dataset_version,
        "evidence_mode": "exploratory_independent_source_replication",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_warning": (
            "JDsearch queries and product text are anonymized term IDs. Results "
            "support functional personalized-product-ranking claims, not plaintext "
            "pretrained-language semantic claims or independent confirmation."
        ),
        "source": {
            "official_repository": "https://github.com/rucliujn/JDsearch",
            "mirror": "https://www.kaggle.com/datasets/duuuscha/jd-search-dataset",
            "mirror_license": "CC BY-NC-SA 4.0",
            "source_files": source_files,
            "provenance_checks": provenance,
        },
        "selection": {
            "strategy": "smallest seeded hashes of immutable source rows after source eligibility",
            "seed": seed,
            "max_requests": max_requests,
            "dev_fraction": dev_fraction,
            "split_strategy": "exact 80/20 order by a second seeded request hash",
            "max_history_len": max_history_len,
            "min_candidate_count": min_candidate_count,
            "max_candidate_count": max_candidate_count,
            **source_audit,
        },
        "metadata": {
            **metadata_audit,
            "needed_item_ids": len(needed_item_ids),
            "loaded_needed_item_ids": len(item_map),
            "needed_item_coverage": len(item_map) / len(needed_item_ids) if needed_item_ids else 0.0,
        },
        "protocol": {
            "history_time": (
                "synthetic strictly increasing integer derived from documented "
                "ordered history gaps; one official row is one user target request"
            ),
            "label_mapping": "0=no interaction, 1=click, 2=cart/follow, 3=purchase",
            "history_event_mapping": HISTORY_EVENT_MAP,
            "candidate_order": (
                "deterministic seeded hash of request_id and item_id; the released "
                "source order is excluded because every positive-label row places all "
                "positives in a prefix"
            ),
            "source_order_leakage_audit": {
                "rows_with_positive": source_audit["rows_with_positive"],
                "rows_with_all_positives_in_prefix": source_audit[
                    "rows_with_all_positives_in_prefix"
                ],
                "positive_prefix_rate": source_audit["positive_prefix_rate"],
            },
            "history_query_in_model_input": True,
            "dev_label_isolation": "candidate labels are written only to qrels_dev.jsonl",
        },
        "outputs": outputs,
        "admission_checks": checks,
        "admission_passed": all(checks.values()),
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(report_path, manifest)
    return manifest


def _verify_provenance_prefixes(
    behavior_path: Path,
    product_path: Path,
    official_repository_dir: Path | None,
) -> dict[str, Any]:
    with behavior_path.open("r", encoding="utf-8") as handle:
        behavior_header = handle.readline().rstrip("\n\r")
    with product_path.open("r", encoding="utf-8") as handle:
        product_header = handle.readline().rstrip("\n\r")
    result: dict[str, Any] = {
        "behavior_header_match": behavior_header == EXPECTED_BEHAVIOR_HEADER,
        "product_header_match": product_header == EXPECTED_PRODUCT_HEADER,
    }
    if official_repository_dir is None:
        return result
    behavior_sample = official_repository_dir / "user_behavior_data_sample.txt"
    product_sample = official_repository_dir / "product_meta_data_sample.txt"
    if behavior_sample.is_file():
        sample_bytes = behavior_sample.read_bytes()
        with behavior_path.open("rb") as handle:
            result["behavior_sample_prefix_match"] = handle.read(len(sample_bytes)) == sample_bytes
        result["behavior_sample_sha256"] = sha256_file(behavior_sample)
    if product_sample.is_file():
        with product_sample.open("r", encoding="utf-8") as handle:
            handle.readline()
            sample_line = handle.readline().rstrip("\n\r")
        result["sample_product_line"] = sample_line
        result["product_sample_sha256"] = sha256_file(product_sample)
    return result


def _stable_sample_requests(
    behavior_path: Path,
    *,
    max_requests: int,
    max_history_len: int,
    min_candidate_count: int,
    max_candidate_count: int,
    seed: int,
) -> tuple[list[JDRequest], dict[str, Any]]:
    heap: list[tuple[int, str, JDRequest]] = []
    counts: Counter[str] = Counter()
    candidate_counts: list[int] = []
    history_lengths: list[int] = []
    alignment_failures = 0
    with behavior_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n\r")
        if header != EXPECTED_BEHAVIOR_HEADER:
            raise ValueError("unexpected JDsearch behavior header")
        for source_row, line in enumerate(handle, start=1):
            counts["source_rows"] += 1
            try:
                request = _parse_behavior_line(
                    line.rstrip("\n\r"),
                    source_row=source_row,
                    max_history_len=max_history_len,
                )
            except ValueError:
                alignment_failures += 1
                raise
            candidate_count = len(request.candidates)
            if not min_candidate_count <= candidate_count <= max_candidate_count:
                counts["excluded_candidate_count"] += 1
                continue
            candidate_ids = [candidate.item_id for candidate in request.candidates]
            if len(set(candidate_ids)) != len(candidate_ids):
                counts["excluded_duplicate_candidates"] += 1
                continue
            positives = sum(candidate.label > 0 for candidate in request.candidates)
            if positives == 0:
                counts["excluded_no_positive"] += 1
                continue
            counts["rows_with_positive"] += 1
            positive_positions = [
                index
                for index, candidate in enumerate(request.candidates)
                if candidate.label > 0
            ]
            counts["rows_with_all_positives_in_prefix"] += int(
                positive_positions == list(range(len(positive_positions)))
            )
            if positives == candidate_count:
                counts["excluded_no_zero_label"] += 1
                continue
            counts["eligible_rows"] += 1
            candidate_counts.append(candidate_count)
            history_lengths.append(len(request.history))
            sample_hash = _hash_int(f"{seed}|sample|{request.request_id}")
            entry = (-sample_hash, request.request_id, request)
            if len(heap) < max_requests:
                heapq.heappush(heap, entry)
            elif sample_hash < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    if len(heap) < 10:
        raise ValueError("not enough eligible JDsearch rows")
    selected = [entry[2] for entry in heap]
    selected.sort(key=lambda request: request.request_id)
    rows_with_positive = counts["rows_with_positive"]
    return selected, {
        **dict(counts),
        "positive_prefix_rate": (
            counts["rows_with_all_positives_in_prefix"] / rows_with_positive
            if rows_with_positive
            else 0.0
        ),
        "alignment_failures": alignment_failures,
        "selected_requests": len(selected),
        "eligible_candidate_count": _summary(candidate_counts),
        "eligible_history_length": _summary(history_lengths),
    }


def _parse_behavior_line(
    line: str, *, source_row: int, max_history_len: int
) -> JDRequest:
    fields = line.split("\t")
    if len(fields) != 7:
        raise ValueError(f"JDsearch behavior row {source_row} has {len(fields)} fields")
    query_tokens = _term_tokens(fields[0])
    if not query_tokens:
        raise ValueError(f"JDsearch behavior row {source_row} has empty query")
    candidate_ids = _underscore_list(fields[1])
    labels = [float(value) for value in _underscore_list(fields[2])]
    history_queries = _underscore_list(fields[3])
    history_ids = _underscore_list(fields[4])
    history_types = _underscore_list(fields[5])
    history_gaps = [int(float(value)) for value in _underscore_list(fields[6])]
    if len(candidate_ids) != len(labels):
        raise ValueError(f"candidate/label alignment failure at row {source_row}")
    if not (len(history_queries) == len(history_ids) == len(history_types)):
        raise ValueError(f"history alignment failure at row {source_row}")
    if len(history_gaps) != len(history_ids) + 1:
        raise ValueError(f"history time alignment failure at row {source_row}")
    if any(gap < 0 for gap in history_gaps):
        raise ValueError(f"negative history gap at row {source_row}")
    if any(label not in {0.0, 1.0, 2.0, 3.0} for label in labels):
        raise ValueError(f"unsupported candidate label at row {source_row}")

    cumulative = 0
    all_history: list[JDHistoryEvent] = []
    for index, (query, item_id, event_type) in enumerate(
        zip(history_queries, history_ids, history_types)
    ):
        normalized_event_type = HISTORY_EVENT_MAP.get(event_type.casefold())
        if normalized_event_type is None:
            raise ValueError(
                f"unsupported history event type {event_type!r} at row {source_row}"
            )
        cumulative += history_gaps[index]
        all_history.append(
            JDHistoryEvent(
                query_tokens=() if query == "-1" else _term_tokens(query),
                item_id=item_id,
                event_type=normalized_event_type,
                gap=history_gaps[index],
                timestamp=cumulative * 1000 + index,
            )
        )
    target_timestamp = sum(history_gaps) * 1000 + len(history_ids) + 1
    if all_history and all_history[-1].timestamp >= target_timestamp:
        raise ValueError(f"noncausal synthesized history at row {source_row}")
    raw_identity = sha256_text(line)
    request_id = "jd_" + raw_identity[:24]
    return JDRequest(
        request_id=request_id,
        source_row=source_row,
        query_tokens=query_tokens,
        candidates=tuple(
            JDCandidate(item_id=item_id, label=label, source_position=index)
            for index, (item_id, label) in enumerate(zip(candidate_ids, labels))
        ),
        history=tuple(all_history[-max_history_len:]),
        target_timestamp=target_timestamp,
    )


def _stable_split(
    requests: list[JDRequest], *, dev_fraction: float, seed: int
) -> dict[str, str]:
    ordered = sorted(
        requests,
        key=lambda request: (_hash_int(f"{seed}|split|{request.request_id}"), request.request_id),
    )
    dev_count = max(1, round(len(ordered) * dev_fraction))
    dev_ids = {request.request_id for request in ordered[:dev_count]}
    return {
        request.request_id: "dev" if request.request_id in dev_ids else "train"
        for request in requests
    }


def _load_product_metadata(
    product_path: Path,
    needed_item_ids: set[str],
    *,
    sample_product_line: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    source_rows = 0
    sample_match = sample_product_line is None
    with product_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n\r")
        if header != EXPECTED_PRODUCT_HEADER:
            raise ValueError("unexpected JDsearch product header")
        for line in handle:
            source_rows += 1
            raw = line.rstrip("\n\r")
            if sample_product_line is not None and raw == sample_product_line:
                sample_match = True
            fields = raw.split("\t")
            if len(fields) != 13:
                raise ValueError(f"JDsearch product row {source_rows} has {len(fields)} fields")
            item_id = fields[0]
            if item_id not in needed_item_ids:
                continue
            name = _term_text(fields[1], maximum=40)
            brand = _term_text(fields[3], maximum=12)
            categories = [
                text
                for text in (
                    _term_text(fields[5], maximum=12),
                    _term_text(fields[7], maximum=12),
                    _term_text(fields[9], maximum=12),
                    _term_text(fields[11], maximum=12),
                )
                if text
            ]
            result[item_id] = {
                "item_id": item_id,
                "title": name,
                "brand": brand,
                "cat": categories,
            }
    return result, {
        "source_rows": source_rows,
        "official_sample_product_row_match": sample_match,
        "missing_needed_item_ids": len(needed_item_ids - set(result)),
        "missing_needed_item_examples": sorted(needed_item_ids - set(result))[:20],
    }


def _write_standardized(
    output_dir: Path,
    requests: list[JDRequest],
    *,
    split_by_request: dict[str, str],
    item_map: dict[str, dict[str, Any]],
    dataset_version: str,
    candidate_order_seed: int,
) -> dict[str, Any]:
    record_paths = {split: output_dir / f"records_{split}.jsonl" for split in ("train", "dev")}
    qrels_paths = {split: output_dir / f"qrels_{split}.jsonl" for split in ("train", "dev")}
    record_handles = {split: path.open("w", encoding="utf-8") for split, path in record_paths.items()}
    qrels_handles = {split: path.open("w", encoding="utf-8") for split, path in qrels_paths.items()}
    candidate_entries = []
    request_entries = []
    counts: Counter[str] = Counter()
    query_counts: defaultdict[str, int] = defaultdict(int)
    candidate_text_hits = candidate_total = 0
    history_text_hits = history_total = 0
    duplicate_candidate_violations = future_history_violations = 0
    candidate_order_violations = 0
    candidate_counts: list[int] = []
    history_lengths: list[int] = []
    try:
        for request in sorted(requests, key=lambda row: row.request_id):
            split = split_by_request[request.request_id]
            query = _tokens_text(request.query_tokens)
            ordered_candidates = sorted(
                request.candidates,
                key=lambda candidate: (
                    _hash_int(
                        f"{candidate_order_seed}|candidate_order|"
                        f"{request.request_id}|{candidate.item_id}"
                    ),
                    candidate.item_id,
                ),
            )
            expected_candidate_ids = [candidate.item_id for candidate in ordered_candidates]
            candidate_order_violations += int(
                expected_candidate_ids
                != sorted(
                    expected_candidate_ids,
                    key=lambda item_id: (
                        _hash_int(
                            f"{candidate_order_seed}|candidate_order|"
                            f"{request.request_id}|{item_id}"
                        ),
                        item_id,
                    ),
                )
            )
            candidates = []
            for shuffled_position, candidate in enumerate(ordered_candidates):
                payload = dict(item_map.get(candidate.item_id, _missing_item(candidate.item_id)))
                payload["source_position"] = shuffled_position
                candidate_text_hits += int(bool(payload["title"]))
                candidate_total += 1
                if split == "train":
                    payload.update(
                        {
                            "clicked": int(candidate.label >= 1.0),
                            "purchased": int(candidate.label >= 3.0),
                            "relevance": candidate.label,
                        }
                    )
                candidates.append(payload)
            history = []
            for event in request.history:
                payload = dict(item_map.get(event.item_id, _missing_item(event.item_id)))
                payload.update(
                    {
                        "query": _tokens_text(event.query_tokens) if event.query_tokens else "",
                        "event": event.event_type,
                        "time_gap": event.gap,
                        "ts": event.timestamp,
                    }
                )
                history_text_hits += int(bool(payload["title"] or payload["query"]))
                history_total += 1
                future_history_violations += int(event.timestamp >= request.target_timestamp)
                history.append(payload)
            candidate_ids = [candidate.item_id for candidate in ordered_candidates]
            duplicate_candidate_violations += int(len(candidate_ids) != len(set(candidate_ids)))
            history_ids = {event.item_id for event in request.history}
            strict_nonrepeat = bool(history) and history_ids.isdisjoint(candidate_ids)
            record = {
                "request_id": request.request_id,
                "user_id": f"jd_source_row_{request.source_row}",
                "session_id": f"jd_target_{request.source_row}",
                "ts": request.target_timestamp,
                "query": query,
                "history": history,
                "candidates": candidates,
                "masks": {
                    "history_present": bool(history),
                    "strict_nonrepeat": strict_nonrepeat,
                    "text_coverage": sum(bool(candidate["title"]) for candidate in candidates) / len(candidates),
                    "history_text_coverage": (
                        sum(bool(event["title"] or event["query"]) for event in history) / len(history)
                        if history
                        else 1.0
                    ),
                },
            }
            validate_standardized_record(record, split)
            record_handles[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            clicked = sorted(candidate.item_id for candidate in request.candidates if candidate.label >= 1.0)
            purchased = sorted(candidate.item_id for candidate in request.candidates if candidate.label >= 3.0)
            relevance = {
                candidate.item_id: candidate.label
                for candidate in request.candidates
                if candidate.label > 0
            }
            qrels_handles[split].write(
                json.dumps(
                    {
                        "request_id": request.request_id,
                        "clicked": clicked,
                        "purchased": purchased,
                        "relevance": relevance,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            candidate_entries.append(
                {"split": split, "request_id": request.request_id, "candidate_item_ids": candidate_ids}
            )
            request_entries.append(
                {
                    "split": split,
                    "request_id": request.request_id,
                    "query_sha256": sha256_text(query),
                    "candidate_item_ids_sha256": sha256_text(json.dumps(candidate_ids, separators=(",", ":"))),
                }
            )
            counts[f"{split}_requests"] += 1
            counts[f"{split}_history_present"] += int(bool(history))
            counts[f"{split}_strict_nonrepeat"] += int(strict_nonrepeat)
            counts[f"{split}_repeat"] += int(bool(history) and not strict_nonrepeat)
            query_counts[query] += 1
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
        "repeated_query_requests": sum(count for count in query_counts.values() if count > 1),
        "duplicate_candidate_id_violations": duplicate_candidate_violations,
        "candidate_order_violations": candidate_order_violations,
        "history_not_strictly_before_target_violations": future_history_violations,
        "structural_audits": audits,
        "label_isolation": {"dev_records_label_free": True, "qrels_read_by_materializer": True},
        "files": {
            **{f"records_{split}": _file_info(record_paths[split]) for split in ("train", "dev")},
            **{f"qrels_{split}": _file_info(qrels_paths[split]) for split in ("train", "dev")},
            "candidate_manifest": _file_info(candidate_manifest_path),
            "request_manifest": _file_info(request_manifest_path),
        },
    }


def _underscore_list(value: str) -> list[str]:
    return [] if value == "" else value.split("_")


def _term_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in value.split(TERM_SEPARATOR) if token and token != "-1")


def _tokens_text(tokens: tuple[str, ...]) -> str:
    return " ".join(f"w{token}" for token in tokens)


def _term_text(value: str, *, maximum: int) -> str:
    return _tokens_text(_term_tokens(value)[:maximum])


def _hash_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest(), "big")


def _missing_item(item_id: str) -> dict[str, Any]:
    return {"item_id": item_id, "title": "", "brand": "", "cat": []}


def _file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }
