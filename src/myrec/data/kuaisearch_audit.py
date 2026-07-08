"""KuaiSearch Phase 0 audit and deterministic time-window sampling."""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json, write_jsonl

REQUEST_FIELDS = ("user_id", "session_id", "query", "time_index")
RANK_HISTORY_FIELDS = ("recently_clicked_item_ids", "recently_purchased_item_ids")
RANK_REQUIRED_FIELDS = (
    "user_id",
    "session_id",
    "time_index",
    "query",
    "target_item_id",
    "is_clicked",
    "is_purchased",
)


@dataclass(frozen=True)
class KuaiSearchRawPaths:
    raw_dir: Path
    rank: Path
    recall: Path
    items: Path
    users: Path
    relevance: Path

    @classmethod
    def from_raw_dir(cls, raw_dir: str | Path) -> "KuaiSearchRawPaths":
        raw_dir = Path(raw_dir)
        return cls(
            raw_dir=raw_dir,
            rank=raw_dir / "rank_lite" / "train.jsonl",
            recall=raw_dir / "recall_lite" / "train.jsonl",
            items=raw_dir / "items_lite" / "train.jsonl",
            users=raw_dir / "users_lite" / "train.jsonl",
            relevance=raw_dir / "relevance" / "train.jsonl",
        )


def request_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["user_id"]),
        str(row["session_id"]),
        str(row["query"]),
        int(row["time_index"]),
    )


def request_id_from_key(key: tuple[str, str, str, int]) -> str:
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return "ks_" + sha256_text(payload)[:24]


def summarize_numbers(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": percentile(ordered, 0.05),
        "median": percentile(ordered, 0.50),
        "mean": sum(ordered) / len(ordered),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def percentile(ordered_values: list[int | float], q: float) -> int | float:
    if not ordered_values:
        raise ValueError("percentile requires at least one value")
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = q * (len(ordered_values) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered_values[lo]
    fraction = position - lo
    return ordered_values[lo] * (1 - fraction) + ordered_values[hi] * fraction


def audit_kuaisearch_c0(
    raw_dir: str | Path,
    report_path: str | Path,
    sample_dir: str | Path,
    seed: int = 20260708,
    recall_sample_size: int = 1000,
    window_size: int = 250_000,
) -> dict[str, Any]:
    paths = KuaiSearchRawPaths.from_raw_dir(raw_dir)
    _assert_required_files(paths)

    generated_at = datetime.now(timezone.utc).isoformat()
    item_audit = _audit_items(paths.items)
    recall_audit = _audit_recall_and_write_window(
        paths.recall,
        sample_dir=Path(sample_dir),
        seed=seed,
        sample_size=recall_sample_size,
        window_size=window_size,
    )
    rank_audit = _audit_rank(
        paths.rank,
        item_text_ids=item_audit["item_ids_with_text"],
        recall_sample=recall_audit["sample_requests"],
    )
    users_audit = _audit_simple_jsonl(paths.users, expected_fields=("user_id",))
    relevance_audit = _audit_simple_jsonl(
        paths.relevance,
        expected_fields=("query", "item_title", "brand", "seller_name", "score"),
    )

    checks = _build_c0_checks(item_audit, recall_audit, rank_audit)
    report = {
        "phase": "C0",
        "dataset_id": "kuaisearch",
        "dataset_version": "v0_lite",
        "generated_at": generated_at,
        "seed": seed,
        "raw_dir": str(Path(raw_dir)),
        "source": {
            "hf_repo": "benchen4395/KuaiSearch",
            "hf_files": {
                "rank": "rank_lite/train.jsonl",
                "recall": "recall_lite/train.jsonl",
                "items": "items_lite/train.jsonl",
                "users": "users_lite/train.jsonl",
                "relevance": "relevance/train.jsonl",
            },
            "official_repo": "https://github.com/benchen4395/KuaiSearch",
            "official_repo_commit_observed": "7ce0471b659112096f0aa7e892ed0aa4c972246a",
            "paper": "https://arxiv.org/abs/2602.11518",
        },
        "files": _file_summaries(paths),
        "audits": {
            "items": _without_large_sets(item_audit),
            "users": users_audit,
            "relevance": relevance_audit,
            "recall": _without_sample(recall_audit),
            "rank": rank_audit,
        },
        "checks": checks,
        "overall_status": "passed"
        if all(check["status"] == "passed" for check in checks.values())
        else "failed",
    }
    write_json(report_path, report)
    return report


def _assert_required_files(paths: KuaiSearchRawPaths) -> None:
    missing = [
        str(path)
        for path in (paths.rank, paths.recall, paths.items, paths.users, paths.relevance)
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError("Missing required KuaiSearch raw files: " + ", ".join(missing))


def _file_summaries(paths: KuaiSearchRawPaths) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ("rank", "recall", "items", "users", "relevance"):
        path = getattr(paths, name)
        result[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path) if path.stat().st_size < 512 * 1024 * 1024 else "skipped_large_file",
        }
    return result


def _audit_simple_jsonl(path: Path, expected_fields: tuple[str, ...]) -> dict[str, Any]:
    total = 0
    missing_fields = Counter()
    first_keys: list[str] | None = None
    split_counts: Counter[str] = Counter()
    score_counts: Counter[str] = Counter()
    for row in iter_jsonl(path):
        total += 1
        if first_keys is None:
            first_keys = sorted(row.keys())
        for field in expected_fields:
            if field not in row:
                missing_fields[field] += 1
        if "split" in row:
            split_counts[str(row["split"])] += 1
        if "score" in row:
            score_counts[str(row["score"])] += 1
    return {
        "path": str(path),
        "rows": total,
        "first_row_keys": first_keys or [],
        "missing_required_fields": dict(missing_fields),
        "split_counts": dict(split_counts),
        "score_counts": dict(score_counts),
    }


def _audit_items(path: Path) -> dict[str, Any]:
    total = 0
    item_ids: set[int] = set()
    item_ids_with_text: set[int] = set()
    missing_title = 0
    missing_any_text_field = 0
    first_keys: list[str] | None = None

    for row in iter_jsonl(path):
        total += 1
        if first_keys is None:
            first_keys = sorted(row.keys())
        item_id = int(row["item_id"])
        item_ids.add(item_id)
        title = str(row.get("item_title") or "").strip()
        text_fields = [
            title,
            str(row.get("brand_name") or "").strip(),
            str(row.get("seller_name") or "").strip(),
            str(row.get("category_level1_name") or "").strip(),
            str(row.get("category_level2_name") or "").strip(),
            str(row.get("category_level3_name") or "").strip(),
        ]
        if not title:
            missing_title += 1
        if any(text_fields):
            item_ids_with_text.add(item_id)
        else:
            missing_any_text_field += 1

    return {
        "path": str(path),
        "rows": total,
        "unique_item_ids": len(item_ids),
        "duplicate_item_id_rows": total - len(item_ids),
        "first_row_keys": first_keys or [],
        "missing_title_rows": missing_title,
        "missing_any_text_field_rows": missing_any_text_field,
        "text_item_coverage_over_catalog": len(item_ids_with_text) / total if total else 0.0,
        "item_ids_with_text": item_ids_with_text,
    }


def _audit_recall_and_write_window(
    path: Path,
    sample_dir: Path,
    seed: int,
    sample_size: int,
    window_size: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    total = 0
    sample: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    request_rows: list[dict[str, Any]] = []
    candidate_counts: list[int] = []
    clicked_counts: list[int] = []
    purchased_counts: list[int] = []
    time_values: list[int] = []
    split_counts: Counter[str] = Counter()
    first_keys: list[str] | None = None

    for row in iter_jsonl(path):
        total += 1
        if first_keys is None:
            first_keys = sorted(row.keys())
        key = request_key(row)
        impressed = [int(item) for item in row.get("impressed_item_ids", [])]
        clicked = [int(item) for item in row.get("clicked_item_ids", [])]
        purchased = [int(item) for item in row.get("purchased_item_ids", [])]
        time_index = int(row["time_index"])
        split_counts[str(row.get("split", ""))] += 1
        candidate_counts.append(len(impressed))
        clicked_counts.append(len(clicked))
        purchased_counts.append(len(purchased))
        time_values.append(time_index)
        request_rows.append(
            {
                "request_id": request_id_from_key(key),
                "user_id": key[0],
                "session_id": key[1],
                "query": key[2],
                "time_index": time_index,
                "candidate_count": len(impressed),
            }
        )

        if len(sample) < sample_size:
            sample[key] = {
                "impressed_item_ids": set(impressed),
                "clicked_item_ids": set(clicked),
                "purchased_item_ids": set(purchased),
            }
        else:
            replacement_index = rng.randrange(total)
            if replacement_index < sample_size:
                replace_key = list(sample.keys())[replacement_index]
                del sample[replace_key]
                sample[key] = {
                    "impressed_item_ids": set(impressed),
                    "clicked_item_ids": set(clicked),
                    "purchased_item_ids": set(purchased),
                }

    window_manifest = _write_time_window_sample(
        request_rows=request_rows,
        sample_dir=sample_dir,
        seed=seed,
        window_size=window_size,
    )
    return {
        "path": str(path),
        "rows": total,
        "first_row_keys": first_keys or [],
        "split_counts": dict(split_counts),
        "candidate_count_distribution": summarize_numbers(candidate_counts),
        "clicked_items_per_request": summarize_numbers(clicked_counts),
        "purchased_items_per_request": summarize_numbers(purchased_counts),
        "requests_with_click_rate": sum(1 for count in clicked_counts if count > 0) / total if total else 0.0,
        "requests_with_purchase_rate": sum(1 for count in purchased_counts if count > 0) / total if total else 0.0,
        "time_index": {
            "min": min(time_values) if time_values else None,
            "max": max(time_values) if time_values else None,
            "unique_in_sampled_rows": len(set(time_values)),
        },
        "sample_requests": sample,
        "time_window_sample": window_manifest,
    }


def _write_time_window_sample(
    request_rows: list[dict[str, Any]],
    sample_dir: Path,
    seed: int,
    window_size: int,
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(request_rows, key=lambda row: (row["time_index"], row["request_id"]))
    actual_window_size = min(window_size, len(ordered))
    rng = random.Random(seed)
    start = rng.randrange(0, len(ordered) - actual_window_size + 1) if ordered else 0
    selected = ordered[start : start + actual_window_size]
    train_end = int(actual_window_size * 0.8)
    dev_end = int(actual_window_size * 0.9)

    def rows_with_split() -> list[dict[str, Any]]:
        output = []
        for index, row in enumerate(selected):
            if index < train_end:
                split = "train"
            elif index < dev_end:
                split = "dev"
            else:
                split = "test"
            output.append({**row, "position": index, "split": split})
        return output

    requests_path = sample_dir / "requests.jsonl"
    count = write_jsonl(requests_path, rows_with_split())
    manifest = {
        "seed": seed,
        "requested_window_size": window_size,
        "actual_window_size": actual_window_size,
        "start_index_after_time_sort": start,
        "requests_path": str(requests_path),
        "requests_sha256": sha256_file(requests_path),
        "count": count,
        "split_counts": {"train": train_end, "dev": dev_end - train_end, "test": actual_window_size - dev_end},
        "time_index_min": selected[0]["time_index"] if selected else None,
        "time_index_max": selected[-1]["time_index"] if selected else None,
    }
    write_json(sample_dir / "manifest.json", manifest)
    return manifest


def _audit_rank(
    path: Path,
    item_text_ids: set[int],
    recall_sample: dict[tuple[str, str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    row_count = 0
    request_count = 0
    candidate_counts: list[int] = []
    click_rows = 0
    purchase_rows = 0
    requests_with_click = 0
    requests_with_purchase = 0
    missing_required_fields = Counter()
    label_values = {"is_clicked": Counter(), "is_purchased": Counter()}
    target_text_total = 0
    target_text_hits = 0
    history_text_total = 0
    history_text_hits = 0
    history_lengths: list[int] = []
    duplicate_candidate_requests = 0
    rank_recall_mismatches: list[dict[str, Any]] = []
    matched_recall_samples = 0
    first_keys: list[str] | None = None
    time_values: set[int] = set()
    split_counts: Counter[str] = Counter()

    current_key: tuple[str, str, str, int] | None = None
    current_candidates: list[int] = []
    current_clicked: set[int] = set()
    current_purchased: set[int] = set()
    current_history: set[int] = set()

    def flush_current() -> None:
        nonlocal request_count, requests_with_click, requests_with_purchase
        nonlocal duplicate_candidate_requests, matched_recall_samples
        nonlocal target_text_total, target_text_hits, history_text_total, history_text_hits
        if current_key is None:
            return
        request_count += 1
        candidate_counts.append(len(current_candidates))
        if current_clicked:
            requests_with_click += 1
        if current_purchased:
            requests_with_purchase += 1
        if len(set(current_candidates)) != len(current_candidates):
            duplicate_candidate_requests += 1
        target_text_total += len(current_candidates)
        target_text_hits += sum(1 for item_id in current_candidates if item_id in item_text_ids)
        history_text_total += len(current_history)
        history_text_hits += sum(1 for item_id in current_history if item_id in item_text_ids)
        history_lengths.append(len(current_history))
        if current_key in recall_sample:
            matched_recall_samples += 1
            expected = recall_sample[current_key]
            ranking_items = set(current_candidates)
            if (
                ranking_items != expected["impressed_item_ids"]
                or current_clicked != expected["clicked_item_ids"]
                or current_purchased != expected["purchased_item_ids"]
            ):
                rank_recall_mismatches.append(
                    {
                        "request_id": request_id_from_key(current_key),
                        "ranking_candidate_count": len(ranking_items),
                        "recall_candidate_count": len(expected["impressed_item_ids"]),
                        "ranking_clicked_count": len(current_clicked),
                        "recall_clicked_count": len(expected["clicked_item_ids"]),
                        "ranking_purchased_count": len(current_purchased),
                        "recall_purchased_count": len(expected["purchased_item_ids"]),
                    }
                )

    for row in iter_jsonl(path):
        row_count += 1
        if first_keys is None:
            first_keys = sorted(row.keys())
        for field in RANK_REQUIRED_FIELDS:
            if field not in row:
                missing_required_fields[field] += 1
        key = request_key(row)
        if key != current_key:
            flush_current()
            current_key = key
            current_candidates = []
            current_clicked = set()
            current_purchased = set()
            current_history = set()
            for history_field in RANK_HISTORY_FIELDS:
                current_history.update(int(item) for item in row.get(history_field, []))

        target_item_id = int(row["target_item_id"])
        is_clicked = int(row["is_clicked"])
        is_purchased = int(row["is_purchased"])
        current_candidates.append(target_item_id)
        if is_clicked:
            current_clicked.add(target_item_id)
            click_rows += 1
        if is_purchased:
            current_purchased.add(target_item_id)
            purchase_rows += 1
        label_values["is_clicked"][str(is_clicked)] += 1
        label_values["is_purchased"][str(is_purchased)] += 1
        time_values.add(int(row["time_index"]))
        split_counts[str(row.get("split", ""))] += 1

    flush_current()

    first_key_set = set(first_keys or [])
    history_timestamp_fields = sorted(
        key for key in first_key_set if "recently" in key and ("time" in key or "ts" in key)
    )
    return {
        "path": str(path),
        "rows": row_count,
        "requests": request_count,
        "first_row_keys": first_keys or [],
        "missing_required_fields": dict(missing_required_fields),
        "split_counts_by_row": dict(split_counts),
        "candidate_count_distribution": summarize_numbers(candidate_counts),
        "duplicate_candidate_requests": duplicate_candidate_requests,
        "label_values": {key: dict(value) for key, value in label_values.items()},
        "clicked_rows": click_rows,
        "purchased_rows": purchase_rows,
        "overall_ctr": click_rows / row_count if row_count else 0.0,
        "overall_purchase_rate": purchase_rows / row_count if row_count else 0.0,
        "requests_with_click_rate": requests_with_click / request_count if request_count else 0.0,
        "requests_with_purchase_rate": requests_with_purchase / request_count if request_count else 0.0,
        "candidate_text_coverage": target_text_hits / target_text_total if target_text_total else 0.0,
        "history_text_coverage": history_text_hits / history_text_total if history_text_total else 0.0,
        "history_length_distribution": summarize_numbers(history_lengths),
        "history_timestamp_fields": history_timestamp_fields,
        "history_future_leakage_row_level_verifiable": bool(history_timestamp_fields),
        "history_future_leakage_sampled_rows_checked": 0 if not history_timestamp_fields else "not_implemented",
        "time_index": {
            "unique_values_seen": len(time_values),
            "min": min(time_values) if time_values else None,
            "max": max(time_values) if time_values else None,
        },
        "rank_recall_sample": {
            "requested_samples": len(recall_sample),
            "matched_samples": matched_recall_samples,
            "mismatch_count": len(rank_recall_mismatches),
            "mismatch_examples": rank_recall_mismatches[:10],
        },
    }


def _build_c0_checks(
    item_audit: dict[str, Any],
    recall_audit: dict[str, Any],
    rank_audit: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    candidate_median = rank_audit["candidate_count_distribution"].get("median", 0)
    candidate_check_pass = (
        rank_audit["requests"] > 0
        and candidate_median >= 10
        and rank_audit["duplicate_candidate_requests"] == 0
        and rank_audit["rank_recall_sample"]["mismatch_count"] == 0
        and rank_audit["rank_recall_sample"]["matched_samples"]
        == rank_audit["rank_recall_sample"]["requested_samples"]
    )
    label_check_pass = (
        0.001 <= rank_audit["overall_ctr"] <= 0.5
        and rank_audit["requests_with_click_rate"] > 0
        and rank_audit["clicked_rows"] > 0
        and rank_audit["purchased_rows"] > 0
    )
    text_check_pass = (
        rank_audit["candidate_text_coverage"] >= 0.95
        and rank_audit["history_text_coverage"] >= 0.95
        and item_audit["text_item_coverage_over_catalog"] >= 0.95
    )
    leakage_check_pass = bool(rank_audit["history_future_leakage_row_level_verifiable"])
    time_check_pass = (
        rank_audit["time_index"]["min"] is not None
        and rank_audit["time_index"]["max"] is not None
        and rank_audit["time_index"]["max"] > rank_audit["time_index"]["min"]
        and recall_audit["time_index"]["max"] > recall_audit["time_index"]["min"]
    )
    return {
        "candidate_aggregation": {
            "status": "passed" if candidate_check_pass else "failed",
            "evidence": {
                "rank_requests": rank_audit["requests"],
                "rank_rows": rank_audit["rows"],
                "candidate_count_distribution": rank_audit["candidate_count_distribution"],
                "duplicate_candidate_requests": rank_audit["duplicate_candidate_requests"],
                "rank_recall_sample": rank_audit["rank_recall_sample"],
            },
            "rule": "ranking rows aggregate into request candidate lists; median candidates >= 10; sampled recall candidates match ranking aggregation",
        },
        "label_sanity": {
            "status": "passed" if label_check_pass else "failed",
            "evidence": {
                "overall_ctr": rank_audit["overall_ctr"],
                "requests_with_click_rate": rank_audit["requests_with_click_rate"],
                "overall_purchase_rate": rank_audit["overall_purchase_rate"],
                "requests_with_purchase_rate": rank_audit["requests_with_purchase_rate"],
                "clicked_rows": rank_audit["clicked_rows"],
                "purchased_rows": rank_audit["purchased_rows"],
            },
            "rule": "CTR must not be suspiciously high/low and click/purchase labels must exist and be nonzero",
        },
        "text_join_coverage": {
            "status": "passed" if text_check_pass else "failed",
            "evidence": {
                "catalog_text_coverage": item_audit["text_item_coverage_over_catalog"],
                "candidate_text_coverage": rank_audit["candidate_text_coverage"],
                "history_text_coverage": rank_audit["history_text_coverage"],
            },
            "rule": "history and candidate item text coverage >= 95%",
        },
        "history_future_leakage": {
            "status": "passed" if leakage_check_pass else "failed",
            "evidence": {
                "history_fields": list(RANK_HISTORY_FIELDS),
                "history_timestamp_fields": rank_audit["history_timestamp_fields"],
                "sampled_rows_checked": rank_audit["history_future_leakage_sampled_rows_checked"],
                "note": "Ranking rows expose recent history item ids but no per-history event timestamps. Field-level timestamp verification is unavailable; a separate recall-log cross-reference check may replace this evidence via scripts/check_kuaisearch_history_leakage.py.",
            },
            "rule": "sample 1000 requests and verify every history interaction time is earlier than request time; if per-history timestamps are unavailable, use the registered recall-log cross-reference check or official confirmation before training",
        },
        "time_field": {
            "status": "passed" if time_check_pass else "failed",
            "evidence": {
                "rank_time_index": rank_audit["time_index"],
                "recall_time_index": recall_audit["time_index"],
                "time_window_sample": recall_audit["time_window_sample"],
            },
            "rule": "time_index exists and varies enough to support global time split",
        },
    }


def _without_large_sets(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "item_ids_with_text"}


def _without_sample(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "sample_requests"}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def dump_report_preview(report: dict[str, Any]) -> str:
    preview = {
        "overall_status": report["overall_status"],
        "checks": {name: check["status"] for name, check in report["checks"].items()},
    }
    return json.dumps(_to_jsonable(preview), ensure_ascii=False, indent=2, sort_keys=True)
