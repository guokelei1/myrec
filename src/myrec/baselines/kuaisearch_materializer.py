"""Materialize KuaiSearch public raw files into the official ranking format."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


AGE_BUCKET_TO_OFFICIAL_AGE = {
    "0-11": "0-11",
    "12-17": "12-17",
    "18-23": "18-23",
    "24-30": "24-30",
    "31-40": "31-40",
    "41-49": "41-49",
    "50+": "50+",
}
DEFAULT_GENDER = "M"
DEFAULT_AGE = "31-40"
OFFICIAL_BGE_ENCODER = "BAAI/bge-small-zh-v1.5"


def materialize_official_ranking_format(
    raw_dir: str | Path,
    output_root: str | Path,
    *,
    max_rank_rows: int | None = None,
    split_policy: str = "last_time_fraction",
    test_fraction: float = 0.10,
    test_time_min: int | None = None,
    min_target_coverage: float = 0.999,
) -> dict[str, Any]:
    """Write ``data/rank.jsonl``, ``data/corpus.jsonl``, and ``data/users.jsonl``.

    The official ranking loader at the locked KuaiSearch commit reads JSONL
    files from ``--data_dir`` but reads embeddings from ``./data``. This
    materializer therefore writes the official JSONL files under
    ``output_root/data`` and records that embedding generation must place or
    move ``query_emb.npy``, ``session_id2idx.json``, ``item_title_emb.npy``, and
    ``item_id2idx.json`` into the same directory before official training.

    ``last_time_fraction`` is an explicit, auditable proxy for the paper's
    last-day split when the public raw file contains only ``split=train`` rows:
    rows at or above the time-index threshold that yields the latest
    ``test_fraction`` of materialized ranking rows are marked ``split=test``.
    Ties at the threshold are included in test, so the actual test fraction may
    exceed the requested value. If the exact day boundary is later recovered,
    use ``last_time_cutoff`` with that ``test_time_min`` instead.
    """

    raw_dir = Path(raw_dir)
    output_root = Path(output_root)
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "rank": raw_dir / "rank_lite" / "train.jsonl",
        "items": raw_dir / "items_lite" / "train.jsonl",
        "users": raw_dir / "users_lite" / "train.jsonl",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing KuaiSearch raw {name} file: {path}")

    scan = _scan_rank_rows(
        paths["rank"],
        max_rank_rows=max_rank_rows,
        split_policy=split_policy,
        test_fraction=test_fraction,
        test_time_min=test_time_min,
    )
    loaded_item_ids, corpus_rows = _write_needed_corpus(
        paths["items"],
        data_dir / "corpus.jsonl",
        scan["needed_item_ids"],
    )
    user_map, user_stats = _load_needed_users(paths["users"], scan["needed_user_ids"])
    _fill_missing_users(user_map, scan["needed_user_ids"], user_stats)

    target_items = scan["target_item_ids"]
    covered_targets = target_items & loaded_item_ids
    target_coverage = len(covered_targets) / len(target_items) if target_items else 1.0
    status = "passed" if target_coverage >= min_target_coverage else "failed"

    users_rows = _write_users(data_dir / "users.jsonl", user_map)
    rank_stats = _write_rank(
        paths["rank"],
        data_dir / "rank.jsonl",
        loaded_item_ids=loaded_item_ids,
        max_rank_rows=max_rank_rows,
        split_threshold=scan["split_threshold"],
        split_policy=split_policy,
    )
    actual_test_fraction = (
        rank_stats["split_counts"].get("test", 0) / rank_stats["rows"]
        if rank_stats["rows"]
        else 0.0
    )

    manifest = {
        "status": status,
        "raw_dir": str(raw_dir),
        "output_root": str(output_root),
        "official_commit": "7ce0471b659112096f0aa7e892ed0aa4c972246a",
        "encoder_policy": {
            "model": OFFICIAL_BGE_ENCODER,
            "reason": "locked official ranking/data/process.py uses this BGE model; the paper BERT wording is recorded as a known protocol difference",
        },
        "split": {
            "policy": split_policy,
            "test_fraction": test_fraction,
            "actual_test_fraction": actual_test_fraction,
            "test_time_min": scan["split_threshold"],
            "tie_handling": "rows with time_index >= test_time_min are test; threshold ties are included in test",
            "source": (
                "public rank_lite/train.jsonl has no test rows; this is an explicit "
                "last-time proxy pending an upstream-confirmed last-day boundary"
            ),
        },
        "age_mapping": {
            "source_field": "age_bucket",
            "target_field": "age",
            "mapping": AGE_BUCKET_TO_OFFICIAL_AGE,
            "default_age": DEFAULT_AGE,
            "default_gender": DEFAULT_GENDER,
        },
        "target_item_coverage": {
            "unique_target_items": len(target_items),
            "covered_unique_target_items": len(covered_targets),
            "coverage_rate": target_coverage,
            "min_required_to_continue": min_target_coverage,
            "missing_target_examples": [str(value) for value in sorted(target_items - covered_targets)[:20]],
        },
        "counts": {
            "rank_rows_scanned": scan["rank_rows_scanned"],
            "rank_rows_written": rank_stats["rows"],
            "corpus_rows_written": corpus_rows,
            "users_rows_written": users_rows,
            "unique_needed_items": len(scan["needed_item_ids"]),
            "loaded_needed_items": len(loaded_item_ids),
            "unique_needed_users": len(scan["needed_user_ids"]),
            "loaded_or_synthetic_users": len(user_map),
        },
        "rank": rank_stats,
        "users": user_stats,
        "embedding_placement": {
            "official_process_script": "baselines/kuaisearch_official/ranking/data/process.py",
            "loader_reads_from": "./data",
            "required_files": [
                "data/query_emb.npy",
                "data/session_id2idx.json",
                "data/item_title_emb.npy",
                "data/item_id2idx.json",
            ],
            "decision": "place or move official process outputs into output_root/data; no source patch required for the path mismatch",
        },
        "files": {
            "rank": _file_info(data_dir / "rank.jsonl"),
            "corpus": _file_info(data_dir / "corpus.jsonl"),
            "users": _file_info(data_dir / "users.jsonl"),
        },
    }
    manifest_path = output_root / "materializer_manifest.json"
    write_json(manifest_path, manifest)
    manifest["files"]["manifest"] = _file_info(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def _scan_rank_rows(
    rank_path: Path,
    *,
    max_rank_rows: int | None,
    split_policy: str,
    test_fraction: float,
    test_time_min: int | None,
) -> dict[str, Any]:
    if split_policy not in {"last_time_fraction", "last_time_cutoff"}:
        raise ValueError("split_policy must be last_time_fraction or last_time_cutoff")
    if split_policy == "last_time_cutoff" and test_time_min is None:
        raise ValueError("last_time_cutoff requires test_time_min")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")

    needed_item_ids: set[int] = set()
    target_item_ids: set[int] = set()
    needed_user_ids: set[int] = set()
    times: list[int] = []
    rows = 0
    for row in iter_jsonl(rank_path):
        rows += 1
        uid = int(row["user_id"])
        target = int(row["target_item_id"])
        needed_user_ids.add(uid)
        target_item_ids.add(target)
        needed_item_ids.add(target)
        for item_id in row.get("recently_clicked_item_ids", []) or []:
            needed_item_ids.add(int(item_id))
        time_index = int(row["time_index"])
        times.append(time_index)
        if max_rank_rows is not None and rows >= max_rank_rows:
            break

    if split_policy == "last_time_cutoff":
        threshold = int(test_time_min)  # type: ignore[arg-type]
    else:
        ordered = sorted(times)
        first_test_index = max(0, int(len(ordered) * (1.0 - test_fraction)))
        threshold = ordered[min(first_test_index, len(ordered) - 1)] if ordered else 0

    return {
        "rank_rows_scanned": rows,
        "needed_item_ids": needed_item_ids,
        "target_item_ids": target_item_ids,
        "needed_user_ids": needed_user_ids,
        "split_threshold": threshold,
    }


def _official_item_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": int(row["item_id"]),
        "item_title": str(row.get("item_title") or ""),
        "brand_id": int(row.get("brand_id") or 0),
        "brand_name": str(row.get("brand_name") or ""),
        "seller_id": int(row.get("seller_id") or 0),
        "seller_name": str(row.get("seller_name") or ""),
        "category_level1_id": int(row.get("category_level1_id") or 0),
        "category_level1_name": str(row.get("category_level1_name") or ""),
        "category_level2_id": int(row.get("category_level2_id") or 0),
        "category_level2_name": str(row.get("category_level2_name") or ""),
        "category_level3_id": int(row.get("category_level3_id") or 0),
        "category_level3_name": str(row.get("category_level3_name") or ""),
    }


def _write_needed_corpus(items_path: Path, output_path: Path, needed_item_ids: set[int]) -> tuple[set[int], int]:
    loaded_item_ids: set[int] = set()
    rows = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(items_path):
            item_id = int(row["item_id"])
            if item_id not in needed_item_ids:
                continue
            handle.write(json.dumps(_official_item_row(row), ensure_ascii=False, sort_keys=True) + "\n")
            loaded_item_ids.add(item_id)
            rows += 1
            if rows == len(needed_item_ids):
                break
    return loaded_item_ids, rows


def _load_needed_users(users_path: Path, needed_user_ids: set[int]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    user_map: dict[int, dict[str, Any]] = {}
    stats = {
        "source_rows_read": 0,
        "age_bucket_to_age": Counter(),
        "invalid_gender_coerced": 0,
        "invalid_age_coerced": 0,
        "synthetic_missing_users": 0,
    }
    for row in iter_jsonl(users_path):
        stats["source_rows_read"] += 1
        user_id = int(row["user_id"])
        if user_id not in needed_user_ids:
            continue
        gender = row.get("gender")
        if gender not in {"M", "F"}:
            gender = DEFAULT_GENDER
            stats["invalid_gender_coerced"] += 1
        age_bucket = str(row.get("age_bucket") or "")
        age = AGE_BUCKET_TO_OFFICIAL_AGE.get(age_bucket)
        if age is None:
            age = DEFAULT_AGE
            stats["invalid_age_coerced"] += 1
        stats["age_bucket_to_age"][f"{age_bucket}->{age}"] += 1
        user_map[user_id] = {
            **row,
            "user_id": user_id,
            "gender": gender,
            "age": age,
            "age_bucket": age_bucket,
        }
        if len(user_map) == len(needed_user_ids):
            break
    stats["age_bucket_to_age"] = dict(stats["age_bucket_to_age"])
    return user_map, stats


def _fill_missing_users(user_map: dict[int, dict[str, Any]], needed_user_ids: set[int], stats: dict[str, Any]) -> None:
    for user_id in sorted(needed_user_ids - set(user_map)):
        user_map[user_id] = {
            "user_id": user_id,
            "gender": DEFAULT_GENDER,
            "age": DEFAULT_AGE,
            "age_bucket": DEFAULT_AGE,
            "synthetic_missing_user": True,
        }
        stats["synthetic_missing_users"] += 1


def _write_users(path: Path, user_map: dict[int, dict[str, Any]]) -> int:
    rows = 0
    with path.open("w", encoding="utf-8") as handle:
        for user_id in sorted(user_map):
            handle.write(json.dumps(user_map[user_id], ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
    return rows


def _write_rank(
    rank_path: Path,
    output_path: Path,
    *,
    loaded_item_ids: set[int],
    max_rank_rows: int | None,
    split_threshold: int,
    split_policy: str,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": 0,
        "split_counts": dict(Counter()),
        "label_counts": dict(Counter()),
        "missing_target_rows": 0,
        "missing_target_examples": [],
    }
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    missing_examples: list[int] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(rank_path):
            target = int(row["target_item_id"])
            if target not in loaded_item_ids:
                stats["missing_target_rows"] += 1
                if len(missing_examples) < 20:
                    missing_examples.append(target)
            split = _assign_split(row, split_threshold=split_threshold, split_policy=split_policy)
            out = dict(row)
            out["target_item_id"] = target
            out["user_id"] = int(out["user_id"])
            out["session_id"] = str(out["session_id"])
            out["split"] = split
            handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            stats["rows"] += 1
            split_counts[split] += 1
            label = 1 if int(row.get("is_clicked", 0) or 0) == 1 or int(row.get("is_purchased", 0) or 0) == 1 else 0
            label_counts[str(label)] += 1
            if max_rank_rows is not None and stats["rows"] >= max_rank_rows:
                break
    stats["split_counts"] = dict(split_counts)
    stats["label_counts"] = dict(label_counts)
    stats["missing_target_examples"] = missing_examples
    return stats


def _assign_split(row: dict[str, Any], *, split_threshold: int, split_policy: str) -> str:
    if split_policy in {"last_time_fraction", "last_time_cutoff"}:
        return "test" if int(row["time_index"]) >= split_threshold else "train"
    return str(row.get("split") or "train")


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/kuaisearch")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-rank-rows", type=int)
    parser.add_argument("--split-policy", choices=["last_time_fraction", "last_time_cutoff"], default="last_time_fraction")
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--test-time-min", type=int)
    parser.add_argument("--min-target-coverage", type=float, default=0.999)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize_official_ranking_format(
        raw_dir=args.raw_dir,
        output_root=args.output_root,
        max_rank_rows=args.max_rank_rows,
        split_policy=args.split_policy,
        test_fraction=args.test_fraction,
        test_time_min=args.test_time_min,
        min_target_coverage=args.min_target_coverage,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
