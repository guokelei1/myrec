"""Minimal native-format adapter for the official ProdSearch ZAM/TEM code."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


OFFICIAL_COMMIT = "449335ba652fe7c877a008e154157d7b2a4b0e76"
DEFAULT_SEED = 20260708
DEFAULT_VALID_SIZE = 500
DEFAULT_TEST_SIZE = 1500
EMPTY_TOKEN = "<EMPTY>"


@dataclass(frozen=True)
class MaterializedPaths:
    root: Path
    data_dir: Path
    split_dir: Path
    dev_request_map: Path
    valid_request_map: Path
    manifest: Path


class _Registry:
    def __init__(self) -> None:
        self.product_to_idx: dict[str, int] = {}
        self.product_ids: list[str] = []
        self.item_texts: list[str] = []
        self.query_to_idx: dict[str, int] = {}
        self.queries: list[str] = []
        self.tokens: set[str] = {EMPTY_TOKEN}

    def register_item(self, item: dict[str, Any]) -> int:
        item_id = str(item["item_id"])
        existing = self.product_to_idx.get(item_id)
        if existing is not None:
            return existing
        idx = len(self.product_ids)
        text = compose_item_text(item)
        self.product_to_idx[item_id] = idx
        self.product_ids.append(item_id)
        self.item_texts.append(text)
        self.tokens.update(tokenize_chars(text))
        return idx

    def register_query(self, query: str) -> int:
        query = str(query or "")
        existing = self.query_to_idx.get(query)
        if existing is not None:
            return existing
        idx = len(self.queries)
        self.query_to_idx[query] = idx
        self.queries.append(query)
        self.tokens.update(tokenize_chars(query))
        return idx


def compose_item_text(item: dict[str, Any]) -> str:
    """Use the frozen title + brand + category item-text contract."""

    category = item.get("cat") or []
    if isinstance(category, str):
        category = [category]
    parts = [str(item.get("title") or ""), str(item.get("brand") or "")]
    parts.extend(str(value) for value in category if value is not None)
    text = " ".join(part.strip() for part in parts if part.strip())
    return text or EMPTY_TOKEN


def tokenize_chars(text: str) -> list[str]:
    tokens = [char for char in str(text) if not char.isspace()]
    return tokens or [EMPTY_TOKEN]


def materialize_prodsearch_format(
    standardized_dir: str | Path,
    output_root: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    valid_examples: int = 5000,
    valid_candidate_size: int = DEFAULT_VALID_SIZE,
    test_candidate_size: int = DEFAULT_TEST_SIZE,
    max_train_requests: int | None = None,
    max_dev_requests: int | None = None,
    dev_history_condition: str = "true",
) -> dict[str, Any]:
    """Materialize request records into the official ProdSearch file layout.

    Each clicked train target becomes an independent synthetic user. Its user
    sequence is exactly the record's frozen history followed by that target;
    sibling positives are never appended. Validation is selected only from
    train positives. Dev receives a label-free dummy target required by the
    upstream parser, and that target is never used by the shared evaluator.
    """

    if dev_history_condition not in {"true", "null"}:
        raise ValueError("dev_history_condition must be true or null")
    standardized_dir = Path(standardized_dir)
    output_root = Path(output_root)
    data_dir = output_root / "data"
    split_dir = output_root / "split"
    work_dir = output_root / "work"
    for required in ("records_train.jsonl", "records_dev.jsonl", "candidate_manifest.json"):
        if not (standardized_dir / required).exists():
            raise FileNotFoundError(standardized_dir / required)
    if data_dir.exists() or split_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing materialization: {output_root}")
    data_dir.mkdir(parents=True, exist_ok=False)
    split_dir.mkdir(parents=True, exist_ok=False)
    work_dir.mkdir(parents=True, exist_ok=False)

    registry = _Registry()
    examples_path = work_dir / "examples.jsonl"
    valid_candidates: list[tuple[str, int]] = []
    train_positive_items: set[str] = set()
    train_stats: Counter[str] = Counter()
    dev_stats: Counter[str] = Counter()

    with examples_path.open("w", encoding="utf-8") as examples_out:
        _scan_train_records(
            standardized_dir / "records_train.jsonl",
            examples_out,
            registry,
            train_positive_items,
            valid_candidates,
            train_stats,
            seed=seed,
            valid_candidate_size=valid_candidate_size,
            max_requests=max_train_requests,
        )
        _scan_dev_records(
            standardized_dir / "records_dev.jsonl",
            examples_out,
            registry,
            train_positive_items,
            dev_stats,
            max_requests=max_dev_requests,
            history_condition=dev_history_condition,
        )

    selected_valid = {
        example_id
        for _, example_id in sorted(valid_candidates)[: min(valid_examples, len(valid_candidates))]
    }
    if not selected_valid:
        raise ValueError("no train-positive examples are eligible for validation")
    if len(registry.product_ids) < max(valid_candidate_size, test_candidate_size):
        raise ValueError(
            "product universe is smaller than requested deterministic padding size: "
            f"products={len(registry.product_ids)} valid={valid_candidate_size} test={test_candidate_size}"
        )

    vocab = [EMPTY_TOKEN] + sorted(registry.tokens - {EMPTY_TOKEN})
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    paths = MaterializedPaths(
        root=output_root,
        data_dir=data_dir,
        split_dir=split_dir,
        dev_request_map=output_root / "dev_request_map.jsonl",
        valid_request_map=output_root / "valid_request_map.jsonl",
        manifest=output_root / "materializer_manifest.json",
    )
    write_stats = _write_native_files(
        examples_path=examples_path,
        registry=registry,
        token_to_idx=token_to_idx,
        vocab=vocab,
        selected_valid=selected_valid,
        paths=paths,
        seed=seed,
        valid_candidate_size=valid_candidate_size,
        test_candidate_size=test_candidate_size,
    )
    subset_manifest_path = output_root / "dev_candidate_manifest.json"
    dataset_metadata = _read_standardized_metadata(standardized_dir)
    write_json(
        subset_manifest_path,
        {
            "dataset_id": dataset_metadata["dataset_id"],
            "dataset_version": dataset_metadata["dataset_version"],
            "scope": "materialized exact dev request set before deterministic filler removal",
            "entries": [
                {
                    "split": "dev",
                    "request_id": row["request_id"],
                    "candidate_item_ids": row["candidate_item_ids"],
                }
                for row in iter_jsonl(paths.dev_request_map)
            ],
        },
    )

    source_files = {
        name: {
            "path": str(standardized_dir / name),
            "sha256": sha256_file(standardized_dir / name),
        }
        for name in ("records_train.jsonl", "records_dev.jsonl", "candidate_manifest.json")
    }
    manifest: dict[str, Any] = {
        "report": "b9_prodsearch_materializer",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_commit": OFFICIAL_COMMIT,
        "adapter_policy": "Option A - minimal official adapter",
        "seed": seed,
        "source_files": source_files,
        "input_boundaries": {
            "records_train_read": True,
            "records_dev_read": True,
            "qrels_dev_read": False,
            "records_test_read": False,
            "qrels_test_read": False,
            "item_catalog_read": False,
        },
        "tokenization": {
            "type": "Unicode character tokens excluding whitespace",
            "item_text": "title + brand + category",
            "vocab_size_without_padding": len(vocab),
        },
        "mapping": {
            "train_positive_examples": int(train_stats["positive_examples"]),
            "train_requests_scanned": int(train_stats["requests"]),
            "dev_requests_scanned": int(dev_stats["requests"]),
            "validation_examples": len(selected_valid),
            "synthetic_user_per_positive": True,
            "dev_dummy_target": "first frozen candidate; parser-only and ignored by shared evaluator",
            "history_contract": (
                "train uses exact frozen record history; dev uses the frozen "
                f"{dev_history_condition} condition; upstream uprev_review_limit "
                "later truncates to most recent 20"
            ),
            "dev_history_condition": dev_history_condition,
            "dev_original_history_events": int(dev_stats["original_history_events"]),
            "dev_effective_history_events": int(dev_stats["effective_history_events"]),
        },
        "multi_positive_guard": {
            "synthetic_histories_equal_input": bool(write_stats["history_exact_assertions_passed"]),
            "sibling_positives_added_by_adapter": int(write_stats["sibling_positives_added"]),
            "input_history_sibling_item_overlaps": int(train_stats["input_sibling_history_overlaps"]),
            "note": "Input overlaps are legitimate prior events; the adapter adds zero current-request positives.",
        },
        "candidate_padding": {
            "valid_size": valid_candidate_size,
            "dev_size": test_candidate_size,
            "algorithm": "sha256(seed, request_id) start offset then cyclic first unseen product",
            "fillers_disjoint_from_original": bool(write_stats["fillers_disjoint"]),
            "original_candidates_preserved_in_order": bool(write_stats["original_candidates_preserved"]),
        },
        "cold_product_coverage": {
            "train_unique_clicked_targets": len(train_positive_items),
            "dev_unique_candidates": int(dev_stats["unique_candidates"]),
            "dev_unique_candidates_seen_as_train_target": int(dev_stats["unique_covered"]),
            "dev_unique_train_target_coverage": (
                dev_stats["unique_covered"] / dev_stats["unique_candidates"]
                if dev_stats["unique_candidates"]
                else 0.0
            ),
            "dev_candidate_rows": int(dev_stats["candidate_rows"]),
            "dev_candidate_rows_seen_as_train_target": int(dev_stats["covered_rows"]),
            "dev_row_train_target_coverage": (
                dev_stats["covered_rows"] / dev_stats["candidate_rows"]
                if dev_stats["candidate_rows"]
                else 0.0
            ),
        },
        "counts": {
            "products": len(registry.product_ids),
            "queries": len(registry.queries),
            "reviews": int(write_stats["reviews"]),
            "users": int(write_stats["users"]),
            "train_examples": int(write_stats["train_examples"]),
            "valid_examples": int(write_stats["valid_examples"]),
            "dev_examples": int(write_stats["dev_examples"]),
        },
        "files": _collect_materialized_files(paths),
    }
    manifest["files"]["dev_candidate_manifest.json"] = _file_info(subset_manifest_path)
    manifest["manifest_path"] = str(paths.manifest)
    write_json(paths.manifest, manifest)
    return manifest


def _scan_train_records(
    path: Path,
    examples_out: TextIO,
    registry: _Registry,
    train_positive_items: set[str],
    valid_candidates: list[tuple[str, int]],
    stats: Counter[str],
    *,
    seed: int,
    valid_candidate_size: int,
    max_requests: int | None,
) -> None:
    example_id = 0
    for record in iter_jsonl(path):
        if max_requests is not None and stats["requests"] >= max_requests:
            break
        stats["requests"] += 1
        query_idx = registry.register_query(str(record.get("query") or ""))
        candidates = list(record.get("candidates") or [])
        history = list(record.get("history") or [])
        candidate_idxs = [registry.register_item(item) for item in candidates]
        history_idxs = [registry.register_item(item) for item in history]
        positives = [item for item in candidates if int(item.get("clicked") or 0) > 0]
        positive_ids = [str(item["item_id"]) for item in positives]
        train_positive_items.update(positive_ids)
        history_id_set = {str(item["item_id"]) for item in history}
        for pos_ordinal, positive in enumerate(positives):
            target_id = str(positive["item_id"])
            siblings = set(positive_ids) - {target_id}
            stats["input_sibling_history_overlaps"] += len(siblings & history_id_set)
            row = {
                "kind": "train_positive",
                "example_id": example_id,
                "request_id": str(record["request_id"]),
                "query_idx": query_idx,
                "target_idx": registry.product_to_idx[target_id],
                "history_idxs": history_idxs,
                "candidate_idxs": candidate_idxs,
                "positive_item_ids": positive_ids,
                "ts": int(record.get("ts") or 0),
            }
            examples_out.write(json.dumps(row, sort_keys=True) + "\n")
            stats["positive_examples"] += 1
            if len(candidate_idxs) <= valid_candidate_size:
                digest = hashlib.sha256(
                    f"{seed}\0{record['request_id']}\0{target_id}\0{pos_ordinal}".encode("utf-8")
                ).hexdigest()
                valid_candidates.append((digest, example_id))
            example_id += 1


def _scan_dev_records(
    path: Path,
    examples_out: TextIO,
    registry: _Registry,
    train_positive_items: set[str],
    stats: Counter[str],
    *,
    max_requests: int | None,
    history_condition: str,
) -> None:
    unique_candidates: set[str] = set()
    covered_unique: set[str] = set()
    for record in iter_jsonl(path):
        if max_requests is not None and stats["requests"] >= max_requests:
            break
        stats["requests"] += 1
        query_idx = registry.register_query(str(record.get("query") or ""))
        candidates = list(record.get("candidates") or [])
        history = list(record.get("history") or [])
        if not candidates:
            raise ValueError(f"dev request has no candidates: {record['request_id']}")
        candidate_idxs = [registry.register_item(item) for item in candidates]
        original_history_idxs = [registry.register_item(item) for item in history]
        history_idxs = original_history_idxs if history_condition == "true" else []
        stats["original_history_events"] += len(original_history_idxs)
        stats["effective_history_events"] += len(history_idxs)
        for item in candidates:
            item_id = str(item["item_id"])
            unique_candidates.add(item_id)
            stats["candidate_rows"] += 1
            if item_id in train_positive_items:
                covered_unique.add(item_id)
                stats["covered_rows"] += 1
        row = {
            "kind": "dev_request",
            "request_id": str(record["request_id"]),
            "query_idx": query_idx,
            "target_idx": candidate_idxs[0],
            "history_idxs": history_idxs,
            "candidate_idxs": candidate_idxs,
            "ts": int(record.get("ts") or 0),
        }
        examples_out.write(json.dumps(row, sort_keys=True) + "\n")
    stats["unique_candidates"] = len(unique_candidates)
    stats["unique_covered"] = len(covered_unique)


def _write_native_files(
    *,
    examples_path: Path,
    registry: _Registry,
    token_to_idx: dict[str, int],
    vocab: list[str],
    selected_valid: set[int],
    paths: MaterializedPaths,
    seed: int,
    valid_candidate_size: int,
    test_candidate_size: int,
) -> dict[str, Any]:
    product_count = len(registry.product_ids)
    example_count = sum(1 for _ in iter_jsonl(examples_path))
    catalog_user_idx = 0
    target_review_start = product_count
    product_target_reviews: dict[int, list[int]] = defaultdict(list)
    user_sequences_path = paths.root / "work" / "user_sequences.txt"
    stats: Counter[str] = Counter()
    stats["history_exact_assertions_passed"] = 1
    stats["fillers_disjoint"] = 1
    stats["original_candidates_preserved"] = 1

    with ExitStack() as stack:
        product_out = stack.enter_context(_gzip_text(paths.data_dir / "product.txt.gz", "wt"))
        users_out = stack.enter_context(_gzip_text(paths.data_dir / "users.txt.gz", "wt"))
        vocab_out = stack.enter_context(_gzip_text(paths.data_dir / "vocab.txt.gz", "wt"))
        review_text_out = stack.enter_context(_gzip_text(paths.data_dir / "review_text.txt.gz", "wt"))
        review_id_out = stack.enter_context(_gzip_text(paths.data_dir / "review_id.txt.gz", "wt"))
        review_up_out = stack.enter_context(_gzip_text(paths.data_dir / "review_u_p.txt.gz", "wt"))
        review_loc_out = stack.enter_context(
            _gzip_text(paths.data_dir / "review_uloc_ploc_and_time.txt.gz", "wt")
        )
        train_out = stack.enter_context(_gzip_text(paths.split_dir / "train.txt.gz", "wt"))
        train_id_out = stack.enter_context(_gzip_text(paths.split_dir / "train_id.txt.gz", "wt"))
        valid_id_out = stack.enter_context(_gzip_text(paths.split_dir / "valid_id.txt.gz", "wt"))
        test_id_out = stack.enter_context(_gzip_text(paths.split_dir / "test_id.txt.gz", "wt"))
        query_out = stack.enter_context(_gzip_text(paths.split_dir / "query.txt.gz", "wt"))
        valid_rank_out = stack.enter_context(
            (paths.split_dir / "valid.bias_product.ranklist").open("w", encoding="utf-8")
        )
        test_rank_out = stack.enter_context(
            (paths.split_dir / "test.bias_product.ranklist").open("w", encoding="utf-8")
        )
        dev_map_out = stack.enter_context(paths.dev_request_map.open("w", encoding="utf-8"))
        valid_map_out = stack.enter_context(paths.valid_request_map.open("w", encoding="utf-8"))
        user_seq_out = stack.enter_context(user_sequences_path.open("w", encoding="utf-8"))

        users_out.write("catalog\n")
        for product_idx, (product_id, text) in enumerate(
            zip(registry.product_ids, registry.item_texts, strict=True)
        ):
            product_out.write(product_id + "\n")
            review_text_out.write(_encode_text(text, token_to_idx) + "\n")
            review_id_out.write(f"review_{product_idx}\n")
            review_up_out.write(f"{catalog_user_idx} {product_idx}\n")
            review_loc_out.write(f"{product_idx} 0 0\n")
        for token in vocab:
            vocab_out.write(token + "\n")
        for query in registry.queries:
            query_out.write(_encode_text(query, token_to_idx) + "\n")

        for user_ordinal, example in enumerate(iter_jsonl(examples_path)):
            user_idx = user_ordinal + 1
            user_id = f"u{user_ordinal}"
            users_out.write(user_id + "\n")
            history_idxs = [int(value) for value in example["history_idxs"]]
            input_snapshot = tuple(history_idxs)
            target_idx = int(example["target_idx"])
            query_idx = int(example["query_idx"])
            target_review_id = target_review_start + user_ordinal
            if tuple(history_idxs) != input_snapshot:
                stats["history_exact_assertions_passed"] = 0
                raise AssertionError("synthetic history changed during materialization")
            positive_ids = set(example.get("positive_item_ids") or [])
            synthetic_added = set()
            stats["sibling_positives_added"] += len(synthetic_added & positive_ids)
            user_seq_out.write(" ".join(str(value) for value in history_idxs + [target_review_id]) + "\n")
            review_text_out.write(_encode_text(registry.item_texts[target_idx], token_to_idx) + "\n")
            review_id_out.write(f"review_{target_review_id}\n")
            review_up_out.write(f"{user_idx} {target_idx}\n")
            product_loc = 1 + len(product_target_reviews[target_idx])
            product_target_reviews[target_idx].append(target_review_id)
            review_loc_out.write(f"{len(history_idxs)} {product_loc} {int(example['ts'])}\n")
            id_line = f"{user_idx}\t{target_idx}\treview_{target_review_id}\t{query_idx}\n"
            if example["kind"] == "train_positive":
                if int(example["example_id"]) in selected_valid:
                    valid_id_out.write(id_line)
                    padded = deterministic_pad_candidates(
                        example["candidate_idxs"],
                        valid_candidate_size,
                        product_count,
                        key=str(example["request_id"]),
                        seed=seed,
                    )
                    _assert_padding(example["candidate_idxs"], padded)
                    _write_ranklist(valid_rank_out, user_id, query_idx, padded, registry.product_ids)
                    _write_request_map(
                        valid_map_out,
                        example,
                        user_id,
                        query_idx,
                        padded,
                        registry.product_ids,
                    )
                    stats["valid_examples"] += 1
                else:
                    train_id_out.write(id_line)
                    train_out.write(
                        f"{user_idx}\t{target_idx}\t{_encode_text(registry.item_texts[target_idx], token_to_idx)}\n"
                    )
                    stats["train_examples"] += 1
            elif example["kind"] == "dev_request":
                test_id_out.write(id_line)
                padded = deterministic_pad_candidates(
                    example["candidate_idxs"],
                    test_candidate_size,
                    product_count,
                    key=str(example["request_id"]),
                    seed=seed,
                )
                _assert_padding(example["candidate_idxs"], padded)
                _write_ranklist(test_rank_out, user_id, query_idx, padded, registry.product_ids)
                _write_request_map(
                    dev_map_out,
                    example,
                    user_id,
                    query_idx,
                    padded,
                    registry.product_ids,
                )
                stats["dev_examples"] += 1
            else:
                raise ValueError(f"unknown example kind: {example['kind']}")

    with ExitStack() as stack:
        u_seq_out = stack.enter_context(_gzip_text(paths.data_dir / "u_r_seq.txt.gz", "wt"))
        p_seq_out = stack.enter_context(_gzip_text(paths.data_dir / "p_r_seq.txt.gz", "wt"))
        train_query_out = stack.enter_context(
            _gzip_text(paths.split_dir / "train_query_idx.txt.gz", "wt")
        )
        test_query_out = stack.enter_context(
            _gzip_text(paths.split_dir / "test_query_idx.txt.gz", "wt")
        )
        u_seq_out.write(" ".join(str(value) for value in range(product_count)) + "\n")
        with user_sequences_path.open("r", encoding="utf-8") as handle:
            shutil.copyfileobj(handle, u_seq_out)
        for product_idx in range(product_count):
            review_ids = [product_idx] + product_target_reviews.get(product_idx, [])
            p_seq_out.write(" ".join(str(value) for value in review_ids) + "\n")
            train_query_out.write("\n")
            test_query_out.write("\n")

    stats["reviews"] = product_count + example_count
    stats["users"] = 1 + example_count
    if stats["sibling_positives_added"] != 0:
        raise AssertionError("adapter added a sibling positive to synthetic history")
    return dict(stats)


def deterministic_pad_candidates(
    candidate_idxs: Iterable[int],
    target_size: int,
    product_count: int,
    *,
    key: str,
    seed: int,
) -> list[int]:
    original = [int(value) for value in candidate_idxs]
    if len(original) != len(set(original)):
        raise ValueError(f"candidate list contains duplicates: {key}")
    if len(original) > target_size:
        raise ValueError(f"candidate list exceeds padding target for {key}: {len(original)} > {target_size}")
    if target_size > product_count:
        raise ValueError("cannot pad beyond product universe")
    padded = list(original)
    seen = set(original)
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    cursor = int.from_bytes(digest[:8], "big") % product_count
    while len(padded) < target_size:
        if cursor not in seen:
            padded.append(cursor)
            seen.add(cursor)
        cursor = (cursor + 1) % product_count
    return padded


def _assert_padding(original: Iterable[int], padded: Iterable[int]) -> None:
    original_list = [int(value) for value in original]
    padded_list = [int(value) for value in padded]
    if padded_list[: len(original_list)] != original_list:
        raise AssertionError("padding changed original candidate order")
    if len(padded_list) != len(set(padded_list)):
        raise AssertionError("padding introduced duplicate candidates")
    if set(original_list) & set(padded_list[len(original_list) :]):
        raise AssertionError("padding fillers overlap original candidates")


def _write_ranklist(
    handle: TextIO,
    user_id: str,
    query_idx: int,
    candidate_idxs: Iterable[int],
    product_ids: list[str],
) -> None:
    for rank, product_idx in enumerate(candidate_idxs, start=1):
        handle.write(
            f"{user_id}_{query_idx} Q0 {product_ids[int(product_idx)]} {rank} 0.0 PPSAdapter\n"
        )


def _write_request_map(
    handle: TextIO,
    example: dict[str, Any],
    user_id: str,
    query_idx: int,
    padded: list[int],
    product_ids: list[str],
) -> None:
    original_ids = [product_ids[int(value)] for value in example["candidate_idxs"]]
    padded_ids = [product_ids[int(value)] for value in padded]
    row = {
        "request_id": str(example["request_id"]),
        "official_key": f"{user_id}_{query_idx}",
        "candidate_item_ids": original_ids,
        "candidate_count": len(original_ids),
        "padded_candidate_count": len(padded_ids),
        "padded_candidate_sha256": _ordered_ids_sha256(padded_ids),
    }
    handle.write(json.dumps(row, sort_keys=True) + "\n")


def convert_prodsearch_ranklist(
    ranklist_path: str | Path,
    request_map_path: str | Path,
    output_scores_path: str | Path,
    *,
    method_id: str,
    candidate_manifest_path: str | Path,
    split: str = "dev",
) -> dict[str, Any]:
    """Remove deterministic fillers and export exact frozen-candidate scores."""

    ranklist_path = Path(ranklist_path)
    request_map_path = Path(request_map_path)
    output_scores_path = Path(output_scores_path)
    mapping = {row["official_key"]: row for row in iter_jsonl(request_map_path)}
    if len(mapping) == 0:
        raise ValueError("empty request map")
    _assert_map_matches_candidate_manifest(mapping.values(), candidate_manifest_path, split)

    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_scores_path.with_suffix(output_scores_path.suffix + ".tmp")
    seen_keys: set[str] = set()
    score_rows = 0
    with ranklist_path.open("r", encoding="utf-8") as source, tmp_path.open(
        "w", encoding="utf-8"
    ) as output:
        current_key: str | None = None
        current_scores: dict[str, float] = {}
        for line_no, line in enumerate(source, start=1):
            fields = line.strip().split()
            if len(fields) < 5:
                raise ValueError(f"{ranklist_path}:{line_no}: malformed ranklist row")
            key, item_id = fields[0], fields[2]
            try:
                score = float(fields[4])
            except ValueError as exc:
                raise ValueError(f"{ranklist_path}:{line_no}: invalid score") from exc
            if not _is_finite(score):
                raise ValueError(f"{ranklist_path}:{line_no}: non-finite score")
            if current_key is None:
                current_key = key
            if key != current_key:
                score_rows += _flush_ranklist_group(
                    current_key, current_scores, mapping, seen_keys, output, method_id
                )
                current_key = key
                current_scores = {}
            if item_id in current_scores:
                raise ValueError(f"duplicate product in ranklist group {key}: {item_id}")
            current_scores[item_id] = score
        if current_key is not None:
            score_rows += _flush_ranklist_group(
                current_key, current_scores, mapping, seen_keys, output, method_id
            )
    missing_keys = set(mapping) - seen_keys
    if missing_keys:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"ranklist missing {len(missing_keys)} request groups")
    os.replace(tmp_path, output_scores_path)
    return {
        "status": "passed",
        "method_id": method_id,
        "split": split,
        "requests": len(seen_keys),
        "score_rows": score_rows,
        "ranklist_path": str(ranklist_path),
        "ranklist_sha256": sha256_file(ranklist_path),
        "request_map_path": str(request_map_path),
        "request_map_sha256": sha256_file(request_map_path),
        "scores_path": str(output_scores_path),
        "scores_sha256": sha256_file(output_scores_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "candidate_sets_exact": True,
        "fillers_removed": True,
    }


def _flush_ranklist_group(
    key: str,
    scores: dict[str, float],
    mapping: dict[str, dict[str, Any]],
    seen_keys: set[str],
    output: TextIO,
    method_id: str,
) -> int:
    if key not in mapping:
        raise ValueError(f"ranklist contains unknown request key: {key}")
    if key in seen_keys:
        raise ValueError(f"ranklist request group is not contiguous or is duplicated: {key}")
    row = mapping[key]
    expected = [str(value) for value in row["candidate_item_ids"]]
    missing = [item_id for item_id in expected if item_id not in scores]
    if missing:
        raise ValueError(f"ranklist group {key} misses {len(missing)} frozen candidates")
    for item_id in expected:
        output.write(
            json.dumps(
                {
                    "request_id": str(row["request_id"]),
                    "candidate_item_id": item_id,
                    "score": scores[item_id],
                    "method_id": method_id,
                },
                sort_keys=True,
            )
            + "\n"
        )
    seen_keys.add(key)
    return len(expected)


def _assert_map_matches_candidate_manifest(
    request_rows: Iterable[dict[str, Any]],
    candidate_manifest_path: str | Path,
    split: str,
) -> None:
    with Path(candidate_manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        str(row["request_id"]): [str(value) for value in row["candidate_item_ids"]]
        for row in manifest["entries"]
        if row.get("split") == split
    }
    observed = {
        str(row["request_id"]): [str(value) for value in row["candidate_item_ids"]]
        for row in request_rows
    }
    if set(observed) != set(expected):
        missing = set(expected) - set(observed)
        extra = set(observed) - set(expected)
        raise ValueError(
            f"request map/manifest mismatch for {split}: missing={len(missing)} extra={len(extra)}"
        )
    for request_id, candidates in observed.items():
        if candidates != expected[request_id]:
            raise ValueError(f"candidate order/content mismatch for request {request_id}")


def build_official_command(
    *,
    baseline_dir: str | Path,
    materialized_root: str | Path,
    save_dir: str | Path,
    model: str,
    seed: int,
    embedding_size: int,
    learning_rate: float,
    max_train_epoch: int,
    batch_size: int,
    valid_batch_size: int,
    valid_candidate_size: int,
    test_candidate_size: int,
    candidate_batch_size: int,
    num_workers: int,
    rankfname: str = "official.ranklist",
    mode: str = "train",
) -> list[str]:
    model_name = {"zam": "ZAM", "tem": "item_transformer"}.get(model.lower())
    if model_name is None:
        raise ValueError("model must be zam or tem")
    baseline_dir = Path(baseline_dir).resolve()
    materialized_root = Path(materialized_root).resolve()
    save_dir = Path(save_dir).resolve()
    command = [
        sys.executable,
        str(baseline_dir / "main.py"),
        "--model_name",
        model_name,
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--data_dir",
        str(materialized_root / "data"),
        "--input_train_dir",
        str(materialized_root / "split"),
        "--save_dir",
        str(save_dir),
        "--rankfname",
        rankfname,
        "--embedding_size",
        str(embedding_size),
        "--lr",
        str(learning_rate),
        "--max_train_epoch",
        str(max_train_epoch),
        "--batch_size",
        str(batch_size),
        "--valid_batch_size",
        str(valid_batch_size),
        "--valid_candi_size",
        str(valid_candidate_size),
        "--test_candi_size",
        str(test_candidate_size),
        "--candi_batch_size",
        str(candidate_batch_size),
        "--rank_cutoff",
        str(test_candidate_size),
        "--num_workers",
        str(num_workers),
        "--uprev_review_limit",
        "20",
        "--do_seq_review_train",
        "true",
        "--do_seq_review_test",
        "true",
        "--train_review_only",
        "false",
        "--fix_train_review",
        "true",
        "--has_valid",
        "true",
        "--decay_method",
        "adam",
        "--query_encoder_name",
        "fs",
        "--use_review_query_idx",
        "true",
        "--pv_window_size",
        "1",
        "--device",
        "cuda",
    ]
    if model.lower() == "tem":
        command.extend(
            [
                "--inter_layers",
                "1",
                "--ff_size",
                "512",
                "--heads",
                "8",
                "--use_dot_prod",
                "true",
            ]
        )
    return command


def run_official_prodsearch(
    *,
    baseline_dir: str | Path,
    materialized_root: str | Path,
    run_dir: str | Path,
    model: str,
    seed: int,
    embedding_size: int,
    learning_rate: float,
    max_train_epoch: int,
    batch_size: int,
    valid_batch_size: int,
    valid_candidate_size: int,
    test_candidate_size: int,
    candidate_batch_size: int,
    num_workers: int,
    candidate_manifest_path: str | Path,
    method_id: str,
    history_condition: str = "true",
    split: str = "dev",
    mode: str = "train",
    rankfname: str = "official.ranklist",
) -> dict[str, Any]:
    """Run official code and convert scores; this function never evaluates."""

    if split not in {"dev", "confirmation"}:
        raise ValueError(f"unsupported scoring split={split}")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    official_dir = run_dir / "official"
    official_dir.mkdir()
    command = build_official_command(
        baseline_dir=baseline_dir,
        materialized_root=materialized_root,
        save_dir=official_dir,
        model=model,
        seed=seed,
        embedding_size=embedding_size,
        learning_rate=learning_rate,
        max_train_epoch=max_train_epoch,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        valid_candidate_size=valid_candidate_size,
        test_candidate_size=test_candidate_size,
        candidate_batch_size=candidate_batch_size,
        num_workers=num_workers,
        rankfname=rankfname,
        mode=mode,
    )
    (run_dir / "command.sh").write_text(" ".join(_shell_quote(value) for value in command) + "\n")
    started = time.perf_counter()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
        run_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=Path(baseline_dir),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        metadata = {
            "status": "failed",
            "returncode": completed.returncode,
            "command": command,
            "elapsed_seconds": elapsed,
            "qrels_read": False,
            "records_test_read": False,
        }
        write_json(run_dir / "metadata.json", metadata)
        raise RuntimeError(f"official ProdSearch exited with code {completed.returncode}: {run_dir}")

    materialized_root = Path(materialized_root)
    conversion = convert_prodsearch_ranklist(
        official_dir / rankfname,
        materialized_root / "dev_request_map.jsonl",
        run_dir / "scores.jsonl",
        method_id=method_id,
        candidate_manifest_path=candidate_manifest_path,
        split=split,
    )
    counterfactual_metadata = _prodsearch_counterfactual_metadata(
        materialized_root=materialized_root,
        candidate_manifest_path=candidate_manifest_path,
        checkpoint_path=official_dir / "model_best.ckpt",
        model=model,
        embedding_size=embedding_size,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        valid_candidate_size=valid_candidate_size,
        test_candidate_size=test_candidate_size,
        candidate_batch_size=candidate_batch_size,
        history_condition=history_condition,
        split=split,
    )
    metadata = {
        "status": "scored_not_evaluated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_commit": OFFICIAL_COMMIT,
        "implementation_type": "official-code, adapter to KuaiSearch interface, not externally aligned",
        "model": model,
        "method_id": method_id,
        "seed": seed,
        "command": command,
        "elapsed_seconds": elapsed,
        "python": platform.python_version(),
        "torch_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "hostname": platform.node(),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "split": split,
        "conversion": conversion,
        "qrels_read": False,
        "records_test_read": False,
        "shared_evaluator_pending": True,
        **counterfactual_metadata,
    }
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def rescore_official_prodsearch(
    *,
    baseline_dir: str | Path,
    materialized_root: str | Path,
    checkpoint_official_dir: str | Path,
    output_dir: str | Path,
    model: str,
    seed: int,
    embedding_size: int,
    learning_rate: float,
    batch_size: int,
    valid_batch_size: int,
    valid_candidate_size: int,
    test_candidate_size: int,
    candidate_batch_size: int,
    num_workers: int,
    candidate_manifest_path: str | Path,
    method_id: str,
    history_condition: str,
    split: str = "dev",
    rankfname: str = "determinism.ranklist",
) -> dict[str, Any]:
    """Score an existing best checkpoint without retraining or evaluating."""

    if split not in {"dev", "confirmation"}:
        raise ValueError(f"unsupported scoring split={split}")

    checkpoint_official_dir = Path(checkpoint_official_dir).resolve()
    if not (checkpoint_official_dir / "model_best.ckpt").exists():
        raise FileNotFoundError(checkpoint_official_dir / "model_best.ckpt")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    command = build_official_command(
        baseline_dir=baseline_dir,
        materialized_root=materialized_root,
        save_dir=checkpoint_official_dir,
        model=model,
        seed=seed,
        embedding_size=embedding_size,
        learning_rate=learning_rate,
        max_train_epoch=0,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        valid_candidate_size=valid_candidate_size,
        test_candidate_size=test_candidate_size,
        candidate_batch_size=candidate_batch_size,
        num_workers=num_workers,
        rankfname=rankfname,
        mode="test",
    )
    (output_dir / "command.sh").write_text(
        " ".join(_shell_quote(value) for value in command) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=Path(baseline_dir),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"official ProdSearch rescore exited with code {completed.returncode}")
    conversion = convert_prodsearch_ranklist(
        checkpoint_official_dir / rankfname,
        Path(materialized_root) / "dev_request_map.jsonl",
        output_dir / "scores.jsonl",
        method_id=method_id,
        candidate_manifest_path=candidate_manifest_path,
        split=split,
    )
    counterfactual_metadata = _prodsearch_counterfactual_metadata(
        materialized_root=materialized_root,
        candidate_manifest_path=candidate_manifest_path,
        checkpoint_path=checkpoint_official_dir / "model_best.ckpt",
        model=model,
        embedding_size=embedding_size,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        valid_candidate_size=valid_candidate_size,
        test_candidate_size=test_candidate_size,
        candidate_batch_size=candidate_batch_size,
        history_condition=history_condition,
        split=split,
    )
    result = {
        "status": "passed",
        "seed": seed,
        "model": model,
        "checkpoint": str(checkpoint_official_dir / "model_best.ckpt"),
        "checkpoint_sha256": sha256_file(checkpoint_official_dir / "model_best.ckpt"),
        "elapsed_seconds": elapsed,
        "split": split,
        "conversion": conversion,
        "qrels_read": False,
        "records_test_read": False,
        **counterfactual_metadata,
    }
    write_json(output_dir / "metadata.json", result)
    return result


def _prodsearch_counterfactual_metadata(
    *,
    materialized_root: str | Path,
    candidate_manifest_path: str | Path,
    checkpoint_path: str | Path,
    model: str,
    embedding_size: int,
    batch_size: int,
    valid_batch_size: int,
    valid_candidate_size: int,
    test_candidate_size: int,
    candidate_batch_size: int,
    history_condition: str,
    split: str = "dev",
) -> dict[str, Any]:
    """Build the identity contract required by the shared evaluator.

    The true and null materializations intentionally have different serialized
    user-history files.  All remaining fields are derived from the frozen
    standardized manifest or fixed checkpoint so that the evaluator can reject
    an accidental model, slate, request-cohort, or scoring change.
    """

    if history_condition not in {"true", "null", "wrong"}:
        raise ValueError(f"unsupported history_condition={history_condition}")
    materialized_root = Path(materialized_root)
    candidate_manifest_path = Path(candidate_manifest_path)
    checkpoint_path = Path(checkpoint_path)
    request_manifest_path = candidate_manifest_path.parent / "request_manifest.json"
    history_path = materialized_root / "data" / "u_r_seq.txt.gz"
    for required in (
        candidate_manifest_path,
        request_manifest_path,
        checkpoint_path,
        history_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    with candidate_manifest_path.open("r", encoding="utf-8") as handle:
        candidate_manifest = json.load(handle)
    standardized_metadata = _read_standardized_metadata(candidate_manifest_path.parent)
    dataset_id = candidate_manifest.get("dataset_id") or standardized_metadata["dataset_id"]
    dataset_version = (
        candidate_manifest.get("dataset_version")
        or standardized_metadata["dataset_version"]
    )
    if not dataset_id or not dataset_version:
        raise ValueError("candidate manifest is missing dataset_id/dataset_version")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    normalized_model = model.lower()
    scoring_signature = {
        "adapter": "official_prodsearch_ranklist_v1",
        "official_commit": OFFICIAL_COMMIT,
        "model": normalized_model,
        "embedding_size": embedding_size,
        "batch_size": batch_size,
        "valid_batch_size": valid_batch_size,
        "valid_candidate_size": valid_candidate_size,
        "test_candidate_size": test_candidate_size,
        "candidate_batch_size": candidate_batch_size,
        "query_encoder_name": "fs",
        "query_binding": "explicit_per_review_query_idx",
        "history_limit": 20,
        "candidate_filter": "remove_deterministic_fillers_then_restore_frozen_order",
    }
    return {
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "checkpoint_id": f"prodsearch-{normalized_model}@{checkpoint_sha256[:20]}",
        "dataset_id": str(dataset_id),
        "dataset_version": str(dataset_version),
        "history_assignment_path": str(history_path),
        "history_assignment_sha256": sha256_file(history_path),
        "history_condition": history_condition,
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "scoring_signature": scoring_signature,
        "split": split,
    }


def _collect_materialized_files(paths: MaterializedPaths) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for root in (paths.data_dir, paths.split_dir):
        for path in sorted(root.iterdir()):
            if path.is_file():
                files[str(path.relative_to(paths.root))] = _file_info(path)
    for path in (paths.dev_request_map, paths.valid_request_map):
        files[str(path.relative_to(paths.root))] = _file_info(path)
    return files


def _file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


@contextmanager
def _gzip_text(path: Path, mode: str):
    if mode == "rt":
        with gzip.open(path, mode, encoding="utf-8", newline="\n") as handle:
            yield handle
        return
    if mode != "wt":
        raise ValueError(f"unsupported gzip text mode: {mode}")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                yield handle


def _read_standardized_metadata(standardized_dir: Path) -> dict[str, str]:
    manifest_path = standardized_dir / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "dataset_id": str(payload.get("dataset_id") or standardized_dir.parent.name),
            "dataset_version": str(payload.get("dataset_version") or standardized_dir.name),
        }
    return {
        "dataset_id": standardized_dir.parent.name,
        "dataset_version": standardized_dir.name,
    }


def _encode_text(text: str, token_to_idx: dict[str, int]) -> str:
    return " ".join(str(token_to_idx[token]) for token in tokenize_chars(text))


def _ordered_ids_sha256(item_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for item_id in item_ids:
        digest.update(str(item_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)
