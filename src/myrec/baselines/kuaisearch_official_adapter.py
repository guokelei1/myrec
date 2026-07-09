"""Stage-B adapter for the KuaiSearch official ranking baseline."""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


DEFAULT_GENDER = "M"
DEFAULT_AGE = "31-40"
OFFICIAL_BGE_ENCODER = "BAAI/bge-small-zh-v1.5"
UNKNOWN_CATEGORY = "UNKNOWN"
CATEGORY_LIMITS = (94, 1073, 6469)
EMBEDDING_FILES = (
    "query_emb.npy",
    "session_id2idx.json",
    "item_title_emb.npy",
    "item_id2idx.json",
)


def materialize_b5o_stageb_format(
    standardized_dir: str | Path,
    output_root: str | Path,
    *,
    split: str = "dev",
    max_train_records: int | None = None,
    max_score_records: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize standardized PPS data into the locked official rank format.

    The official loader uses ``session_id`` as the query-embedding key. PPS
    request ids are unique and stable, so the adapter writes ``request_id`` into
    that official field and preserves the raw session id as ``original_session_id``.
    Dev rows are written to ``score_<split>.jsonl`` with dummy labels and are
    never consumed by official training.
    """

    standardized_dir = Path(standardized_dir)
    output_root = Path(output_root)
    data_dir = output_root / "data"
    manifest_path = output_root / "stageb_materializer_manifest.json"
    if (
        manifest_path.exists()
        and not overwrite
        and max_train_records is None
        and max_score_records is None
        and all((data_dir / name).exists() for name in ("rank.jsonl", "score_" + split + ".jsonl", "corpus.jsonl", "users.jsonl", "query_texts.jsonl"))
    ):
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    output_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = standardized_dir / "records_train.jsonl"
    score_path = standardized_dir / f"records_{split}.jsonl"
    item_catalog_path = standardized_dir / "item_catalog.jsonl"
    for path in (train_path, score_path, item_catalog_path):
        if not path.exists():
            raise FileNotFoundError(path)

    scan = _scan_records_for_stageb(
        train_path=train_path,
        score_path=score_path,
        max_train_records=max_train_records,
        max_score_records=max_score_records,
    )
    category_maps = _build_category_maps(scan["category_values"])

    users_rows = _write_stageb_users(data_dir / "users.jsonl", scan["needed_user_ids"])
    corpus_stats = _write_stageb_corpus(
        item_catalog_path=item_catalog_path,
        output_path=data_dir / "corpus.jsonl",
        needed_item_ids=scan["needed_item_ids"],
        item_snapshots=scan["item_snapshots"],
        category_maps=category_maps,
    )
    query_stats = _write_query_texts(data_dir / "query_texts.jsonl", scan["query_by_request_id"])
    rank_stats = _write_stageb_rank(
        records_path=train_path,
        output_path=data_dir / "rank.jsonl",
        max_records=max_train_records,
    )
    score_stats = _write_stageb_score_input(
        records_path=score_path,
        output_path=data_dir / f"score_{split}.jsonl",
        split=split,
        max_records=max_score_records,
    )

    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if corpus_stats["missing_item_count"] == 0 else "missing_items_filled_from_records",
        "adapter": "myrec.baselines.kuaisearch_official_adapter",
        "identity_label": "official-code, proxy-aligned (last-time 10% split)",
        "standardized_dir": str(standardized_dir),
        "output_root": str(output_root),
        "split": split,
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "input_files": {
            "records_train": _file_info(train_path),
            f"records_{split}": _file_info(score_path),
            "item_catalog": _file_info(item_catalog_path),
            "candidate_manifest": _file_info(candidate_manifest_path),
        },
        "qrels_read": False,
        "train_limits": {
            "max_train_records": max_train_records,
            "max_score_records": max_score_records,
        },
        "official_field_mapping": {
            "session_id": "request_id; used only as the official query embedding key",
            "original_session_id": "standardized record session_id, preserved for audit",
            "recently_clicked_item_ids": "frozen record history click item ids, truncated by the official loader to 20",
            "recently_purchased_item_ids": "frozen record history purchase item ids; retained in JSONL but ignored by the locked official loader",
            "target_item_id": "candidate item_id from standardized fixed candidates",
            "is_clicked": "records_train.candidates.clicked for train rows; zero dummy for scoring rows",
            "is_purchased": "records_train.candidates.purchased for train rows; zero dummy for scoring rows",
        },
        "user_defaults": {
            "reason": "standardized records do not expose official demographic fields",
            "gender": DEFAULT_GENDER,
            "age": DEFAULT_AGE,
            "all_needed_users_written": users_rows == len(scan["needed_user_ids"]),
        },
        "category_mapping": {
            "source": "standardized candidate/history/category text",
            "unknown_bucket": 0,
            "limits_excluding_unknown": CATEGORY_LIMITS,
            "unique_counts": [len(category_maps[level]) - 1 for level in range(3)],
        },
        "embedding_policy": {
            "encoder": OFFICIAL_BGE_ENCODER,
            "pooling": "CLS token, matching locked ranking/data/process.py",
            "dtype": "float16",
            "max_length": 32,
            "cache_dir": str(data_dir),
        },
        "counts": {
            "needed_items": len(scan["needed_item_ids"]),
            "needed_users": len(scan["needed_user_ids"]),
            "query_keys": len(scan["query_by_request_id"]),
            "rank_rows": rank_stats["rows"],
            "score_rows": score_stats["rows"],
            "train_records": rank_stats["records"],
            f"{split}_records": score_stats["records"],
        },
        "rank": rank_stats,
        "score_input": score_stats,
        "query_texts": query_stats,
        "corpus": corpus_stats,
        "files": {
            "rank": _file_info(data_dir / "rank.jsonl"),
            "score_input": _file_info(data_dir / f"score_{split}.jsonl"),
            "corpus": _file_info(data_dir / "corpus.jsonl"),
            "users": _file_info(data_dir / "users.jsonl"),
            "query_texts": _file_info(data_dir / "query_texts.jsonl"),
        },
    }
    write_json(manifest_path, manifest)
    manifest["files"]["manifest"] = _file_info(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def ensure_b5o_stageb_embeddings(
    output_root: str | Path,
    *,
    batch_size: int = 1024,
    max_length: int = 32,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate BGE query/title embeddings in the official loader paths."""

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    output_root = Path(output_root)
    data_dir = output_root / "data"
    query_input = data_dir / "query_texts.jsonl"
    corpus_input = data_dir / "corpus.jsonl"
    if not query_input.exists() or not corpus_input.exists():
        raise FileNotFoundError("materialized query_texts.jsonl and corpus.jsonl are required")

    existing = all((data_dir / name).exists() for name in EMBEDDING_FILES)
    manifest_path = output_root / "stageb_embedding_manifest.json"
    if existing and manifest_path.exists() and not overwrite:
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(OFFICIAL_BGE_ENCODER, trust_remote_code=True)
    model = AutoModel.from_pretrained(OFFICIAL_BGE_ENCODER, trust_remote_code=True).to(device)
    model.eval()

    started = time.perf_counter()
    query_count = _count_lines(query_input)
    item_count = _count_lines(corpus_input)
    query_shape = _encode_jsonl_texts(
        input_path=query_input,
        id_field="session_id",
        text_field="query",
        count=query_count,
        tokenizer=tokenizer,
        model=model,
        device=device,
        npy_path=data_dir / "query_emb.npy",
        mapping_path=data_dir / "session_id2idx.json",
        batch_size=batch_size,
        max_length=max_length,
    )
    item_shape = _encode_jsonl_texts(
        input_path=corpus_input,
        id_field="item_id",
        text_field="item_title",
        count=item_count,
        tokenizer=tokenizer,
        model=model,
        device=device,
        npy_path=data_dir / "item_title_emb.npy",
        mapping_path=data_dir / "item_id2idx.json",
        batch_size=batch_size,
        max_length=max_length,
    )
    elapsed = time.perf_counter() - started
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encoder": OFFICIAL_BGE_ENCODER,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "elapsed_seconds": elapsed,
        "query_shape": query_shape,
        "item_shape": item_shape,
        "files": {name: _file_info(data_dir / name) for name in EMBEDDING_FILES},
    }
    write_json(manifest_path, manifest)
    manifest["files"]["manifest"] = _file_info(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def train_and_score_b5o(
    config: dict[str, Any],
    run_id: str,
    *,
    model_name: str,
    seed: int,
    split: str = "dev",
    runs_dir: str | Path = "runs",
    artifact_root: str | Path | None = None,
    max_train_records: int | None = None,
    max_score_records: int | None = None,
    overwrite_materialized: bool = False,
    overwrite_embeddings: bool = False,
    skip_embeddings: bool = False,
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the locked official model and export PPS fixed-candidate scores."""

    if model_name not in {"DNN", "DCNv2", "DIN"}:
        raise ValueError("model_name must be one of DNN, DCNv2, DIN")

    started = time.perf_counter()
    standardized_dir = Path(config["standardized_dir"]).resolve()
    artifact_root = Path(artifact_root or config["stage_b"]["artifact_root"]).resolve()
    runs_dir = Path(runs_dir)
    run_dir = (runs_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    materializer_manifest = materialize_b5o_stageb_format(
        standardized_dir=standardized_dir,
        output_root=artifact_root,
        split=split,
        max_train_records=max_train_records,
        max_score_records=max_score_records,
        overwrite=overwrite_materialized,
    )
    embedding_manifest = None
    if not skip_embeddings:
        embedding_manifest = ensure_b5o_stageb_embeddings(
            artifact_root,
            batch_size=int(config["stage_b"]["embedding"]["batch_size"]),
            max_length=int(config["stage_b"]["embedding"]["max_length"]),
            overwrite=overwrite_embeddings,
        )

    official = _import_official_modules(config)
    torch = official["torch"]
    _set_global_seed(seed, torch)

    hp = _official_hyperparams(config, hyperparams or {})
    data_dir_arg = "data"
    previous_cwd = Path.cwd()
    previous_argv = sys.argv[:]
    os.chdir(artifact_root)
    try:
        sys.argv = [previous_argv[0]]
        data_proc = official["TrainingDataProcessor"](
            dataset_name_or_path=data_dir_arg,
            batch_size=int(hp["batch_size"]),
            max_history_len=int(hp["max_history_len"]),
            valid_ratio=float(hp["valid_ratio"]),
            seed=seed,
        )
        train_loader = data_proc.get_train_dataloader()
        valid_loader = data_proc.get_valid_dataloader()
        test_loader = data_proc.get_test_dataloader()
        model = _build_official_model(official, model_name=model_name, config=config, hyperparams=hp)
        trainer = official["DCNAccelerateTrainer"](
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            lr=float(hp["lr"]),
            weight_decay=float(hp["weight_decay"]),
            num_epochs=int(hp["num_epochs"]),
            early_stop_patience=int(hp["early_stop_patience"]),
            mixed_precision=str(hp["mixed_precision"]),
            gradient_accumulation_steps=int(hp["gradient_accumulation_steps"]),
            max_grad_norm=float(hp["max_grad_norm"]),
            log_with=None,
            save_dir=str(run_dir / "official_checkpoints"),
        )
        trainer.train()
        trainer.load_checkpoint()
        score_stats = _score_stageb_split(
            data_proc=data_proc,
            trainer=trainer,
            torch=torch,
            score_input_path=artifact_root / "data" / f"score_{split}.jsonl",
            output_path=run_dir / "scores.jsonl",
            method_id=config["method_id"],
            model_name=model_name,
            batch_size=int(hp["batch_size"]),
            max_score_records=max_score_records,
        )
        internal_test = trainer.evaluate_test()
        internal_metrics = {
            "best_valid_loss": _jsonable(getattr(trainer, "best_valid_loss", None)),
            "best_valid_auc": _jsonable(getattr(trainer, "best_valid_auc", None)),
            "test_logloss_on_internal_train_split": _jsonable(internal_test[0]),
            "test_auc_on_internal_train_split": _jsonable(internal_test[1]),
        }
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    _copy_if_exists(config.get("_config_path"), config_snapshot_path)
    elapsed = time.perf_counter() - started
    metadata = {
        "adapter": "myrec.baselines.kuaisearch_official_adapter",
        "artifact_root": str(artifact_root),
        "candidate_manifest_path": str(standardized_dir / "candidate_manifest.json"),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "config_path": config.get("_config_path"),
        "config_sha256": sha256_file(config["_config_path"]) if config.get("_config_path") else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": config["dataset_id"],
        "dataset_version": config["dataset_version"],
        "env_group": config["environment_group"],
        "env_name": config["environment_name"],
        "git_dirty": None,
        "hostname": platform.node(),
        "implementation_type": config["implementation_type"],
        "identity_label": config.get("identity_label"),
        "input_fields_used": config["stage_b"]["input_fields_used"],
        "internal_train_metrics": internal_metrics,
        "materializer_manifest": materializer_manifest,
        "embedding_manifest": embedding_manifest,
        "method_id": config["method_id"],
        "model_name": model_name,
        "official_hyperparams": hp,
        "package_versions": _package_versions(),
        "python": platform.python_version(),
        "qrels_read": False,
        "run_id": run_id,
        "score_stats": score_stats,
        "seed": seed,
        "split": split,
        "timing": {"train_and_score_seconds": elapsed},
        "train_interactions": config["stage_b"]["train_interactions"],
        "upstream": config["upstream"],
    }
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def _scan_records_for_stageb(
    *,
    train_path: Path,
    score_path: Path,
    max_train_records: int | None,
    max_score_records: int | None,
) -> dict[str, Any]:
    needed_item_ids: set[str] = set()
    needed_user_ids: set[str] = set()
    query_by_request_id: dict[str, str] = {}
    item_snapshots: dict[str, dict[str, Any]] = {}
    category_values = [set() for _ in range(3)]

    def consume_record(record: dict[str, Any]) -> None:
        request_id = str(record["request_id"])
        query_by_request_id[request_id] = str(record.get("query") or "")
        needed_user_ids.add(str(record["user_id"]))
        for item in list(record.get("history") or []) + list(record.get("candidates") or []):
            item_id = str(item["item_id"])
            needed_item_ids.add(item_id)
            item_snapshots.setdefault(item_id, item)
            for level, value in enumerate(_cat_levels(item)):
                if value:
                    category_values[level].add(value)

    train_records = 0
    for record in iter_jsonl(train_path):
        if max_train_records is not None and train_records >= max_train_records:
            break
        consume_record(record)
        train_records += 1
    score_records = 0
    for record in iter_jsonl(score_path):
        if max_score_records is not None and score_records >= max_score_records:
            break
        consume_record(record)
        score_records += 1
    return {
        "needed_item_ids": needed_item_ids,
        "needed_user_ids": needed_user_ids,
        "query_by_request_id": query_by_request_id,
        "item_snapshots": item_snapshots,
        "category_values": category_values,
        "train_records": train_records,
        "score_records": score_records,
    }


def _build_category_maps(category_values: list[set[str]]) -> list[dict[str, int]]:
    maps = []
    for level, values in enumerate(category_values):
        values = {value for value in values if value and value != UNKNOWN_CATEGORY}
        limit = CATEGORY_LIMITS[level]
        if len(values) > limit:
            raise ValueError(f"category level {level + 1} has {len(values)} values > official limit {limit}")
        mapping = {UNKNOWN_CATEGORY: 0}
        for idx, value in enumerate(sorted(values), start=1):
            mapping[value] = idx
        maps.append(mapping)
    return maps


def _write_stageb_users(path: Path, needed_user_ids: set[str]) -> int:
    rows = 0
    with path.open("w", encoding="utf-8") as handle:
        for user_id in sorted(needed_user_ids, key=_int_sort_key):
            row = {
                "age": DEFAULT_AGE,
                "gender": DEFAULT_GENDER,
                "stageb_default_user_features": True,
                "user_id": int(user_id),
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
    return rows


def _write_stageb_corpus(
    *,
    item_catalog_path: Path,
    output_path: Path,
    needed_item_ids: set[str],
    item_snapshots: dict[str, dict[str, Any]],
    category_maps: list[dict[str, int]],
) -> dict[str, Any]:
    written: set[str] = set()
    rows = 0
    unknown_category_counts = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(item_catalog_path):
            item_id = str(row["item_id"])
            if item_id not in needed_item_ids:
                continue
            out, unknowns = _official_corpus_row(row, category_maps)
            handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            written.add(item_id)
            rows += 1
            unknown_category_counts.update(unknowns)
            if rows == len(needed_item_ids):
                break
        for item_id in sorted(needed_item_ids - written, key=_int_sort_key):
            out, unknowns = _official_corpus_row(item_snapshots[item_id], category_maps)
            handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
            unknown_category_counts.update(unknowns)
    return {
        "rows": rows,
        "missing_item_count": len(needed_item_ids - written),
        "missing_filled_from_record_examples": sorted(needed_item_ids - written, key=_int_sort_key)[:20],
        "unknown_category_counts": dict(unknown_category_counts),
    }


def _official_corpus_row(row: dict[str, Any], category_maps: list[dict[str, int]]) -> tuple[dict[str, Any], Counter[str]]:
    item_id = int(row["item_id"])
    cats = _cat_levels(row)
    unknowns: Counter[str] = Counter()
    cat_ids = []
    for level in range(3):
        value = cats[level]
        mapped = category_maps[level].get(value, 0)
        if mapped == 0 and value != UNKNOWN_CATEGORY:
            unknowns[f"level{level + 1}"] += 1
        cat_ids.append(mapped)
    title = row.get("title", row.get("item_title", "")) or ""
    brand = row.get("brand", row.get("brand_name", "")) or ""
    seller = row.get("seller", row.get("seller_name", "")) or ""
    return (
        {
            "brand_id": 0,
            "brand_name": str(brand),
            "category_level1_id": cat_ids[0],
            "category_level1_name": cats[0],
            "category_level2_id": cat_ids[1],
            "category_level2_name": cats[1],
            "category_level3_id": cat_ids[2],
            "category_level3_name": cats[2],
            "item_id": item_id,
            "item_title": str(title),
            "seller_id": 0,
            "seller_name": str(seller),
        },
        unknowns,
    )


def _write_query_texts(path: Path, query_by_request_id: dict[str, str]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8") as handle:
        for request_id in sorted(query_by_request_id):
            row = {"query": query_by_request_id[request_id], "session_id": request_id}
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(query_by_request_id)}


def _write_stageb_rank(
    *,
    records_path: Path,
    output_path: Path,
    max_records: int | None,
) -> dict[str, Any]:
    records = 0
    rows = 0
    label_counts = Counter()
    history_lengths = []
    with output_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(records_path):
            if max_records is not None and records >= max_records:
                break
            histories = _history_channels(record)
            history_lengths.append(len(histories["recently_clicked_item_ids"]))
            for candidate in record.get("candidates") or []:
                label = 1 if int(candidate.get("clicked") or 0) or int(candidate.get("purchased") or 0) else 0
                row = _official_rank_row(record, candidate, histories, split="train", label=label)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                label_counts[str(label)] += 1
                rows += 1
            records += 1
    return {
        "records": records,
        "rows": rows,
        "label_counts": dict(label_counts),
        "history_len_max": max(history_lengths) if history_lengths else 0,
        "history_len_mean": sum(history_lengths) / len(history_lengths) if history_lengths else 0.0,
        "split": "train",
    }


def _write_stageb_score_input(
    *,
    records_path: Path,
    output_path: Path,
    split: str,
    max_records: int | None,
) -> dict[str, Any]:
    records = 0
    rows = 0
    history_lengths = []
    with output_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(records_path):
            if max_records is not None and records >= max_records:
                break
            histories = _history_channels(record)
            history_lengths.append(len(histories["recently_clicked_item_ids"]))
            for candidate in record.get("candidates") or []:
                row = _official_rank_row(record, candidate, histories, split=split, label=0)
                row["candidate_item_id"] = str(candidate["item_id"])
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                rows += 1
            records += 1
    return {
        "records": records,
        "rows": rows,
        "history_len_max": max(history_lengths) if history_lengths else 0,
        "history_len_mean": sum(history_lengths) / len(history_lengths) if history_lengths else 0.0,
        "labels": "dummy_zero_for_label_free_scoring",
        "split": split,
    }


def _official_rank_row(
    record: dict[str, Any],
    candidate: dict[str, Any],
    histories: dict[str, list[int]],
    *,
    split: str,
    label: int,
) -> dict[str, Any]:
    return {
        "is_clicked": int(candidate.get("clicked") or 0) if split == "train" else 0,
        "is_purchased": int(candidate.get("purchased") or 0) if split == "train" else 0,
        "original_session_id": str(record.get("session_id") or ""),
        "query": str(record.get("query") or ""),
        "recently_clicked_item_ids": histories["recently_clicked_item_ids"],
        "recently_purchased_item_ids": histories["recently_purchased_item_ids"],
        "request_id": str(record["request_id"]),
        "session_id": str(record["request_id"]),
        "split": split,
        "target_item_id": int(candidate["item_id"]),
        "time_index": int(record.get("ts") or 0),
        "user_id": int(record["user_id"]),
        "stageb_label": label,
    }


def _history_channels(record: dict[str, Any]) -> dict[str, list[int]]:
    clicked = []
    purchased = []
    for event in record.get("history") or []:
        target = purchased if str(event.get("event")) == "purchase" else clicked
        target.append(int(event["item_id"]))
    return {
        "recently_clicked_item_ids": clicked[-50:],
        "recently_purchased_item_ids": purchased[-50:],
    }


def _encode_jsonl_texts(
    *,
    input_path: Path,
    id_field: str,
    text_field: str,
    count: int,
    tokenizer: Any,
    model: Any,
    device: str,
    npy_path: Path,
    mapping_path: Path,
    batch_size: int,
    max_length: int,
) -> list[int]:
    import numpy as np
    import torch

    id2idx: dict[str, int] = {}
    memmap = None
    offset = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            batch_ids.append(str(row[id_field]))
            batch_texts.append(str(row.get(text_field) or ""))
            if len(batch_ids) >= batch_size:
                memmap, offset = _encode_batch(
                    batch_ids=batch_ids,
                    batch_texts=batch_texts,
                    id2idx=id2idx,
                    memmap=memmap,
                    offset=offset,
                    count=count,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    npy_path=npy_path,
                    max_length=max_length,
                    torch=torch,
                    np=np,
                )
                batch_ids, batch_texts = [], []
    if batch_ids:
        memmap, offset = _encode_batch(
            batch_ids=batch_ids,
            batch_texts=batch_texts,
            id2idx=id2idx,
            memmap=memmap,
            offset=offset,
            count=count,
            tokenizer=tokenizer,
            model=model,
            device=device,
            npy_path=npy_path,
            max_length=max_length,
            torch=torch,
            np=np,
        )
    if memmap is None:
        raise ValueError(f"no rows to encode in {input_path}")
    memmap.flush()
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(id2idx, handle, ensure_ascii=False)
    return [count, int(memmap.shape[1])]


def _encode_batch(
    *,
    batch_ids: list[str],
    batch_texts: list[str],
    id2idx: dict[str, int],
    memmap: Any,
    offset: int,
    count: int,
    tokenizer: Any,
    model: Any,
    device: str,
    npy_path: Path,
    max_length: int,
    torch: Any,
    np: Any,
) -> tuple[Any, int]:
    with torch.no_grad():
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = model(**encoded)
        cls_emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float16)
    if memmap is None:
        memmap = np.lib.format.open_memmap(
            npy_path,
            mode="w+",
            dtype=np.float16,
            shape=(count, cls_emb.shape[1]),
        )
    end = offset + len(batch_ids)
    memmap[offset:end] = cls_emb
    for idx, unique_id in enumerate(batch_ids, start=offset):
        id2idx[unique_id] = idx
    return memmap, end


def _import_official_modules(config: dict[str, Any]) -> dict[str, Any]:
    official_dir = (Path(config["upstream"]["local_dir"]) / "ranking").resolve()
    sys.path.insert(0, str(official_dir))
    import torch
    from datasets import TrainingDataProcessor
    from models import DCNModel, DINModel, DNNModel
    from trainer import DCNAccelerateTrainer

    return {
        "torch": torch,
        "TrainingDataProcessor": TrainingDataProcessor,
        "DCNModel": DCNModel,
        "DINModel": DINModel,
        "DNNModel": DNNModel,
        "DCNAccelerateTrainer": DCNAccelerateTrainer,
    }


def _build_official_model(
    official: dict[str, Any],
    *,
    model_name: str,
    config: dict[str, Any],
    hyperparams: dict[str, Any],
) -> Any:
    model_params = {
        "config": {},
        "num_cross_layers": int(hyperparams["num_cross_layers"]),
        "hidden_size": int(hyperparams["hidden_size"]),
        "dropout_rate": float(hyperparams["dropout_rate"]),
        "user_id_embedding_dim": int(hyperparams["user_id_embedding_dim"]),
    }
    if model_name == "DNN":
        return official["DNNModel"](**model_params)
    if model_name == "DCNv2":
        return official["DCNModel"](**model_params, version="v2")
    if model_name == "DIN":
        return official["DINModel"](**model_params)
    raise ValueError(model_name)


def _score_stageb_split(
    *,
    data_proc: Any,
    trainer: Any,
    torch: Any,
    score_input_path: Path,
    output_path: Path,
    method_id: str,
    model_name: str,
    batch_size: int,
    max_score_records: int | None,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader, IterableDataset

    class ScoreDataset(IterableDataset):
        def __iter__(self) -> Iterable[dict[str, Any]]:
            rows = 0
            with score_input_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if max_score_records is not None and rows >= max_score_records:
                        break
                    rows += 1
                    yield json.loads(line)

    def collate_score(batch: list[dict[str, Any]]) -> tuple[Any, Any, Any, list[dict[str, str]]]:
        query_features, user_features, item_features, _labels = data_proc.collate_fn(batch)
        metas = [
            {
                "candidate_item_id": str(sample["target_item_id"]),
                "request_id": str(sample["request_id"]),
            }
            for sample in batch
        ]
        return query_features, user_features, item_features, metas

    loader = DataLoader(
        ScoreDataset(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_score,
        num_workers=0,
        pin_memory=True,
    )
    model = trainer.model
    model.eval()
    device = trainer.accelerator.device
    score_rows = 0
    request_ids: set[str] = set()
    score_min = None
    score_max = None
    with output_path.open("w", encoding="utf-8") as handle:
        for query_features, user_features, item_features, metas in loader:
            query_features = _move_to_device(query_features, device)
            user_features = _move_to_device(user_features, device)
            item_features = _move_to_device(item_features, device)
            with torch.no_grad():
                logits = model(query_features, user_features, item_features)
                probs = torch.sigmoid(logits).detach().cpu().view(-1).tolist()
            for meta, score in zip(metas, probs):
                request_ids.add(meta["request_id"])
                score_min = score if score_min is None else min(score_min, score)
                score_max = score if score_max is None else max(score_max, score)
                row = {
                    "candidate_item_id": meta["candidate_item_id"],
                    "method_id": method_id,
                    "model_name": model_name,
                    "request_id": meta["request_id"],
                    "score": float(score),
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                score_rows += 1
    return {
        "request_count": len(request_ids),
        "score_rows": score_rows,
        "scores_path": str(output_path),
        "scores_sha256": sha256_file(output_path),
        "score_min": score_min,
        "score_max": score_max,
    }


def _official_hyperparams(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    hp = dict(config["stage_b"]["official_defaults"])
    hp.update({key: value for key, value in overrides.items() if value is not None})
    return hp


def _set_global_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _move_to_device(value: Any, device: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
    except Exception:
        pass
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(item, device) for item in value)
    return value


def _history_event_type(event: dict[str, Any]) -> str:
    return str(event.get("event") or event.get("event_type") or "click")


def _cat_levels(row: dict[str, Any]) -> list[str]:
    cats = row.get("cat")
    if cats is None:
        cats = [
            row.get("category_level1_name"),
            row.get("category_level2_name"),
            row.get("category_level3_name"),
        ]
    result = [str(value or UNKNOWN_CATEGORY) for value in list(cats)[:3]]
    while len(result) < 3:
        result.append(UNKNOWN_CATEGORY)
    return result


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _int_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):020d}")
    except ValueError:
        return (1, value)


def _copy_if_exists(source: str | Path | None, target: Path) -> None:
    if source is None:
        return
    source_path = Path(source)
    if source_path.exists():
        shutil.copyfile(source_path, target)


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "accelerate", "transformers", "numpy"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[package] = "not_importable"
    return versions


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)
