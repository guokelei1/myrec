"""Cross-encoder reranker scorer for PPS Batch 2."""

from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.baselines.core import document_text
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


def write_cross_encoder_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    model_name: str = "BAAI/bge-reranker-base",
    cache_folder: str | Path = "models/huggingface/cross_encoders",
    device: str = "cuda:0",
    batch_size: int = 256,
    max_length: int = 256,
    pair_chunk_size: int = 8192,
) -> dict[str, Any]:
    """Score each fixed candidate with a query-document cross encoder."""

    import sentence_transformers
    import torch
    from sentence_transformers import CrossEncoder

    standardized_dir = Path(standardized_dir)
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)

    records_path = standardized_dir / f"records_{split}.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    scores_path = run_dir / "scores.jsonl"

    model = CrossEncoder(
        model_name,
        cache_folder=str(cache_folder),
        device=device,
        max_length=max_length,
        model_kwargs={"torch_dtype": "auto"},
    )

    rows = 0
    requests = 0
    started = time.perf_counter()
    pair_buffer: list[tuple[str, str]] = []
    key_buffer: list[tuple[str, str]] = []

    def flush(handle: Any) -> None:
        nonlocal rows
        if not pair_buffer:
            return
        predicted = model.predict(
            pair_buffer,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for (request_id, item_id), score in zip(key_buffer, predicted):
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"non-finite score for {request_id} {item_id}: {value}")
            handle.write(
                json.dumps(
                    {
                        "candidate_item_id": item_id,
                        "method_id": "b3_cross_encoder_zero_shot",
                        "request_id": request_id,
                        "score": value,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            rows += 1
        pair_buffer.clear()
        key_buffer.clear()

    with scores_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(records_path):
            requests += 1
            request_id = str(record["request_id"])
            query = str(record.get("query") or "")
            for candidate in record["candidates"]:
                item_id = str(candidate["item_id"])
                pair_buffer.append((query, document_text(candidate)))
                key_buffer.append((request_id, item_id))
                if len(pair_buffer) >= pair_chunk_size:
                    flush(handle)
        flush(handle)

    elapsed = time.perf_counter() - started
    metadata = {
        "batch_size": batch_size,
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "cache_folder": str(cache_folder),
        "config_path": str(config_path) if config_path else None,
        "dataset_id": "kuaisearch",
        "dataset_version": "v0_lite",
        "device": device,
        "document_template": "title + brand + seller + cat_l1 + cat_l2 + cat_l3",
        "elapsed_seconds": elapsed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_fields_used": [
            "query",
            "candidates.title",
            "candidates.brand",
            "candidates.seller",
            "candidates.cat",
            "candidates.item_id",
        ],
        "latency": {
            "candidate_pairs_per_second": rows / elapsed if elapsed else None,
            "seconds_total": elapsed,
        },
        "max_length": max_length,
        "method_id": "b3_cross_encoder_zero_shot",
        "model_name": model_name,
        "package_versions": {
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
        },
        "pair_chunk_size": pair_chunk_size,
        "qrels_read": False,
        "request_count": requests,
        "run_id": run_id,
        "score_definition": "cross-encoder relevance score for (query, B1 document text)",
        "score_rows": rows,
        "split": split,
        "standardized_dir": str(standardized_dir),
        "tuning": {
            "class": "zero-shot",
            "dev_evals_used": 1,
            "prompt_or_template_changes": 0,
        },
    }
    _copy_config(config_path, run_dir)
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def _copy_config(config_path: str | Path | None, run_dir: Path) -> None:
    if not config_path:
        return
    config_path = Path(config_path)
    if config_path.exists():
        shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")
