"""Dense bi-encoder zero-shot scorer for Batch 1."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.baselines.core import document_text
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


def write_dense_biencoder_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    model_name: str = "BAAI/bge-small-zh-v1.5",
    cache_folder: str | Path = "models/huggingface/sentence_transformers",
    device: str = "cuda:0",
    batch_size: int = 256,
    max_seq_length: int = 256,
    query_prefix: str = "",
) -> dict[str, Any]:
    """Encode queries and candidate documents, then score by cosine/dot product."""

    import numpy as np
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    standardized_dir = Path(standardized_dir)
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)

    records_path = standardized_dir / f"records_{split}.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    loaded = _load_records_for_dense(records_path, query_prefix=query_prefix)

    model = SentenceTransformer(model_name, cache_folder=str(cache_folder), device=device)
    model.max_seq_length = max_seq_length

    item_ids = sorted(loaded["item_texts"])
    item_texts = [loaded["item_texts"][item_id] for item_id in item_ids]
    item_embeddings = model.encode(
        item_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        loaded["queries"],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}

    scores_path = run_dir / "scores.jsonl"
    rows = 0
    with scores_path.open("w", encoding="utf-8") as handle:
        for query_index, request in enumerate(loaded["requests"]):
            request_id = request["request_id"]
            query_embedding = query_embeddings[query_index]
            indices = [item_index[item_id] for item_id in request["candidate_item_ids"]]
            candidate_embeddings = item_embeddings[np.asarray(indices)]
            scores = candidate_embeddings @ query_embedding
            for item_id, score in zip(request["candidate_item_ids"], scores):
                handle.write(
                    json.dumps(
                        {
                            "candidate_item_id": item_id,
                            "method_id": "b2z_dense_biencoder",
                            "request_id": request_id,
                            "score": float(score),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                rows += 1

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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_fields_used": [
            "query",
            "candidates.title",
            "candidates.brand",
            "candidates.seller",
            "candidates.cat",
        ],
        "item_text_conflicts": loaded["item_text_conflicts"],
        "max_seq_length": max_seq_length,
        "method_id": "b2z_dense_biencoder",
        "model_name": model_name,
        "package_versions": {
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
        },
        "qrels_read": False,
        "query_prefix": query_prefix,
        "request_count": len(loaded["requests"]),
        "run_id": run_id,
        "score_definition": "dot product of L2-normalized query and candidate document embeddings",
        "score_rows": rows,
        "split": split,
        "standardized_dir": str(standardized_dir),
        "unique_item_texts": len(item_ids),
    }
    _copy_config(config_path, run_dir)
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def _load_records_for_dense(path: Path, query_prefix: str) -> dict[str, Any]:
    item_texts = {}
    item_text_conflicts = 0
    requests = []
    queries = []
    for record in iter_jsonl(path):
        request_id = str(record["request_id"])
        candidate_item_ids = []
        for candidate in record["candidates"]:
            item_id = str(candidate["item_id"])
            text = document_text(candidate)
            previous = item_texts.setdefault(item_id, text)
            if previous != text:
                item_text_conflicts += 1
            candidate_item_ids.append(item_id)
        requests.append({"request_id": request_id, "candidate_item_ids": candidate_item_ids})
        queries.append(query_prefix + str(record.get("query") or ""))
    return {
        "item_text_conflicts": item_text_conflicts,
        "item_texts": item_texts,
        "queries": queries,
        "requests": requests,
    }


def _copy_config(config_path: str | Path | None, run_dir: Path) -> None:
    if not config_path:
        return
    config_path = Path(config_path)
    if config_path.exists():
        shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")
