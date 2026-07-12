"""Label-free C38 structure collection and frozen BGE state encoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from train.selection import (
    candidate_key_sha256,
    load_blind_records,
    read_json,
    sha256_file,
    write_json,
)


def collect_label_free_features(
    *,
    records_path: str | Path,
    selection_path: str | Path,
    output_root: str | Path,
    roles: Iterable[str] = ("fit", "internal_A"),
) -> dict[str, Any]:
    records_path = Path(records_path)
    selection_path = Path(selection_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selection = read_json(selection_path)
    if sha256_file(records_path) != selection["records_sha256"]:
        raise RuntimeError("C38 blind records changed after selection")
    records = load_blind_records(records_path)
    selected_indices: list[int] = []
    role_names = list(roles)
    for role in role_names:
        row = selection["roles"][role]
        indices = [int(value) for value in row["indices"]]
        if candidate_key_sha256(records, indices) != row["candidate_key_sha256"]:
            raise RuntimeError(f"C38 candidate hash changed for {role}")
        selected_indices.extend(indices)
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("C38 feature roles overlap")

    item_text: dict[str, str] = {}
    text_conflicts = 0
    for index in selected_indices:
        record = records[index]
        wrong_index = int(selection["wrong_donors"][str(index)]["donor_index"])
        for item in [*record["candidates"], *record["history"], *records[wrong_index]["history"]]:
            item_id = str(item["item_id"])
            text = _item_text(item)
            previous = item_text.get(item_id)
            if previous is not None and previous != text:
                text_conflicts += 1
                text = max((previous, text), key=lambda value: (len(value), value))
            item_text[item_id] = text
    item_ids = sorted(item_text)
    item_position = {item_id: position for position, item_id in enumerate(item_ids)}

    candidate_offsets = [0]
    candidate_item_positions: list[int] = []
    true_history_offsets = [0]
    true_history_item_positions: list[int] = []
    wrong_history_offsets = [0]
    wrong_history_item_positions: list[int] = []
    request_ids: list[str] = []
    queries: list[str] = []
    for index in selected_indices:
        record = records[index]
        request_ids.append(str(record["request_id"]))
        queries.append(str(record["query"]))
        candidates = [item_position[str(item["item_id"])] for item in record["candidates"]]
        true_history = [item_position[str(item["item_id"])] for item in record["history"]]
        wrong_index = int(selection["wrong_donors"][str(index)]["donor_index"])
        wrong_history = [
            item_position[str(item["item_id"])] for item in records[wrong_index]["history"]
        ]
        candidate_item_positions.extend(candidates)
        true_history_item_positions.extend(true_history)
        wrong_history_item_positions.extend(wrong_history)
        candidate_offsets.append(len(candidate_item_positions))
        true_history_offsets.append(len(true_history_item_positions))
        wrong_history_offsets.append(len(wrong_history_item_positions))

    array_paths = {
        "feature_request_indices": output_root / "feature_request_indices.npy",
        "candidate_offsets": output_root / "candidate_offsets.npy",
        "candidate_item_positions": output_root / "candidate_item_positions.npy",
        "true_history_offsets": output_root / "true_history_offsets.npy",
        "true_history_item_positions": output_root / "true_history_item_positions.npy",
        "wrong_history_offsets": output_root / "wrong_history_offsets.npy",
        "wrong_history_item_positions": output_root / "wrong_history_item_positions.npy",
    }
    arrays = {
        "feature_request_indices": np.asarray(selected_indices, dtype=np.int64),
        "candidate_offsets": np.asarray(candidate_offsets, dtype=np.int64),
        "candidate_item_positions": np.asarray(candidate_item_positions, dtype=np.int32),
        "true_history_offsets": np.asarray(true_history_offsets, dtype=np.int64),
        "true_history_item_positions": np.asarray(true_history_item_positions, dtype=np.int32),
        "wrong_history_offsets": np.asarray(wrong_history_offsets, dtype=np.int64),
        "wrong_history_item_positions": np.asarray(wrong_history_item_positions, dtype=np.int32),
    }
    for name, array in arrays.items():
        np.save(array_paths[name], array, allow_pickle=False)

    request_path = output_root / "requests.jsonl"
    with request_path.open("w", encoding="utf-8") as handle:
        for position, (index, request_id, query) in enumerate(
            zip(selected_indices, request_ids, queries)
        ):
            handle.write(
                json.dumps(
                    {
                        "position": position,
                        "record_index": index,
                        "request_id": request_id,
                        "text": query,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    item_path = output_root / "items.jsonl"
    with item_path.open("w", encoding="utf-8") as handle:
        for position, item_id in enumerate(item_ids):
            handle.write(
                json.dumps(
                    {
                        "position": position,
                        "item_id": item_id,
                        "text": item_text[item_id],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    manifest = {
        "candidate_id": "c38",
        "stage": "label_free_feature_collection",
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "roles": role_names,
        "requests": len(selected_indices),
        "items": len(item_ids),
        "candidate_rows": len(candidate_item_positions),
        "true_history_rows": len(true_history_item_positions),
        "wrong_history_rows": len(wrong_history_item_positions),
        "item_text_conflicts_resolved": text_conflicts,
        "text_construction": "title + brand + nonempty category path",
        "label_access": {
            "records_train_blind_opened": True,
            "records_train_labels_opened": False,
            "dev_test_records_labels_qrels_opened": False,
        },
        "files": {
            **{
                name: _file_info(path)
                for name, path in array_paths.items()
            },
            "requests": _file_info(request_path),
            "items": _file_info(item_path),
        },
    }
    manifest_path = output_root / "feature_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def encode_feature_shard(
    *,
    feature_root: str | Path,
    snapshot: str | Path,
    output_path: str | Path,
    shard_id: int,
    num_shards: int,
    device: str,
    item_batch_size: int = 256,
    query_batch_size: int = 128,
    item_max_length: int = 128,
    query_max_length: int = 256,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
) -> dict[str, Any]:
    if not (0 <= shard_id < num_shards):
        raise ValueError("invalid C38 embedding shard")
    feature_root = Path(feature_root)
    snapshot = Path(snapshot)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True).to(device).eval()
    items = _read_text_shard(
        feature_root / "items.jsonl",
        shard_id=shard_id,
        num_shards=num_shards,
        prefix="",
    )
    queries = _read_text_shard(
        feature_root / "requests.jsonl",
        shard_id=shard_id,
        num_shards=num_shards,
        prefix=query_prefix,
    )
    item_positions, item_embeddings = _encode_rows(
        items,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=item_batch_size,
        max_length=item_max_length,
    )
    query_positions, query_embeddings = _encode_rows(
        queries,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=query_batch_size,
        max_length=query_max_length,
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            item_positions=item_positions,
            item_embeddings=item_embeddings.astype(np.float16),
            query_positions=query_positions,
            query_embeddings=query_embeddings.astype(np.float16),
        )
    temporary.replace(output_path)
    report = {
        "candidate_id": "c38",
        "stage": "embedding_shard",
        "shard_id": shard_id,
        "num_shards": num_shards,
        "physical_device": device,
        "snapshot": str(snapshot),
        "snapshot_config_sha256": sha256_file(snapshot / "config.json"),
        "query_prefix": query_prefix,
        "item_max_length": item_max_length,
        "query_max_length": query_max_length,
        "item_positions": len(item_positions),
        "query_positions": len(query_positions),
        "embedding_dim": int(item_embeddings.shape[1] if len(item_embeddings) else query_embeddings.shape[1]),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(output_path.with_suffix(".json"), report)
    return report


def finalize_embeddings(
    *,
    feature_root: str | Path,
    shard_paths: Iterable[str | Path],
    embedding_dim: int,
    repeat_boost: float = 3.0,
) -> dict[str, Any]:
    feature_root = Path(feature_root)
    manifest = read_json(feature_root / "feature_manifest.json")
    item_embeddings = np.empty((int(manifest["items"]), embedding_dim), dtype=np.float16)
    query_embeddings = np.empty((int(manifest["requests"]), embedding_dim), dtype=np.float16)
    item_seen = np.zeros(len(item_embeddings), dtype=bool)
    query_seen = np.zeros(len(query_embeddings), dtype=bool)
    shard_info = []
    for raw_path in shard_paths:
        path = Path(raw_path)
        with np.load(path, allow_pickle=False) as shard:
            item_positions = np.asarray(shard["item_positions"], dtype=np.int64)
            query_positions = np.asarray(shard["query_positions"], dtype=np.int64)
            if item_seen[item_positions].any() or query_seen[query_positions].any():
                raise ValueError("C38 embedding shards overlap")
            item_embeddings[item_positions] = shard["item_embeddings"]
            query_embeddings[query_positions] = shard["query_embeddings"]
            item_seen[item_positions] = True
            query_seen[query_positions] = True
        shard_info.append({"path": str(path), "sha256": sha256_file(path)})
    if not item_seen.all() or not query_seen.all():
        raise ValueError("C38 embedding shards do not cover every position")
    item_path = feature_root / "item_embeddings.npy"
    query_path = feature_root / "query_embeddings.npy"
    np.save(item_path, item_embeddings, allow_pickle=False)
    np.save(query_path, query_embeddings, allow_pickle=False)

    request_indices = np.load(feature_root / "feature_request_indices.npy", mmap_mode="r")
    candidate_offsets = np.load(feature_root / "candidate_offsets.npy", mmap_mode="r")
    candidate_positions = np.load(
        feature_root / "candidate_item_positions.npy",
        mmap_mode="r",
    )
    true_history_offsets = np.load(feature_root / "true_history_offsets.npy", mmap_mode="r")
    true_history_positions = np.load(
        feature_root / "true_history_item_positions.npy",
        mmap_mode="r",
    )
    if len(candidate_offsets) != len(request_indices) + 1:
        raise ValueError("C38 candidate offsets differ")
    base_scores = np.empty(len(candidate_positions), dtype=np.float32)
    for row in range(len(request_indices)):
        start, stop = int(candidate_offsets[row]), int(candidate_offsets[row + 1])
        query = np.asarray(query_embeddings[row], dtype=np.float32)
        row_candidate_positions = np.asarray(candidate_positions[start:stop], dtype=np.int64)
        candidates = np.asarray(item_embeddings[row_candidate_positions], dtype=np.float32)
        history_start = int(true_history_offsets[row])
        history_stop = int(true_history_offsets[row + 1])
        history_set = set(
            int(value) for value in true_history_positions[history_start:history_stop]
        )
        recurrence = np.asarray(
            [float(int(position) in history_set) for position in row_candidate_positions],
            dtype=np.float32,
        )
        base_scores[start:stop] = candidates @ query + float(repeat_boost) * recurrence
    base_path = feature_root / "base_scores.npy"
    np.save(base_path, base_scores, allow_pickle=False)
    output = {
        "candidate_id": "c38",
        "stage": "frozen_embedding_finalize",
        "embedding_dim": embedding_dim,
        "items": len(item_embeddings),
        "requests": len(query_embeddings),
        "candidate_rows": len(base_scores),
        "repeat_boost": float(repeat_boost),
        "shards": shard_info,
        "files": {
            "item_embeddings": _file_info(item_path),
            "query_embeddings": _file_info(query_path),
            "base_scores": _file_info(base_path),
        },
        "finite": bool(
            np.isfinite(item_embeddings).all()
            and np.isfinite(query_embeddings).all()
            and np.isfinite(base_scores).all()
        ),
    }
    write_json(feature_root / "embedding_manifest.json", output)
    return output


def _item_text(item: Mapping[str, Any]) -> str:
    categories = item.get("cat")
    category_text = " > ".join(
        str(value) for value in categories if value
    ) if isinstance(categories, list) else ""
    parts = [str(item.get("title") or ""), str(item.get("brand") or ""), category_text]
    return " ".join(part.strip() for part in parts if part.strip())


def _read_text_shard(
    path: Path,
    *,
    shard_id: int,
    num_shards: int,
    prefix: str,
) -> list[tuple[int, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            position = int(row["position"])
            if position % num_shards == shard_id:
                rows.append((position, prefix + str(row["text"])))
    return rows


def _encode_rows(
    rows: list[tuple[int, str]],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray([position for position, _ in rows], dtype=np.int64)
    if not rows:
        dim = int(model.config.hidden_size)
        return positions, np.empty((0, dim), dtype=np.float32)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            texts = [text for _, text in rows[start : start + batch_size]]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            state = model(**encoded).last_hidden_state[:, 0]
            outputs.append(F.normalize(state.float(), dim=-1, eps=1e-6).cpu().numpy())
    return positions, np.concatenate(outputs).astype(np.float32, copy=False)


def _file_info(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
