#!/usr/bin/env python
"""Prepare label-safe C02 structural arrays and frozen query states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from transformers import AutoModel

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.data import _request_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="systems/02_history_hyperadapter/configs/screen.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def deepest_category(values: Any) -> str:
    categories = [str(value) for value in (values or [])]
    for value in reversed(categories):
        if value and value.upper() != "UNKNOWN":
            return value
    return "__UNKNOWN__"


def main() -> int:
    started = time.time()
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_config(config)
    paths = config["paths"]
    feature_root = Path(paths["feature_root"])
    feature_root.mkdir(parents=True, exist_ok=True)
    expected_hash = str(config["integrity"]["candidate_manifest_sha256"])
    candidate_manifest = Path(paths["candidate_manifest"])
    actual_hash = sha256_file(candidate_manifest)
    if actual_hash != expected_hash:
        raise ValueError(f"candidate manifest hash mismatch: {actual_hash}")

    category_map: dict[str, int] = {"__UNKNOWN__": 0}
    category_samples: dict[int, list[int]] = defaultdict(list)
    sample_seen: dict[int, set[int]] = defaultdict(set)
    split_reports: dict[str, Any] = {}
    user_ids_by_split: dict[str, list[str]] = {}
    for split in ("train", "dev"):
        report, user_ids = _materialize_structure(
            config,
            split,
            category_map,
            category_samples,
            sample_seen,
        )
        split_reports[split] = report
        user_ids_by_split[split] = user_ids
        if split == "train":
            write_json(feature_root / "category_map.json", category_map)

    checkpoint = torch.load(
        Path(paths["d2t_checkpoint"]), map_location="cpu", weights_only=False
    )
    model_state = checkpoint["model_state"]
    item_adapter_weight = model_state["item_adapter.weight"].float().cpu().numpy()
    np.save(feature_root / "item_adapter_weight.npy", item_adapter_weight)
    logit_scale = float(model_state["logit_scale"].exp().clamp(1.0, 100.0))
    write_json(feature_root / "base_parameters.json", {"logit_scale": logit_scale})

    query_reports = _encode_queries(config, model_state, args.device)
    centroid_report = _write_category_centroids(
        config, category_map, category_samples
    )
    dev_base_report = _materialize_dev_base_scores(config)

    local_files: dict[str, Any] = {}
    for path in sorted(feature_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            local_files[str(path.relative_to(feature_root))] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    manifest = {
        "analysis_id": config["analysis_id"],
        "candidate_id": config["candidate_id"],
        "candidate_manifest": {
            "path": str(candidate_manifest),
            "sha256": actual_hash,
        },
        "category_centroids": centroid_report,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dev_base_scores": dev_base_report,
        "d2t_checkpoint": {
            "path": str(paths["d2t_checkpoint"]),
            "sha256": sha256_file(Path(paths["d2t_checkpoint"])),
        },
        "files": local_files,
        "label_boundary": {
            "dev_records_are_label_free": True,
            "separated_evaluation_labels_read": False,
            "held_out_test_data_read": False,
        },
        "query_embeddings": query_reports,
        "splits": split_reports,
        "status": "passed",
        "elapsed_seconds": time.time() - started,
    }
    write_json(feature_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _materialize_structure(
    config: dict[str, Any],
    split: str,
    category_map: dict[str, int],
    category_samples: dict[int, list[int]],
    sample_seen: dict[int, set[int]],
) -> tuple[dict[str, Any], list[str]]:
    paths = config["paths"]
    shared_root = Path(paths["shared_packed_data"])
    feature_root = Path(paths["feature_root"])
    shared = shared_root / split
    output = feature_root / split
    output.mkdir(parents=True, exist_ok=True)
    expected_ids = _request_ids(shared / "request_ids.jsonl")
    expected_set = set(expected_ids)
    candidate_offsets = np.load(shared / "candidate_offsets.npy", mmap_mode="r")
    candidate_indices = np.load(
        shared / "candidate_embedding_indices.npy", mmap_mode="r"
    )
    candidate_item_ids = np.load(shared / "candidate_item_ids.npy", mmap_mode="r")
    history_offsets = np.load(shared / "history_offsets.npy", mmap_mode="r")
    history_indices = np.load(
        shared / "history_embedding_indices.npy", mmap_mode="r"
    )
    candidate_categories = np.full(len(candidate_indices), -1, dtype=np.int32)
    history_categories = np.full(len(history_indices), -1, dtype=np.int32)
    history_item_ids = np.full(len(history_indices), -1, dtype=np.int64)
    history_event_weights = np.zeros(len(history_indices), dtype=np.float32)
    observed_ids: list[str] = []
    user_ids: list[str] = []
    candidate_cursor = 0
    history_cursor = 0
    max_samples = int(config["data"]["category_centroid_items"])
    records_path = Path(paths["standardized_dir"]) / f"records_{split}.jsonl"
    for record in iter_jsonl(records_path):
        request_id = str(record["request_id"])
        if request_id not in expected_set:
            continue
        request_index = len(observed_ids)
        if request_index >= len(expected_ids) or expected_ids[request_index] != request_id:
            raise ValueError(f"packed/record order mismatch at {request_id}")
        observed_ids.append(request_id)
        user_ids.append(str(record["user_id"]))
        cs = int(candidate_offsets[request_index])
        ce = int(candidate_offsets[request_index + 1])
        candidates = list(record["candidates"])
        if len(candidates) != ce - cs:
            raise ValueError(f"candidate count mismatch for {request_id}")
        for offset, candidate in enumerate(candidates):
            flat = cs + offset
            if int(candidate_item_ids[flat]) != int(candidate["item_id"]):
                raise ValueError(f"candidate identity mismatch for {request_id}")
            category = deepest_category(candidate.get("cat"))
            category_id = _category_id(category_map, category, split)
            candidate_categories[flat] = category_id
            if split == "train":
                _sample_category_item(
                    category_id,
                    int(candidate_indices[flat]),
                    category_samples,
                    sample_seen,
                    max_samples,
                )
            candidate_cursor += 1

        hs = int(history_offsets[request_index])
        he = int(history_offsets[request_index + 1])
        events = list(record.get("history") or [])
        if len(events) != he - hs:
            raise ValueError(f"history count mismatch for {request_id}")
        size = len(events)
        for offset, event in enumerate(events):
            flat = hs + offset
            history_item_ids[flat] = int(event["item_id"])
            reverse_age = size - offset
            recency = 1.0 / math.sqrt(reverse_age)
            event_multiplier = 1.5 if str(event.get("event")) == "purchase" else 1.0
            history_event_weights[flat] = recency * event_multiplier
            category = deepest_category(event.get("cat"))
            category_id = _category_id(category_map, category, split)
            history_categories[flat] = category_id
            if split == "train":
                _sample_category_item(
                    category_id,
                    int(history_indices[flat]),
                    category_samples,
                    sample_seen,
                    max_samples,
                )
            history_cursor += 1
    if observed_ids != expected_ids:
        raise ValueError(
            f"record coverage mismatch for {split}: {len(observed_ids)} != {len(expected_ids)}"
        )
    if candidate_cursor != len(candidate_indices) or history_cursor != len(history_indices):
        raise ValueError(f"row coverage mismatch for {split}")
    if np.any(candidate_categories < 0) or np.any(history_categories < 0):
        raise ValueError(f"unassigned category IDs for {split}")

    wrong = _wrong_donors(expected_ids, user_ids, history_offsets, config["seed"])
    np.save(output / "candidate_category_ids.npy", candidate_categories)
    np.save(output / "history_category_ids.npy", history_categories)
    np.save(output / "history_item_ids.npy", history_item_ids)
    np.save(output / "history_event_weights.npy", history_event_weights)
    np.save(output / "wrong_request_indices.npy", wrong)
    history_counts = np.diff(history_offsets)
    return (
        {
            "candidate_rows": len(candidate_indices),
            "history_present": int(np.count_nonzero(history_counts)),
            "history_rows": len(history_indices),
            "records_path": str(records_path),
            "requests": len(expected_ids),
            "wrong_different_user_violations": int(
                sum(
                    history_counts[index] > 0
                    and user_ids[index] == user_ids[int(wrong[index])]
                    for index in range(len(expected_ids))
                )
            ),
        },
        user_ids,
    )


def _category_id(mapping: dict[str, int], category: str, split: str) -> int:
    if category in mapping:
        return mapping[category]
    if split == "dev":
        return mapping["__UNKNOWN__"]
    value = len(mapping)
    mapping[category] = value
    return value


def _sample_category_item(
    category_id: int,
    embedding_index: int,
    samples: dict[int, list[int]],
    seen: dict[int, set[int]],
    limit: int,
) -> None:
    if len(samples[category_id]) >= limit or embedding_index in seen[category_id]:
        return
    seen[category_id].add(embedding_index)
    samples[category_id].append(embedding_index)


def _wrong_donors(
    request_ids: list[str],
    user_ids: list[str],
    history_offsets: np.ndarray,
    seed: int,
) -> np.ndarray:
    counts = np.diff(history_offsets)
    present = [index for index, count in enumerate(counts) if count > 0]
    present.sort(
        key=lambda index: hashlib.sha256(
            f"{seed}:{request_ids[index]}".encode("utf-8")
        ).hexdigest()
    )
    if len(present) < 2:
        raise ValueError("not enough history-present requests for wrong donors")
    wrong = np.arange(len(request_ids), dtype=np.int64)
    shift = max(1, len(present) // 3)
    for position, index in enumerate(present):
        donor = present[(position + shift) % len(present)]
        attempts = 0
        while user_ids[donor] == user_ids[index] and attempts < len(present):
            donor = present[(position + shift + attempts + 1) % len(present)]
            attempts += 1
        if user_ids[donor] == user_ids[index]:
            raise ValueError(f"unable to find different-user donor for {request_ids[index]}")
        wrong[index] = donor
    return wrong


def _encode_queries(
    config: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    device: str,
) -> dict[str, Any]:
    paths = config["paths"]
    encoder = AutoModel.from_pretrained(
        config["base"]["model_name"],
        local_files_only=bool(config["base"]["local_files_only"]),
    )
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(device).eval()
    batch_size = int(config["base"]["query_embedding_batch_size"])
    reports: dict[str, Any] = {}
    for split in ("train", "dev"):
        token_root = Path(paths["shared_query_tokens"])
        input_ids = np.load(token_root / f"{split}_input_ids.npy", mmap_mode="r")
        attention = np.load(
            token_root / f"{split}_attention_mask.npy", mmap_mode="r"
        )
        output_path = Path(paths["feature_root"]) / split / "query_embeddings.npy"
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(input_ids), int(config["model"]["input_dim"])),
        )
        with torch.inference_mode():
            for start in range(0, len(input_ids), batch_size):
                end = min(start + batch_size, len(input_ids))
                ids = torch.from_numpy(
                    np.asarray(input_ids[start:end], dtype=np.int64)
                ).to(device)
                mask = torch.from_numpy(
                    np.asarray(attention[start:end], dtype=np.int64)
                ).to(device)
                hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                query = F.normalize(hidden[:, 0, :].float(), dim=-1, eps=1e-6)
                output[start:end] = query.cpu().numpy().astype(np.float16)
        output.flush()
        reports[split] = {
            "path": str(output_path),
            "rows": len(input_ids),
            "sha256": sha256_file(output_path),
        }
    del encoder
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return reports


def _write_category_centroids(
    config: dict[str, Any],
    category_map: dict[str, int],
    category_samples: dict[int, list[int]],
) -> dict[str, Any]:
    embeddings = np.load(config["paths"]["shared_item_embeddings"], mmap_mode="r")
    centroids = np.zeros((len(category_map), embeddings.shape[1]), dtype=np.float32)
    nonempty = 0
    for category_id in range(len(category_map)):
        indices = category_samples.get(category_id, [])
        if not indices:
            continue
        values = np.asarray(embeddings[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
        centroid = values.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid /= norm
        centroids[category_id] = centroid
        nonempty += 1
    output = Path(config["paths"]["feature_root"]) / "category_centroids.npy"
    np.save(output, centroids.astype(np.float16))
    return {
        "categories": len(category_map),
        "nonempty": nonempty,
        "path": str(output),
        "sha256": sha256_file(output),
    }


def _materialize_dev_base_scores(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    shared = Path(paths["shared_packed_data"]) / "dev"
    request_ids = _request_ids(shared / "request_ids.jsonl")
    candidate_offsets = np.load(shared / "candidate_offsets.npy", mmap_mode="r")
    candidate_item_ids = np.load(shared / "candidate_item_ids.npy", mmap_mode="r")
    scores_path = Path(paths["d2p_dev_scores"])
    output_path = Path(paths["feature_root"]) / "dev" / "base_scores.npy"
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float64, shape=(len(candidate_item_ids),)
    )
    score_rows = iter_jsonl(scores_path)
    rows = 0
    for request_index, request_id in enumerate(request_ids):
        start = int(candidate_offsets[request_index])
        end = int(candidate_offsets[request_index + 1])
        for flat in range(start, end):
            row = next(score_rows, None)
            if row is None:
                raise ValueError("D2p score file ended early")
            expected_item = str(int(candidate_item_ids[flat]))
            if (
                str(row["request_id"]) != request_id
                or str(row["candidate_item_id"]) != expected_item
            ):
                raise ValueError(
                    "D2p score order mismatch: "
                    f"expected {request_id}/{expected_item}, got "
                    f"{row.get('request_id')}/{row.get('candidate_item_id')}"
                )
            output[flat] = float(row["score"])
            rows += 1
    if next(score_rows, None) is not None:
        raise ValueError("D2p score file contains extra rows")
    output.flush()
    return {
        "path": str(output_path),
        "rows": rows,
        "sha256": sha256_file(output_path),
        "source_path": str(scores_path),
        "source_sha256": sha256_file(scores_path),
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("candidate_id") != "c02":
        raise ValueError("configuration is not C02")
    if int(config.get("physical_gpu", -1)) != 1:
        raise ValueError("C02 must use physical GPU 1")
    if int(config.get("seed", -1)) != 20260708:
        raise ValueError("unexpected seed")
    if not str(config.get("run_id", "")).startswith("20260710_kuaisearch_c02_"):
        raise ValueError("invalid C02 run prefix")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen query-state preparation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("C02 process must see exactly one GPU")


if __name__ == "__main__":
    raise SystemExit(main())
