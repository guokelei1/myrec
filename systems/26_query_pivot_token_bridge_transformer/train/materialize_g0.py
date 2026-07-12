"""Materialize label-free token arrays and reuse only C25 compact fit labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.locking import verify_proposal_lock  # noqa: E402
from train.real_data import copy_compact_labels  # noqa: E402
from train.structure import (  # noqa: E402
    FEATURE_ROLES,
    PackedStructure,
    atomic_json,
    candidate_key_sha256,
    load_config,
    read_json,
    sha256_file,
)


def save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / name
    np.save(path, value)
    loaded = np.load(path, mmap_mode="r")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
    }


def assert_cuda(config: Mapping[str, Any], device: str) -> None:
    physical = int(config["resources"]["physical_gpu"])
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C26 GPU registration mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C26 requires one visible CUDA GPU")


def tokenize_items(
    *,
    corpus_path: str | Path,
    item_mapping_path: str | Path,
    item_indices: np.ndarray,
    tokenizer: Any,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mapping = read_json(item_mapping_path)
    token_ids = np.zeros((len(item_indices), max_length), dtype=np.int32)
    attention = np.zeros((len(item_indices), max_length), dtype=bool)
    content = np.zeros((len(item_indices), max_length), dtype=bool)
    special = np.asarray(sorted(set(int(value) for value in tokenizer.all_special_ids)), dtype=np.int64)
    pointer = 0
    positions: list[int] = []
    titles: list[str] = []

    def flush() -> None:
        if not titles:
            return
        encoded = tokenizer(
            titles,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="np",
        )
        values = np.asarray(encoded["input_ids"], dtype=np.int32)
        masks = np.asarray(encoded["attention_mask"], dtype=bool)
        token_ids[positions] = values
        attention[positions] = masks
        content[positions] = masks & ~np.isin(values, special)
        positions.clear()
        titles.clear()

    with Path(corpus_path).open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if pointer >= len(item_indices):
                break
            target = int(item_indices[pointer])
            if row < target:
                continue
            if row != target:
                raise ValueError(f"C26 corpus row skipped: {row}/{target}")
            value = json.loads(line)
            item_id = str(value["item_id"])
            if int(mapping[item_id]) != row:
                raise ValueError("C26 corpus/item mapping differs")
            positions.append(pointer)
            titles.append(str(value.get("item_title", "")))
            pointer += 1
            if len(titles) >= 4096:
                flush()
    flush()
    if pointer != len(item_indices):
        raise ValueError(f"C26 corpus coverage differs: {pointer}/{len(item_indices)}")
    return token_ids, attention, content, {
        "items": len(item_indices),
        "nonempty_content": int(content.any(axis=1).sum()),
        "content_tokens": int(content.sum()),
        "special_token_ids": special.tolist(),
        "mapping_entries": len(mapping),
    }


def materialize(config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    config = load_config(config_path, require_selection=True)
    assert_cuda(config, device)
    _, proposal_hash = verify_proposal_lock(config)
    paths = config["paths"]
    root = Path(paths["artifact_root"])
    report_path = root / "g0_report.json"
    if report_path.exists():
        raise FileExistsError("immutable C26 G0 exists")
    for name, expected_name in (
        ("c25_selection", "c25_selection_sha256"),
        ("c25_g0_report", "c25_g0_report_sha256"),
        ("c25_train_report", "c25_train_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("query_token_manifest", "query_token_manifest_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected_name]:
            raise ValueError(f"C26 registered source changed: {name}")
    selection = read_json(paths["selection"])
    c25_g0 = read_json(paths["c25_g0_report"])
    c25_outcome = read_json(paths["c25_train_report"])
    if c25_outcome.get("internal_A_labels_opened") is not False or c25_outcome.get(
        "delayed_B_labels_opened"
    ) is not False:
        raise PermissionError("C26 source A/B labels are not untouched")
    data = PackedStructure(paths["packed_train_root"])
    feature_indices = np.asarray(
        [int(value) for role in FEATURE_ROLES for value in selection["roles"][role]["indices"]],
        dtype=np.int64,
    )
    c25_root = Path(paths["c25_artifact_root"])
    source_feature_indices = np.load(c25_root / "feature_request_indices.npy", mmap_mode="r")
    if not np.array_equal(feature_indices, source_feature_indices):
        raise ValueError("C26/C25 feature request order differs")
    item_indices = np.asarray(
        np.load(c25_root / "item_embedding_indices.npy", mmap_mode="r"), dtype=np.int64
    )
    score_offsets = np.asarray(
        np.load(c25_root / "feature_candidate_offsets.npy", mmap_mode="r"), dtype=np.int64
    )
    base_scores = np.asarray(np.load(c25_root / "base_scores.npy", mmap_mode="r"), dtype=np.float32)
    for name in ("feature_request_indices.npy", "item_embedding_indices.npy", "feature_candidate_offsets.npy", "base_scores.npy", "fit_request_indices.npy", "fit_label_offsets.npy", "fit_labels.npy"):
        expected = c25_g0["outputs"][name]["sha256"]
        if sha256_file(c25_root / name) != expected:
            raise RuntimeError(f"C26 frozen C25 source changed: {name}")

    tokenizer = AutoTokenizer.from_pretrained(paths["bge_snapshot"], local_files_only=True)
    tokenization = config["tokenization"]
    item_token_ids, item_attention, item_content, item_audit = tokenize_items(
        corpus_path=paths["corpus"],
        item_mapping_path=paths["item_id2idx"],
        item_indices=item_indices,
        tokenizer=tokenizer,
        max_length=int(tokenization["max_item_tokens"]),
    )
    query_source_ids = np.load(Path(paths["query_tokens"]) / "train_input_ids.npy", mmap_mode="r")
    query_source_attention = np.load(
        Path(paths["query_tokens"]) / "train_attention_mask.npy", mmap_mode="r"
    )
    query_length = int(tokenization["max_query_tokens"])
    query_ids = np.asarray(query_source_ids[feature_indices, :query_length], dtype=np.int32)
    query_attention = np.asarray(
        query_source_attention[feature_indices, :query_length], dtype=bool
    )
    special = np.asarray(sorted(set(int(value) for value in tokenizer.all_special_ids)))
    query_content = query_attention & ~np.isin(query_ids, special)

    encoder = AutoModel.from_pretrained(paths["bge_snapshot"], local_files_only=True)
    embeddings = encoder.embeddings.word_embeddings.weight.detach().cpu().float().numpy()
    padding_idx = int(tokenizer.pad_token_id)
    del encoder

    fit_indices = [int(value) for value in selection["roles"]["fit"]["indices"]]
    fit_labels = copy_compact_labels(c25_root, fit_indices)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_request_indices.npy": save_array(root, "feature_request_indices.npy", feature_indices),
        "item_embedding_indices.npy": save_array(root, "item_embedding_indices.npy", item_indices),
        "feature_candidate_offsets.npy": save_array(
            root, "feature_candidate_offsets.npy", score_offsets
        ),
        "base_scores.npy": save_array(root, "base_scores.npy", base_scores),
        "query_token_ids.npy": save_array(root, "query_token_ids.npy", query_ids),
        "query_attention_mask.npy": save_array(
            root, "query_attention_mask.npy", query_attention
        ),
        "query_content_mask.npy": save_array(root, "query_content_mask.npy", query_content),
        "item_token_ids.npy": save_array(root, "item_token_ids.npy", item_token_ids),
        "item_attention_mask.npy": save_array(
            root, "item_attention_mask.npy", item_attention
        ),
        "item_content_mask.npy": save_array(root, "item_content_mask.npy", item_content),
        "word_embeddings.npy": save_array(root, "word_embeddings.npy", embeddings),
        "fit_request_indices.npy": save_array(
            root, "fit_request_indices.npy", fit_labels.request_indices
        ),
        "fit_label_offsets.npy": save_array(root, "fit_label_offsets.npy", fit_labels.offsets),
        "fit_labels.npy": save_array(root, "fit_labels.npy", fit_labels.values),
    }
    report = {
        "candidate_id": "c26",
        "gate": "G0",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "proposal_lock_sha256": proposal_hash,
        "selection_sha256": paths["selection_sha256"],
        "feature_roles": list(FEATURE_ROLES),
        "fit_labels_reused_from_c25_compact": True,
        "original_train_label_array_opened": False,
        "internal_A_labels_opened": False,
        "delayed_B_labels_opened": False,
        "escrow_features_or_labels_opened": False,
        "train_records_parsed": False,
        "dev_test_qrels_metrics_read": False,
        "candidate_rows": len(base_scores),
        "candidate_key_sha256": candidate_key_sha256(data, feature_indices),
        "base_scores_sha256_matches_c25": outputs["base_scores.npy"]["sha256"]
        == c25_g0["outputs"]["base_scores.npy"]["sha256"],
        "tokenization": {
            "snapshot": paths["bge_snapshot"],
            "padding_idx": padding_idx,
            "vocab_size": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "query_nonempty_content": int(query_content.any(axis=1).sum()),
            **item_audit,
        },
        "outputs": outputs,
        "physical_gpu": int(config["resources"]["physical_gpu"]),
        "primary_dev_evaluator_calls": 0,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    materialize(args.config, args.device)


if __name__ == "__main__":
    main()
