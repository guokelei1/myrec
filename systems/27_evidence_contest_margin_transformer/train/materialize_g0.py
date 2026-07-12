"""Bind frozen C26 token inputs and materialize compact C27 feature rows."""

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
        raise RuntimeError("C27 GPU registration mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C27 requires one visible CUDA GPU")


def materialize(config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    config = load_config(config_path, require_selection=True)
    assert_cuda(config, device)
    _, proposal_hash = verify_proposal_lock(config)
    paths = config["paths"]
    root = Path(paths["artifact_root"])
    report_path = root / "g0_report.json"
    if report_path.exists():
        raise FileExistsError("immutable C27 G0 exists")
    for name, expected_name in (
        ("c26_selection", "c26_selection_sha256"),
        ("c26_g0_report", "c26_g0_report_sha256"),
        ("c26_train_report", "c26_train_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected_name]:
            raise ValueError(f"C27 registered source changed: {name}")
    selection = read_json(paths["selection"])
    c26_g0 = read_json(paths["c26_g0_report"])
    c26_outcome = read_json(paths["c26_train_report"])
    if c26_outcome.get("internal_A_labels_opened") is not False or c26_outcome.get(
        "delayed_B_labels_opened"
    ) is not False:
        raise PermissionError("C27 source A/B labels are not untouched")
    data = PackedStructure(paths["packed_train_root"])
    feature_indices = np.asarray(
        [int(value) for role in FEATURE_ROLES for value in selection["roles"][role]["indices"]],
        dtype=np.int64,
    )
    c26_root = Path(paths["c26_artifact_root"])
    source_indices = np.load(c26_root / "feature_request_indices.npy", mmap_mode="r")
    source_position = {int(index): row for row, index in enumerate(source_indices)}
    if any(int(index) not in source_position for index in feature_indices):
        raise ValueError("C27 selected request absent from C26 token features")
    positions = np.asarray([source_position[int(index)] for index in feature_indices])
    source_offsets = np.load(c26_root / "feature_candidate_offsets.npy", mmap_mode="r")
    source_base = np.load(c26_root / "base_scores.npy", mmap_mode="r")
    offsets = [0]
    base_rows: list[np.ndarray] = []
    for request_index, source_row in zip(feature_indices, positions):
        start, stop = int(source_offsets[source_row]), int(source_offsets[source_row + 1])
        expected = int(data.candidate_offsets[request_index + 1] - data.candidate_offsets[request_index])
        if stop - start != expected:
            raise ValueError("C27/C26 candidate row length differs")
        base_rows.append(np.asarray(source_base[start:stop], dtype=np.float32).copy())
        offsets.append(offsets[-1] + expected)
    base_scores = np.concatenate(base_rows)
    feature_offsets = np.asarray(offsets, dtype=np.int64)
    query_ids = np.asarray(
        np.load(c26_root / "query_token_ids.npy", mmap_mode="r")[positions], dtype=np.int32
    )
    query_attention = np.asarray(
        np.load(c26_root / "query_attention_mask.npy", mmap_mode="r")[positions], dtype=bool
    )
    query_content = np.asarray(
        np.load(c26_root / "query_content_mask.npy", mmap_mode="r")[positions], dtype=bool
    )

    fit_indices = [int(value) for value in selection["roles"]["fit"]["indices"]]
    fit_labels = copy_compact_labels(c26_root, fit_indices)
    token_names = (
        "item_embedding_indices.npy",
        "item_token_ids.npy",
        "item_attention_mask.npy",
        "item_content_mask.npy",
        "word_embeddings.npy",
    )
    bound_inputs = {}
    for name in token_names:
        expected = c26_g0["outputs"][name]["sha256"]
        path = c26_root / name
        if sha256_file(path) != expected:
            raise RuntimeError(f"C27 frozen C26 token input changed: {name}")
        bound_inputs[name] = {"path": str(path), "sha256": expected}

    item_indices = np.load(c26_root / "item_embedding_indices.npy", mmap_mode="r")
    required: list[np.ndarray] = []
    for request_index in feature_indices:
        required.extend((data.candidate_indices(int(request_index)), data.history_indices(int(request_index))))
    for role in ("fit", "internal_A", "delayed_B"):
        for donor in selection["wrong_history_donors"][role]["indices"]:
            required.append(data.history_indices(int(donor)))
    required_indices = np.unique(np.concatenate(required))
    token_positions = np.searchsorted(item_indices, required_indices)
    if bool((token_positions >= len(item_indices)).any()) or not np.array_equal(
        item_indices[token_positions], required_indices
    ):
        raise RuntimeError("C27 required item token coverage differs")

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_request_indices.npy": save_array(root, "feature_request_indices.npy", feature_indices),
        "feature_candidate_offsets.npy": save_array(
            root, "feature_candidate_offsets.npy", feature_offsets
        ),
        "base_scores.npy": save_array(root, "base_scores.npy", base_scores),
        "query_token_ids.npy": save_array(root, "query_token_ids.npy", query_ids),
        "query_attention_mask.npy": save_array(
            root, "query_attention_mask.npy", query_attention
        ),
        "query_content_mask.npy": save_array(root, "query_content_mask.npy", query_content),
        "fit_request_indices.npy": save_array(
            root, "fit_request_indices.npy", fit_labels.request_indices
        ),
        "fit_label_offsets.npy": save_array(root, "fit_label_offsets.npy", fit_labels.offsets),
        "fit_labels.npy": save_array(root, "fit_labels.npy", fit_labels.values),
    }
    report = {
        "candidate_id": "c27",
        "gate": "G0",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "proposal_lock_sha256": proposal_hash,
        "selection_sha256": paths["selection_sha256"],
        "feature_roles": list(FEATURE_ROLES),
        "fit_labels_reused_from_c26_compact": True,
        "original_train_label_array_opened": False,
        "internal_A_labels_opened": False,
        "delayed_B_labels_opened": False,
        "escrow_features_or_labels_opened": False,
        "dev_test_qrels_metrics_read": False,
        "candidate_rows": len(base_scores),
        "candidate_key_sha256": candidate_key_sha256(data, feature_indices),
        "required_item_token_rows": len(required_indices),
        "query_nonempty_content": int(query_content.any(axis=1).sum()),
        "tokenization": {
            "padding_idx": int(c26_g0["tokenization"]["padding_idx"]),
            "vocab_size": int(c26_g0["tokenization"]["vocab_size"]),
            "embedding_dim": int(c26_g0["tokenization"]["embedding_dim"]),
        },
        "outputs": outputs,
        "bound_inputs": bound_inputs,
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
