#!/usr/bin/env python
"""Prepare label-free HSO folds/donors, then freeze the execution lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.history_signal_observability import (  # noqa: E402
    FrozenSemanticStore,
    PackedObservabilityData,
    atomic_json,
    fold_for_user,
    load_config,
    sha256_array,
    sha256_file,
)


SOURCE_KEYS = ("module", "prepare_script", "run_script", "summarize_script", "protocol")
INPUT_FILES = (
    "packed_manifest",
    "train_candidate_labels",
    "raw_query_embeddings",
    "raw_item_embeddings",
    "request_metadata",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "freeze", "stage-labels"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_users(path: Path, request_ids: list[str]) -> list[str]:
    positions = {request_id: index for index, request_id in enumerate(request_ids)}
    users: list[str | None] = [None] * len(request_ids)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            position = positions.get(str(row["request_id"]))
            if position is not None:
                users[position] = str(row["user_id"])
    if any(value is None for value in users):
        raise ValueError("HSO request metadata does not cover packed train")
    return [str(value) for value in users]


def history_present_indices(data: PackedObservabilityData) -> np.ndarray:
    counts = np.diff(data.history_offsets)
    return np.flatnonzero(counts > 0).astype(np.int64)


def matched_donors(
    data: PackedObservabilityData,
    features: FrozenSemanticStore,
    users: list[str],
    targets: np.ndarray,
    pool: np.ndarray,
    *,
    topk: int,
    batch_size: int,
    ratio_max: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    pool_query = torch.from_numpy(features.query(pool)).to(device)
    pool_query = F.normalize(pool_query.float(), dim=-1, eps=1e-6)
    pool_transpose = pool_query.transpose(0, 1).contiguous()
    pool_lengths = np.diff(data.history_offsets)[pool]
    target_lengths = np.diff(data.history_offsets)[targets]
    donors = np.empty(len(targets), dtype=np.int64)
    selected_similarity = np.empty(len(targets), dtype=np.float32)
    ratio_matched = np.zeros(len(targets), dtype=bool)
    different_user = np.zeros(len(targets), dtype=bool)
    effective_topk = min(int(topk), len(pool))
    for start in range(0, len(targets), int(batch_size)):
        stop = min(start + int(batch_size), len(targets))
        query = torch.from_numpy(features.query(targets[start:stop])).to(device)
        query = F.normalize(query.float(), dim=-1, eps=1e-6)
        similarities, neighbor_positions = torch.topk(
            query @ pool_transpose, k=effective_topk, dim=-1
        )
        positions = neighbor_positions.cpu().numpy()
        values = similarities.cpu().numpy()
        for local, target in enumerate(targets[start:stop]):
            target = int(target)
            target_user = users[target]
            target_length = max(1, int(target_lengths[start + local]))
            chosen: tuple[int, float, bool] | None = None
            fallback: tuple[int, float, bool] | None = None
            for rank, pool_position in enumerate(positions[local]):
                donor = int(pool[int(pool_position)])
                if donor == target or users[donor] == target_user:
                    continue
                ratio = max(
                    target_length / max(1, int(pool_lengths[int(pool_position)])),
                    max(1, int(pool_lengths[int(pool_position)])) / target_length,
                )
                candidate = (donor, float(values[local, rank]), ratio <= ratio_max)
                if fallback is None:
                    fallback = candidate
                if ratio <= ratio_max:
                    chosen = candidate
                    break
            if chosen is None:
                chosen = fallback
            if chosen is None:
                raise ValueError(f"HSO cannot find different-user donor for {target}")
            donors[start + local] = chosen[0]
            selected_similarity[start + local] = chosen[1]
            ratio_matched[start + local] = chosen[2]
            different_user[start + local] = users[chosen[0]] != target_user
    return donors, {
        "targets": len(targets),
        "pool": len(pool),
        "topk": effective_topk,
        "different_user_rate": float(different_user.mean()),
        "history_length_ratio_matched_rate": float(ratio_matched.mean()),
        "query_cosine_mean": float(selected_similarity.mean()),
        "query_cosine_median": float(np.median(selected_similarity)),
        "query_cosine_min": float(selected_similarity.min()),
    }


def save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / name
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def prepare(config: dict[str, Any], device_name: str) -> None:
    paths = config["paths"]
    output = ROOT / paths["artifact_root"]
    manifest_path = output / "selection_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    data = PackedObservabilityData(ROOT / paths["packed_train_root"])
    features = FrozenSemanticStore(
        data,
        ROOT / paths["raw_query_embeddings"],
        ROOT / paths["raw_item_embeddings"],
    )
    users = load_users(ROOT / paths["request_metadata"], data.request_ids)
    selected = data.strict_nonrepeat_indices()
    if data.labels_opened:
        raise PermissionError("HSO selection opened train labels")
    folds = int(config["selection"]["folds"])
    namespace = str(config["selection"]["fold_namespace"])
    assignments = np.asarray(
        [fold_for_user(users[int(index)], namespace, folds) for index in selected],
        dtype=np.int8,
    )
    pool = history_present_indices(data)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("HSO donor preparation requires available CUDA")
    donors, donor_audit = matched_donors(
        data,
        features,
        users,
        selected,
        pool,
        topk=int(config["selection"]["donor_topk"]),
        batch_size=int(config["selection"]["donor_query_batch"]),
        ratio_max=float(config["selection"]["donor_history_length_ratio_max"]),
        device=device,
    )
    user_values = np.asarray([users[int(index)] for index in selected])
    fold_rows = []
    overlap = True
    for fold in range(folds):
        evaluation = selected[assignments == fold]
        fit = selected[assignments != fold]
        fit_users = {users[int(value)] for value in fit}
        evaluation_users = {users[int(value)] for value in evaluation}
        disjoint = not bool(fit_users & evaluation_users)
        overlap = overlap and disjoint
        fold_rows.append(
            {
                "fold": fold,
                "fit_requests": len(fit),
                "evaluation_requests": len(evaluation),
                "fit_users": len(fit_users),
                "evaluation_users": len(evaluation_users),
                "user_disjoint": disjoint,
                "evaluation_candidate_hash": data.candidate_hash(evaluation),
            }
        )
    files = {
        "strict_indices.npy": save_array(output, "strict_indices.npy", selected),
        "fold_assignments.npy": save_array(output, "fold_assignments.npy", assignments),
        "wrong_request_indices.npy": save_array(
            output, "wrong_request_indices.npy", donors
        ),
    }
    users_path = output / "strict_request_users.jsonl"
    with users_path.open("w", encoding="utf-8") as handle:
        for index, user in zip(selected, user_values):
            handle.write(
                json.dumps(
                    {"request_index": int(index), "user_id": str(user)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    files["strict_request_users.jsonl"] = {
        "path": str(users_path.relative_to(ROOT)),
        "sha256": sha256_file(users_path),
        "rows": len(selected),
    }
    manifest = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "label_free_selection",
        "strict_nonrepeat_requests": len(selected),
        "history_present_requests": len(pool),
        "folds": fold_rows,
        "all_fold_users_disjoint": overlap,
        "donor_audit": donor_audit,
        "candidate_rows": int(
            sum(
                int(data.candidate_offsets[int(index) + 1] - data.candidate_offsets[int(index)])
                for index in selected
            )
        ),
        "history_rows": int(
            sum(
                int(data.history_offsets[int(index) + 1] - data.history_offsets[int(index)])
                for index in selected
            )
        ),
        "selection_sha256": sha256_array(selected),
        "assignment_sha256": sha256_array(assignments),
        "labels_opened": data.labels_opened,
        "dev_test_qrels_opened": False,
        "files": files,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def freeze(config: dict[str, Any], config_path: Path) -> None:
    paths = config["paths"]
    lock_path = ROOT / paths["execution_lock"]
    if lock_path.exists():
        raise FileExistsError(lock_path)
    selection_path = ROOT / paths["artifact_root"] / "selection_manifest.json"
    if not selection_path.exists():
        raise FileNotFoundError(selection_path)
    source_paths = {"config": config_path}
    source_paths.update({key: ROOT / paths[key] for key in SOURCE_KEYS})
    input_paths = {key: ROOT / paths[key] for key in INPUT_FILES}
    lock = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "authorize_four_mode_three_fold_train_internal_observability_run",
        "source_sha256": {
            key: sha256_file(value) for key, value in source_paths.items()
        },
        "input_sha256": {
            key: sha256_file(value) for key, value in input_paths.items()
        },
        "selection_manifest_sha256": sha256_file(selection_path),
        "outcome_boundary": {
            "fit_fold_train_labels": True,
            "heldout_fold_labels_before_all_scores": False,
            "dev_test_qrels": False,
        },
    }
    atomic_json(lock_path, lock)
    print(
        json.dumps(
            {"path": str(lock_path), "sha256": sha256_file(lock_path)},
            sort_keys=True,
        )
    )


def verify_lock(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    paths = config["paths"]
    lock_path = ROOT / paths["execution_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source_paths = {"config": config_path}
    source_paths.update({key: ROOT / paths[key] for key in SOURCE_KEYS})
    input_paths = {key: ROOT / paths[key] for key in INPUT_FILES}
    observed_sources = {key: sha256_file(value) for key, value in source_paths.items()}
    observed_inputs = {key: sha256_file(value) for key, value in input_paths.items()}
    selection_path = ROOT / paths["artifact_root"] / "selection_manifest.json"
    if observed_sources != lock["source_sha256"]:
        raise RuntimeError("HSO source changed after execution lock")
    if observed_inputs != lock["input_sha256"]:
        raise RuntimeError("HSO input changed after execution lock")
    if sha256_file(selection_path) != lock["selection_manifest_sha256"]:
        raise RuntimeError("HSO selection changed after execution lock")
    return lock


def stage_labels(config: dict[str, Any], config_path: Path) -> None:
    verify_lock(config, config_path)
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    manifest_path = root / "fold_label_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    data = PackedObservabilityData(ROOT / paths["packed_train_root"])
    selected = np.load(root / "strict_indices.npy", mmap_mode="r")
    assignments = np.load(root / "fold_assignments.npy", mmap_mode="r")
    source_labels = data.open_train_labels()
    files: dict[str, Any] = {}
    folds: list[dict[str, Any]] = []
    for fold in range(int(config["selection"]["folds"])):
        fit = np.asarray(selected[np.asarray(assignments) != fold], dtype=np.int64)
        offsets = [0]
        rows: list[np.ndarray] = []
        for index_value in fit:
            index = int(index_value)
            start, stop = (
                int(data.candidate_offsets[index]),
                int(data.candidate_offsets[index + 1]),
            )
            row = np.asarray(source_labels[start:stop], dtype=np.uint8).copy()
            if not bool((row > 0).any()):
                raise ValueError("HSO fit staging found a zero-positive request")
            rows.append(row)
            offsets.append(offsets[-1] + len(row))
        values = np.concatenate(rows) if rows else np.empty(0, dtype=np.uint8)
        outputs = {
            f"fit_fold{fold}_request_indices.npy": fit,
            f"fit_fold{fold}_label_offsets.npy": np.asarray(offsets, dtype=np.int64),
            f"fit_fold{fold}_labels.npy": values,
        }
        for name, value in outputs.items():
            files[name] = save_array(root, name, value)
        evaluation = np.asarray(
            selected[np.asarray(assignments) == fold], dtype=np.int64
        )
        folds.append(
            {
                "fold": fold,
                "fit_requests": len(fit),
                "evaluation_requests": len(evaluation),
                "fit_evaluation_overlap": int(
                    len(set(fit.tolist()) & set(evaluation.tolist()))
                ),
                "fit_candidate_rows": len(values),
                "fit_positive_rows": int((values > 0).sum()),
            }
        )
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "post_lock_fit_fold_label_staging",
        "source_train_labels_opened_by_stager": True,
        "heldout_labels_written_to_fit_artifacts": False,
        "dev_test_qrels_opened": False,
        "folds": folds,
        "files": files,
    }
    atomic_json(manifest_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.stage == "prepare":
        prepare(config, args.device)
    elif args.stage == "freeze":
        freeze(config, config_path)
    else:
        stage_labels(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
