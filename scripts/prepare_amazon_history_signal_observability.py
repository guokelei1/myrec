#!/usr/bin/env python
"""Prepare, freeze, and stage compact Amazon HSO train-fold labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.amazon_history_signal_observability import (  # noqa: E402
    AmazonObservabilityData,
    load_all_feature_labels,
    load_feature_users,
)
from myrec.analysis.history_signal_observability import (  # noqa: E402
    atomic_json,
    fold_for_user,
    sha256_array,
    sha256_file,
)


SOURCE_KEYS = (
    "protocol",
    "core_module",
    "amazon_module",
    "prepare_script",
    "run_script",
    "summarize_script",
    "shared_run_script",
    "shared_summarizer",
)
FEATURE_FILES = (
    "feature_request_indices.npy",
    "candidate_offsets.npy",
    "candidate_item_positions.npy",
    "true_history_offsets.npy",
    "true_history_item_positions.npy",
    "wrong_history_offsets.npy",
    "wrong_history_item_positions.npy",
    "item_embeddings.npy",
    "query_embeddings.npy",
    "items.jsonl",
    "requests.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("prepare", "freeze", "stage-labels"), required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Amazon HSO config must be a mapping")
    if tuple(value["model"]["modes"]) != ("full", "text", "id"):
        raise ValueError("Amazon HSO modes differ")
    for key in ("dev", "test", "qrels"):
        if bool(value["authorization"][key]):
            raise PermissionError(f"Amazon HSO unauthorized split: {key}")
    return value


def source_paths(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths = config["paths"]
    output = {"config": config_path}
    output.update({key: ROOT / paths[key] for key in SOURCE_KEYS})
    return output


def input_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = config["paths"]
    feature_root = ROOT / paths["feature_root"]
    output = {
        "feature_manifest": ROOT / paths["feature_manifest"],
        "embedding_manifest": ROOT / paths["embedding_manifest"],
        "records_train_blind": ROOT / paths["records_train_blind"],
        "records_train": ROOT / paths["records_train"],
    }
    output.update({f"feature_{name}": feature_root / name for name in FEATURE_FILES})
    return output


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


def prepare(config: dict[str, Any]) -> None:
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    manifest_path = root / "selection_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    root.mkdir(parents=True, exist_ok=True)
    data = AmazonObservabilityData(ROOT / paths["feature_root"])
    users = load_feature_users(ROOT / paths["records_train_blind"], data)
    selected = data.strict_nonrepeat_indices()
    folds = int(config["selection"]["folds"])
    namespace = str(config["selection"]["fold_namespace"])
    assignments = np.asarray(
        [fold_for_user(users[int(index)], namespace, folds) for index in selected],
        dtype=np.int8,
    )
    fold_rows = []
    for fold in range(folds):
        fit = selected[assignments != fold]
        evaluation = selected[assignments == fold]
        fit_users = {users[int(index)] for index in fit}
        evaluation_users = {users[int(index)] for index in evaluation}
        fold_rows.append(
            {
                "fold": fold,
                "fit_requests": len(fit),
                "evaluation_requests": len(evaluation),
                "fit_users": len(fit_users),
                "evaluation_users": len(evaluation_users),
                "user_disjoint": not bool(fit_users & evaluation_users),
                "evaluation_candidate_hash": data.candidate_hash(evaluation),
            }
        )
    files = {
        "strict_indices.npy": save_array(root, "strict_indices.npy", selected),
        "fold_assignments.npy": save_array(root, "fold_assignments.npy", assignments),
    }
    users_path = root / "strict_request_users.jsonl"
    with users_path.open("w", encoding="utf-8") as handle:
        for index in selected:
            handle.write(
                json.dumps(
                    {"request_index": int(index), "user_id": users[int(index)]},
                    sort_keys=True,
                )
                + "\n"
            )
    files["strict_request_users.jsonl"] = {
        "path": str(users_path.relative_to(ROOT)),
        "sha256": sha256_file(users_path),
        "rows": len(selected),
    }
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "label_free_selection_from_frozen_c38_features",
        "feature_requests": len(data.request_ids),
        "strict_nonrepeat_requests": len(selected),
        "repeat_removed": len(data.request_ids) - len(selected),
        "candidate_rows": int(
            sum(len(data.candidates(int(index))) for index in selected)
        ),
        "history_rows": int(
            sum(len(data.history(int(index))[0]) for index in selected)
        ),
        "selection_sha256": sha256_array(selected),
        "assignment_sha256": sha256_array(assignments),
        "folds": fold_rows,
        "all_fold_users_disjoint": all(row["user_disjoint"] for row in fold_rows),
        "stored_wrong_history_available": True,
        "labels_opened": data.labels_opened,
        "dev_test_qrels_opened": False,
        "files": files,
    }
    atomic_json(manifest_path, report)
    print(json.dumps(report, sort_keys=True))


def freeze(config: dict[str, Any], config_path: Path) -> None:
    paths = config["paths"]
    lock_path = ROOT / paths["execution_lock"]
    if lock_path.exists():
        raise FileExistsError(lock_path)
    selection_path = ROOT / paths["artifact_root"] / "selection_manifest.json"
    if not selection_path.exists():
        raise FileNotFoundError(selection_path)
    lock = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "authorize_three_mode_three_fold_amazon_observability_run",
        "source_sha256": {
            key: sha256_file(path) for key, path in source_paths(config, config_path).items()
        },
        "input_sha256": {
            key: sha256_file(path) for key, path in input_paths(config).items()
        },
        "selection_manifest_sha256": sha256_file(selection_path),
        "outcome_boundary": {
            "compact_fit_labels": True,
            "heldout_fold_labels_before_all_scores": False,
            "dev_test_qrels": False,
        },
    }
    atomic_json(lock_path, lock)
    print(json.dumps({"path": str(lock_path), "sha256": sha256_file(lock_path)}, sort_keys=True))


def verify_lock(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    paths = config["paths"]
    lock_path = ROOT / paths["execution_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if {
        key: sha256_file(path) for key, path in source_paths(config, config_path).items()
    } != lock["source_sha256"]:
        raise RuntimeError("Amazon HSO source changed after lock")
    if {key: sha256_file(path) for key, path in input_paths(config).items()} != lock[
        "input_sha256"
    ]:
        raise RuntimeError("Amazon HSO input changed after lock")
    selection_path = ROOT / paths["artifact_root"] / "selection_manifest.json"
    if sha256_file(selection_path) != lock["selection_manifest_sha256"]:
        raise RuntimeError("Amazon HSO selection changed after lock")
    return lock, sha256_file(lock_path)


def stage_labels(config: dict[str, Any], config_path: Path) -> None:
    verify_lock(config, config_path)
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    manifest_path = root / "fold_label_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    data = AmazonObservabilityData(ROOT / paths["feature_root"])
    all_labels = load_all_feature_labels(ROOT / paths["records_train"], data)
    selected = np.load(root / "strict_indices.npy", mmap_mode="r")
    assignments = np.load(root / "fold_assignments.npy", mmap_mode="r")
    files: dict[str, Any] = {}
    folds = []
    for fold in range(int(config["selection"]["folds"])):
        fit = np.asarray(selected[np.asarray(assignments) != fold], dtype=np.int64)
        offsets = [0]
        rows = []
        for index in fit:
            row = all_labels[int(index)]
            rows.append(row.astype(np.uint8))
            offsets.append(offsets[-1] + len(row))
        values = np.concatenate(rows).astype(np.uint8, copy=False)
        outputs = {
            f"fit_fold{fold}_request_indices.npy": fit,
            f"fit_fold{fold}_label_offsets.npy": np.asarray(offsets, dtype=np.int64),
            f"fit_fold{fold}_labels.npy": values,
        }
        for name, value in outputs.items():
            files[name] = save_array(root, name, value)
        folds.append(
            {
                "fold": fold,
                "fit_requests": len(fit),
                "fit_candidate_rows": len(values),
                "fit_positive_rows": int((values > 0).sum()),
                "evaluation_requests": int((np.asarray(assignments) == fold).sum()),
            }
        )
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "post_lock_compact_fit_label_staging",
        "source_train_labels_opened_by_stager": True,
        "heldout_labels_written_to_fit_artifacts": False,
        "dev_test_qrels_opened": False,
        "folds": folds,
        "files": files,
    }
    atomic_json(manifest_path, report)
    print(json.dumps(report, sort_keys=True))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "freeze":
        freeze(config, config_path)
    else:
        stage_labels(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
