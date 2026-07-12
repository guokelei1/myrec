#!/usr/bin/env python
"""Freeze and execute HSO's same-checkpoint, still-unlabeled recovery readout."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from myrec.analysis.history_signal_observability import (  # noqa: E402
    PackedObservabilityData,
    atomic_json,
    sha256_file,
)
from summarize_history_signal_observability import (  # noqa: E402
    comparison,
    derived_seed,
    load_users,
    request_ndcg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("HSO-R1 config must be a mapping")
    for key in ("training", "rescoring", "checkpoint_selection", "dev", "test", "qrels"):
        if bool(value["authorization"][key]):
            raise PermissionError(f"HSO-R1 unauthorized action: {key}")
    return value


def source_paths(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths = config["paths"]
    root = ROOT / paths["source_artifact_root"]
    values = {
        "recovery_config": config_path,
        "recovery_protocol": ROOT / paths["protocol"],
        "recovery_script": ROOT / paths["script"],
        "base_summarizer": ROOT / paths["base_summarizer"],
        "source_config": ROOT / paths["source_config"],
        "source_execution_lock": ROOT / paths["source_execution_lock"],
        "selection_manifest": root / "selection_manifest.json",
        "fold_label_manifest": root / "fold_label_manifest.json",
        "strict_indices": root / "strict_indices.npy",
        "fold_assignments": root / "fold_assignments.npy",
        "strict_users": root / "strict_request_users.jsonl",
    }
    for mode in config["evaluation"]["modes"]:
        for fold in range(int(config["evaluation"]["folds"])):
            values[f"{mode}_fold{fold}_report"] = root / f"{mode}_fold{fold}_report.json"
            values[f"{mode}_fold{fold}_scores"] = root / f"{mode}_fold{fold}_scores.npz"
    for fold in range(int(config["evaluation"]["folds"])):
        values[f"null_fold{fold}_report"] = root / f"null_fold{fold}_report.json"
    return values


def preoutcome_audit(config: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / config["paths"]["source_artifact_root"]
    folds = int(config["evaluation"]["folds"])
    source_modes: dict[str, Any] = {}
    base_equal: dict[str, bool] = {}
    for fold in range(folds):
        reference_request: np.ndarray | None = None
        reference_offsets: np.ndarray | None = None
        reference_base: np.ndarray | None = None
        for mode in config["evaluation"]["modes"]:
            report = json.loads(
                (root / f"{mode}_fold{fold}_report.json").read_text(encoding="utf-8")
            )
            scores = np.load(root / f"{mode}_fold{fold}_scores.npz", allow_pickle=False)
            checks = {
                "mechanics": bool(report["passed_mechanics"]),
                "heldout_labels_closed": not bool(report["scoring"]["heldout_labels_opened"]),
                "score_hash": sha256_file(root / f"{mode}_fold{fold}_scores.npz")
                == report["scoring"]["score_sha256"],
            }
            source_modes[f"{mode}_fold{fold}"] = checks
            if reference_request is None:
                reference_request = np.asarray(scores["request_indices"])
                reference_offsets = np.asarray(scores["offsets"])
                reference_base = np.asarray(scores["base"])
            else:
                base_equal[f"{mode}_fold{fold}"] = bool(
                    np.array_equal(reference_request, scores["request_indices"])
                    and np.array_equal(reference_offsets, scores["offsets"])
                    and np.array_equal(reference_base, scores["base"])
                )
    null_failures: dict[str, Any] = {}
    for fold in range(folds):
        report = json.loads(
            (root / f"null_fold{fold}_report.json").read_text(encoding="utf-8")
        )
        other_checks = {
            key: value for key, value in report["checks"].items() if key != "loss_decreased"
        }
        null_failures[f"fold{fold}"] = {
            "loss_decreased": bool(report["checks"]["loss_decreased"]),
            "all_other_mechanics": all(other_checks.values()),
            "heldout_labels_closed": not bool(report["scoring"]["heldout_labels_opened"]),
        }
    checks = {
        "all_history_modes_mechanical": all(
            all(row.values()) for row in source_modes.values()
        ),
        "base_identical_across_modes": all(base_equal.values()),
        "null_failed_loss_all_folds": all(
            not row["loss_decreased"] for row in null_failures.values()
        ),
        "null_other_mechanics_passed": all(
            row["all_other_mechanics"] for row in null_failures.values()
        ),
        "all_heldout_labels_closed": all(
            row["heldout_labels_closed"] for row in null_failures.values()
        )
        and all(row["heldout_labels_closed"] for row in source_modes.values()),
        "no_prior_recovery_report": not (ROOT / config["paths"]["report"]).exists(),
    }
    return {
        "checks": checks,
        "source_modes": source_modes,
        "base_equal": base_equal,
        "null_failures": null_failures,
    }


def freeze(config: dict[str, Any], config_path: Path) -> None:
    lock_path = ROOT / config["paths"]["recovery_lock"]
    if lock_path.exists():
        raise FileExistsError(lock_path)
    audit = preoutcome_audit(config)
    if not all(audit["checks"].values()):
        raise RuntimeError("HSO-R1 pre-outcome audit failed")
    sources = source_paths(config, config_path)
    lock = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "authorize_same_checkpoint_null_and_fixed_base_recovery_readout",
        "source_sha256": {key: sha256_file(path) for key, path in sources.items()},
        "preoutcome_audit": audit,
        "outcome_boundary": {
            "training": False,
            "rescoring": False,
            "checkpoint_selection": False,
            "heldout_fold_labels_before_lock": False,
            "dev_test_qrels": False,
        },
    }
    atomic_json(lock_path, lock)
    print(json.dumps({"path": str(lock_path), "sha256": sha256_file(lock_path)}, sort_keys=True))


def verify_lock(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    lock_path = ROOT / config["paths"]["recovery_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    observed = {
        key: sha256_file(path) for key, path in source_paths(config, config_path).items()
    }
    if observed != lock["source_sha256"]:
        raise RuntimeError("HSO-R1 source changed after recovery lock")
    if not all(lock["preoutcome_audit"]["checks"].values()):
        raise RuntimeError("HSO-R1 lock did not pass pre-outcome audit")
    return lock, sha256_file(lock_path)


def evaluate(config: dict[str, Any], config_path: Path) -> None:
    _, lock_hash = verify_lock(config, config_path)
    paths = config["paths"]
    artifact_root = ROOT / paths["source_artifact_root"]
    report_path = ROOT / paths["report"]
    if report_path.exists():
        raise FileExistsError(report_path)
    selected = np.load(artifact_root / "strict_indices.npy", mmap_mode="r")
    assignments = np.load(artifact_root / "fold_assignments.npy", mmap_mode="r")
    users = load_users(artifact_root / "strict_request_users.jsonl", selected)
    folds = int(config["evaluation"]["folds"])
    data = PackedObservabilityData(ROOT / paths["packed_train_root"])
    if data.labels_opened:
        raise PermissionError("HSO-R1 labels opened before lock verification")
    labels = data.open_train_labels()
    methods = ["base"]
    for mode in config["evaluation"]["modes"]:
        methods.extend(
            [f"{mode}_true", f"{mode}_wrong", f"{mode}_shuffle", f"{mode}_null"]
        )
    per_request = {name: np.empty(len(selected), dtype=np.float64) for name in methods}
    position = {int(index): row for row, index in enumerate(selected)}
    for fold in range(folds):
        score_sets = {
            mode: np.load(
                artifact_root / f"{mode}_fold{fold}_scores.npz", allow_pickle=False
            )
            for mode in config["evaluation"]["modes"]
        }
        reference = score_sets[str(config["evaluation"]["modes"][0])]
        for local, index_value in enumerate(reference["request_indices"]):
            index = int(index_value)
            output_row = position[index]
            start, stop = int(reference["offsets"][local]), int(
                reference["offsets"][local + 1]
            )
            source_start, source_stop = int(data.candidate_offsets[index]), int(
                data.candidate_offsets[index + 1]
            )
            item_ids = data.candidate_ids(index)
            row_labels = np.asarray(labels[source_start:source_stop], dtype=np.float32)
            per_request["base"][output_row] = request_ndcg(
                data.request_ids[index], item_ids, reference["base"][start:stop], row_labels
            )
            for mode, scores in score_sets.items():
                mode_start, mode_stop = int(scores["offsets"][local]), int(
                    scores["offsets"][local + 1]
                )
                if mode_stop - mode_start != stop - start:
                    raise ValueError("HSO-R1 candidate count differs")
                for scenario in ("true", "wrong", "shuffle", "null"):
                    per_request[f"{mode}_{scenario}"][output_row] = request_ndcg(
                        data.request_ids[index],
                        item_ids,
                        scores[scenario][mode_start:mode_stop],
                        row_labels,
                    )
    metrics = {
        name: {
            "ndcg@10": float(values.mean()),
            "fold_ndcg@10": [
                float(values[np.asarray(assignments) == fold].mean())
                for fold in range(folds)
            ],
        }
        for name, values in per_request.items()
    }
    comparisons: dict[str, Any] = {}
    samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    for mode in config["evaluation"]["modes"]:
        pairs = {
            f"{mode}_true_minus_base": (f"{mode}_true", "base"),
            f"{mode}_true_minus_own_null": (f"{mode}_true", f"{mode}_null"),
            f"{mode}_true_minus_wrong": (f"{mode}_true", f"{mode}_wrong"),
            f"{mode}_true_minus_shuffle": (f"{mode}_true", f"{mode}_shuffle"),
        }
        for name, (left, right) in pairs.items():
            comparisons[name] = comparison(
                per_request[left],
                per_request[right],
                users,
                assignments,
                folds=folds,
                samples=samples,
                seed=derived_seed(seed, name),
            )
    minimum = float(config["evaluation"]["observable_min_ndcg"])
    checks: dict[str, Any] = {}
    observable: dict[str, bool] = {}
    for mode in config["evaluation"]["modes"]:
        base = comparisons[f"{mode}_true_minus_base"]
        null = comparisons[f"{mode}_true_minus_own_null"]
        wrong = comparisons[f"{mode}_true_minus_wrong"]
        row = {
            "base_minimum_effect": base["mean"] >= minimum,
            "base_ci_positive": base["user_cluster_95_ci"][0] > 0,
            "base_all_folds_positive": base["all_folds_positive"],
            "null_minimum_effect": null["mean"] >= minimum,
            "null_ci_positive": null["user_cluster_95_ci"][0] > 0,
            "null_all_folds_positive": null["all_folds_positive"],
            "wrong_ci_positive": wrong["user_cluster_95_ci"][0] > 0,
            "wrong_all_folds_positive": wrong["all_folds_positive"],
            "source_mechanics": True,
        }
        checks[str(mode)] = row
        observable[str(mode)] = all(row.values())
    if observable.get("text", False):
        decision = "semantic_history_observable_freeze_semantic_architecture_primitive"
    elif observable.get("id", False):
        decision = "collaborative_identity_observable_freeze_internal_id_memory_primitive"
    elif observable.get("full", False):
        decision = "joint_carrier_observable_freeze_coupled_carrier_primitive"
    else:
        decision = "no_kuai_source_observable_run_preregistered_amazon_counterpart"
    metric_path = ROOT / paths["per_request_metrics"]
    with metric_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=np.asarray(selected, dtype=np.int64),
            fold_assignments=np.asarray(assignments, dtype=np.int8),
            **{name: values.astype(np.float32) for name, values in per_request.items()},
        )
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "same_checkpoint_recovery_outcome",
        "decision": decision,
        "recovery_lock_sha256": lock_hash,
        "requests": len(selected),
        "users": len(np.unique(users)),
        "metrics": metrics,
        "comparisons": comparisons,
        "observable": observable,
        "observability_checks": checks,
        "label_boundary": {
            "heldout_fold_labels_opened_only_after_recovery_lock": True,
            "training_or_rescoring_after_lock": False,
            "dev_test_qrels_opened": False,
        },
        "per_request_metrics": {
            "path": str(metric_path.relative_to(ROOT)),
            "sha256": sha256_file(metric_path),
        },
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "decision": decision,
                "observable": observable,
                "base": {
                    mode: comparisons[f"{mode}_true_minus_base"]
                    for mode in config["evaluation"]["modes"]
                },
                "null": {
                    mode: comparisons[f"{mode}_true_minus_own_null"]
                    for mode in config["evaluation"]["modes"]
                },
                "wrong": {
                    mode: comparisons[f"{mode}_true_minus_wrong"]
                    for mode in config["evaluation"]["modes"]
                },
                "report": str(report_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.stage == "freeze":
        freeze(config, config_path)
    else:
        evaluate(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
