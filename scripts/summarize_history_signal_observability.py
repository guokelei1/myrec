#!/usr/bin/env python
"""Open staged OOF labels only after all HSO modes/folds have scored."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.history_signal_observability import (  # noqa: E402
    MODES,
    PackedObservabilityData,
    atomic_json,
    load_config,
    sha256_file,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def request_ndcg(
    request_id: str,
    item_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    positive = {
        str(item_id) for item_id, label in zip(item_ids, labels) if float(label) > 0
    }
    ranked = sort_candidates(
        request_id,
        [
            ScoredCandidate(str(item_id), float(score))
            for item_id, score in zip(item_ids, scores)
        ],
    )
    return ndcg_at_k([value.item_id for value in ranked], positive, 10)


def fold_summary(values: np.ndarray, assignments: np.ndarray, folds: int) -> list[float]:
    return [
        float(np.asarray(values)[np.asarray(assignments) == fold].mean())
        for fold in range(int(folds))
    ]


def cluster_bootstrap(
    differences: np.ndarray,
    users: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    unique, inverse = np.unique(users, return_inverse=True)
    sums = np.bincount(inverse, weights=differences, minlength=len(unique)).astype(
        np.float64
    )
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 128):
        stop = min(samples, start + 128)
        draw = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        means[start:stop] = sums[draw].sum(-1) / counts[draw].sum(-1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "requests": len(differences),
        "users": len(unique),
        "mean": float(np.mean(differences)),
        "user_cluster_95_ci": [float(low), float(high)],
        "samples": int(samples),
        "seed": int(seed),
    }


def comparison(
    left: np.ndarray,
    right: np.ndarray,
    users: np.ndarray,
    assignments: np.ndarray,
    *,
    folds: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    result = cluster_bootstrap(difference, users, samples=samples, seed=seed)
    result["fold_means"] = fold_summary(difference, assignments, folds)
    result["all_folds_positive"] = all(value > 0 for value in result["fold_means"])
    return result


def derived_seed(base: int, name: str) -> int:
    value = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
    return int(base) + value % 100000


def load_users(path: Path, selected: np.ndarray) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    indices = np.asarray([int(row["request_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(indices, np.asarray(selected, dtype=np.int64)):
        raise ValueError("HSO user rows differ from strict selection")
    return np.asarray([str(row["user_id"]) for row in rows])


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    artifact_root = ROOT / paths["artifact_root"]
    report_path = ROOT / paths["report"]
    if report_path.exists():
        raise FileExistsError(report_path)
    lock_path = ROOT / paths["execution_lock"]
    lock_hash = sha256_file(lock_path)
    selected = np.load(artifact_root / "strict_indices.npy", mmap_mode="r")
    assignments = np.load(artifact_root / "fold_assignments.npy", mmap_mode="r")
    folds = int(config["selection"]["folds"])
    users = load_users(artifact_root / "strict_request_users.jsonl", selected)
    score_files: dict[tuple[str, int], Any] = {}
    mechanics: dict[str, Any] = {}
    parameter_counts: set[int] = set()
    prechecks: dict[str, bool] = {}
    for mode in MODES:
        for fold in range(folds):
            key = f"{mode}_fold{fold}"
            report_file = artifact_root / f"{key}_report.json"
            score_file = artifact_root / f"{key}_scores.npz"
            if not report_file.exists() or not score_file.exists():
                raise FileNotFoundError(f"HSO incomplete output: {key}")
            row = json.loads(report_file.read_text(encoding="utf-8"))
            expected = np.asarray(selected[np.asarray(assignments) == fold], dtype=np.int64)
            scores = np.load(score_file, allow_pickle=False)
            checks = {
                "execution_lock": row["execution_lock_sha256"] == lock_hash,
                "mechanics": bool(row["passed_mechanics"]),
                "score_hash": sha256_file(score_file) == row["scoring"]["score_sha256"],
                "request_order": np.array_equal(scores["request_indices"], expected),
                "candidate_rows": int(scores["offsets"][-1])
                == int(row["scoring"]["candidate_rows"]),
                "heldout_labels_closed": not bool(row["scoring"]["heldout_labels_opened"]),
            }
            prechecks[key] = all(checks.values())
            mechanics[key] = checks
            parameter_counts.add(int(row["parameters"]["total"]))
            score_files[(mode, fold)] = scores
    prechecks["all_outputs_present_before_label_open"] = len(score_files) == len(MODES) * folds
    prechecks["equal_parameter_count"] = len(parameter_counts) == 1
    if not all(prechecks.values()):
        raise RuntimeError("HSO pre-label output audit failed")

    data = PackedObservabilityData(ROOT / paths["packed_train_root"])
    labels = data.open_train_labels()
    methods = ["base", "null"]
    for mode in ("full", "text", "id"):
        methods.extend(
            [f"{mode}_true", f"{mode}_wrong", f"{mode}_shuffle", f"{mode}_null"]
        )
    per_request = {name: np.empty(len(selected), dtype=np.float64) for name in methods}
    position = {int(index): row for row, index in enumerate(selected)}
    for fold in range(folds):
        null_scores = score_files[("null", fold)]
        mode_scores = {
            mode: score_files[(mode, fold)] for mode in ("full", "text", "id")
        }
        for local, index_value in enumerate(null_scores["request_indices"]):
            index = int(index_value)
            global_position = position[index]
            start, stop = int(null_scores["offsets"][local]), int(
                null_scores["offsets"][local + 1]
            )
            source_start, source_stop = int(data.candidate_offsets[index]), int(
                data.candidate_offsets[index + 1]
            )
            item_ids = data.candidate_ids(index)
            row_labels = np.asarray(labels[source_start:source_stop], dtype=np.float32)
            per_request["base"][global_position] = request_ndcg(
                data.request_ids[index], item_ids, null_scores["base"][start:stop], row_labels
            )
            per_request["null"][global_position] = request_ndcg(
                data.request_ids[index], item_ids, null_scores["null"][start:stop], row_labels
            )
            for mode, scores in mode_scores.items():
                mode_start, mode_stop = int(scores["offsets"][local]), int(
                    scores["offsets"][local + 1]
                )
                if mode_stop - mode_start != stop - start:
                    raise ValueError("HSO mode candidate count differs")
                for scenario in ("true", "wrong", "shuffle", "null"):
                    per_request[f"{mode}_{scenario}"][global_position] = request_ndcg(
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
    samples = int(config["evaluation"]["bootstrap_samples"])
    base_seed = int(config["evaluation"]["bootstrap_seed"])
    comparisons: dict[str, Any] = {}
    for mode in ("full", "text", "id"):
        pairs = {
            f"{mode}_true_minus_null_model": (f"{mode}_true", "null"),
            f"{mode}_true_minus_wrong": (f"{mode}_true", f"{mode}_wrong"),
            f"{mode}_true_minus_own_null": (f"{mode}_true", f"{mode}_null"),
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
                seed=derived_seed(base_seed, name),
            )
    for name, left, right in (
        ("full_true_minus_text_true", "full_true", "text_true"),
        ("full_true_minus_id_true", "full_true", "id_true"),
        ("null_minus_base", "null", "base"),
    ):
        comparisons[name] = comparison(
            per_request[left],
            per_request[right],
            users,
            assignments,
            folds=folds,
            samples=samples,
            seed=derived_seed(base_seed, name),
        )
    minimum = float(config["evaluation"]["observable_min_ndcg"])
    observable: dict[str, bool] = {}
    observability_checks: dict[str, Any] = {}
    for mode in ("full", "text", "id"):
        utility = comparisons[f"{mode}_true_minus_null_model"]
        identity = comparisons[f"{mode}_true_minus_wrong"]
        checks = {
            "minimum_effect": utility["mean"] >= minimum,
            "utility_ci_positive": utility["user_cluster_95_ci"][0] > 0,
            "utility_all_folds_positive": utility["all_folds_positive"],
            "identity_ci_positive": identity["user_cluster_95_ci"][0] > 0,
            "identity_all_folds_positive": identity["all_folds_positive"],
            "all_mechanics": all(prechecks.values()),
        }
        observability_checks[mode] = checks
        observable[mode] = all(checks.values())
    if observable["text"]:
        branch = "semantic_history_observable_design_semantic_internal_architecture"
    elif observable["id"]:
        branch = "collaborative_identity_observable_design_internal_id_memory"
    elif observable["full"]:
        branch = "joint_carrier_only_observable_design_coupled_semantic_collaborative_memory"
    else:
        branch = "no_kuai_source_observable_run_preregistered_amazon_counterpart"
    metrics_path = artifact_root / "per_request_metrics.npz"
    with metrics_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=np.asarray(selected, dtype=np.int64),
            fold_assignments=np.asarray(assignments, dtype=np.int8),
            **{name: value.astype(np.float32) for name, value in per_request.items()},
        )
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "train_internal_user_crossfit_outcome",
        "decision": branch,
        "execution_lock_sha256": lock_hash,
        "requests": len(selected),
        "users": len(np.unique(users)),
        "folds": folds,
        "candidate_hashes": [
            data.candidate_hash(selected[np.asarray(assignments) == fold])
            for fold in range(folds)
        ],
        "metrics": metrics,
        "comparisons": comparisons,
        "observable": observable,
        "observability_checks": observability_checks,
        "pre_label_checks": prechecks,
        "mechanics": mechanics,
        "parameter_count": next(iter(parameter_counts)),
        "label_boundary": {
            "fit_fold_labels_staged_after_lock": True,
            "heldout_labels_opened_only_after_all_scores": True,
            "dev_test_qrels_opened": False,
        },
        "per_request_metrics": {
            "path": str(metrics_path.relative_to(ROOT)),
            "sha256": sha256_file(metrics_path),
        },
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "decision": branch,
                "observable": observable,
                "utility": {
                    mode: comparisons[f"{mode}_true_minus_null_model"]
                    for mode in ("full", "text", "id")
                },
                "identity": {
                    mode: comparisons[f"{mode}_true_minus_wrong"]
                    for mode in ("full", "text", "id")
                },
                "report": str(report_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
