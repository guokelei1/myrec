#!/usr/bin/env python
"""Open Amazon HSO OOF labels only after all source modes have scored."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from myrec.analysis.amazon_history_signal_observability import (  # noqa: E402
    AmazonObservabilityData,
    load_all_feature_labels,
)
from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402
from prepare_amazon_history_signal_observability import load_config, verify_lock  # noqa: E402
from summarize_history_signal_observability import (  # noqa: E402
    comparison,
    derived_seed,
    load_users,
    request_ndcg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    _, lock_hash = verify_lock(config, config_path)
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    report_path = ROOT / paths["report"]
    if report_path.exists():
        raise FileExistsError(report_path)
    selected = np.load(root / "strict_indices.npy", mmap_mode="r")
    assignments = np.load(root / "fold_assignments.npy", mmap_mode="r")
    users = load_users(root / "strict_request_users.jsonl", selected)
    modes = [str(value) for value in config["model"]["modes"]]
    folds = int(config["selection"]["folds"])
    score_files: dict[tuple[str, int], Any] = {}
    checks: dict[str, bool] = {}
    parameter_counts = set()
    for mode in modes:
        for fold in range(folds):
            key = f"{mode}_fold{fold}"
            row = json.loads((root / f"{key}_report.json").read_text(encoding="utf-8"))
            score_path = root / f"{key}_scores.npz"
            scores = np.load(score_path, allow_pickle=False)
            expected = np.asarray(selected[np.asarray(assignments) == fold], dtype=np.int64)
            checks[key] = bool(
                row["passed_mechanics"]
                and row["execution_lock_sha256"] == lock_hash
                and sha256_file(score_path) == row["scoring"]["score_sha256"]
                and np.array_equal(scores["request_indices"], expected)
                and not row["scoring"]["heldout_labels_opened"]
            )
            parameter_counts.add(int(row["parameters"]["total"]))
            score_files[(mode, fold)] = scores
    checks["all_scores_present"] = len(score_files) == len(modes) * folds
    checks["equal_parameter_count"] = len(parameter_counts) == 1
    checks["all_mechanics"] = all(checks.values())
    if not all(checks.values()):
        raise RuntimeError("Amazon HSO pre-label audit failed")
    for fold in range(folds):
        reference = score_files[(modes[0], fold)]
        for mode in modes[1:]:
            current = score_files[(mode, fold)]
            if not (
                np.array_equal(reference["request_indices"], current["request_indices"])
                and np.array_equal(reference["offsets"], current["offsets"])
                and np.array_equal(reference["base"], current["base"])
            ):
                raise RuntimeError("Amazon HSO base differs across modes")

    data = AmazonObservabilityData(ROOT / paths["feature_root"])
    if data.labels_opened:
        raise PermissionError("Amazon HSO labels opened before all-score audit")
    all_labels = load_all_feature_labels(ROOT / paths["records_train"], data)
    methods = ["base"]
    for mode in modes:
        methods.extend(
            [f"{mode}_true", f"{mode}_wrong", f"{mode}_shuffle", f"{mode}_null"]
        )
    per_request = {name: np.empty(len(selected), dtype=np.float64) for name in methods}
    position = {int(index): row for row, index in enumerate(selected)}
    for fold in range(folds):
        scores_by_mode = {mode: score_files[(mode, fold)] for mode in modes}
        reference = scores_by_mode[modes[0]]
        for local, index_value in enumerate(reference["request_indices"]):
            index = int(index_value)
            output_row = position[index]
            start, stop = int(reference["offsets"][local]), int(
                reference["offsets"][local + 1]
            )
            item_ids = data.candidate_ids(index)
            labels = all_labels[index]
            per_request["base"][output_row] = request_ndcg(
                data.request_ids[index], item_ids, reference["base"][start:stop], labels
            )
            for mode, scores in scores_by_mode.items():
                mode_start, mode_stop = int(scores["offsets"][local]), int(
                    scores["offsets"][local + 1]
                )
                if mode_stop - mode_start != stop - start:
                    raise ValueError("Amazon HSO mode candidate count differs")
                for scenario in ("true", "wrong", "shuffle", "null"):
                    per_request[f"{mode}_{scenario}"][output_row] = request_ndcg(
                        data.request_ids[index],
                        item_ids,
                        scores[scenario][mode_start:mode_stop],
                        labels,
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
    for mode in modes:
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
    observability_checks: dict[str, Any] = {}
    observable: dict[str, bool] = {}
    for mode in modes:
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
        observability_checks[mode] = row
        observable[mode] = all(row.values())
    if observable["text"]:
        decision = "amazon_semantic_history_observable_freeze_semantic_architecture_primitive"
    elif observable["id"]:
        decision = "amazon_collaborative_identity_observable_freeze_id_memory_primitive"
    elif observable["full"]:
        decision = "amazon_joint_carrier_observable_freeze_coupled_carrier_primitive"
    else:
        decision = "no_cross_domain_source_observable_change_data_contract_or_narrow_claim"
    metrics_path = root / "per_request_metrics.npz"
    with metrics_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=np.asarray(selected, dtype=np.int64),
            fold_assignments=np.asarray(assignments, dtype=np.int8),
            **{name: values.astype(np.float32) for name, values in per_request.items()},
        )
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "amazon_train_internal_user_crossfit_outcome",
        "decision": decision,
        "execution_lock_sha256": lock_hash,
        "requests": len(selected),
        "users": len(np.unique(users)),
        "metrics": metrics,
        "comparisons": comparisons,
        "observable": observable,
        "observability_checks": observability_checks,
        "pre_label_checks": checks,
        "parameter_count": next(iter(parameter_counts)),
        "label_boundary": {
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
                "decision": decision,
                "observable": observable,
                "base": {
                    mode: comparisons[f"{mode}_true_minus_base"] for mode in modes
                },
                "null": {
                    mode: comparisons[f"{mode}_true_minus_own_null"] for mode in modes
                },
                "wrong": {
                    mode: comparisons[f"{mode}_true_minus_wrong"] for mode in modes
                },
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
