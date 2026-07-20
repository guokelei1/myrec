"""Registered 96-cell D1 region-level decoding synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.deep_dive_native_evaluator import benjamini_hochberg
from myrec.mechanism.deep_dive_representation_analysis import ALL_POSITIONS
from myrec.mechanism.deep_dive_representation_evaluator import BLOCK_REGIONS
from myrec.utils.hashing import sha256_file


BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20_260_715
FAMILY_SIZE = 96


def synthesize_d1_region_decoding(
    evaluation_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Combine fixed Q2/Q3 cells and apply one frozen 96-unit BH family."""

    if set(evaluation_dirs) != {"q2", "q3"}:
        raise ValueError("D1 synthesis requires exact q2 and q3 evaluations")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D1 synthesis output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    for model_key in ("q2", "q3"):
        root = Path(evaluation_dirs[model_key])
        metrics_path = root / "metrics.json"
        metrics = _read_json(metrics_path)
        if metrics.get("status") != "completed" or metrics.get("qrels_read") is not True:
            raise ValueError(f"D1 evaluation is incomplete: {model_key}")
        correctness_path = Path(metrics["correctness_path"])
        if metrics.get("correctness_sha256") != sha256_file(correctness_path):
            raise ValueError(f"D1 correctness bytes changed: {model_key}")
        inputs[model_key] = {
            "evaluation_dir": str(root),
            "metrics_sha256": sha256_file(metrics_path),
            "correctness_sha256": sha256_file(correctness_path),
            "method_id": metrics["method_id"],
            "checkpoint_id": metrics["checkpoint_id"],
        }
        with np.load(correctness_path, allow_pickle=False) as payload:
            strict = np.asarray(payload["strict_mask"], dtype=bool)
            folds = np.asarray(payload["folds"], dtype=np.int8)
            clusters = np.asarray(payload["normalized_queries"], dtype=np.str_)
            for position in ALL_POSITIONS:
                for task in ("brand", "category"):
                    labels = np.asarray(payload[f"{task}_labels"], dtype=np.str_)
                    eligible = strict & (labels != "")
                    for region, states in BLOCK_REGIONS.items():
                        arrays = {
                            "full_real": np.stack(
                                [
                                    payload[
                                        _key(
                                            "full",
                                            position,
                                            task,
                                            "real_labels",
                                            state,
                                        )
                                    ]
                                    for state in states
                                ]
                            ),
                            "full_random": np.stack(
                                [
                                    payload[
                                        _key(
                                            "full",
                                            position,
                                            task,
                                            "random_labels",
                                            state,
                                        )
                                    ]
                                    for state in states
                                ]
                            ),
                            "null_real": np.stack(
                                [
                                    payload[
                                        _key(
                                            "null",
                                            position,
                                            task,
                                            "real_labels",
                                            state,
                                        )
                                    ]
                                    for state in states
                                ]
                            ),
                        }
                        for contrast, left, right in (
                            (
                                "real_minus_random",
                                "full_real",
                                "full_random",
                            ),
                            (
                                "full_minus_null_excess",
                                "full_real",
                                "null_real",
                            ),
                        ):
                            inference = region_balanced_accuracy_contrast(
                                arrays[left],
                                arrays[right],
                                labels,
                                clusters,
                                eligible,
                            )
                            fold_estimates = {
                                str(fold): region_balanced_accuracy_point(
                                    arrays[left],
                                    arrays[right],
                                    labels,
                                    eligible & (folds == fold),
                                )
                                for fold in (0, 1)
                            }
                            rows.append(
                                {
                                    "model_key": model_key,
                                    "method_id": metrics["method_id"],
                                    "position": position,
                                    "task": task,
                                    "region": region,
                                    "hidden_state_indices": list(states),
                                    "contrast": contrast,
                                    "expected_sign": "positive",
                                    **inference,
                                    "fold_estimates": fold_estimates,
                                    "point_expected_sign": inference["estimate"] > 0,
                                    "both_folds_expected_sign": all(
                                        value > 0 for value in fold_estimates.values()
                                    ),
                                }
                            )
    if len(rows) != FAMILY_SIZE:
        raise AssertionError(f"D1 family size is {len(rows)}, expected {FAMILY_SIZE}")
    q_values = benjamini_hochberg([float(row["two_sided_p"]) for row in rows])
    for row, q_value in zip(rows, q_values):
        row["bh_q"] = q_value
        row["bh_q_below_0.05"] = q_value < 0.05
        row["registered_gate_passed"] = bool(
            row["point_expected_sign"]
            and row["both_folds_expected_sign"]
            and row["bh_q_below_0.05"]
        )
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d1_region_decoding_synthesis",
        "inputs": inputs,
        "family": {
            "name": "d1_region_decoding",
            "planned_size": FAMILY_SIZE,
            "observed_size": len(rows),
            "multiple_testing": "benjamini_hochberg",
            "alpha": 0.05,
        },
        "bootstrap": {
            "cluster": "normalized_query",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "two_sided_p": "min(1,2*min((1+#draw<=0)/(B+1),(1+#draw>=0)/(B+1)))",
        },
        "block_regions": {key: list(value) for key, value in BLOCK_REGIONS.items()},
        "cells": rows,
        "passed_cells": sum(row["registered_gate_passed"] for row in rows),
        "command": list(command or []),
        "status": "completed",
    }
    _write_json(output_dir / "metrics.json", result)
    return result


def region_balanced_accuracy_contrast(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    mask: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    labels = np.asarray(labels, dtype=np.str_)
    clusters = np.asarray(clusters, dtype=np.str_)
    mask = np.asarray(mask, dtype=bool)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != labels.size:
        raise ValueError("D1 region correctness arrays are misaligned")
    if clusters.shape != labels.shape or mask.shape != labels.shape or not mask.any():
        raise ValueError("D1 region population arrays are invalid")
    labels = labels[mask]
    clusters = clusters[mask]
    left = left[:, mask]
    right = right[:, mask]
    unique_clusters, cluster_inverse = np.unique(clusters, return_inverse=True)
    unique_classes, class_inverse = np.unique(labels, return_inverse=True)
    cluster_count = len(unique_clusters)
    class_count = len(unique_classes)
    joint = cluster_inverse * class_count + class_inverse
    counts = np.bincount(
        joint, minlength=cluster_count * class_count
    ).reshape(cluster_count, class_count)
    left_correct = np.stack(
        [
            np.bincount(joint, weights=row, minlength=cluster_count * class_count).reshape(
                cluster_count, class_count
            )
            for row in left
        ]
    )
    right_correct = np.stack(
        [
            np.bincount(joint, weights=row, minlength=cluster_count * class_count).reshape(
                cluster_count, class_count
            )
            for row in right
        ]
    )
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    batch_size = 64
    for start in range(0, samples, batch_size):
        width = min(batch_size, samples - start)
        selected = rng.integers(
            0, cluster_count, size=(width, cluster_count), endpoint=False
        )
        multiplicity = np.zeros((width, cluster_count), dtype=np.int32)
        rows = np.arange(width)[:, None]
        np.add.at(multiplicity, (rows, selected), 1)
        denominators = multiplicity @ counts
        valid = np.all(denominators > 0, axis=1)
        if not valid.all():
            raise ValueError("D1 cluster draw omitted an observed label class")
        left_ba = np.stack(
            [
                np.mean((multiplicity @ values) / denominators, axis=1)
                for values in left_correct
            ],
            axis=1,
        ).mean(axis=1)
        right_ba = np.stack(
            [
                np.mean((multiplicity @ values) / denominators, axis=1)
                for values in right_correct
            ],
            axis=1,
        ).mean(axis=1)
        draws.append(left_ba - right_ba)
    bootstrap = np.concatenate(draws)
    if bootstrap.size != samples:
        raise AssertionError("D1 bootstrap sample count drifted")
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    lower_tail = (1 + int(np.count_nonzero(bootstrap <= 0))) / (samples + 1)
    upper_tail = (1 + int(np.count_nonzero(bootstrap >= 0))) / (samples + 1)
    return {
        "requests": int(mask.sum()),
        "normalized_query_clusters": cluster_count,
        "classes": class_count,
        "estimate": region_balanced_accuracy_point(
            left, right, labels, np.ones(labels.size, dtype=bool)
        ),
        "ci95": [float(lower), float(upper)],
        "two_sided_p": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "bootstrap_samples": samples,
    }


def region_balanced_accuracy_point(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    labels = np.asarray(labels, dtype=np.str_)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("D1 region point population is empty")
    classes = np.unique(labels[mask])
    left_ba = np.mean(
        [np.mean(left[:, mask][:, labels[mask] == value], axis=1) for value in classes],
        axis=0,
    ).mean()
    right_ba = np.mean(
        [np.mean(right[:, mask][:, labels[mask] == value], axis=1) for value in classes],
        axis=0,
    ).mean()
    return float(left_ba - right_ba)


def _key(condition: str, position: str, task: str, control: str, state: int) -> str:
    return f"{condition}__{position}__{task}__{control}__state_{state}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

