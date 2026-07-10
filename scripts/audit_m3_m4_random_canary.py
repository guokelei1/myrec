#!/usr/bin/env python
"""Audit whether M3/M4 evidence can be reproduced by a random channel."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import iter_jsonl, write_json  # noqa: E402


DEFAULT_RUNS = {
    "query": "20260708_kuaisearch_b2z_bge_small_zh_dev",
    "history": "20260708_kuaisearch_b0b_recent_behavior_dev",
    "static": "20260708_kuaisearch_b7_bge_dev_a02",
    "random": "20260708_kuaisearch_random_c1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--features", default="artifacts/m4/m4_features_dev.parquet")
    parser.add_argument("--output", default="reports/pps_m3_m4_random_canary_audit.json")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def _load_metric_map(path: Path) -> dict[str, float]:
    values = {
        str(row["request_id"]): float(row["ndcg@10"])
        for row in iter_jsonl(path)
    }
    if not values:
        raise ValueError(f"empty per-request metric file: {path}")
    return values


def _choose(values: dict[str, float], order: tuple[str, ...]) -> str:
    return max(order, key=lambda name: (values[name], -order.index(name)))


def _oracle_summary(
    frame: pd.DataFrame,
    order: tuple[str, ...],
    baseline: str = "static",
) -> dict[str, Any]:
    oracle_values = frame[list(order)].max(axis=1)
    baseline_values = frame[baseline]
    delta = float((oracle_values - baseline_values).mean())
    baseline_mean = float(baseline_values.mean())
    choices = [
        _choose({name: float(row[name]) for name in order}, order)
        for _, row in frame.iterrows()
    ]
    return {
        "requests": int(len(frame)),
        "oracle_ndcg@10": float(oracle_values.mean()),
        "baseline_ndcg@10": baseline_mean,
        "headroom_absolute": delta,
        "headroom_relative": delta / baseline_mean if baseline_mean else 0.0,
        "choice_counts": dict(sorted(Counter(choices).items())),
    }


def _predictability(
    frame: pd.DataFrame,
    feature_columns: list[str],
    labels: np.ndarray,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    classes = sorted(set(str(label) for label in labels))
    if len(classes) != 3:
        raise ValueError(f"expected three oracle classes, found {classes}")
    features = frame[feature_columns]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows = []
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(features, labels), start=1
    ):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        model.fit(features.iloc[train_index], labels[train_index])
        probabilities = model.predict_proba(features.iloc[valid_index])
        class_to_column = {
            str(label): index for index, label in enumerate(model.classes_)
        }
        aligned = np.zeros((len(valid_index), len(classes)), dtype=float)
        for class_index, label in enumerate(classes):
            aligned[:, class_index] = probabilities[:, class_to_column[label]]
        valid_labels = labels[valid_index]
        per_channel = {
            label: float(
                roc_auc_score(
                    (valid_labels == label).astype(int), aligned[:, class_index]
                )
            )
            for class_index, label in enumerate(classes)
        }
        fold_rows.append(
            {
                "fold": fold,
                "macro_ovr_auc": float(
                    roc_auc_score(
                        valid_labels,
                        aligned,
                        labels=classes,
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                "per_channel_auc": per_channel,
            }
        )
    macro_values = [row["macro_ovr_auc"] for row in fold_rows]
    return {
        "requests": int(len(frame)),
        "class_counts": dict(sorted(Counter(str(label) for label in labels).items())),
        "macro_ovr_auc_mean": float(np.mean(macro_values)),
        "macro_ovr_auc_std": float(np.std(macro_values, ddof=1)),
        "per_channel_auc": {
            label: float(
                np.mean([row["per_channel_auc"][label] for row in fold_rows])
            )
            for label in classes
        },
        "folds": fold_rows,
    }


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    feature_path = Path(args.features)
    metric_paths = {
        name: runs_dir / run_id / "per_request_metrics.jsonl"
        for name, run_id in DEFAULT_RUNS.items()
    }
    metric_maps = {
        name: _load_metric_map(path) for name, path in metric_paths.items()
    }
    request_ids = set(next(iter(metric_maps.values())))
    if any(set(values) != request_ids for values in metric_maps.values()):
        raise ValueError("per-request metric request_id sets differ")

    features = pd.read_parquet(feature_path)
    if set(features["request_id"].astype(str)) != request_ids:
        raise ValueError("M4 feature and per-request metric request_id sets differ")
    features = features.copy()
    features["request_id"] = features["request_id"].astype(str)
    # Preserve the frozen M4 feature-row order so the seeded shuffled folds
    # exactly reproduce the registered formal M4 result.
    for name, values in metric_maps.items():
        features[name] = [values[request_id] for request_id in features["request_id"]]

    actual_order = ("query", "history", "static")
    random_order = ("query", "random", "static")
    feature_columns = [
        column
        for column in features.columns
        if column not in {"request_id", "split", *DEFAULT_RUNS}
    ]

    scopes = {
        "all": features,
        "history_present": features[features["history_length"] > 0].reset_index(drop=True),
        "history_absent": features[features["history_length"] == 0].reset_index(drop=True),
    }
    oracle = {
        scope: {
            "actual": _oracle_summary(frame, actual_order),
            "random_canary": _oracle_summary(frame, random_order),
            "query_static_only": _oracle_summary(frame, ("query", "static")),
        }
        for scope, frame in scopes.items()
    }

    predictability = {}
    for scope in ("all", "history_present"):
        frame = scopes[scope]
        actual_labels = np.asarray(
            [
                _choose(
                    {name: float(row[name]) for name in actual_order}, actual_order
                )
                for _, row in frame.iterrows()
            ]
        )
        random_labels = np.asarray(
            [
                _choose(
                    {name: float(row[name]) for name in random_order}, random_order
                )
                for _, row in frame.iterrows()
            ]
        )
        predictability[scope] = {
            "actual": _predictability(
                frame, feature_columns, actual_labels, args.folds, args.seed
            ),
            "random_canary": _predictability(
                frame, feature_columns, random_labels, args.folds, args.seed
            ),
        }

    actual_oracle = oracle["all"]["actual"]
    random_oracle = oracle["all"]["random_canary"]
    actual_auc = predictability["all"]["actual"]["macro_ovr_auc_mean"]
    random_auc = predictability["all"]["random_canary"]["macro_ovr_auc_mean"]
    result = {
        "report": "pps_m3_m4_random_canary_audit",
        "analysis_type": "post-hoc read-only construct-validity audit",
        "status": "construct-validity-failed",
        "seed": args.seed,
        "folds": args.folds,
        "metric": "ndcg@10",
        "qrels_read": False,
        "records_test_read": False,
        "sources": {
            "run_ids": DEFAULT_RUNS,
            "per_request_metrics": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in metric_paths.items()
            },
            "features": {
                "path": str(feature_path),
                "sha256": sha256_file(feature_path),
                "columns": feature_columns,
            },
        },
        "oracle": oracle,
        "predictability": predictability,
        "construct_checks": {
            "random_canary_oracle_below_actual": (
                random_oracle["oracle_ndcg@10"] < actual_oracle["oracle_ndcg@10"]
            ),
            "random_canary_auc_below_actual": random_auc < actual_auc,
            "history_absent_actual_headroom_relative": oracle["history_absent"]["actual"][
                "headroom_relative"
            ],
        },
        "key_differences": {
            "random_minus_actual_oracle_ndcg@10": (
                random_oracle["oracle_ndcg@10"] - actual_oracle["oracle_ndcg@10"]
            ),
            "random_minus_actual_headroom_relative": (
                random_oracle["headroom_relative"]
                - actual_oracle["headroom_relative"]
            ),
            "random_minus_actual_macro_ovr_auc": random_auc - actual_auc,
        },
        "conclusion": (
            "A fixed random ranking channel reproduces and exceeds both the M3 oracle "
            "headroom and the M4 label-predictability result. M3/M4 therefore do not, "
            "as currently defined, distinguish learnable personalized evidence utility "
            "from per-request argmax selection noise. Frozen results remain preserved, "
            "but +28.0% and 0.6688 are not usable as positive construct-validity evidence "
            "until the protocol adds a noise control or held-out selection/evaluation."
        ),
    }
    write_json(args.output, result)
    print(json.dumps(result["key_differences"], sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
