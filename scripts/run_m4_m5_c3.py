#!/usr/bin/env python
"""Run M4 predictability, M5 slices, and C3 motivation gate analysis."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
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
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.metrics import ScoredCandidate, sort_candidates  # noqa: E402
from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import iter_jsonl, write_json  # noqa: E402


CHANNEL_ORDER = ["query_b2z", "history_b0b", "static_b7_bge"]
FORBIDDEN_FEATURE_SUBSTRINGS = ("ndcg", "mrr", "recall", "oracle", "label", "per_request")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dev", default="artifacts/m4/m4_features_dev.parquet")
    parser.add_argument("--feature-manifest", default="artifacts/m4/m4_feature_manifest.json")
    parser.add_argument("--m3-summary", default="reports/pps_m3_headroom_summary.json")
    parser.add_argument("--oracle-choices", default="runs/20260708_kuaisearch_m3_oracle_dev/oracle_choices.jsonl")
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(args.features_dev)
    with Path(args.m3_summary).open("r", encoding="utf-8") as handle:
        m3_summary = json.load(handle)
    input_run_ids = m3_summary["input_run_ids"]
    labels, tie_summary = _load_oracle_labels(args.oracle_choices)
    data = features.merge(labels, on="request_id", how="inner", validate="one_to_one")
    if len(data) != len(features):
        raise ValueError(f"feature/oracle row mismatch: features={len(features)} joined={len(data)}")

    feature_cols = [col for col in features.columns if col not in {"request_id", "split"}]
    forbidden_cols = [
        col for col in feature_cols if any(token in col.lower() for token in FORBIDDEN_FEATURE_SUBSTRINGS)
    ]
    if forbidden_cols:
        raise ValueError(f"forbidden label-derived feature columns: {forbidden_cols}")
    X = data[feature_cols]
    y = data["chosen_method"].to_numpy()
    m4_report = _run_m4(
        X=X,
        y=y,
        feature_cols=feature_cols,
        tie_summary=tie_summary,
        b0b_metrics_path=Path(args.runs_dir) / input_run_ids["history_b0b"] / "per_request_metrics.jsonl",
        request_ids=data["request_id"].tolist(),
        seed=args.seed,
        folds=args.folds,
        features_path=Path(args.features_dev),
        feature_manifest_path=Path(args.feature_manifest),
    )
    write_json(reports_dir / "pps_m4_predictability.json", m4_report)

    m5_report, case_review = _run_m5(
        data=data,
        m3_summary=m3_summary,
        standardized_dir=Path(args.standardized_dir),
        runs_dir=Path(args.runs_dir),
        input_run_ids=input_run_ids,
        oracle_choices_path=Path(args.oracle_choices),
        seed=args.seed,
    )
    write_json(reports_dir / "pps_m5_slices.json", m5_report)
    (reports_dir / "pps_m5_case_review.md").write_text(case_review, encoding="utf-8")

    c3_report = _build_c3_report(
        m3_summary=m3_summary,
        m4_report=m4_report,
        m5_report=m5_report,
        reports_dir=reports_dir,
    )
    write_json(reports_dir / "pps_c3_motivation.json", c3_report)
    print(json.dumps(c3_report, ensure_ascii=False, sort_keys=True))
    return 0 if c3_report["status"] == "passed" else 2


def _run_m4(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    tie_summary: dict[str, Any],
    b0b_metrics_path: Path,
    request_ids: list[str],
    seed: int,
    folds: int,
    features_path: Path,
    feature_manifest_path: Path,
) -> dict[str, Any]:
    class_counts = dict(sorted(Counter(y).items()))
    logistic = Pipeline(
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
    tree = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=seed)),
        ]
    )
    models = {
        "logistic_regression": logistic,
        "decision_tree_depth3": tree,
    }
    model_results = {
        name: _cross_val_auc(model, X, y, feature_cols, folds=folds, seed=seed)
        for name, model in models.items()
    }
    shuffled = np.asarray(y).copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    shuffle_result = _cross_val_auc(logistic, X, shuffled, feature_cols, folds=folds, seed=seed)

    leak_values = _load_metric_map(b0b_metrics_path, "ndcg@10")
    X_leak = X.copy()
    X_leak["leak_history_b0b_per_request_ndcg"] = [
        leak_values[request_id] for request_id in request_ids
    ]
    leak_result = _cross_val_auc(logistic, X_leak, y, feature_cols + ["leak_history_b0b_per_request_ndcg"], folds=folds, seed=seed)
    primary_auc = model_results["logistic_regression"]["macro_ovr_auc_mean"]
    shuffle_auc = shuffle_result["macro_ovr_auc_mean"]
    leak_auc = leak_result["macro_ovr_auc_mean"]
    canaries = {
        "label_shuffle": {
            **_compact_auc_result(shuffle_result),
            "expected": "0.50 +/- 0.02",
            "passed": 0.48 <= shuffle_auc <= 0.52,
        },
        "intentional_leak_history_b0b_ndcg": {
            **_compact_auc_result(leak_result),
            "expected": "AUC should increase when an explicitly leaked per-request metric is injected.",
            "formal_primary_auc": primary_auc,
            "delta_vs_formal": leak_auc - primary_auc,
            "passed": leak_auc >= primary_auc + 0.03,
        },
    }
    return {
        "report": "pps_m4_predictability",
        "seed": seed,
        "folds": folds,
        "primary_model": "logistic_regression",
        "status": "passed" if primary_auc >= 0.65 and all(row["passed"] for row in canaries.values()) else "failed",
        "gate": {
            "criterion": "5-fold macro OvR AUC >= 0.65",
            "primary_auc": primary_auc,
            "passed": primary_auc >= 0.65,
            "failure_actions": {
                "0.60_to_0.65": "claim shrinks to static bucketed improvement",
                "below_0.60": "abandon adaptive claim and follow doc/11 C3 fallback",
            },
        },
        "class_counts": class_counts,
        "tie_summary": tie_summary,
        "features": {
            "path": str(features_path),
            "sha256": sha256_file(features_path),
            "manifest": str(feature_manifest_path),
            "manifest_sha256": sha256_file(feature_manifest_path) if feature_manifest_path.exists() else None,
            "columns": feature_cols,
            "missing_indicator_columns": [col for col in feature_cols if col.endswith("_missing")],
            "forbidden_label_derived_columns": [],
        },
        "models": model_results,
        "canaries": canaries,
    }


def _cross_val_auc(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    folds: int,
    seed: int,
) -> dict[str, Any]:
    classes = sorted(CHANNEL_ORDER)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(X, y), start=1):
        fitted = model.fit(X.iloc[train_index], y[train_index])
        proba = fitted.predict_proba(X.iloc[valid_index])
        class_to_col = {label: idx for idx, label in enumerate(fitted.classes_)}
        aligned = np.zeros((len(valid_index), len(classes)), dtype=float)
        for class_index, label in enumerate(classes):
            if label in class_to_col:
                aligned[:, class_index] = proba[:, class_to_col[label]]
        y_valid = y[valid_index]
        macro = roc_auc_score(y_valid, aligned, labels=classes, multi_class="ovr", average="macro")
        per_channel = {
            label: roc_auc_score((y_valid == label).astype(int), aligned[:, idx])
            for idx, label in enumerate(classes)
        }
        fold_rows.append({"fold": fold, "macro_ovr_auc": float(macro), "per_channel_auc": per_channel})
    macro_values = [row["macro_ovr_auc"] for row in fold_rows]
    per_channel_summary = {}
    for label in classes:
        values = [row["per_channel_auc"][label] for row in fold_rows]
        per_channel_summary[label] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    fitted_full = model.fit(X, y)
    return {
        "macro_ovr_auc_mean": float(np.mean(macro_values)),
        "macro_ovr_auc_std": float(np.std(macro_values, ddof=1)) if len(macro_values) > 1 else 0.0,
        "per_channel_auc": per_channel_summary,
        "folds": fold_rows,
        "feature_importance": _feature_importance(fitted_full, feature_cols),
    }


def _feature_importance(model: Pipeline, feature_cols: list[str]) -> list[dict[str, float | str]]:
    fitted = model.named_steps["model"]
    if hasattr(fitted, "coef_"):
        values = np.mean(np.abs(fitted.coef_), axis=0)
    elif hasattr(fitted, "feature_importances_"):
        values = fitted.feature_importances_
    else:
        return []
    rows = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(feature_cols, values)
    ]
    rows.sort(key=lambda row: float(row["importance"]), reverse=True)
    return rows


def _compact_auc_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_ovr_auc_mean": result["macro_ovr_auc_mean"],
        "macro_ovr_auc_std": result["macro_ovr_auc_std"],
        "per_channel_auc": result["per_channel_auc"],
    }


def _run_m5(
    data: pd.DataFrame,
    m3_summary: dict[str, Any],
    standardized_dir: Path,
    runs_dir: Path,
    input_run_ids: dict[str, str],
    oracle_choices_path: Path,
    seed: int,
) -> tuple[dict[str, Any], str]:
    metric_maps = {
        channel: _load_metric_map(runs_dir / run_id / "per_request_metrics.jsonl", "ndcg@10")
        for channel, run_id in input_run_ids.items()
    }
    oracle_values = {
        row["request_id"]: float(row["oracle_metric"]) for row in iter_jsonl(oracle_choices_path)
    }
    slice_frame = data[["request_id", "query_click_entropy", "history_candidate_cat_overlap"]].copy()
    slice_frame["query_train_missing"] = data["query_train_missing"].astype(int)
    for channel, values in metric_maps.items():
        slice_frame[channel] = [values[request_id] for request_id in slice_frame["request_id"]]
    slice_frame["oracle"] = [oracle_values[request_id] for request_id in slice_frame["request_id"]]
    choices = _load_choice_map(oracle_choices_path)
    slice_frame["oracle_choice"] = [choices[request_id] for request_id in slice_frame["request_id"]]
    slice_frame["entropy_bucket"] = _three_bins(slice_frame["query_click_entropy"], ["low", "medium", "high"])
    slice_frame["overlap_bucket"] = _three_bins(slice_frame["history_candidate_cat_overlap"], ["low", "medium", "high"])
    entropy_rows = _slice_summary(
        slice_frame,
        bucket_col="entropy_bucket",
        value_col="query_click_entropy",
        primary_col="query_b2z",
        oracle_col="oracle",
    )
    overlap_rows = _slice_summary(
        slice_frame,
        bucket_col="overlap_bucket",
        value_col="history_candidate_cat_overlap",
        primary_col="history_b0b",
        oracle_col="oracle",
    )
    low_entropy_gap = entropy_rows["low"]["primary_gap_to_oracle"]
    high_entropy_gap = entropy_rows["high"]["primary_gap_to_oracle"]
    low_overlap_history = overlap_rows["low"]["primary_mean"]
    high_overlap_history = overlap_rows["high"]["primary_mean"]
    low_overlap_gap = overlap_rows["low"]["primary_gap_to_oracle"]
    high_overlap_gap = overlap_rows["high"]["primary_gap_to_oracle"]
    checks = {
        "e1_high_entropy_query_gap_gt_low_entropy": high_entropy_gap > low_entropy_gap,
        "e2_low_overlap_history_mean_lt_high_overlap": low_overlap_history < high_overlap_history,
        "e2_low_overlap_history_gap_gt_high_overlap": low_overlap_gap > high_overlap_gap,
    }
    posthoc = _posthoc_entropy_diagnostics(slice_frame)
    cases = _sample_case_ids(slice_frame, seed=seed)
    case_review = _build_case_review(
        cases=cases,
        standardized_dir=standardized_dir,
        runs_dir=runs_dir,
        input_run_ids=input_run_ids,
        metric_maps=metric_maps,
        oracle_values=oracle_values,
    )
    report = {
        "report": "pps_m5_slices",
        "seed": seed,
        "metric": "ndcg@10",
        "source_artifacts": {
            "m3_summary": "reports/pps_m3_headroom_summary.json",
            "oracle_choices": str(oracle_choices_path),
            "input_run_ids": input_run_ids,
        },
        "num_requests": int(len(slice_frame)),
        "entropy_slices": entropy_rows,
        "history_candidate_overlap_slices": overlap_rows,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
        "frozen_e1_negative_result": {
            "definition": "train cross-user click entropy bucket should stratify query-only gap to oracle",
            "high_entropy_query_to_oracle_gap": high_entropy_gap,
            "low_entropy_query_to_oracle_gap": low_entropy_gap,
            "direction_holds": high_entropy_gap > low_entropy_gap,
            "interpretation": (
                "The frozen single-variable entropy proxy did not stratify the E1 failure. "
                "This is retained as a negative result and is not replaced by post-hoc variants."
            ),
        },
        "posthoc_exploratory_entropy_diagnostics": posthoc,
        "case_review": {
            "path": "reports/pps_m5_case_review.md",
            "sampled_groups": sorted(cases),
            "cases": sum(len(values) for values in cases.values()),
            "qrels_dev_use": "M-series case-review only; not used by scoring or training code.",
        },
        "m3_reference": {
            "oracle_metric": m3_summary["oracle_metric"],
            "best_global_method": m3_summary["best_global_method"],
            "best_global_metric": m3_summary["best_global_metric"],
        },
    }
    return report, case_review


def _posthoc_entropy_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    total = len(frame)
    coverage = {
        "dev_requests": int(total),
        "query_seen_in_train_requests": int((frame["query_train_missing"] == 0).sum()),
        "query_seen_in_train_rate": float((frame["query_train_missing"] == 0).mean()),
        "query_missing_in_train_requests": int((frame["query_train_missing"] == 1).sum()),
        "query_missing_in_train_rate": float((frame["query_train_missing"] == 1).mean()),
        "bucket_sizes": {
            bucket: int((frame["entropy_bucket"] == bucket).sum())
            for bucket in ["low", "medium", "high"]
        },
    }
    difficulty = {}
    channel_choice = {}
    for bucket in ["low", "medium", "high"]:
        subset = frame[frame["entropy_bucket"] == bucket]
        difficulty[bucket] = {
            "requests": int(len(subset)),
            "query_b2z_mean_ndcg10": float(subset["query_b2z"].mean()),
            "history_b0b_mean_ndcg10": float(subset["history_b0b"].mean()),
            "static_b7_bge_mean_ndcg10": float(subset["static_b7_bge"].mean()),
            "oracle_mean_ndcg10": float(subset["oracle"].mean()),
            "best_available_input_mean_ndcg10": float(
                subset[["query_b2z", "history_b0b", "static_b7_bge"]].max(axis=1).mean()
            ),
        }
        counts = Counter(subset["oracle_choice"])
        channel_choice[bucket] = {
            channel: {
                "count": int(counts[channel]),
                "rate": float(counts[channel] / len(subset)) if len(subset) else 0.0,
            }
            for channel in CHANNEL_ORDER
        }
    oracle_minus_query = frame["oracle"] - frame["query_b2z"]
    history_minus_query = frame["history_b0b"] - frame["query_b2z"]
    history_choice = (frame["oracle_choice"] == "history_b0b").astype(float)
    spearman = {
        "entropy_vs_oracle_minus_query_b2z": _spearman(frame["query_click_entropy"], oracle_minus_query),
        "entropy_vs_history_minus_query_b2z": _spearman(frame["query_click_entropy"], history_minus_query),
        "entropy_vs_history_optimal_indicator": _spearman(frame["query_click_entropy"], history_choice),
        "doc11_insight2_thresholds": {
            "rho_ge_0.4": "supports Consensus Law",
            "rho_lt_0.2": "falsifies Consensus Law",
        },
        "headline_use": "post-hoc exploratory warning only; does not change the frozen M5-E1 failed result",
    }
    rho = spearman["entropy_vs_oracle_minus_query_b2z"]
    if rho >= 0.4:
        verdict = "supports"
    elif rho < 0.2:
        verdict = "falsified"
    else:
        verdict = "inconclusive"
    spearman["consensus_law_warning_verdict"] = verdict
    return {
        "status": "post-hoc exploratory, authorized 2026-07-10",
        "coverage": coverage,
        "difficulty_by_entropy_bucket": difficulty,
        "oracle_choice_by_entropy_bucket": channel_choice,
        "insight2_consensus_law_spearman": spearman,
        "prohibited_uses": [
            "Do not switch to relative gap and claim E1 passed.",
            "Do not alter bucket boundaries and claim E1 passed.",
            "Do not alter entropy definition and claim E1 passed.",
        ],
    }


def _slice_summary(
    frame: pd.DataFrame,
    bucket_col: str,
    value_col: str,
    primary_col: str,
    oracle_col: str,
) -> dict[str, Any]:
    rows = {}
    for bucket in ["low", "medium", "high"]:
        subset = frame[frame[bucket_col] == bucket]
        primary_mean = float(subset[primary_col].mean())
        oracle_mean = float(subset[oracle_col].mean())
        rows[bucket] = {
            "requests": int(len(subset)),
            f"{value_col}_min": float(subset[value_col].min()),
            f"{value_col}_max": float(subset[value_col].max()),
            "primary_channel": primary_col,
            "primary_mean": primary_mean,
            "oracle_mean": oracle_mean,
            "primary_gap_to_oracle": oracle_mean - primary_mean,
            "primary_gap_to_oracle_relative": (oracle_mean - primary_mean) / primary_mean
            if primary_mean
            else 0.0,
            "query_b2z_mean": float(subset["query_b2z"].mean()),
            "history_b0b_mean": float(subset["history_b0b"].mean()),
            "static_b7_bge_mean": float(subset["static_b7_bge"].mean()),
        }
    return rows


def _three_bins(values: pd.Series, labels: list[str]) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=3, labels=labels)


def _sample_case_ids(frame: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    cases: dict[str, list[str]] = {}
    for column, prefix in [("entropy_bucket", "entropy"), ("overlap_bucket", "overlap")]:
        for bucket in ["low", "medium", "high"]:
            request_ids = frame.loc[frame[column] == bucket, "request_id"].tolist()
            rng.shuffle(request_ids)
            cases[f"{prefix}_{bucket}"] = sorted(request_ids[:5])
    return cases


def _build_case_review(
    cases: dict[str, list[str]],
    standardized_dir: Path,
    runs_dir: Path,
    input_run_ids: dict[str, str],
    metric_maps: dict[str, dict[str, float]],
    oracle_values: dict[str, float],
) -> str:
    selected_ids = {request_id for values in cases.values() for request_id in values}
    records = _load_records_for_cases(standardized_dir / "records_dev.jsonl", selected_ids)
    qrels = _load_qrels_for_cases(standardized_dir / "qrels_dev.jsonl", selected_ids)
    score_maps = {
        channel: _load_scores_for_cases(runs_dir / run_id / "scores.jsonl", selected_ids)
        for channel, run_id in input_run_ids.items()
    }
    lines = [
        "# PPS M5 Case Review",
        "",
        "Scope: M-series qualitative slice review. `qrels_dev` is read only here for case inspection, not by any scoring or training code.",
        "",
    ]
    for group, request_ids in sorted(cases.items()):
        lines.extend([f"## {group}", ""])
        for request_id in request_ids:
            record = records[request_id]
            history = record.get("history", [])
            positives = qrels.get(request_id, {"clicked": [], "purchased": []})
            metrics = {channel: metric_maps[channel][request_id] for channel in CHANNEL_ORDER}
            metric_text = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
            lines.extend(
                [
                    f"- `{request_id}` query={json.dumps(record.get('query', ''), ensure_ascii=False)} "
                    f"history_len={len(history)} oracle={oracle_values[request_id]:.4f}; {metric_text}",
                    f"  clicked={positives['clicked'][:5]} purchased={positives['purchased'][:5]}",
                    f"  history_tail={_history_tail(history)}",
                ]
            )
            for channel in CHANNEL_ORDER:
                top_titles = _top_titles(record, score_maps[channel][request_id])
                lines.append(f"  top5_{channel}: {top_titles}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _history_tail(history: list[dict[str, Any]]) -> str:
    tail = history[-3:]
    values = []
    for item in tail:
        title = str(item.get("title") or "")[:32]
        cats = "/".join(str(part) for part in item.get("cat", []) if str(part).upper() != "UNKNOWN")
        values.append(f"{title} [{cats}]")
    return "; ".join(values) if values else "EMPTY"


def _top_titles(record: dict[str, Any], scores: dict[str, float]) -> str:
    item_by_id = {str(candidate["item_id"]): candidate for candidate in record.get("candidates", [])}
    ranked = sort_candidates(
        str(record["request_id"]),
        [ScoredCandidate(item_id=item_id, score=score) for item_id, score in scores.items()],
    )
    values = []
    for candidate in ranked[:5]:
        item = item_by_id[candidate.item_id]
        title = str(item.get("title") or "")[:28]
        values.append(f"{candidate.item_id}:{title}")
    return " | ".join(values)


def _load_oracle_labels(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    ties = 0
    total = 0
    for row in iter_jsonl(path):
        values = {name: float(value) for name, value in row["values"].items()}
        best = max(values.values())
        ties += int(sum(math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12) for value in values.values()) > 1)
        total += 1
        rows.append({"request_id": str(row["request_id"]), "chosen_method": str(row["chosen_method"])})
    return pd.DataFrame(rows), {
        "tie_requests": ties,
        "total_requests": total,
        "tie_rate": ties / total if total else 0.0,
        "tie_rule": "M3 channel order: query_b2z, history_b0b, static_b7_bge",
    }


def _load_metric_map(path: str | Path, metric: str) -> dict[str, float]:
    return {str(row["request_id"]): float(row[metric]) for row in iter_jsonl(path)}


def _load_choice_map(path: str | Path) -> dict[str, str]:
    return {str(row["request_id"]): str(row["chosen_method"]) for row in iter_jsonl(path)}


def _spearman(left: pd.Series, right: pd.Series) -> float:
    value = left.corr(right, method="spearman")
    return float(value) if value == value else 0.0


def _load_records_for_cases(path: Path, request_ids: set[str]) -> dict[str, dict[str, Any]]:
    records = {}
    for record in iter_jsonl(path):
        request_id = str(record["request_id"])
        if request_id in request_ids:
            records[request_id] = record
    return records


def _load_qrels_for_cases(path: Path, request_ids: set[str]) -> dict[str, dict[str, list[str]]]:
    qrels = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in request_ids:
            qrels[request_id] = {
                "clicked": [str(item_id) for item_id in row.get("clicked", [])],
                "purchased": [str(item_id) for item_id in row.get("purchased", [])],
            }
    return qrels


def _load_scores_for_cases(path: Path, request_ids: set[str]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in request_ids:
            scores[request_id][str(row["candidate_item_id"])] = float(row["score"])
    return dict(scores)


def _build_c3_report(
    m3_summary: dict[str, Any],
    m4_report: dict[str, Any],
    m5_report: dict[str, Any],
    reports_dir: Path,
) -> dict[str, Any]:
    m3_passed = m3_summary.get("gate_status") == "passed"
    m4_passed = bool(m4_report["gate"]["passed"])
    m5_passed = m5_report["status"] == "passed"
    canaries_passed = all(row["passed"] for row in m4_report["canaries"].values())
    checks = {
        "m3_headroom": {
            "passed": m3_passed,
            "evidence": {
                "headroom_relative": m3_summary["headroom_relative"],
                "ci95_relative": m3_summary["bootstrap"]["ci95_relative"],
                "split_half": m3_summary["split_half"],
                "choice_distribution": m3_summary["choice_distribution"],
            },
        },
        "m4_auc": {
            "passed": m4_passed,
            "evidence": {
                "primary_model": m4_report["primary_model"],
                "macro_ovr_auc_mean": m4_report["gate"]["primary_auc"],
                "criterion": m4_report["gate"]["criterion"],
            },
        },
        "m4_canaries": {
            "passed": canaries_passed,
            "evidence": m4_report["canaries"],
        },
        "m5_slice_direction": {
            "passed": m5_passed,
            "evidence": m5_report["checks"],
        },
    }
    frozen_status = "passed" if all(row["passed"] for row in checks.values()) else "failed"
    adjudication = _build_c3_adjudication(m3_summary, m4_report, m5_report, frozen_status)
    final_status = adjudication["post_adjudication_status"]
    return {
        "report": "pps_c3_motivation",
        "status": final_status,
        "frozen_gate_status_before_adjudication": frozen_status,
        "metric": "ndcg@10",
        "checks": checks,
        "adjudication": adjudication,
        "conclusion": (
            "C3 passed: oracle headroom exists, the best evidence channel is predictable from cheap request features, and M5 slices support the E1/E2 failure directions."
            if all(row["passed"] for row in checks.values())
            else adjudication["conclusion"]
        ),
        "evidence_files": {
            "m3": "reports/pps_m3_headroom_summary.json",
            "m4": str(reports_dir / "pps_m4_predictability.json"),
            "m5": str(reports_dir / "pps_m5_slices.json"),
            "m5_cases": str(reports_dir / "pps_m5_case_review.md"),
        },
    }


def _build_c3_adjudication(
    m3_summary: dict[str, Any],
    m4_report: dict[str, Any],
    m5_report: dict[str, Any],
    frozen_status: str,
) -> dict[str, Any]:
    e1_failed = not bool(m5_report["checks"]["e1_high_entropy_query_gap_gt_low_entropy"])
    e2_passed = (
        bool(m5_report["checks"]["e2_low_overlap_history_mean_lt_high_overlap"])
        and bool(m5_report["checks"]["e2_low_overlap_history_gap_gt_high_overlap"])
    )
    m3_passed = m3_summary.get("gate_status") == "passed"
    m4_passed = bool(m4_report["gate"]["passed"])
    canaries_passed = all(row["passed"] for row in m4_report["canaries"].values())
    post_adjudication_passed = (
        frozen_status == "passed"
        or (m3_passed and m4_passed and canaries_passed and e1_failed and e2_passed)
    )
    top_importance = m4_report["models"]["logistic_regression"]["feature_importance"][:10]
    entropy_rank = next(
        (
            index
            for index, row in enumerate(m4_report["models"]["logistic_regression"]["feature_importance"], start=1)
            if row["feature"] == "query_click_entropy"
        ),
        None,
    )
    return {
        "authorization": {
            "source": "user handling decision in chat on 2026-07-10",
            "date": "2026-07-10",
            "rule_invoked": "doc/11 C3 M5 failure action: insight wording rewrite and re-review",
        },
        "frozen_m5_e1_result_preserved": m5_report["frozen_e1_negative_result"],
        "rewritten_e1_evidence": {
            "new_wording": (
                "Query-only failure is supported by direct M3 per-request oracle slicing, "
                "not by the single-variable entropy bucket proxy."
            ),
            "direct_m3_evidence": {
                "history_optimal_request_rate": m3_summary["choice_distribution"]["history_b0b"]["rate"],
                "history_optimal_request_count": m3_summary["choice_distribution"]["history_b0b"]["count"],
                "source": "reports/pps_m3_bidirectional_slice.json",
                "query_loss_on_history_optimal_slice_relative": -0.5705109460931871,
            },
            "negative_proxy_result": m5_report["frozen_e1_negative_result"],
        },
        "m4_feature_importance_observation": {
            "query_click_entropy_rank": entropy_rank,
            "top_10": top_importance,
            "interpretation": (
                "M4 passes with a multivariate feature set; query click entropy is not the dominant signal, "
                "which is consistent with the failed single-variable entropy proxy."
            ),
        },
        "insight2_warning": {
            "text": (
                "The Phase 4 Consensus Law falsification relies on the same entropy mechanism and is at risk. "
                "The post-hoc Spearman diagnostic is recorded as an early warning, not as a headline result."
            ),
            "posthoc_spearman": m5_report["posthoc_exploratory_entropy_diagnostics"][
                "insight2_consensus_law_spearman"
            ],
        },
        "post_adjudication_status": "passed" if post_adjudication_passed else "failed",
        "step2_step3_unlocked": bool(post_adjudication_passed),
        "conclusion": (
            "C3 passed after the pre-registered doc/11 rewrite-and-review action: M3 passed, M4 passed, "
            "M5-E2 passed, and M5-E1 is retained as a frozen negative entropy-proxy result while E1 is "
            "supported by direct M3 per-request slicing."
            if post_adjudication_passed
            else "C3 remains failed after adjudication; do not start Step 2/3."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
