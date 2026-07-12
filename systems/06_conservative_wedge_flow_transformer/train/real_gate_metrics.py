"""Shared-metric summaries and paired tests for the C06 real gate."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


def paired_bootstrap(
    differences: np.ndarray, *, samples: int, seed: int
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired bootstrap requires a nonempty vector")
    if not np.isfinite(values).all() or samples <= 0:
        raise ValueError("invalid paired-bootstrap input")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    # Chunking avoids a 10,000 x 1,200 integer allocation in long-lived runs.
    chunk = 512
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "requests": len(values),
        "mean": float(values.mean()),
        "samples": int(samples),
        "seed": int(seed),
        "percentile_95_ci": [float(low), float(high)],
    }


def request_fold(request_id: str, *, seed: int, folds: int) -> int:
    if folds <= 1:
        raise ValueError("fold count must exceed one")
    payload = f"c06_real_fold:{seed}:{request_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16) % folds


def fold_differences(
    request_ids: Sequence[str],
    differences: np.ndarray,
    *,
    seed: int,
    folds: int,
) -> list[dict[str, Any]]:
    values = np.asarray(differences, dtype=np.float64)
    if len(request_ids) != len(values):
        raise ValueError("request/fold-value length mismatch")
    result = []
    for fold in range(folds):
        selected = np.asarray(
            [request_fold(str(request_id), seed=seed, folds=folds) == fold for request_id in request_ids],
            dtype=bool,
        )
        if not selected.any():
            raise ValueError(f"empty C06 hash fold: {fold}")
        result.append(
            {
                "fold": fold,
                "requests": int(selected.sum()),
                "mean_difference": float(values[selected].mean()),
            }
        )
    return result


def compare_primary(
    *,
    request_ids: Sequence[str],
    primary: np.ndarray,
    references: Mapping[str, np.ndarray],
    bootstrap_samples: int,
    bootstrap_seed: int,
    folds: int,
) -> dict[str, Any]:
    primary_values = np.asarray(primary, dtype=np.float64)
    if len(primary_values) != len(request_ids):
        raise ValueError("primary/request length mismatch")
    result: dict[str, Any] = {}
    for offset, (name, raw_reference) in enumerate(references.items()):
        reference = np.asarray(raw_reference, dtype=np.float64)
        if reference.shape != primary_values.shape:
            raise ValueError(f"reference shape mismatch: {name}")
        differences = primary_values - reference
        row = paired_bootstrap(
            differences,
            samples=bootstrap_samples,
            seed=bootstrap_seed + offset,
        )
        row["hash_folds"] = fold_differences(
            request_ids,
            differences,
            seed=bootstrap_seed,
            folds=folds,
        )
        result[name] = row
    return result


def order_change_summary(
    *,
    base_rankings: Sequence[Sequence[str]],
    personalized_rankings: Sequence[Sequence[str]],
) -> dict[str, Any]:
    if len(base_rankings) != len(personalized_rankings) or not base_rankings:
        raise ValueError("ranking collections must be nonempty and aligned")
    any_changes = 0
    top10_changes = 0
    for base, personalized in zip(base_rankings, personalized_rankings):
        if list(base) != list(personalized):
            any_changes += 1
        if set(base[:10]) != set(personalized[:10]):
            top10_changes += 1
    count = len(base_rankings)
    return {
        "requests": count,
        "requests_with_any_order_change": any_changes,
        "requests_with_any_order_change_fraction": any_changes / count,
        "requests_with_top10_membership_change": top10_changes,
        "requests_with_top10_membership_change_fraction": top10_changes / count,
    }


def clicked_minus_unclicked(
    *,
    deltas: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
) -> np.ndarray:
    if len(deltas) != len(labels):
        raise ValueError("delta/label request mismatch")
    rows = []
    for delta, label in zip(deltas, labels):
        delta = np.asarray(delta, dtype=np.float64)
        label = np.asarray(label, dtype=np.float64)
        if delta.shape != label.shape:
            raise ValueError("delta/label candidate mismatch")
        clicked = label > 0.0
        if clicked.any() and (~clicked).any():
            rows.append(float(delta[clicked].mean() - delta[~clicked].mean()))
    if not rows:
        raise ValueError("no request has both clicked and unclicked candidates")
    return np.asarray(rows, dtype=np.float64)
