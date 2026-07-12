"""Paired train-internal statistics for the C21 signal gate."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


def paired_bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("C21 paired bootstrap requires a finite nonempty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        draws = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "requests": len(array),
        "mean": float(array.mean()),
        "samples": int(samples),
        "seed": int(seed),
        "percentile_95_ci": [float(low), float(high)],
    }


def request_fold(request_id: str, *, seed: int, folds: int) -> int:
    payload = f"c21_train_path_fold:{seed}:{request_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def fold_means(
    request_ids: Sequence[str],
    differences: np.ndarray,
    *,
    seed: int,
    folds: int,
) -> list[dict[str, Any]]:
    values = np.asarray(differences, dtype=np.float64)
    if len(request_ids) != len(values) or folds <= 1:
        raise ValueError("invalid C21 hash folds")
    result: list[dict[str, Any]] = []
    for fold in range(folds):
        selected = np.asarray(
            [request_fold(str(request_id), seed=seed, folds=folds) == fold for request_id in request_ids],
            dtype=bool,
        )
        if not selected.any():
            raise ValueError(f"empty C21 hash fold: {fold}")
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
    samples: int,
    seed: int,
    folds: int,
) -> dict[str, Any]:
    primary_values = np.asarray(primary, dtype=np.float64)
    output: dict[str, Any] = {}
    for offset, (name, reference) in enumerate(references.items()):
        difference = primary_values - np.asarray(reference, dtype=np.float64)
        row = paired_bootstrap(difference, samples=samples, seed=seed + offset)
        row["hash_folds"] = fold_means(request_ids, difference, seed=seed, folds=folds)
        output[name] = row
    return output


def retention_bootstrap(
    clean_gain: np.ndarray,
    corrupted_gain: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    clean = np.asarray(clean_gain, dtype=np.float64)
    corrupt = np.asarray(corrupted_gain, dtype=np.float64)
    if clean.shape != corrupt.shape or clean.ndim != 1 or not len(clean):
        raise ValueError("invalid C21 corruption retention arrays")
    if not np.isfinite(clean).all() or not np.isfinite(corrupt).all():
        raise ValueError("nonfinite C21 corruption retention arrays")
    point_denominator = float(clean.mean())
    point = float(corrupt.mean() / point_denominator) if point_denominator > 0.0 else float("inf")
    rng = np.random.default_rng(seed)
    ratios = np.empty(samples, dtype=np.float64)
    invalid = 0
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        draws = rng.integers(0, len(clean), size=(stop - start, len(clean)))
        clean_means = clean[draws].mean(axis=1)
        corrupt_means = corrupt[draws].mean(axis=1)
        valid = clean_means > 0.0
        invalid += int((~valid).sum())
        ratios[start:stop] = np.where(valid, corrupt_means / clean_means, np.inf)
    low, high = np.percentile(ratios, [2.5, 97.5])
    return {
        "requests": len(clean),
        "clean_gain_mean": point_denominator,
        "corrupted_gain_mean": float(corrupt.mean()),
        "retention": point,
        "samples": int(samples),
        "seed": int(seed),
        "bootstrap_nonpositive_clean_denominator_draws": invalid,
        "percentile_95_ci": [float(low), float(high)],
    }


def clicked_minus_unclicked(
    deltas: Sequence[np.ndarray], labels: Sequence[np.ndarray]
) -> np.ndarray:
    if len(deltas) != len(labels):
        raise ValueError("C21 delta/label request count mismatch")
    output: list[float] = []
    for delta, label in zip(deltas, labels):
        delta_array = np.asarray(delta, dtype=np.float64)
        label_array = np.asarray(label, dtype=np.float64)
        if delta_array.shape != label_array.shape:
            raise ValueError("C21 delta/label candidate mismatch")
        clicked = label_array > 0.0
        if clicked.any() and (~clicked).any():
            output.append(float(delta_array[clicked].mean() - delta_array[~clicked].mean()))
    if not output:
        raise ValueError("C21 has no mixed clicked/unclicked request")
    return np.asarray(output, dtype=np.float64)
