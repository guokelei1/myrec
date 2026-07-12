"""Paired statistics and ranking diagnostics for C23."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


def paired_bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("C23 bootstrap requires a finite nonempty vector")
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
    payload = f"c23_recurrence_reset_fold:{seed}:{request_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


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
        fold_rows = []
        for fold in range(folds):
            selected = np.asarray(
                [
                    request_fold(str(request_id), seed=seed, folds=folds) == fold
                    for request_id in request_ids
                ],
                dtype=bool,
            )
            if not selected.any():
                raise ValueError(f"empty C23 hash fold: {fold}")
            fold_rows.append(
                {
                    "fold": fold,
                    "requests": int(selected.sum()),
                    "mean_difference": float(difference[selected].mean()),
                }
            )
        row["hash_folds"] = fold_rows
        output[name] = row
    return output


def retention_bootstrap(
    clean_gain: np.ndarray,
    corrupt_gain: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    clean = np.asarray(clean_gain, dtype=np.float64)
    corrupt = np.asarray(corrupt_gain, dtype=np.float64)
    if clean.shape != corrupt.shape or clean.ndim != 1 or not len(clean):
        raise ValueError("invalid C23 retention arrays")
    if not np.isfinite(clean).all() or not np.isfinite(corrupt).all():
        raise ValueError("nonfinite C23 retention arrays")
    clean_mean = float(clean.mean())
    if clean_mean <= 0.0:
        return {
            "applicable": False,
            "requests": len(clean),
            "clean_gain_mean": clean_mean,
            "corrupted_gain_mean": float(corrupt.mean()),
            "retention": None,
            "percentile_95_ci": None,
            "reason": "nonpositive clean gain",
        }
    rng = np.random.default_rng(seed)
    finite_ratios: list[float] = []
    invalid = 0
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        draws = rng.integers(0, len(clean), size=(stop - start, len(clean)))
        clean_means = clean[draws].mean(axis=1)
        corrupt_means = corrupt[draws].mean(axis=1)
        valid = clean_means > 0.0
        invalid += int((~valid).sum())
        finite_ratios.extend((corrupt_means[valid] / clean_means[valid]).tolist())
    if not finite_ratios:
        return {
            "applicable": False,
            "requests": len(clean),
            "clean_gain_mean": clean_mean,
            "corrupted_gain_mean": float(corrupt.mean()),
            "retention": None,
            "percentile_95_ci": None,
            "reason": "all bootstrap clean denominators nonpositive",
        }
    low, high = np.percentile(np.asarray(finite_ratios), [2.5, 97.5])
    return {
        "applicable": True,
        "requests": len(clean),
        "clean_gain_mean": clean_mean,
        "corrupted_gain_mean": float(corrupt.mean()),
        "retention": float(corrupt.mean() / clean_mean),
        "samples": int(samples),
        "seed": int(seed),
        "bootstrap_nonpositive_clean_denominator_draws": invalid,
        "percentile_95_ci": [float(low), float(high)],
    }


def clicked_minus_unclicked(
    corrections: Sequence[np.ndarray], labels: Sequence[np.ndarray]
) -> np.ndarray:
    if len(corrections) != len(labels):
        raise ValueError("C23 correction/label row count differs")
    output: list[float] = []
    for correction, label in zip(corrections, labels):
        delta = np.asarray(correction, dtype=np.float64)
        target = np.asarray(label, dtype=np.float64)
        clicked = target > 0.0
        if clicked.any() and (~clicked).any():
            output.append(float(delta[clicked].mean() - delta[~clicked].mean()))
    if not output:
        raise ValueError("C23 has no mixed clicked/unclicked request")
    return np.asarray(output, dtype=np.float64)
