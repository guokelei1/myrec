from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


def bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("C26 bootstrap values invalid")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "requests": len(values),
        "mean": float(values.mean()),
        "samples": samples,
        "seed": seed,
        "percentile_95_ci": [float(low), float(high)],
    }


def compare(
    request_ids: Sequence[str],
    primary: np.ndarray,
    references: Mapping[str, np.ndarray],
    *,
    samples: int,
    seed: int,
    folds: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for offset, (name, reference) in enumerate(references.items()):
        difference = np.asarray(primary) - np.asarray(reference)
        row = bootstrap(difference, samples=samples, seed=seed + offset)
        fold_rows = []
        for fold in range(folds):
            selected = np.asarray(
                [
                    int.from_bytes(
                        hashlib.sha256(f"c26-fold:{seed}:{request_id}".encode()).digest()[:8],
                        "big",
                    )
                    % folds
                    == fold
                    for request_id in request_ids
                ],
                dtype=bool,
            )
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


def retention(clean: np.ndarray, corrupt: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    clean, corrupt = np.asarray(clean), np.asarray(corrupt)
    clean_mean = float(clean.mean())
    if clean_mean <= 0:
        return {
            "applicable": False,
            "clean_gain_mean": clean_mean,
            "corrupted_gain_mean": float(corrupt.mean()),
            "retention": None,
            "percentile_95_ci": None,
        }
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    invalid = 0
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        draws = rng.integers(0, len(clean), size=(stop - start, len(clean)))
        denominators = clean[draws].mean(1)
        numerators = corrupt[draws].mean(1)
        valid = denominators > 0
        invalid += int((~valid).sum())
        ratios.extend((numerators[valid] / denominators[valid]).tolist())
    if not ratios:
        return {
            "applicable": False,
            "clean_gain_mean": clean_mean,
            "corrupted_gain_mean": float(corrupt.mean()),
            "retention": None,
            "percentile_95_ci": None,
        }
    low, high = np.percentile(ratios, [2.5, 97.5])
    return {
        "applicable": True,
        "clean_gain_mean": clean_mean,
        "corrupted_gain_mean": float(corrupt.mean()),
        "retention": float(corrupt.mean() / clean_mean),
        "percentile_95_ci": [float(low), float(high)],
        "invalid_denominator_draws": invalid,
    }


def clicked_direction(corrections: Sequence[np.ndarray], labels: Sequence[np.ndarray]) -> np.ndarray:
    output = []
    for correction, target in zip(corrections, labels):
        correction, target = np.asarray(correction), np.asarray(target)
        positive = target > 0
        if positive.any() and (~positive).any():
            output.append(float(correction[positive].mean() - correction[~positive].mean()))
    if not output:
        raise ValueError("C26 clicked-direction surface empty")
    return np.asarray(output, dtype=np.float64)
