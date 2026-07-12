"""Frozen C39 paired-bootstrap and ranking helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


def bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("C39 bootstrap values invalid")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(axis=1)
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
    output = {}
    for offset, (name, reference) in enumerate(references.items()):
        difference = np.asarray(primary) - np.asarray(reference)
        row = bootstrap(difference, samples=samples, seed=seed + offset)
        fold_rows = []
        for fold in range(folds):
            selected = np.asarray(
                [
                    int.from_bytes(
                        hashlib.sha256(f"c39-fold:{seed}:{request_id}".encode()).digest()[:8],
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


def clicked_direction(
    corrections: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
) -> np.ndarray:
    output = []
    for correction, label in zip(corrections, labels):
        correction = np.asarray(correction)
        positive = np.asarray(label) > 0
        if positive.any() and (~positive).any():
            output.append(float(correction[positive].mean() - correction[~positive].mean()))
    if not output:
        raise ValueError("C39 clicked-direction surface empty")
    return np.asarray(output, dtype=np.float64)
