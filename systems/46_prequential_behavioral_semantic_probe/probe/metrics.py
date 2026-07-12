from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


def bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("C46 bootstrap input invalid")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"requests": len(values), "mean": float(values.mean()), "samples": samples, "seed": seed, "percentile_95_ci": [float(low), float(high)]}


def compare(
    request_ids: Sequence[str],
    primary: np.ndarray,
    references: Mapping[str, np.ndarray],
    *,
    samples: int,
    seed: int,
    folds: int,
) -> dict[str, Any]:
    result = {}
    for offset, (name, reference) in enumerate(references.items()):
        difference = np.asarray(primary) - np.asarray(reference)
        row = bootstrap(difference, samples=samples, seed=seed + offset)
        fold_rows = []
        for fold in range(folds):
            selected = np.asarray(
                [
                    int.from_bytes(hashlib.sha256(f"c46:{seed}:{request_id}".encode()).digest()[:8], "big") % folds == fold
                    for request_id in request_ids
                ],
                dtype=bool,
            )
            fold_rows.append({"fold": fold, "requests": int(selected.sum()), "mean_difference": float(difference[selected].mean())})
        row["hash_folds"] = fold_rows
        result[name] = row
    return result


def order(values: np.ndarray, item_ids: Sequence[str]) -> tuple[int, ...]:
    return tuple(sorted(range(len(values)), key=lambda i: (-float(values[i]), str(item_ids[i]))))


def order_change_fraction(
    first: Sequence[np.ndarray], second: Sequence[np.ndarray], item_ids: Sequence[Sequence[str]]
) -> float:
    return float(np.mean([order(a, ids) != order(b, ids) for a, b, ids in zip(first, second, item_ids)]))


def clicked_direction(scores: Sequence[np.ndarray], labels: Sequence[np.ndarray]) -> np.ndarray:
    values = []
    for score, label in zip(scores, labels):
        positive = np.asarray(label) > 0
        if positive.any() and (~positive).any():
            values.append(float(np.asarray(score)[positive].mean() - np.asarray(score)[~positive].mean()))
    if not values:
        raise ValueError("C46 clicked direction empty")
    return np.asarray(values, dtype=np.float64)
