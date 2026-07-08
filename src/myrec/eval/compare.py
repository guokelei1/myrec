"""Shared paired bootstrap comparison."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from myrec.utils.jsonl import iter_jsonl, write_json


def compare_per_request_metrics(
    run_a_path: str | Path,
    run_b_path: str | Path,
    output_path: str | Path,
    metric: str = "ndcg@10",
    samples: int = 10000,
    seed: int = 20260708,
    request_ids_path: str | Path | None = None,
) -> dict[str, Any]:
    a = _load_metric_map(run_a_path, metric)
    b = _load_metric_map(run_b_path, metric)
    if request_ids_path is not None:
        request_ids = _load_request_ids(request_ids_path)
        a = {request_id: value for request_id, value in a.items() if request_id in request_ids}
        b = {request_id: value for request_id, value in b.items() if request_id in request_ids}
        if len(a) != len(request_ids) or len(b) != len(request_ids):
            raise ValueError(
                f"subset coverage mismatch: subset={len(request_ids)} a={len(a)} b={len(b)}"
            )
    if set(a) != set(b):
        raise ValueError("per-request metric request_id sets differ")
    request_ids = sorted(a)
    diffs = [a[request_id] - b[request_id] for request_id in request_ids]
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    bootstrap = []
    n = len(diffs)
    for _ in range(samples):
        bootstrap.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    bootstrap.sort()
    lo = bootstrap[int(0.025 * samples)]
    hi = bootstrap[min(samples - 1, int(0.975 * samples))]
    result = {
        "metric": metric,
        "num_requests": n,
        "delta": point,
        "ci95": [lo, hi],
        "samples": samples,
        "seed": seed,
        "significant_a_gt_b": lo > 0,
    }
    if request_ids_path is not None:
        result["request_ids_path"] = str(request_ids_path)
    write_json(output_path, result)
    return result


def _load_metric_map(path: str | Path, metric: str) -> dict[str, float]:
    values = {}
    for row in iter_jsonl(path):
        values[str(row["request_id"])] = float(row[metric])
    return values


def _load_request_ids(path: str | Path) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}
