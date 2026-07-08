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
) -> dict[str, Any]:
    a = _load_metric_map(run_a_path, metric)
    b = _load_metric_map(run_b_path, metric)
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
    write_json(output_path, result)
    return result


def _load_metric_map(path: str | Path, metric: str) -> dict[str, float]:
    values = {}
    for row in iter_jsonl(path):
        values[str(row["request_id"])] = float(row[metric])
    return values
