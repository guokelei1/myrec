"""Surface summaries for response-direction interventions."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from myrec.eval.response_direction_intervention import (
    aggregate_direction_interventions,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


CI_METRICS = (
    "mean_actual_minus_null_ndcg@10",
    "mean_actual_minus_random_ndcg@10",
    "mean_aligned_minus_actual_ndcg@10",
    "mean_aligned_minus_null_ndcg@10",
    "direction_conversion_efficiency",
)


def summarize_response_direction_intervention_surfaces(
    per_request_path: str | Path,
    records_path: str | Path,
    surfaces_dir: str | Path,
    output_path: str | Path,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 20260714,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    per_request_path = Path(per_request_path)
    records_path = Path(records_path)
    surfaces_dir = Path(surfaces_dir)
    rows = {str(row["request_id"]): row for row in iter_jsonl(per_request_path)}
    cluster_keys = _load_cluster_keys(records_path)
    if set(rows) != set(cluster_keys):
        raise ValueError("per-request results and records have different coverage")

    surfaces = {}
    empty_surfaces: list[str] = []
    for path in sorted(surfaces_dir.glob("*.txt")):
        name = path.stem
        request_ids = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing = request_ids - set(rows)
        if missing:
            raise ValueError(f"surface {name} has unknown request IDs")
        selected = [rows[request_id] for request_id in sorted(request_ids)]
        if not selected:
            empty_surfaces.append(name)
            continue
        aggregate = aggregate_direction_interventions(selected)
        surface_seed = seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        aggregate["bootstrap_ci95"] = {
            "request": _bootstrap_ci(
                selected,
                lambda row: str(row["request_id"]),
                bootstrap_samples,
                surface_seed,
            ),
            "user_cluster": _bootstrap_ci(
                selected,
                lambda row: cluster_keys[str(row["request_id"])]["user_id"],
                bootstrap_samples,
                surface_seed + 1,
            ),
            "query_cluster": _bootstrap_ci(
                selected,
                lambda row: cluster_keys[str(row["request_id"])]["query"],
                bootstrap_samples,
                surface_seed + 2,
            ),
        }
        aggregate["surface_request_ids_path"] = str(path)
        aggregate["surface_request_ids_sha256"] = sha256_file(path)
        surfaces[name] = aggregate
    result = {
        "analysis_type": "response_direction_intervention_surface_summary",
        "bootstrap_samples": bootstrap_samples,
        "ci_metrics": list(CI_METRICS),
        "empty_surfaces": empty_surfaces,
        "per_request_path": str(per_request_path),
        "per_request_sha256": sha256_file(per_request_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "seed": seed,
        "surfaces": surfaces,
    }
    write_json(output_path, result)
    return result


def _bootstrap_ci(
    rows: list[dict[str, Any]],
    cluster_key: Callable[[dict[str, Any]], str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[cluster_key(row)].append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    draws = {metric: [] for metric in CI_METRICS}
    for _ in range(samples):
        sampled = [
            row
            for _index in range(len(keys))
            for row in clusters[keys[rng.randrange(len(keys))]]
        ]
        metrics = aggregate_direction_interventions(sampled)
        for metric in CI_METRICS:
            value = metrics.get(metric)
            if value is not None:
                draws[metric].append(float(value))
    result: dict[str, Any] = {"num_clusters": len(keys)}
    for metric, values in draws.items():
        if not values:
            result[metric] = None
            continue
        values.sort()
        result[metric] = [
            values[int(0.025 * len(values))],
            values[min(len(values) - 1, int(0.975 * len(values)))],
        ]
    return result


def _load_cluster_keys(records_path: Path) -> dict[str, dict[str, str]]:
    result = {}
    for row in iter_jsonl(records_path):
        request_id = str(row["request_id"])
        result[request_id] = {
            "query": "".join(str(row.get("query", "")).lower().split()),
            "user_id": str(row["user_id"]),
        }
    return result
