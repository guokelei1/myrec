"""Per-request oracle headroom analysis."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from myrec.utils.jsonl import iter_jsonl, write_json


def run_per_request_oracle(
    method_metric_paths: dict[str, str | Path],
    output_dir: str | Path,
    metric: str = "ndcg@10",
    bootstrap_samples: int = 10000,
    seed: int = 20260708,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_maps = {name: _load_metric_map(path, metric) for name, path in method_metric_paths.items()}
    request_ids = _shared_request_ids(metric_maps)
    global_means = {
        name: sum(values[request_id] for request_id in request_ids) / len(request_ids)
        for name, values in metric_maps.items()
    }
    best_global_method = max(global_means, key=lambda name: global_means[name])
    choices = []
    choice_counts = Counter()
    oracle_values = []
    best_global_values = []
    method_order = list(method_metric_paths)
    for request_id in request_ids:
        values = {name: metric_maps[name][request_id] for name in method_order}
        chosen_method = max(method_order, key=lambda name: (values[name], -method_order.index(name)))
        chosen_value = values[chosen_method]
        choices.append(
            {
                "chosen_method": chosen_method,
                "oracle_metric": chosen_value,
                "request_id": request_id,
                "values": values,
            }
        )
        choice_counts[chosen_method] += 1
        oracle_values.append(chosen_value)
        best_global_values.append(metric_maps[best_global_method][request_id])

    oracle_mean = sum(oracle_values) / len(oracle_values)
    best_global_mean = global_means[best_global_method]
    diffs = [oracle - baseline for oracle, baseline in zip(oracle_values, best_global_values)]
    delta = sum(diffs) / len(diffs)
    ci = _bootstrap_ci(diffs, samples=bootstrap_samples, seed=seed)
    split_half = _split_half_headroom(
        request_ids=request_ids,
        oracle_by_request={row["request_id"]: row["oracle_metric"] for row in choices},
        baseline_by_request=metric_maps[best_global_method],
        baseline_mean=best_global_mean,
        seed=seed,
    )
    choice_distribution = {
        name: {
            "count": choice_counts[name],
            "rate": choice_counts[name] / len(request_ids),
        }
        for name in method_order
    }
    max_choice_rate = max(row["rate"] for row in choice_distribution.values()) if choice_distribution else 0.0
    relative = delta / best_global_mean if best_global_mean else 0.0
    ci_relative = [value / best_global_mean if best_global_mean else 0.0 for value in ci]
    gate_checks = {
        "headroom_relative_ge_5pct": relative >= 0.05,
        "bootstrap_ci_low_ge_2pct": ci_relative[0] >= 0.02,
        "split_halves_ge_2pct": all(row["relative_headroom"] >= 0.02 for row in split_half.values()),
        "no_single_channel_over_90pct": max_choice_rate <= 0.90,
    }
    summary = {
        "best_global_method": best_global_method,
        "best_global_metric": best_global_mean,
        "bootstrap": {
            "ci95": ci,
            "ci95_relative": ci_relative,
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "choice_distribution": choice_distribution,
        "delta": delta,
        "gate_checks": gate_checks,
        "gate_status": "passed" if all(gate_checks.values()) else "failed",
        "headroom_relative": relative,
        "method_global_metrics": global_means,
        "metric": metric,
        "num_requests": len(request_ids),
        "oracle_metric": oracle_mean,
        "split_half": split_half,
    }
    _write_choices(output_dir / "oracle_choices.jsonl", choices)
    write_json(output_dir / "headroom_summary.json", summary)
    return summary


def _load_metric_map(path: str | Path, metric: str) -> dict[str, float]:
    values = {}
    for row in iter_jsonl(path):
        values[str(row["request_id"])] = float(row[metric])
    if not values:
        raise ValueError(f"empty per-request metric file: {path}")
    return values


def _shared_request_ids(metric_maps: dict[str, dict[str, float]]) -> list[str]:
    iterator = iter(metric_maps.values())
    request_ids = set(next(iterator))
    for values in iterator:
        if set(values) != request_ids:
            raise ValueError("per-request metric files have different request_id sets")
    return sorted(request_ids)


def _bootstrap_ci(diffs: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(diffs)
    values = []
    for _ in range(samples):
        values.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    values.sort()
    return [values[int(0.025 * samples)], values[min(samples - 1, int(0.975 * samples))]]


def _split_half_headroom(
    request_ids: list[str],
    oracle_by_request: dict[str, float],
    baseline_by_request: dict[str, float],
    baseline_mean: float,
    seed: int,
) -> dict[str, Any]:
    shuffled = list(request_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    halves = {"first": shuffled[:midpoint], "second": shuffled[midpoint:]}
    result = {}
    for name, half_ids in halves.items():
        oracle_mean = sum(oracle_by_request[request_id] for request_id in half_ids) / len(half_ids)
        baseline_half_mean = sum(baseline_by_request[request_id] for request_id in half_ids) / len(half_ids)
        delta = oracle_mean - baseline_half_mean
        result[name] = {
            "baseline_metric": baseline_half_mean,
            "delta": delta,
            "oracle_metric": oracle_mean,
            "relative_headroom": delta / baseline_mean if baseline_mean else 0.0,
            "requests": len(half_ids),
        }
    return result


def _write_choices(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
