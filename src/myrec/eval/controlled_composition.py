"""Request-level diagnostics for controlled history composition.

The module consumes evaluator outputs only.  It does not read qrels or score
models, which keeps the analysis layer separate from training and scoring.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Iterable


ACCOUNTING_METRICS = (
    "base_retention",
    "history_utility",
    "net_value",
    "additive_gain",
    "full_repair",
    "repair_shortfall",
    "nonpositive_history",
)

DESCRIPTIVE_ACCOUNTING_METRICS = (
    "null_path_gap",
    "same_checkpoint_recovery",
    "end_model_gap",
    "additive_gain",
    "recovered_to_qc",
    "recovery_shortfall",
    "nonpositive_recovery",
)


def build_endpoint_aligned_accounting_rows(
    qc_rows: dict[str, dict[str, Any]],
    response_rows: dict[str, dict[str, Any]],
    request_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Join QC and true/null evaluator rows under one registered endpoint."""

    result = []
    for request_id in sorted(set(request_ids)):
        if request_id not in qc_rows:
            raise ValueError(f"QC endpoint metrics missing request_id={request_id}")
        if request_id not in response_rows:
            raise ValueError(f"history response missing request_id={request_id}")
        qc_row = qc_rows[request_id]
        response_row = response_rows[request_id]
        qc = float(qc_row["ndcg@10"])
        null = float(response_row["null_ndcg@10"])
        true = float(response_row["true_ndcg@10"])
        null_path_gap = null - qc
        recovery = true - null
        end_model_gap = true - qc
        if not math.isclose(
            null_path_gap + recovery,
            end_model_gap,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"endpoint accounting identity failed for {request_id}")
        qc_positive = bool(qc_row.get("positive_eligible"))
        response_positive = bool(response_row.get("positive_eligible"))
        if qc_positive != response_positive:
            raise ValueError(
                f"positive eligibility mismatch for request_id={request_id}"
            )
        result.append(
            {
                "request_id": request_id,
                "positive_eligible": response_positive,
                "direction_eligible": int(
                    response_row.get("direction_preferred_pairs", 0)
                )
                > 0,
                "qc_ndcg@10": qc,
                "null_ndcg@10": null,
                "true_ndcg@10": true,
                "null_path_gap": null_path_gap,
                "same_checkpoint_recovery": recovery,
                "end_model_gap": end_model_gap,
                "additive_gain": float(null_path_gap >= 0 and recovery > 0),
                "recovered_to_qc": float(
                    null_path_gap < 0 and recovery > 0 and end_model_gap >= 0
                ),
                "recovery_shortfall": float(
                    null_path_gap < 0 and recovery > 0 and end_model_gap < 0
                ),
                "nonpositive_recovery": float(recovery <= 0),
            }
        )
    if not result:
        raise ValueError("endpoint-aligned accounting population is empty")
    return result


def summarize_endpoint_aligned_accounting(
    rows: list[dict[str, Any]],
    cluster_by_request: dict[str, str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize descriptive accounting without assigning a causal mechanism."""

    _validate_bootstrap(bootstrap_samples)
    means = {
        metric: _mean_present(rows, metric)
        for metric in DESCRIPTIVE_ACCOUNTING_METRICS
    }
    intervals = cluster_bootstrap_mean_ci(
        rows,
        cluster_by_request,
        DESCRIPTIVE_ACCOUNTING_METRICS,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "bootstrap_ci95": intervals,
        "means": means,
        "num_direction_eligible_requests": sum(
            bool(row["direction_eligible"]) for row in rows
        ),
        "num_positive_eligible_requests": sum(
            bool(row["positive_eligible"]) for row in rows
        ),
        "num_requests": len(rows),
    }


def summarize_partition_contributions(
    *,
    all_rows: list[dict[str, Any]],
    partition_rows: dict[str, list[dict[str, Any]]],
    metric: str,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Decompose one all-request mean over an arbitrary disjoint partition."""

    all_ids = {str(row["request_id"]) for row in all_rows}
    union: set[str] = set()
    for name, rows in partition_rows.items():
        ids = {str(row["request_id"]) for row in rows}
        overlap = union & ids
        if overlap:
            raise ValueError(
                f"partition surface {name} overlaps prior surfaces: {sorted(overlap)[:5]}"
            )
        union.update(ids)
    if union != all_ids:
        raise ValueError(
            "partition does not cover all requests: "
            f"missing={sorted(all_ids - union)[:5]} extra={sorted(union - all_ids)[:5]}"
        )

    all_mean = _mean_present(all_rows, metric)
    if all_mean is None:
        raise ValueError(f"metric {metric} is absent from all rows")
    total = len(all_rows)
    surfaces = {}
    contributions = []
    for name, rows in partition_rows.items():
        mean = _mean_present(rows, metric)
        contribution = len(rows) / total * mean if mean is not None else 0.0
        contributions.append(contribution)
        surfaces[name] = {
            "contribution": contribution,
            "mean": mean,
            "num_requests": len(rows),
            "prevalence": len(rows) / total,
        }
    reconstructed = sum(contributions)
    if not math.isclose(all_mean, reconstructed, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"partition contribution identity failed: {all_mean} != {reconstructed}"
        )
    absolute_total = sum(abs(value) for value in contributions)
    for values in surfaces.values():
        values["absolute_contribution_share"] = (
            abs(values["contribution"]) / absolute_total if absolute_total else None
        )
        values["signed_share_of_all_mean"] = (
            values["contribution"] / all_mean if abs(all_mean) > tolerance else None
        )
    return {
        "all_mean": all_mean,
        "metric": metric,
        "reconstructed_mean": reconstructed,
        "surfaces": surfaces,
    }


def build_accounting_rows(
    qc_rows: dict[str, dict[str, Any]],
    null_rows: dict[str, dict[str, Any]],
    true_rows: dict[str, dict[str, Any]],
    request_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Join shared-evaluator QC/FULL metrics into exact request accounting."""

    result = []
    for request_id in sorted(set(request_ids)):
        if request_id not in qc_rows:
            raise ValueError(f"QC metrics missing request_id={request_id}")
        if request_id not in null_rows:
            raise ValueError(f"FULL-null metrics missing request_id={request_id}")
        if request_id not in true_rows:
            raise ValueError(f"FULL-true metrics missing request_id={request_id}")
        qc = float(qc_rows[request_id]["ndcg@10"])
        null = float(null_rows[request_id]["ndcg@10"])
        true = float(true_rows[request_id]["ndcg@10"])
        base_retention = null - qc
        history_utility = true - null
        net_value = true - qc
        if not math.isclose(
            base_retention + history_utility,
            net_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"accounting identity failed for {request_id}")
        result.append(
            {
                "request_id": request_id,
                "qc_ndcg@10": qc,
                "null_ndcg@10": null,
                "true_ndcg@10": true,
                "base_retention": base_retention,
                "history_utility": history_utility,
                "net_value": net_value,
                "additive_gain": float(base_retention >= 0 and history_utility > 0),
                "full_repair": float(
                    base_retention < 0 and history_utility > 0 and net_value >= 0
                ),
                "repair_shortfall": float(
                    base_retention < 0 and history_utility > 0 and net_value < 0
                ),
                "nonpositive_history": float(history_utility <= 0),
            }
        )
    if not result:
        raise ValueError("accounting population is empty")
    return result


def summarize_accounting(
    rows: list[dict[str, Any]],
    cluster_by_request: dict[str, str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize accounting means, state shares, and query-cluster CIs."""

    _validate_bootstrap(bootstrap_samples)
    means = {metric: _mean_present(rows, metric) for metric in ACCOUNTING_METRICS}
    intervals = cluster_bootstrap_mean_ci(
        rows,
        cluster_by_request,
        ACCOUNTING_METRICS,
        samples=bootstrap_samples,
        seed=seed,
    )
    positive_history = [row for row in rows if row["history_utility"] > 0]
    damaged_then_helped = [
        row
        for row in positive_history
        if row["base_retention"] < 0
    ]
    repayment = {
        "positive_history_request_rate": len(positive_history) / len(rows),
        "base_damage_rate": sum(row["base_retention"] < 0 for row in rows)
        / len(rows),
        "positive_history_on_damaged_base_rate": (
            len(damaged_then_helped) / len(positive_history)
            if positive_history
            else None
        ),
        "repair_shortfall_given_damaged_then_helped": (
            sum(row["net_value"] < 0 for row in damaged_then_helped)
            / len(damaged_then_helped)
            if damaged_then_helped
            else None
        ),
    }
    return {
        "bootstrap_ci95": intervals,
        "means": means,
        "num_requests": len(rows),
        "repayment_diagnostics": repayment,
    }


def summarize_recurrence_masking(
    *,
    all_rows: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
    strict_nonrepeat_rows: list[dict[str, Any]],
    no_history_rows: list[dict[str, Any]],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Decompose all-request history utility by disjoint request surface."""

    all_ids = {str(row["request_id"]) for row in all_rows}
    repeat_ids = {str(row["request_id"]) for row in repeat_rows}
    strict_ids = {str(row["request_id"]) for row in strict_nonrepeat_rows}
    no_history_ids = {str(row["request_id"]) for row in no_history_rows}
    if repeat_ids & strict_ids or repeat_ids & no_history_ids or strict_ids & no_history_ids:
        raise ValueError("repeat, strict-nonrepeat, and no-history surfaces overlap")
    if repeat_ids | strict_ids | no_history_ids != all_ids:
        raise ValueError("request surfaces do not partition the all population")

    all_utility = _mean_present(all_rows, "history_utility")
    repeat_utility = _mean_present(repeat_rows, "history_utility")
    strict_utility = _mean_present(strict_nonrepeat_rows, "history_utility")
    no_history_utility = _mean_present(no_history_rows, "history_utility")
    total = len(all_rows)
    repeat_contribution = (
        len(repeat_rows) / total * repeat_utility if repeat_utility is not None else 0.0
    )
    strict_contribution = (
        len(strict_nonrepeat_rows) / total * strict_utility
        if strict_utility is not None
        else 0.0
    )
    no_history_contribution = (
        len(no_history_rows) / total * no_history_utility
        if no_history_utility is not None
        else 0.0
    )
    reconstructed = repeat_contribution + strict_contribution + no_history_contribution
    if not math.isclose(all_utility, reconstructed, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"surface utility decomposition failed: {all_utility} != {reconstructed}"
        )
    nonzero_absolute = abs(repeat_contribution) + abs(strict_contribution)
    return {
        "all_history_utility": all_utility,
        "no_history": {
            "contribution": no_history_contribution,
            "mean_history_utility": no_history_utility,
            "num_requests": len(no_history_rows),
            "prevalence": len(no_history_rows) / total,
        },
        "reconstructed_history_utility": reconstructed,
        "repeat": {
            "absolute_contribution_share": (
                abs(repeat_contribution) / nonzero_absolute
                if nonzero_absolute
                else None
            ),
            "contribution": repeat_contribution,
            "mean_history_utility": repeat_utility,
            "num_requests": len(repeat_rows),
            "prevalence": len(repeat_rows) / total,
        },
        "strict_nonrepeat": {
            "absolute_contribution_share": (
                abs(strict_contribution) / nonzero_absolute
                if nonzero_absolute
                else None
            ),
            "contribution": strict_contribution,
            "mean_history_utility": strict_utility,
            "num_requests": len(strict_nonrepeat_rows),
            "prevalence": len(strict_nonrepeat_rows) / total,
        },
    }


def build_response_curve(
    response_rows: dict[str, dict[str, Any]],
    request_ids: Iterable[str],
    cluster_by_request: dict[str, str],
    *,
    intervention_rows: dict[str, dict[str, Any]] | None,
    bins: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Build equal-count response-magnitude bins and outcome associations."""

    if bins < 2:
        raise ValueError("response curve needs at least two bins")
    _validate_bootstrap(bootstrap_samples)
    joined = []
    for request_id in sorted(set(request_ids)):
        if request_id not in response_rows:
            raise ValueError(f"response metrics missing request_id={request_id}")
        source = response_rows[request_id]
        response = source.get("normalized_response_rms")
        if response is None or not math.isfinite(float(response)):
            continue
        row = {
            "request_id": request_id,
            "normalized_response_rms": float(response),
            "pairwise_directional_accuracy": _optional_float(
                source.get("pairwise_directional_accuracy")
            ),
            "signed_delta_alignment": _optional_float(
                source.get("signed_delta_alignment")
            ),
            "true_minus_null_ndcg@10": float(source["true_minus_null_ndcg@10"]),
        }
        if intervention_rows is not None:
            if request_id not in intervention_rows:
                raise ValueError(
                    f"intervention metrics missing request_id={request_id}"
                )
            row["actual_minus_random_ndcg@10"] = float(
                intervention_rows[request_id]["actual_minus_random_ndcg@10"]
            )
        joined.append(row)
    if len(joined) < bins:
        raise ValueError("not enough finite response rows for requested bins")

    outcome_metrics = [
        "pairwise_directional_accuracy",
        "signed_delta_alignment",
        "true_minus_null_ndcg@10",
    ]
    if intervention_rows is not None:
        outcome_metrics.append("actual_minus_random_ndcg@10")

    ordered = sorted(
        joined,
        key=lambda row: (row["normalized_response_rms"], row["request_id"]),
    )
    curve = []
    for index in range(bins):
        start = index * len(ordered) // bins
        end = (index + 1) * len(ordered) // bins
        selected = ordered[start:end]
        metric_names = ("normalized_response_rms", *outcome_metrics)
        curve.append(
            {
                "bin": index + 1,
                "bootstrap_ci95": cluster_bootstrap_mean_ci(
                    selected,
                    cluster_by_request,
                    metric_names,
                    samples=bootstrap_samples,
                    seed=seed + index,
                ),
                "max_response": selected[-1]["normalized_response_rms"],
                "means": {
                    metric: _mean_present(selected, metric)
                    for metric in metric_names
                },
                "min_response": selected[0]["normalized_response_rms"],
                "num_requests": len(selected),
            }
        )

    associations = {
        metric: {
            "num_complete_requests": sum(row.get(metric) is not None for row in joined),
            "spearman": spearman_correlation(
                [
                    (row["normalized_response_rms"], row[metric])
                    for row in joined
                    if row.get(metric) is not None
                ]
            ),
        }
        for metric in outcome_metrics
    }
    return {
        "associations": associations,
        "bins": curve,
        "num_requests": len(joined),
        "response_metric": "normalized_response_rms",
    }


def cluster_bootstrap_mean_ci(
    rows: list[dict[str, Any]],
    cluster_by_request: dict[str, str],
    metrics: Iterable[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float] | None]:
    """Query-cluster bootstrap confidence intervals for request-level means."""

    _validate_bootstrap(samples)
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        request_id = str(row["request_id"])
        if request_id not in cluster_by_request:
            raise ValueError(f"cluster key missing request_id={request_id}")
        clusters[cluster_by_request[request_id]].append(row)
    keys = sorted(clusters)
    if not keys:
        raise ValueError("cannot bootstrap an empty population")
    metric_names = tuple(metrics)
    draws: dict[str, list[float]] = {metric: [] for metric in metric_names}
    rng = random.Random(seed)
    for _ in range(samples):
        selected = [
            row
            for _index in range(len(keys))
            for row in clusters[keys[rng.randrange(len(keys))]]
        ]
        for metric in metric_names:
            value = _mean_present(selected, metric)
            if value is not None:
                draws[metric].append(value)
    return {metric: _percentile_ci(values) for metric, values in draws.items()}


def cluster_bootstrap_group_mean_contrast(
    rows: list[dict[str, Any]],
    cluster_by_request: dict[str, str],
    *,
    group_field: str,
    value_field: str,
    left_group: str,
    right_group: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap ``mean(left) - mean(right)`` over request clusters.

    The two group means are recomputed inside every cluster-bootstrap draw;
    this is intentionally different from averaging signed rows, which would
    weight the contrast by the relative group sizes.
    """

    _validate_bootstrap(samples)
    if left_group == right_group:
        raise ValueError("contrast groups must be different")
    selected_rows = [
        row for row in rows if row.get(group_field) in (left_group, right_group)
    ]
    if not selected_rows:
        raise ValueError("group contrast population is empty")

    by_group = {
        group: [row for row in selected_rows if row[group_field] == group]
        for group in (left_group, right_group)
    }
    for group, group_rows in by_group.items():
        if not group_rows:
            raise ValueError(f"group contrast has no rows for {group!r}")
        if any(row.get(value_field) is None for row in group_rows):
            raise ValueError(
                f"group contrast metric {value_field!r} is missing for {group!r}"
            )

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        request_id = str(row["request_id"])
        if request_id not in cluster_by_request:
            raise ValueError(f"cluster key missing request_id={request_id}")
        clusters[cluster_by_request[request_id]].append(row)
    keys = sorted(clusters)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        draw_rows = [
            row
            for _index in range(len(keys))
            for row in clusters[keys[rng.randrange(len(keys))]]
        ]
        left = [
            float(row[value_field])
            for row in draw_rows
            if row[group_field] == left_group
        ]
        right = [
            float(row[value_field])
            for row in draw_rows
            if row[group_field] == right_group
        ]
        if left and right:
            draws.append(sum(left) / len(left) - sum(right) / len(right))
    if not draws:
        raise ValueError("no bootstrap draw contained both contrast groups")

    left_values = [float(row[value_field]) for row in by_group[left_group]]
    right_values = [float(row[value_field]) for row in by_group[right_group]]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    return {
        "bootstrap_ci95": _percentile_ci(draws),
        "bootstrap_samples": samples,
        "left": {
            "group": left_group,
            "mean": left_mean,
            "num_requests": len(left_values),
        },
        "num_clusters": len(keys),
        "point_estimate": left_mean - right_mean,
        "right": {
            "group": right_group,
            "mean": right_mean,
            "num_requests": len(right_values),
        },
        "valid_bootstrap_samples": len(draws),
    }


def spearman_correlation(pairs: list[tuple[float, float]]) -> float | None:
    """Spearman rank correlation with average ranks for ties."""

    if len(pairs) < 2:
        return None
    left = [float(pair[0]) for pair in pairs]
    right = [float(pair[1]) for pair in pairs]
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks)
    )
    left_energy = sum((value - left_mean) ** 2 for value in left_ranks)
    right_energy = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator else None


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _mean_present(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return sum(values) / len(values) if values else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _percentile_ci(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values.sort()
    return [
        values[int(0.025 * len(values))],
        values[min(len(values) - 1, int(0.975 * len(values)))],
    ]


def _validate_bootstrap(samples: int) -> None:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
