"""Label-oracle interventions on a fixed history-response score budget."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

from myrec.eval.metrics import ScoredCandidate, sort_candidates


@dataclass(frozen=True)
class DirectionInterventionCandidate:
    item_id: str
    true_score: float
    null_score: float
    gain: float


def request_direction_intervention(
    request_id: str,
    candidates: Iterable[DirectionInterventionCandidate],
    *,
    random_permutations: int = 128,
    seed: int = 20260714,
    k: int = 10,
) -> dict[str, Any]:
    """Reassign the observed delta multiset while holding its values fixed.

    The gain-aligned condition is a label-oracle diagnostic, not a deployable
    ranking method. It preserves the exact per-request history-response values
    and changes only which candidate receives each value.
    """

    if random_permutations <= 0:
        raise ValueError("random_permutations must be positive")
    rows = list(candidates)
    if len(rows) < 2:
        raise ValueError("direction intervention requires at least two candidates")
    item_ids = [row.item_id for row in rows]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("candidate item_id values must be unique")
    for row in rows:
        if row.gain < 0:
            raise ValueError("candidate gain must be non-negative")
        if not all(
            math.isfinite(value)
            for value in (row.true_score, row.null_score, row.gain)
        ):
            raise ValueError(f"non-finite candidate value for item_id={row.item_id}")

    null_scores = [row.null_score for row in rows]
    true_scores = [row.true_score for row in rows]
    gains = [row.gain for row in rows]
    delta = [true - null for true, null in zip(true_scores, null_scores)]
    aligned_delta = _gain_ordered_delta(
        delta,
        gains,
        null_scores,
        item_ids,
        reverse=False,
    )
    anti_aligned_delta = _gain_ordered_delta(
        delta,
        gains,
        null_scores,
        item_ids,
        reverse=True,
    )
    aligned_scores = [base + value for base, value in zip(null_scores, aligned_delta)]
    anti_aligned_scores = [
        base + value for base, value in zip(null_scores, anti_aligned_delta)
    ]

    stable_seed = seed + int(
        hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16], 16
    )
    rng = random.Random(stable_seed)
    random_ndcg = []
    for _ in range(random_permutations):
        permuted = list(delta)
        rng.shuffle(permuted)
        random_scores = [base + value for base, value in zip(null_scores, permuted)]
        random_ndcg.append(_ndcg(request_id, item_ids, random_scores, gains, k))

    null_ndcg = _ndcg(request_id, item_ids, null_scores, gains, k)
    actual_ndcg = _ndcg(request_id, item_ids, true_scores, gains, k)
    aligned_ndcg = _ndcg(request_id, item_ids, aligned_scores, gains, k)
    anti_aligned_ndcg = _ndcg(
        request_id,
        item_ids,
        anti_aligned_scores,
        gains,
        k,
    )
    random_mean = sum(random_ndcg) / len(random_ndcg)
    distinct_gain_pairs = sum(
        gains[left] != gains[right]
        for left in range(len(gains))
        for right in range(left + 1, len(gains))
    )
    return {
        "actual_minus_null_ndcg@10": actual_ndcg - null_ndcg,
        "actual_minus_random_ndcg@10": actual_ndcg - random_mean,
        "actual_ndcg@10": actual_ndcg,
        "aligned_minus_actual_ndcg@10": aligned_ndcg - actual_ndcg,
        "aligned_minus_null_ndcg@10": aligned_ndcg - null_ndcg,
        "aligned_ndcg@10": aligned_ndcg,
        "anti_aligned_ndcg@10": anti_aligned_ndcg,
        "candidate_count": len(rows),
        "direction_eligible": distinct_gain_pairs > 0,
        "distinct_gain_pairs": distinct_gain_pairs,
        "gain_levels": len(set(gains)),
        "null_ndcg@10": null_ndcg,
        "random_mean_ndcg@10": random_mean,
        "random_permutations": random_permutations,
        "request_id": request_id,
        "response_delta_energy": sum(value * value for value in delta),
        "response_delta_multiset_sha256": _float_multiset_sha256(delta),
    }


def aggregate_direction_interventions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty direction interventions")
    eligible = [row for row in rows if row["direction_eligible"]]
    result: dict[str, Any] = {
        "direction_eligible_rate": len(eligible) / len(rows),
        "num_direction_eligible_requests": len(eligible),
        "num_requests": len(rows),
    }
    metric_names = (
        "actual_minus_null_ndcg@10",
        "actual_minus_random_ndcg@10",
        "actual_ndcg@10",
        "aligned_minus_actual_ndcg@10",
        "aligned_minus_null_ndcg@10",
        "aligned_ndcg@10",
        "anti_aligned_ndcg@10",
        "null_ndcg@10",
        "random_mean_ndcg@10",
    )
    for name in metric_names:
        result[f"mean_{name}"] = _mean(rows, name)
        result[f"eligible_mean_{name}"] = _mean(eligible, name)
    denominator = (
        result["mean_aligned_ndcg@10"] - result["mean_random_mean_ndcg@10"]
    )
    result["direction_conversion_efficiency"] = (
        (result["mean_actual_ndcg@10"] - result["mean_random_mean_ndcg@10"])
        / denominator
        if abs(denominator) > 1e-12
        else None
    )
    eligible_denominator = (
        result["eligible_mean_aligned_ndcg@10"]
        - result["eligible_mean_random_mean_ndcg@10"]
    )
    result["eligible_direction_conversion_efficiency"] = (
        (
            result["eligible_mean_actual_ndcg@10"]
            - result["eligible_mean_random_mean_ndcg@10"]
        )
        / eligible_denominator
        if eligible and abs(eligible_denominator) > 1e-12
        else None
    )
    return result


def _gain_ordered_delta(
    delta: list[float],
    gains: list[float],
    null_scores: list[float],
    item_ids: list[str],
    *,
    reverse: bool,
) -> list[float]:
    """Assign the unchanged delta multiset monotonically by observed gain."""

    ordered_values = sorted(delta)
    if reverse:
        candidate_order = sorted(
            range(len(delta)),
            key=lambda index: (-gains[index], -null_scores[index], item_ids[index]),
        )
    else:
        candidate_order = sorted(
            range(len(delta)),
            key=lambda index: (gains[index], -null_scores[index], item_ids[index]),
        )
    assigned = [0.0] * len(delta)
    for index, value in zip(candidate_order, ordered_values):
        assigned[index] = value
    return assigned


def _ndcg(
    request_id: str,
    item_ids: list[str],
    scores: list[float],
    gains: list[float],
    k: int,
) -> float:
    gain_by_item = dict(zip(item_ids, gains))
    ranked = sort_candidates(
        request_id,
        [
            ScoredCandidate(item_id=item_id, score=score)
            for item_id, score in zip(item_ids, scores)
        ],
    )
    ranked_gains = [gain_by_item[row.item_id] for row in ranked[:k]]
    ideal_gains = sorted(gains, reverse=True)[:k]
    dcg = sum(
        (2.0**gain - 1.0) / math.log2(rank + 1)
        for rank, gain in enumerate(ranked_gains, 1)
    )
    idcg = sum(
        (2.0**gain - 1.0) / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, 1)
    )
    return dcg / idcg if idcg else 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(float(row[key]) for row in rows) / len(rows)


def _float_multiset_sha256(values: list[float]) -> str:
    payload = "|".join(format(value, ".17g") for value in sorted(values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
