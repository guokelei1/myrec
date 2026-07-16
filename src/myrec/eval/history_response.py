"""Candidate-relative history-response metrics for shared evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from myrec.eval.metrics import ScoredCandidate, sort_candidates


@dataclass(frozen=True)
class ResponseCandidate:
    item_id: str
    true_score: float
    null_score: float
    gain: float
    wrong_score: float | None = None


def request_history_response(
    request_id: str,
    candidates: Iterable[ResponseCandidate],
    activity_epsilon: float,
    k: int = 10,
) -> dict[str, Any]:
    """Measure response activity, direction and utility for one fixed slate."""

    if activity_epsilon < 0:
        raise ValueError("activity_epsilon must be non-negative")
    rows = list(candidates)
    if len(rows) < 2:
        raise ValueError("history-response metrics require at least two candidates")
    item_ids = [row.item_id for row in rows]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("candidate item_id values must be unique")
    for row in rows:
        values = (row.true_score, row.null_score, row.gain)
        if row.wrong_score is not None:
            values += (row.wrong_score,)
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"non-finite response value for item_id={row.item_id}")
        if row.gain < 0:
            raise ValueError("candidate gain must be non-negative")

    true_scores = [row.true_score for row in rows]
    null_scores = [row.null_score for row in rows]
    gains = [row.gain for row in rows]
    delta = [true - null for true, null in zip(true_scores, null_scores)]
    response = _response_components(delta, activity_epsilon)
    direction = _direction_components(delta, gains, activity_epsilon)
    true_ndcg = gain_ndcg_at_k(request_id, item_ids, true_scores, gains, k)
    null_ndcg = gain_ndcg_at_k(request_id, item_ids, null_scores, gains, k)
    null_centered_rms = _centered_rms(null_scores)

    result: dict[str, Any] = {
        "active_pair_fraction": response["active_pair_fraction"],
        "active_pairs": response["active_pairs"],
        "candidate_count": len(rows),
        "common_energy": response["common_energy"],
        "common_mode_ratio": response["common_mode_ratio"],
        "differential_energy": response["differential_energy"],
        "direction_active_pairs": direction["active_pairs"],
        "direction_correct_pairs": direction["correct_pairs"],
        "direction_pair_coverage": direction["active_pair_fraction"],
        "direction_preferred_pairs": direction["preferred_pairs"],
        "direction_wrong_pairs": direction["wrong_pairs"],
        "null_ndcg@10": null_ndcg,
        "normalized_response_rms": (
            response["differential_rms"] / null_centered_rms
            if null_centered_rms > activity_epsilon
            else None
        ),
        "pair_count": response["pair_count"],
        "pairwise_directional_accuracy": direction["directional_accuracy"],
        "positive_eligible": any(gain > 0 for gain in gains),
        "request_id": request_id,
        "response_active": response["active_pairs"] > 0,
        "response_common_mean": response["common_mean"],
        "response_differential_rms": response["differential_rms"],
        "signed_delta_alignment": direction["signed_alignment"],
        "true_minus_null_ndcg@10": true_ndcg - null_ndcg,
        "true_ndcg@10": true_ndcg,
    }

    wrong_scores = [row.wrong_score for row in rows]
    if any(score is not None for score in wrong_scores):
        if not all(score is not None for score in wrong_scores):
            raise ValueError("wrong_score must be present for every candidate or none")
        concrete_wrong = [float(score) for score in wrong_scores if score is not None]
        wrong_delta = [wrong - null for wrong, null in zip(concrete_wrong, null_scores)]
        wrong_direction = _direction_components(wrong_delta, gains, activity_epsilon)
        wrong_ndcg = gain_ndcg_at_k(
            request_id, item_ids, concrete_wrong, gains, k
        )
        result.update(
            {
                "true_minus_wrong_ndcg@10": true_ndcg - wrong_ndcg,
                "true_over_wrong_signed_alignment": _optional_difference(
                    direction["signed_alignment"], wrong_direction["signed_alignment"]
                ),
                "wrong_ndcg@10": wrong_ndcg,
                "wrong_signed_delta_alignment": wrong_direction["signed_alignment"],
            }
        )
    return result


def aggregate_history_response(
    rows: list[dict[str, Any]], utility_epsilon: float
) -> dict[str, Any]:
    """Aggregate per-request response metrics without pooling dependent pairs."""

    if utility_epsilon < 0:
        raise ValueError("utility_epsilon must be non-negative")
    if not rows:
        raise ValueError("cannot aggregate empty history-response rows")
    active = [row for row in rows if row["response_active"]]
    # Older frozen evaluator rows predate the explicit eligibility field.  For
    # those artifacts, distinct-gain pair availability is the narrowest safe
    # compatibility proxy; newly repaired rows always carry the exact field.
    positive_eligible = [
        row
        for row in rows
        if bool(
            row.get(
                "positive_eligible",
                int(row.get("direction_preferred_pairs", 0)) > 0,
            )
        )
    ]
    direction_eligible = [row for row in rows if row["direction_preferred_pairs"] > 0]
    active_direction = [row for row in direction_eligible if row["response_active"]]
    common_energy = sum(float(row["common_energy"]) for row in rows)
    differential_energy = sum(float(row["differential_energy"]) for row in rows)
    total_energy = common_energy + differential_energy

    result = {
        "active_response_harm_rate": (
            sum(
                row["true_minus_null_ndcg@10"] < -utility_epsilon
                for row in active_direction
            )
            / len(active_direction)
            if active_direction
            else None
        ),
        "active_response_precision": (
            sum(
                row["true_minus_null_ndcg@10"] > utility_epsilon
                for row in active_direction
            )
            / len(active_direction)
            if active_direction
            else None
        ),
        "active_response_rate": len(active) / len(rows),
        "common_mode_ratio": common_energy / total_energy if total_energy else None,
        "direction_eligible_rate": len(direction_eligible) / len(rows),
        "mean_active_pair_fraction": _mean_present(rows, "active_pair_fraction"),
        "mean_direction_pair_coverage": _mean_present(rows, "direction_pair_coverage"),
        "mean_pairwise_directional_accuracy": _mean_present(
            rows, "pairwise_directional_accuracy"
        ),
        "mean_signed_delta_alignment": _mean_present(rows, "signed_delta_alignment"),
        "mean_true_minus_null_ndcg@10": _mean_present(rows, "true_minus_null_ndcg@10"),
        "mean_true_minus_null_ndcg@10_positive": _mean_present(
            positive_eligible, "true_minus_null_ndcg@10"
        ),
        "mean_true_ndcg@10": _mean_present(rows, "true_ndcg@10"),
        "mean_true_ndcg@10_positive": _mean_present(
            positive_eligible, "true_ndcg@10"
        ),
        "num_active_requests": len(active),
        "num_active_direction_requests": len(active_direction),
        "num_direction_eligible_requests": len(direction_eligible),
        "num_positive_eligible_requests": len(positive_eligible),
        "num_requests": len(rows),
        "positive_eligible_rate": len(positive_eligible) / len(rows),
    }
    if any("true_minus_wrong_ndcg@10" in row for row in rows):
        if not all("true_minus_wrong_ndcg@10" in row for row in rows):
            raise ValueError("wrong-history metrics must cover every request or none")
        result.update(
            {
                "mean_true_minus_wrong_ndcg@10": _mean_present(
                    rows, "true_minus_wrong_ndcg@10"
                ),
                "mean_true_over_wrong_signed_alignment": _mean_present(
                    rows, "true_over_wrong_signed_alignment"
                ),
            }
        )
    return result


def _response_components(delta: list[float], epsilon: float) -> dict[str, Any]:
    common_mean = sum(delta) / len(delta)
    residual = [value - common_mean for value in delta]
    common_energy = len(delta) * common_mean * common_mean
    differential_energy = sum(value * value for value in residual)
    total_energy = common_energy + differential_energy
    pair_margins = [
        delta[left] - delta[right]
        for left in range(len(delta))
        for right in range(left + 1, len(delta))
    ]
    active_pairs = sum(abs(value) > epsilon for value in pair_margins)
    return {
        "active_pair_fraction": active_pairs / len(pair_margins),
        "active_pairs": active_pairs,
        "common_energy": common_energy,
        "common_mean": common_mean,
        "common_mode_ratio": common_energy / total_energy if total_energy else None,
        "differential_energy": differential_energy,
        "differential_rms": math.sqrt(differential_energy / len(delta)),
        "pair_count": len(pair_margins),
    }


def _direction_components(
    delta: list[float], gains: list[float], epsilon: float
) -> dict[str, Any]:
    preferred_pairs = 0
    active_pairs = 0
    correct_pairs = 0
    wrong_pairs = 0
    weighted_sign = 0.0
    total_weight = 0.0
    for left in range(len(delta)):
        for right in range(left + 1, len(delta)):
            if gains[left] == gains[right]:
                continue
            if gains[left] > gains[right]:
                high, low = left, right
            else:
                high, low = right, left
            preferred_pairs += 1
            weight = gains[high] - gains[low]
            total_weight += weight
            margin = delta[high] - delta[low]
            if margin > epsilon:
                active_pairs += 1
                correct_pairs += 1
                weighted_sign += weight
            elif margin < -epsilon:
                active_pairs += 1
                wrong_pairs += 1
                weighted_sign -= weight
    return {
        "active_pair_fraction": active_pairs / preferred_pairs if preferred_pairs else None,
        "active_pairs": active_pairs,
        "correct_pairs": correct_pairs,
        "directional_accuracy": correct_pairs / active_pairs if active_pairs else None,
        "preferred_pairs": preferred_pairs,
        "signed_alignment": weighted_sign / total_weight if total_weight else None,
        "wrong_pairs": wrong_pairs,
    }


def gain_ndcg_at_k(
    request_id: str,
    item_ids: list[str],
    scores: list[float],
    gains: list[float],
    k: int = 10,
) -> float:
    """Compute gain-aware NDCG with the repository's deterministic tie break."""

    if not (len(item_ids) == len(scores) == len(gains)):
        raise ValueError("item, score, and gain lengths differ")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("candidate item_id values must be unique")
    if any(not math.isfinite(value) for value in (*scores, *gains)):
        raise ValueError("gain-aware NDCG inputs must be finite")
    if any(gain < 0 for gain in gains):
        raise ValueError("candidate gains must be non-negative")
    gain_by_item = dict(zip(item_ids, gains))
    ranked = sort_candidates(
        request_id,
        [ScoredCandidate(item_id=item_id, score=score) for item_id, score in zip(item_ids, scores)],
    )
    ranked_gains = [gain_by_item[row.item_id] for row in ranked[:k]]
    ideal_gains = sorted(gains, reverse=True)[:k]
    dcg = _dcg(ranked_gains)
    idcg = _dcg(ideal_gains)
    return dcg / idcg if idcg else 0.0


def _dcg(gains: list[float]) -> float:
    return sum((2.0**gain - 1.0) / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))


def _centered_rms(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _optional_difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None
