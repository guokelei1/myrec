"""Exact request-common and candidate-relative native-score decomposition."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def decompose_condition_scores(
    scores: Mapping[str, Mapping[str, float]],
    request_ids: Sequence[str],
    candidates: Mapping[str, Sequence[str]],
) -> dict[str, np.ndarray | float]:
    """Decompose each request slate as score = common + relative exactly."""

    common = np.empty(len(request_ids), dtype=np.float64)
    relative_rms = np.empty(len(request_ids), dtype=np.float64)
    relative_by_request: dict[str, np.ndarray] = {}
    maximum_recomposition = 0.0
    maximum_relative_sum = 0.0
    for ordinal, request_id in enumerate(request_ids):
        item_ids = [str(item_id) for item_id in candidates[request_id]]
        values = np.asarray(
            [float(scores[request_id][item_id]) for item_id in item_ids],
            dtype=np.float64,
        )
        if not values.size or not np.isfinite(values).all():
            raise ValueError("readout decomposition requires finite nonempty slates")
        center = float(values.mean())
        relative = values - center
        common[ordinal] = center
        relative_rms[ordinal] = float(np.sqrt(np.mean(relative**2)))
        relative_by_request[request_id] = relative
        maximum_recomposition = max(
            maximum_recomposition,
            float(np.max(np.abs(values - (center + relative)))),
        )
        maximum_relative_sum = max(
            maximum_relative_sum, float(abs(relative.sum()))
        )
    return {
        "common": common,
        "relative_rms": relative_rms,
        "relative_by_request": relative_by_request,
        "maximum_recomposition_abs_error": maximum_recomposition,
        "maximum_relative_sum_abs_error": maximum_relative_sum,
    }


def summarize_decomposition_contrast(
    left: Mapping[str, np.ndarray | Mapping[str, np.ndarray] | float],
    right: Mapping[str, np.ndarray | Mapping[str, np.ndarray] | float],
    request_ids: Sequence[str],
) -> dict[str, object]:
    """Summarize common shift and relative-shape shift without qrels."""

    common_shift = np.asarray(left["common"], dtype=np.float64) - np.asarray(
        right["common"], dtype=np.float64
    )
    left_relative = left["relative_by_request"]
    right_relative = right["relative_by_request"]
    if not isinstance(left_relative, Mapping) or not isinstance(right_relative, Mapping):
        raise ValueError("readout relative decomposition mapping is missing")
    relative_shift_rms = np.asarray(
        [
            np.sqrt(
                np.mean(
                    (
                        np.asarray(left_relative[request_id], dtype=np.float64)
                        - np.asarray(right_relative[request_id], dtype=np.float64)
                    )
                    ** 2
                )
            )
            for request_id in request_ids
        ],
        dtype=np.float64,
    )
    return {
        "common_shift": _summary(common_shift),
        "candidate_relative_shift_rms": _summary(relative_shift_rms),
    }


def condition_decomposition_report(
    condition_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    request_ids: Sequence[str],
    candidates: Mapping[str, Sequence[str]],
    contrasts: Mapping[str, tuple[str, str]],
) -> dict[str, object]:
    """Build a complete qrels-blind decomposition report for fixed contrasts."""

    decompositions = {
        name: decompose_condition_scores(scores, request_ids, candidates)
        for name, scores in condition_scores.items()
    }
    return {
        "algebra": {
            "score_identity": "score_ij = common_i + relative_ij",
            "common_definition": "mean_j(score_ij)",
            "relative_definition": "score_ij - common_i",
            "maximum_recomposition_abs_error": max(
                float(value["maximum_recomposition_abs_error"])
                for value in decompositions.values()
            ),
            "maximum_relative_sum_abs_error": max(
                float(value["maximum_relative_sum_abs_error"])
                for value in decompositions.values()
            ),
        },
        "condition_summaries": {
            name: {
                "common": _summary(np.asarray(value["common"], dtype=np.float64)),
                "candidate_relative_rms": _summary(
                    np.asarray(value["relative_rms"], dtype=np.float64)
                ),
            }
            for name, value in decompositions.items()
        },
        "contrast_summaries": {
            name: summarize_decomposition_contrast(
                decompositions[left], decompositions[right], request_ids
            )
            for name, (left, right) in contrasts.items()
        },
        "qrels_read": False,
        "confirmatory_family_membership": False,
    }


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("readout decomposition summary is invalid")
    return {
        "requests": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "mean_absolute": float(np.abs(values).mean()),
    }
