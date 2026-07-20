"""Qrels-blind cross-cell synthesis of completed D3 head observations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


MODELS = (
    "q2_recranker_generalqwen",
    "q3_tallrec_generalqwen",
)
BLOCKS = (13, 20, 27)
AXIS_SIZES = {"gqa_group": 8, "query_head": 16}
METRICS = ("attention_mass", "o_proj_contribution_norm")


def concentration_summary(values: Sequence[float], *, top_k: int = 3) -> dict[str, Any]:
    """Return scale-free concentration for one nonnegative complete vector."""

    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or not array.size
        or not np.isfinite(array).all()
        or np.any(array < 0)
        or float(array.sum()) <= 0
    ):
        raise ValueError("attention concentration vector is invalid")
    top_k = int(top_k)
    if not 0 < top_k <= array.size:
        raise ValueError("attention concentration top-k is invalid")
    shares = array / array.sum()
    order = sorted(range(array.size), key=lambda index: (-shares[index], index))
    positive = shares[shares > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "values": array.tolist(),
        "shares": shares.tolist(),
        "effective_count_simpson": float(1.0 / np.square(shares).sum()),
        "effective_count_entropy": float(math.exp(entropy)),
        "top_indices": order[:top_k],
        "top_share": float(shares[order[0]]),
        "top_k_share": float(shares[order[:top_k]].sum()),
    }


def summarize_attention_patterns(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize all fixed model/block cells without labels or selection."""

    if (
        metrics.get("analysis_type")
        != "transformer_deep_dive_d3_attention_head_observation"
        or metrics.get("status") != "completed"
        or metrics.get("descriptive_only") is not True
        or metrics.get("qrels_read") is not False
        or metrics.get("source_test_opened") is not False
        or tuple(metrics.get("blocks", ())) != BLOCKS
        or set(metrics.get("results", {})) != set(MODELS)
    ):
        raise ValueError("completed D3 attention-head metrics boundary differs")

    cells = []
    for model_id in MODELS:
        model_results = metrics["results"][model_id]
        if set(model_results) != {str(block) for block in BLOCKS}:
            raise ValueError("D3 attention-head block coverage differs")
        expected_paths = ("prompt",) if model_id == MODELS[0] else ("yes", "no")
        for block in BLOCKS:
            cell = model_results[str(block)]
            if set(cell.get("paths", {})) != set(expected_paths):
                raise ValueError("D3 native path coverage differs")
            axes = {}
            for axis, size in AXIS_SIZES.items():
                vectors = {
                    metric: _mean_path_vector(
                        cell["paths"], expected_paths, metric=metric, axis=axis, size=size
                    )
                    for metric in METRICS
                }
                mass = concentration_summary(vectors["attention_mass"])
                contribution = concentration_summary(
                    vectors["o_proj_contribution_norm"]
                )
                axes[axis] = {
                    "history_attention_mass": mass,
                    "history_o_proj_contribution_norm": contribution,
                    "mass_contribution_pearson": _pearson(
                        vectors["attention_mass"],
                        vectors["o_proj_contribution_norm"],
                    ),
                    "same_top_index": (
                        mass["top_indices"][0] == contribution["top_indices"][0]
                    ),
                    "top3_index_overlap": sorted(
                        set(mass["top_indices"]) & set(contribution["top_indices"])
                    ),
                }
            cells.append(
                {
                    "method_id": model_id,
                    "block_zero_based": block,
                    "paths_averaged": list(expected_paths),
                    "axes": axes,
                    "maximum_manual_attention_low_precision_ratio": float(
                        cell["maximum_manual_attention_low_precision_ratio"]
                    ),
                    "maximum_score_identity_delta": float(
                        cell["maximum_score_identity_delta"]
                    ),
                }
            )
    return {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d3_attention_pattern_synthesis",
        "status": "completed",
        "descriptive_only": True,
        "qrels_read": False,
        "source_test_opened": False,
        "models": list(MODELS),
        "blocks": list(BLOCKS),
        "cells": cells,
        "stability": _stability(cells),
        "interpretation_boundary": (
            "Fixed-cell concentration and cross-cell stability of descriptive native "
            "history attention mass/o-proj contribution only; no head, group, block, "
            "architecture, or causal mechanism is selected from this synthesis."
        ),
    }


def _mean_path_vector(
    paths: Mapping[str, Any],
    path_names: Sequence[str],
    *,
    metric: str,
    axis: str,
    size: int,
) -> list[float]:
    values = []
    for path_name in path_names:
        try:
            vector = paths[path_name]["observations"]["native_readout"]["history"][
                metric
            ][axis]["mean"]
        except (KeyError, TypeError) as exc:
            raise ValueError("D3 attention observation schema differs") from exc
        array = np.asarray(vector, dtype=np.float64)
        if array.shape != (size,) or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("D3 attention observation vector differs")
        values.append(array)
    return np.mean(np.stack(values), axis=0).tolist()


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("attention correlation vectors differ")
    if float(left_array.std()) == 0 or float(right_array.std()) == 0:
        return None
    value = float(np.corrcoef(left_array, right_array)[0, 1])
    if not math.isfinite(value):
        raise ValueError("attention correlation is non-finite")
    return max(-1.0, min(1.0, value))


def _stability(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_axis = {}
    for axis in AXIS_SIZES:
        per_model = {}
        for model_id in MODELS:
            selected = [row for row in cells if row["method_id"] == model_id]
            mass_tops = [
                row["axes"][axis]["history_attention_mass"]["top_indices"][0]
                for row in selected
            ]
            contribution_tops = [
                row["axes"][axis]["history_o_proj_contribution_norm"][
                    "top_indices"
                ][0]
                for row in selected
            ]
            per_model[model_id] = {
                "history_mass_top_by_block": dict(zip(map(str, BLOCKS), mass_tops)),
                "history_contribution_top_by_block": dict(
                    zip(map(str, BLOCKS), contribution_tops)
                ),
                "history_mass_unique_top_count": len(set(mass_tops)),
                "history_contribution_unique_top_count": len(
                    set(contribution_tops)
                ),
            }
        cross_model = {}
        for block in BLOCKS:
            by_model = {
                row["method_id"]: row
                for row in cells
                if row["block_zero_based"] == block
            }
            left = by_model[MODELS[0]]["axes"][axis]
            right = by_model[MODELS[1]]["axes"][axis]
            left_mass = set(left["history_attention_mass"]["top_indices"])
            right_mass = set(right["history_attention_mass"]["top_indices"])
            left_contribution = set(
                left["history_o_proj_contribution_norm"]["top_indices"]
            )
            right_contribution = set(
                right["history_o_proj_contribution_norm"]["top_indices"]
            )
            cross_model[str(block)] = {
                "same_history_mass_top1": (
                    left["history_attention_mass"]["top_indices"][0]
                    == right["history_attention_mass"]["top_indices"][0]
                ),
                "history_mass_top3_jaccard": _jaccard(left_mass, right_mass),
                "same_history_contribution_top1": (
                    left["history_o_proj_contribution_norm"]["top_indices"][0]
                    == right["history_o_proj_contribution_norm"]["top_indices"][0]
                ),
                "history_contribution_top3_jaccard": _jaccard(
                    left_contribution, right_contribution
                ),
            }
        per_axis[axis] = {"per_model": per_model, "cross_model": cross_model}
    return per_axis


def _jaccard(left: set[int], right: set[int]) -> float:
    return float(len(left & right) / len(left | right))
