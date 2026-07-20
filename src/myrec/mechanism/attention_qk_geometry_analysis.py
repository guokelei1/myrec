"""Complete qrels-blind Q/K stage geometry over the frozen D3 grid."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.attention_pattern_analysis import BLOCKS, MODELS


TENSOR_SIZES = {"q": 16, "k": 8}
STAGES = ("pre_norm", "post_norm", "post_rope")
VECTOR_FIELDS = (
    "full_norm",
    "null_norm",
    "full_null_delta_norm",
    "full_null_cosine",
)


def summarize_attention_qk_geometry(metrics: Mapping[str, Any]) -> dict[str, Any]:
    _validate_source(metrics)
    cells = []
    path_cells = []
    maximum_rope_norm_relative_error = 0.0
    for model_id in MODELS:
        path_names = ("prompt",) if model_id == MODELS[0] else ("yes", "no")
        for block in BLOCKS:
            paths = metrics["results"][model_id][str(block)]["paths"]
            if set(paths) != set(path_names):
                raise ValueError("D3 Q/K native path coverage differs")
            tensors = {}
            for tensor, size in TENSOR_SIZES.items():
                positions = _position_set(paths, path_names, tensor)
                tensors[tensor], norm_error = _summarize_tensor(
                    paths,
                    path_names,
                    tensor=tensor,
                    size=size,
                    positions=positions,
                )
                maximum_rope_norm_relative_error = max(
                    maximum_rope_norm_relative_error, norm_error
                )
            cells.append(
                {
                    "method_id": model_id,
                    "block_zero_based": block,
                    "paths_averaged": list(path_names),
                    "tensors": tensors,
                }
            )
            for path_name in path_names:
                path_tensors = {}
                for tensor, size in TENSOR_SIZES.items():
                    positions = _position_set(paths, (path_name,), tensor)
                    path_tensors[tensor], norm_error = _summarize_tensor(
                        paths,
                        (path_name,),
                        tensor=tensor,
                        size=size,
                        positions=positions,
                    )
                    maximum_rope_norm_relative_error = max(
                        maximum_rope_norm_relative_error, norm_error
                    )
                path_cells.append(
                    {
                        "method_id": model_id,
                        "block_zero_based": block,
                        "path": path_name,
                        "tensors": path_tensors,
                    }
                )
    return {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d3_qk_stage_geometry",
        "status": "completed",
        "descriptive_only": True,
        "qrels_read": False,
        "source_test_opened": False,
        "models": list(MODELS),
        "blocks": list(BLOCKS),
        "stages": list(STAGES),
        "cells": cells,
        "path_cells": path_cells,
        "stage_transition_consistency": {
            "qk_rmsnorm_pre_to_post": _stage_transition_consistency(
                path_cells, "pre_norm", "post_norm"
            ),
            "rope_post_norm_to_post_rope": _stage_transition_consistency(
                path_cells, "post_norm", "post_rope"
            ),
        },
        "maximum_rope_norm_relative_l2_error": maximum_rope_norm_relative_error,
        "interpretation_boundary": (
            "Complete fixed-grid full/null Q/K geometry at registered semantic "
            "positions. RoPE-stage differences are descriptive phase geometry, not "
            "a natural-position intervention, causal bottleneck, or layer selector."
        ),
    }


def _stage_transition_consistency(
    path_cells: Sequence[Mapping[str, Any]], from_stage: str, to_stage: str
) -> dict[str, Any]:
    rows = []
    for cell in path_cells:
        for tensor, tensor_values in cell["tensors"].items():
            for position in tensor_values["positions"]:
                before = tensor_values["stages"][from_stage][position]
                after = tensor_values["stages"][to_stage][position]
                rows.append(
                    {
                        "tensor": tensor,
                        "block_zero_based": cell["block_zero_based"],
                        "position": position,
                        "relative_delta_change": float(
                            after["mean_relative_delta"]
                            - before["mean_relative_delta"]
                        ),
                        "cosine_change": float(
                            after["mean_full_null_cosine"]
                            - before["mean_full_null_cosine"]
                        ),
                    }
                )
    return {
        "from_stage": from_stage,
        "to_stage": to_stage,
        "overall": _transition_group(rows, {}),
        "by_tensor": _group_transition_rows(rows, ("tensor",)),
        "by_tensor_block": _group_transition_rows(
            rows, ("tensor", "block_zero_based")
        ),
        "by_position": _group_transition_rows(rows, ("position",)),
        "interpretation_boundary": (
            "Counts every fixed path/tensor/semantic-position comparison; exact "
            "sign consistency is descriptive and is not a causal or significance gate."
        ),
    }


def _group_transition_rows(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        grouped.setdefault(key, []).append(row)
    result = []
    for key in sorted(grouped):
        labels = dict(zip(fields, key))
        result.append({**labels, **_transition_group(grouped[key], labels)})
    return result


def _transition_group(
    rows: Sequence[Mapping[str, Any]], _labels: Mapping[str, Any]
) -> dict[str, Any]:
    relative = np.asarray(
        [row["relative_delta_change"] for row in rows], dtype=np.float64
    )
    cosine = np.asarray([row["cosine_change"] for row in rows], dtype=np.float64)
    if relative.size == 0 or not np.isfinite(relative).all() or not np.isfinite(cosine).all():
        raise ValueError("D3 Q/K stage-transition values differ")
    return {
        "comparisons": int(relative.size),
        "relative_delta_decreased": int(np.sum(relative < 0)),
        "relative_delta_unchanged": int(np.sum(relative == 0)),
        "relative_delta_increased": int(np.sum(relative > 0)),
        "mean_relative_delta_change": float(relative.mean()),
        "minimum_relative_delta_change": float(relative.min()),
        "maximum_relative_delta_change": float(relative.max()),
        "cosine_decreased": int(np.sum(cosine < 0)),
        "cosine_unchanged": int(np.sum(cosine == 0)),
        "cosine_increased": int(np.sum(cosine > 0)),
        "mean_cosine_change": float(cosine.mean()),
        "minimum_cosine_change": float(cosine.min()),
        "maximum_cosine_change": float(cosine.max()),
    }


def _summarize_tensor(
    paths: Mapping[str, Any],
    path_names: Sequence[str],
    *,
    tensor: str,
    size: int,
    positions: Sequence[str],
) -> tuple[dict[str, Any], float]:
    stage_rows = {}
    raw: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for stage in STAGES:
        raw[stage] = {}
        stage_rows[stage] = {}
        for position in positions:
            values = {
                field: _mean_path_vector(
                    paths,
                    path_names,
                    tensor=tensor,
                    stage=stage,
                    field=field,
                    position=position,
                    size=size,
                )
                for field in VECTOR_FIELDS
            }
            raw[stage][position] = values
            reference = 0.5 * (values["full_norm"] + values["null_norm"])
            relative_delta = values["full_null_delta_norm"] / np.maximum(
                reference, 1.0e-12
            )
            stage_rows[stage][position] = {
                "heads": size,
                "mean_full_norm": float(values["full_norm"].mean()),
                "mean_null_norm": float(values["null_norm"].mean()),
                "mean_full_null_delta_norm": float(
                    values["full_null_delta_norm"].mean()
                ),
                "mean_relative_delta": float(relative_delta.mean()),
                "maximum_relative_delta": float(relative_delta.max()),
                "mean_full_null_cosine": float(values["full_null_cosine"].mean()),
                "minimum_full_null_cosine": float(
                    values["full_null_cosine"].min()
                ),
                "maximum_full_null_cosine": float(
                    values["full_null_cosine"].max()
                ),
            }
    rope_effect = {}
    maximum_norm_error = 0.0
    for position in positions:
        post_norm = raw["post_norm"][position]
        post_rope = raw["post_rope"][position]
        full_error = _relative_vector_error(
            post_rope["full_norm"], post_norm["full_norm"]
        )
        null_error = _relative_vector_error(
            post_rope["null_norm"], post_norm["null_norm"]
        )
        maximum_norm_error = max(maximum_norm_error, full_error, null_error)
        rope_effect[position] = {
            "full_norm_relative_l2_error": full_error,
            "null_norm_relative_l2_error": null_error,
            "post_rope_minus_post_norm_mean_full_null_cosine": float(
                post_rope["full_null_cosine"].mean()
                - post_norm["full_null_cosine"].mean()
            ),
            "post_rope_minus_post_norm_mean_delta_norm": float(
                post_rope["full_null_delta_norm"].mean()
                - post_norm["full_null_delta_norm"].mean()
            ),
        }
    return (
        {
            "positions": list(positions),
            "stages": stage_rows,
            "rope_effect": rope_effect,
        },
        maximum_norm_error,
    )


def _validate_source(metrics: Mapping[str, Any]) -> None:
    if (
        metrics.get("analysis_type")
        != "transformer_deep_dive_d3_attention_head_observation"
        or metrics.get("status") != "completed"
        or metrics.get("descriptive_only") is not True
        or metrics.get("qrels_read") is not False
        or metrics.get("source_test_opened") is not False
        or tuple(metrics.get("blocks", ())) != BLOCKS
        or set(metrics.get("results", {})) != set(MODELS)
        or any(
            set(metrics["results"][model_id])
            != {str(block) for block in BLOCKS}
            for model_id in MODELS
        )
    ):
        raise ValueError("completed D3 Q/K source boundary differs")


def _position_set(
    paths: Mapping[str, Any], path_names: Sequence[str], tensor: str
) -> tuple[str, ...]:
    observed = None
    for path_name in path_names:
        geometry = paths[path_name]["qk_geometry"][tensor]
        if set(geometry) != set(STAGES):
            raise ValueError("D3 Q/K stage coverage differs")
        for stage in STAGES:
            if set(geometry[stage]) != set(VECTOR_FIELDS):
                raise ValueError("D3 Q/K vector-field coverage differs")
            for field in VECTOR_FIELDS:
                positions = tuple(sorted(geometry[stage][field]))
                if observed is None:
                    observed = positions
                elif positions != observed:
                    raise ValueError("D3 Q/K semantic-position coverage differs")
    if not observed:
        raise ValueError("D3 Q/K semantic positions are empty")
    return observed


def _mean_path_vector(
    paths: Mapping[str, Any],
    path_names: Sequence[str],
    *,
    tensor: str,
    stage: str,
    field: str,
    position: str,
    size: int,
) -> np.ndarray:
    values = []
    for path_name in path_names:
        try:
            vector = paths[path_name]["qk_geometry"][tensor][stage][field][position][
                "mean"
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError("D3 Q/K geometry schema differs") from exc
        array = np.asarray(vector, dtype=np.float64)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError("D3 Q/K head vector differs")
        if field != "full_null_cosine" and np.any(array < 0):
            raise ValueError("D3 Q/K norm vector is negative")
        if field == "full_null_cosine" and (
            np.any(array < -1.000001) or np.any(array > 1.000001)
        ):
            raise ValueError("D3 Q/K cosine vector is outside [-1,1]")
        values.append(array)
    return np.mean(np.stack(values), axis=0)


def _relative_vector_error(observed: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.linalg.norm(expected))
    error = float(np.linalg.norm(observed - expected))
    value = error / denominator if denominator > 0 else error
    if not math.isfinite(value):
        raise ValueError("D3 Q/K relative norm error is non-finite")
    return value
