"""Qrels-blind scoring kernels for the preregistered N25 SwiGLU boundary."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.attention_edge_scoring import _aggregate_paths, _path_scores
from myrec.mechanism.mlp_group_scoring import build_native_pointwise_paths
from myrec.mechanism.swiglu_formation_interventions import SWIGLU_MODES, SWIGLU_OPERATORS, QwenSwiGLUFormationPatch


SWIGLU_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    *(
        f"{path}_{operator}_{mode}"
        for path in ("full", "null")
        for operator in SWIGLU_OPERATORS
        for mode in SWIGLU_MODES
    ),
)


def score_swiglu_formation_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("SwiGLU candidate chunk is empty")
    full_paths = build_native_pointwise_paths(
        tokenizer, record, candidates, record.history, config, device=device
    )
    null_paths = build_native_pointwise_paths(
        tokenizer, record, candidates, [], config, device=device
    )
    conditions: dict[str, np.ndarray] = {
        "baseline_full": _run_native(model, full_paths),
        "baseline_null": _run_native(model, null_paths),
    }
    summaries: dict[str, Any] = {}
    maximum_identity_delta = 0.0
    for operator in SWIGLU_OPERATORS:
        for mode in SWIGLU_MODES:
            full_name = f"full_{operator}_{mode}"
            null_name = f"null_{operator}_{mode}"
            full_values, full_summary = _run_patch(model, full_paths, block, operator, mode)
            null_values, null_summary = _run_patch(model, null_paths, block, operator, mode)
            conditions[full_name] = full_values
            conditions[null_name] = null_values
            summaries[full_name] = full_summary
            summaries[null_name] = null_summary
            if mode == "identity":
                maximum_identity_delta = max(
                    maximum_identity_delta,
                    float(np.max(np.abs(full_values - conditions["baseline_full"]))),
                    float(np.max(np.abs(null_values - conditions["baseline_null"]))),
                )
    _validate_conditions(conditions, len(candidates))
    return {
        "conditions": conditions,
        "summaries": summaries,
        "maximum_identity_delta": maximum_identity_delta,
        "swiglu_operator_set": list(SWIGLU_OPERATORS),
    }


def _run_native(model: Any, paths: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = []
    for path in paths:
        output = model(
            input_ids=path["ids"],
            attention_mask=path["mask"],
            use_cache=False,
            logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
        )
        values.append(_path_scores(output, path))
    return _aggregate_paths(paths, values)


def _run_patch(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    block: int,
    operator: str,
    mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    summaries = []
    with QwenSwiGLUFormationPatch(model, block, operator, mode) as patch:
        for path in paths:
            patch.arm(path["positions"], sequence_length=path["ids"].shape[1])
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            summaries.append(patch.disarm())
            values.append(_path_scores(output, path))
    return _aggregate_paths(paths, values), summaries


def _validate_conditions(conditions: Mapping[str, np.ndarray], expected_count: int) -> None:
    if set(conditions) != set(SWIGLU_CONDITIONS):
        raise ValueError("SwiGLU condition set differs from registered manifest")
    for name, values in conditions.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (expected_count,) or not np.isfinite(values).all():
            raise FloatingPointError(f"SwiGLU condition is invalid: {name}")

