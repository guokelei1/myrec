"""Qrels-blind scoring kernels for the registered N15/N16 operators.

These functions intentionally stop at per-candidate scalar conditions.  Bundle
integrity, resume, and qrels-gated evaluation remain in the shared runtime and
evaluator layers, just as for N11--N14.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.attention_edge_scoring import (
    _aggregate_paths,
    _build_paths,
    _neutralize_paths,
    _path_scores,
)
from myrec.mechanism.attention_logit_scoring import _assert_shared_prompt_paths
from myrec.mechanism.residual_composition_interventions import (
    RESIDUAL_MODES,
    QwenResidualCompositionPatch,
)
from myrec.mechanism.rmsnorm_interventions import (
    RMSNORM_MODES,
    QwenRMSNormPatch,
)


RESIDUAL_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_residual_identity",
    "null_residual_identity",
    *(f"full_residual_{mode}" for mode in RESIDUAL_MODES[1:]),
    *(f"null_residual_{mode}" for mode in RESIDUAL_MODES[1:]),
)
RMSNORM_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_norm_identity",
    "null_norm_identity",
    *(f"full_norm_{mode}" for mode in RMSNORM_MODES[1:]),
    *(f"null_norm_{mode}" for mode in RMSNORM_MODES[1:]),
)


def score_residual_composition_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    content_control: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    block: int,
    branch: str,
    device: str,
) -> dict[str, Any]:
    """Score full/null paths while scaling one residual branch increment."""

    if not candidates:
        raise ValueError("residual composition candidate chunk is empty")
    if content_control.get("eligible") is not True:
        raise ValueError("residual composition requires content-neutral eligibility")
    full_paths = _build_paths(tokenizer, record, candidates, content_control, config, device=device)
    null_paths = _neutralize_paths(full_paths)
    _assert_q3_path_contract(config, full_paths, null_paths)
    conditions: dict[str, np.ndarray] = {
        "baseline_full": _run_native(model, full_paths),
        "baseline_null": _run_native(model, null_paths),
    }
    summaries: dict[str, Any] = {}
    for mode in RESIDUAL_MODES:
        full_values, full_summary = _run_residual(
            model, full_paths, block=block, branch=branch, mode=mode
        )
        null_values, null_summary = _run_residual(
            model, null_paths, block=block, branch=branch, mode=mode
        )
        full_name = "full_residual_identity" if mode == "identity" else f"full_residual_{mode}"
        null_name = "null_residual_identity" if mode == "identity" else f"null_residual_{mode}"
        conditions[full_name] = full_values
        conditions[null_name] = null_values
        summaries[full_name] = full_summary
        summaries[null_name] = null_summary
    _validate_condition_arrays(conditions, len(candidates), RESIDUAL_CONDITIONS)
    identity = max(
        float(np.max(np.abs(conditions["full_residual_identity"] - conditions["baseline_full"]))),
        float(np.max(np.abs(conditions["null_residual_identity"] - conditions["baseline_null"]))),
    )
    return {
        "conditions": conditions,
        "summaries": summaries,
        "maximum_identity_delta": identity,
        "shared_prompt_path_identity": True,
    }


def score_rmsnorm_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    content_control: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    scope: str,
    block: int | None,
    device: str,
) -> dict[str, Any]:
    """Score full/null paths while changing one RMSNorm factor."""

    if not candidates:
        raise ValueError("RMSNorm candidate chunk is empty")
    if content_control.get("eligible") is not True:
        raise ValueError("RMSNorm scoring requires content-neutral eligibility")
    full_paths = _build_paths(tokenizer, record, candidates, content_control, config, device=device)
    null_paths = _neutralize_paths(full_paths)
    _assert_q3_path_contract(config, full_paths, null_paths)
    conditions: dict[str, np.ndarray] = {
        "baseline_full": _run_native(model, full_paths),
        "baseline_null": _run_native(model, null_paths),
    }
    summaries: dict[str, Any] = {}
    for mode in RMSNORM_MODES:
        full_values, full_summary = _run_rmsnorm(
            model, full_paths, scope=scope, block=block, mode=mode
        )
        null_values, null_summary = _run_rmsnorm(
            model, null_paths, scope=scope, block=block, mode=mode
        )
        full_name = "full_norm_identity" if mode == "identity" else f"full_norm_{mode}"
        null_name = "null_norm_identity" if mode == "identity" else f"null_norm_{mode}"
        conditions[full_name] = full_values
        conditions[null_name] = null_values
        summaries[full_name] = full_summary
        summaries[null_name] = null_summary
    _validate_condition_arrays(conditions, len(candidates), RMSNORM_CONDITIONS)
    identity = max(
        float(np.max(np.abs(conditions["full_norm_identity"] - conditions["baseline_full"]))),
        float(np.max(np.abs(conditions["null_norm_identity"] - conditions["baseline_null"]))),
    )
    return {
        "conditions": conditions,
        "summaries": summaries,
        "maximum_identity_delta": identity,
        "shared_prompt_path_identity": True,
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


def _run_residual(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    *,
    block: int,
    branch: str,
    mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    summaries = []
    with QwenResidualCompositionPatch(model, block, branch, mode) as patch:
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


def _run_rmsnorm(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    block: int | None,
    mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    summaries = []
    with QwenRMSNormPatch(model, scope, block=block, mode=mode) as patch:
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


def _assert_q3_path_contract(
    config: Mapping[str, Any],
    full_paths: Sequence[Mapping[str, Any]],
    null_paths: Sequence[Mapping[str, Any]],
) -> None:
    if str(config.get("method_id")) != "q3_tallrec_generalqwen":
        return
    _assert_shared_prompt_paths(full_paths)
    _assert_shared_prompt_paths(null_paths)


def _validate_condition_arrays(
    conditions: Mapping[str, np.ndarray], expected_count: int, registered: Sequence[str]
) -> None:
    if set(conditions) != set(registered):
        raise ValueError("operator condition set differs from registered manifest")
    for name, values in conditions.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (expected_count,) or not np.isfinite(values).all():
            raise FloatingPointError(f"operator condition is invalid: {name}")
