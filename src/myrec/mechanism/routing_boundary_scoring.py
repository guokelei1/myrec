"""Qrels-blind scoring kernels for the preregistered N17/N18 boundaries.

N17 changes only the selected q_norm or k_norm output before RoPE.  N18
changes only the repeat-KV group assignment before attention.  Both kernels
reuse the frozen path builder and native readout used by earlier waves; no
qrels or outcome-dependent path selection occurs here.
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
from myrec.mechanism.gqa_grouping_interventions import (
    GQA_GROUPING_MODES,
    QwenGQAGroupingIntervention,
)
from myrec.mechanism.qk_head_rmsnorm_interventions import (
    HEAD_NORM_COMPONENTS,
    HEAD_NORM_MODES,
    QwenQKHeadRMSNormPatch,
)


HEAD_NORM_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_head_norm_identity",
    "null_head_norm_identity",
    *(f"full_head_norm_{mode}" for mode in HEAD_NORM_MODES[1:]),
    *(f"null_head_norm_{mode}" for mode in HEAD_NORM_MODES[1:]),
)
GQA_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_gqa_identity",
    "null_gqa_identity",
    *(f"full_gqa_{mode}" for mode in GQA_GROUPING_MODES[1:]),
    *(f"null_gqa_{mode}" for mode in GQA_GROUPING_MODES[1:]),
)


def score_qk_head_norm_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    content_control: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    block: int,
    component: str,
    device: str,
) -> dict[str, Any]:
    """Score one candidate chunk under N17 q_norm or k_norm conditions."""

    if not candidates:
        raise ValueError("N17 candidate chunk is empty")
    component = str(component)
    if component not in HEAD_NORM_COMPONENTS:
        raise ValueError(f"unsupported N17 component={component}")
    if content_control.get("eligible") is not True:
        raise ValueError("N17 requires content-neutral eligibility")
    full_paths = _build_paths(tokenizer, record, candidates, content_control, config, device=device)
    null_paths = _neutralize_paths(full_paths)
    _assert_q3_paths(config, full_paths, null_paths)
    conditions: dict[str, np.ndarray] = {
        "baseline_full": _run_native(model, full_paths),
        "baseline_null": _run_native(model, null_paths),
    }
    summaries: dict[str, Any] = {}
    for mode in HEAD_NORM_MODES:
        full_values, full_summary = _run_head_norm(
            model, full_paths, block=block, component=component, mode=mode
        )
        null_values, null_summary = _run_head_norm(
            model, null_paths, block=block, component=component, mode=mode
        )
        full_name = (
            "full_head_norm_identity"
            if mode == "identity"
            else f"full_head_norm_{mode}"
        )
        null_name = (
            "null_head_norm_identity"
            if mode == "identity"
            else f"null_head_norm_{mode}"
        )
        conditions[full_name] = full_values
        conditions[null_name] = null_values
        summaries[full_name] = full_summary
        summaries[null_name] = null_summary
    _validate_conditions(conditions, len(candidates), HEAD_NORM_CONDITIONS)
    identity = max(
        float(np.max(np.abs(conditions["full_head_norm_identity"] - conditions["baseline_full"]))),
        float(np.max(np.abs(conditions["null_head_norm_identity"] - conditions["baseline_null"]))),
    )
    return {
        "conditions": conditions,
        "summaries": summaries,
        "maximum_identity_delta": identity,
        "shared_prompt_path_identity": True,
    }


def score_gqa_grouping_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    content_control: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    """Score one candidate chunk under N18 repeat-KV grouping conditions."""

    if not candidates:
        raise ValueError("N18 candidate chunk is empty")
    if content_control.get("eligible") is not True:
        raise ValueError("N18 requires content-neutral eligibility")
    full_paths = _build_paths(tokenizer, record, candidates, content_control, config, device=device)
    null_paths = _neutralize_paths(full_paths)
    _assert_q3_paths(config, full_paths, null_paths)
    conditions: dict[str, np.ndarray] = {
        "baseline_full": _run_native(model, full_paths),
        "baseline_null": _run_native(model, null_paths),
    }
    summaries: dict[str, Any] = {}
    for mode in GQA_GROUPING_MODES:
        full_values, full_summary = _run_gqa(
            model, full_paths, block=block, mode=mode
        )
        null_values, null_summary = _run_gqa(
            model, null_paths, block=block, mode=mode
        )
        full_name = "full_gqa_identity" if mode == "identity" else f"full_gqa_{mode}"
        null_name = "null_gqa_identity" if mode == "identity" else f"null_gqa_{mode}"
        conditions[full_name] = full_values
        conditions[null_name] = null_values
        summaries[full_name] = full_summary
        summaries[null_name] = null_summary
    _validate_conditions(conditions, len(candidates), GQA_CONDITIONS)
    identity = max(
        float(np.max(np.abs(conditions["full_gqa_identity"] - conditions["baseline_full"]))),
        float(np.max(np.abs(conditions["null_gqa_identity"] - conditions["baseline_null"]))),
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


def _run_head_norm(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    *,
    block: int,
    component: str,
    mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    summaries = []
    with QwenQKHeadRMSNormPatch(model, block, component, mode) as patch:
        for path in paths:
            patch.arm(
                path["positions"],
                path["starts"],
                path["ends"],
                sequence_length=path["ids"].shape[1],
            )
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            summaries.append(patch.disarm())
            values.append(_path_scores(output, path))
    return _aggregate_paths(paths, values), summaries


def _run_gqa(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    *,
    block: int,
    mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    summaries = []
    with QwenGQAGroupingIntervention(model, block, mode) as intervention:
        for path in paths:
            intervention.arm(
                path["positions"],
                path["starts"],
                path["ends"],
                sequence_length=path["ids"].shape[1],
            )
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            summaries.append(intervention.disarm())
            values.append(_path_scores(output, path))
    return _aggregate_paths(paths, values), summaries


def _assert_q3_paths(
    config: Mapping[str, Any],
    full_paths: Sequence[Mapping[str, Any]],
    null_paths: Sequence[Mapping[str, Any]],
) -> None:
    if str(config.get("method_id")) != "q3_tallrec_generalqwen":
        return
    _assert_shared_prompt_paths(full_paths)
    _assert_shared_prompt_paths(null_paths)


def _validate_conditions(
    conditions: Mapping[str, np.ndarray], expected_count: int, registered: Sequence[str]
) -> None:
    if set(conditions) != set(registered):
        raise ValueError("routing condition set differs from registered manifest")
    for name, values in conditions.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (expected_count,) or not np.isfinite(values).all():
            raise FloatingPointError(f"routing condition is invalid: {name}")
