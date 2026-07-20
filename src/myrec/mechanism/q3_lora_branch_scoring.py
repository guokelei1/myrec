"""Qrels-blind scoring for the N19 complete Q3 LoRA branch boundary."""

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
from myrec.mechanism.q3_lora_branch_interventions import (
    LORA_BRANCH_COMPONENTS,
    LORA_BRANCH_MODES,
    QwenQ3LoraBranchPatch,
)


LORA_BRANCH_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_lora_identity",
    "null_lora_identity",
    *(f"full_lora_{mode}" for mode in LORA_BRANCH_MODES[1:]),
    *(f"null_lora_{mode}" for mode in LORA_BRANCH_MODES[1:]),
)


def score_q3_lora_branch_chunk(
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
    """Score one Q3 candidate chunk with q/v adapter contribution controls."""

    if str(config.get("method_id")) != "q3_tallrec_generalqwen":
        raise ValueError("N19 LoRA branch scoring is Q3-only")
    if not candidates:
        raise ValueError("N19 candidate chunk is empty")
    component = str(component)
    if component not in LORA_BRANCH_COMPONENTS:
        raise ValueError(f"unsupported N19 component={component}")
    if content_control.get("eligible") is not True:
        raise ValueError("N19 requires content-neutral eligibility")
    full_paths = _build_paths(tokenizer, record, candidates, content_control, config, device=device)
    null_paths = _neutralize_paths(full_paths)
    _assert_shared_prompt_paths(full_paths)
    _assert_shared_prompt_paths(null_paths)
    conditions: dict[str, np.ndarray] = {
        "baseline_full": _run_native(model, full_paths),
        "baseline_null": _run_native(model, null_paths),
    }
    summaries: dict[str, Any] = {}
    for mode in LORA_BRANCH_MODES:
        full_values, full_summary = _run_branch(
            model, full_paths, block=block, component=component, mode=mode
        )
        null_values, null_summary = _run_branch(
            model, null_paths, block=block, component=component, mode=mode
        )
        full_name = "full_lora_identity" if mode == "identity" else f"full_lora_{mode}"
        null_name = "null_lora_identity" if mode == "identity" else f"null_lora_{mode}"
        conditions[full_name] = full_values
        conditions[null_name] = null_values
        summaries[full_name] = full_summary
        summaries[null_name] = null_summary
    _validate_conditions(conditions, len(candidates))
    identity = max(
        float(np.max(np.abs(conditions["full_lora_identity"] - conditions["baseline_full"]))),
        float(np.max(np.abs(conditions["null_lora_identity"] - conditions["baseline_null"]))),
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


def _run_branch(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    *,
    block: int,
    component: str,
    mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    summaries = []
    with QwenQ3LoraBranchPatch(model, block, component, mode) as patch:
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


def _validate_conditions(conditions: Mapping[str, np.ndarray], expected_count: int) -> None:
    if set(conditions) != set(LORA_BRANCH_CONDITIONS):
        raise ValueError("N19 condition set differs from registered manifest")
    for name, values in conditions.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (expected_count,) or not np.isfinite(values).all():
            raise FloatingPointError(f"N19 condition is invalid: {name}")
