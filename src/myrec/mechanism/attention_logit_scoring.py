"""Qrels-blind operator-level scaled-QK-logit scoring for Q2/Q3."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.attention_edge_scoring import (
    _aggregate_paths,
    _build_paths,
    _path_scores,
)
from myrec.mechanism.deep_dive_assignments import CONTENT_NEUTRAL_TOKEN_ID
from myrec.mechanism.attention_logit_interventions import (
    LOGIT_MODES,
    QwenAttentionLogitIntervention,
)


ATTENTION_LOGIT_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_qk_identity",
    "null_qk_identity",
    "full_qk_scale_half",
    "null_qk_scale_half",
    "full_qk_scale_double",
    "null_qk_scale_double",
    "full_qk_sign_flip",
    "null_qk_sign_flip",
)
MODE_TO_CONDITION = {
    "identity": ("full_qk_identity", "null_qk_identity"),
    "scale_half": ("full_qk_scale_half", "null_qk_scale_half"),
    "scale_double": ("full_qk_scale_double", "null_qk_scale_double"),
    "sign_flip": ("full_qk_sign_flip", "null_qk_sign_flip"),
}


def score_attention_logit_chunk(
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
    """Score full/null paths under fixed pre-softmax QK-logit operators."""

    if not candidates:
        raise ValueError("attention logit scoring candidate chunk is empty")
    if content_control.get("eligible") is not True:
        raise ValueError("attention logit scoring requires frozen content-neutral eligibility")
    full_paths = _build_paths(
        tokenizer, record, candidates, content_control, config, device=device
    )
    # The null path uses the same frozen span geometry; its tokens are replaced
    # only by the native content-control helper in the existing edge scorer.
    null_paths = [
        {**path, "ids": _neutralize_ids(path, content_control)}
        for path in full_paths
    ]
    shared_prompt_identity = True
    if str(config.get("method_id")) == "q3_tallrec_generalqwen":
        _assert_shared_prompt_paths(full_paths)
        _assert_shared_prompt_paths(null_paths)

    conditions: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    conditions["baseline_full"], summaries["baseline_full"] = _run_native(
        model, full_paths
    )
    conditions["baseline_null"], summaries["baseline_null"] = _run_native(
        model, null_paths
    )
    for mode in LOGIT_MODES:
        full_values, full_summary = _run_intervention(
            model, full_paths, block=block, mode=mode
        )
        null_values, null_summary = _run_intervention(
            model, null_paths, block=block, mode=mode
        )
        full_name, null_name = MODE_TO_CONDITION[mode]
        conditions[full_name] = full_values
        conditions[null_name] = null_values
        summaries[full_name] = full_summary
        summaries[null_name] = null_summary

    for name, values in conditions.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise FloatingPointError(f"attention logit condition is invalid: {name}")
        conditions[name] = values
    identity = max(
        float(np.max(np.abs(conditions["full_qk_identity"] - conditions["baseline_full"]))),
        float(np.max(np.abs(conditions["null_qk_identity"] - conditions["baseline_null"]))),
    )
    return {
        "conditions": conditions,
        "summaries": summaries,
        "maximum_identity_delta": identity,
        "maximum_manual_native_error": max(
            _summary_max(summaries["full_qk_scale_half"]),
            _summary_max(summaries["null_qk_scale_half"]),
            _summary_max(summaries["full_qk_scale_double"]),
            _summary_max(summaries["null_qk_scale_double"]),
            _summary_max(summaries["full_qk_sign_flip"]),
            _summary_max(summaries["null_qk_sign_flip"]),
        ),
        "shared_prompt_path_identity": shared_prompt_identity,
    }


def _run_native(model: Any, paths: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    summaries = []
    for path in paths:
        output = model(
            input_ids=path["ids"],
            attention_mask=path["mask"],
            use_cache=False,
            logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
        )
        values.append(_path_scores(output, path))
        summaries.append({"mode": "native", "rows": int(path["ids"].shape[0])})
    return _aggregate_paths(paths, values), summaries


def _run_intervention(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    *,
    block: int,
    mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    summaries = []
    with QwenAttentionLogitIntervention(model, block, mode) as intervention:
        for path in paths:
            intervention.arm(path["positions"], sequence_length=path["ids"].shape[1])
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            summaries.append(intervention.disarm())
            values.append(_path_scores(output, path))
    return _aggregate_paths(paths, values), summaries


def _neutralize_ids(path: Mapping[str, Any], content_control: Mapping[str, Any]) -> Any:
    """Return the fixed content-neutral path without changing positions/mask."""

    ids = path["ids"].clone()
    starts = path["starts"]
    ends = path["ends"]
    for row in range(ids.shape[0]):
        ids[row, int(starts[row]) : int(ends[row])] = CONTENT_NEUTRAL_TOKEN_ID
    return ids


def _summary_max(values: Sequence[Mapping[str, Any]]) -> float:
    return max(
        float(item.get("manual_baseline_native_max_abs_error", 0.0))
        for item in values
    )


def _assert_shared_prompt_paths(paths: Sequence[Mapping[str, Any]]) -> None:
    if len(paths) != 2:
        raise ValueError("Q3 attention-logit paths must contain Yes and No")
    first, second = paths
    if first["ids"].shape[0] != second["ids"].shape[0]:
        raise ValueError("Q3 Yes/No batch size differs")
    for row in range(first["ids"].shape[0]):
        first_end = int(first["positions"][row].min())
        second_end = int(second["positions"][row].min())
        first_prefix = first["ids"][row, :first_end]
        second_prefix = second["ids"][row, :second_end]
        if first_prefix.shape != second_prefix.shape or not bool(
            (first_prefix == second_prefix).all()
        ):
            raise ValueError("Q3 Yes/No shared prompt path identity failed")
