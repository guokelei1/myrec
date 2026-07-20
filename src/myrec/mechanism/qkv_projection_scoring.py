"""Qrels-blind Q/K/V projection-stage scoring for Q2/Q3."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.attention_edge_scoring import (
    _aggregate_paths,
    _build_paths,
    _path_scores,
)
from myrec.mechanism.attention_logit_scoring import (
    _assert_shared_prompt_paths,
    _neutralize_ids,
)
from myrec.mechanism.qkv_projection_interventions import (
    PROJECTION_COMPONENTS,
    PROJECTION_MODES,
    QwenQKVProjectionIntervention,
)


QKV_PROJECTION_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_q_identity",
    "null_q_identity",
    "full_k_identity",
    "null_k_identity",
    "full_v_identity",
    "null_v_identity",
    "full_q_scale_half",
    "null_q_scale_half",
    "full_q_scale_double",
    "null_q_scale_double",
    "full_q_sign_flip",
    "null_q_sign_flip",
    "full_k_scale_half",
    "null_k_scale_half",
    "full_k_scale_double",
    "null_k_scale_double",
    "full_k_sign_flip",
    "null_k_sign_flip",
    "full_v_scale_half",
    "null_v_scale_half",
    "full_v_scale_double",
    "null_v_scale_double",
    "full_v_sign_flip",
    "null_v_sign_flip",
)

_IDENTITY_CONDITIONS = tuple(
    name
    for component in PROJECTION_COMPONENTS
    for name in (f"full_{component}_identity", f"null_{component}_identity")
)


def score_qkv_projection_chunk(
    model: Any,
    tokenizer: Any,
    record: Any,
    candidates: Sequence[Mapping[str, Any]],
    content_control: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("QKV projection candidate chunk is empty")
    if content_control.get("eligible") is not True:
        raise ValueError("QKV projection scoring requires frozen content-neutral eligibility")
    full_paths = _build_paths(
        tokenizer, record, candidates, content_control, config, device=device
    )
    null_paths = [
        {**path, "ids": _neutralize_ids(path, content_control)}
        for path in full_paths
    ]
    if str(config.get("method_id")) == "q3_tallrec_generalqwen":
        _assert_shared_prompt_paths(full_paths)
        _assert_shared_prompt_paths(null_paths)
    conditions: dict[str, np.ndarray] = {}
    conditions["baseline_full"] = _run_native(model, full_paths)
    conditions["baseline_null"] = _run_native(model, null_paths)
    summaries: dict[str, Any] = {}
    identity = 0.0
    for component in PROJECTION_COMPONENTS:
        for path_kind, paths in (("full", full_paths), ("null", null_paths)):
            name = f"{path_kind}_{component}_identity"
            values, summary = _run_projection(
                model, paths, block=block, component=component, mode="identity"
            )
            conditions[name] = values
            summaries[name] = summary
            baseline_name = f"baseline_{path_kind}"
            identity = max(identity, float(np.max(np.abs(values - conditions[baseline_name]))))
        for mode in PROJECTION_MODES[1:]:
            for path_kind, paths in (("full", full_paths), ("null", null_paths)):
                name = f"{path_kind}_{component}_{mode}"
                values, summary = _run_projection(
                    model, paths, block=block, component=component, mode=mode
                )
                conditions[name] = values
                summaries[name] = summary
    for name in QKV_PROJECTION_CONDITIONS:
        values = np.asarray(conditions[name], dtype=np.float32)
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise FloatingPointError(f"QKV projection condition is invalid: {name}")
        conditions[name] = values
    return {
        "conditions": conditions,
        "summaries": summaries,
        "maximum_identity_delta": float(identity),
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


def _run_projection(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    *,
    block: int,
    component: str,
    mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    summaries = []
    with QwenQKVProjectionIntervention(model, block, component, mode) as intervention:
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
