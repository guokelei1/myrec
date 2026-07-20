"""Qrels-blind gate/up stage intervention scoring for Q2/Q3."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.attention_edge_scoring import _aggregate_paths, _path_scores
from myrec.mechanism.mlp_group_scoring import build_native_pointwise_paths
from myrec.mechanism.mlp_feature_formation import QwenMLPFeatureObserver
from myrec.mechanism.mlp_stage_interventions import QwenMLPStagePatch


MLP_STAGE_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_gate_identity",
    "null_gate_identity",
    "null_gate_from_full",
    "null_up_from_full",
    "null_joint_from_full",
    "full_gate_from_null",
    "full_up_from_null",
    "full_joint_from_null",
    "null_gate_sign_flip",
    "null_up_sign_flip",
)
ACTIVE_STAGE_CONDITIONS = MLP_STAGE_CONDITIONS[4:]


def score_mlp_stage_chunk(
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
        raise ValueError("MLP stage candidate chunk is empty")
    full_paths = build_native_pointwise_paths(
        tokenizer, record, candidates, record.history, config, device=device
    )
    null_paths = build_native_pointwise_paths(
        tokenizer, record, candidates, [], config, device=device
    )
    full = _capture_stage_paths(model, full_paths, block=block)
    null = _capture_stage_paths(model, null_paths, block=block)
    conditions: dict[str, np.ndarray] = {
        "baseline_full": _aggregate_paths(full_paths, full["scores"]),
        "baseline_null": _aggregate_paths(null_paths, null["scores"]),
    }
    identity = 0.0
    conditions["full_gate_identity"], _ = _patched_stage_paths(
        model, full_paths, full, block=block, gate=True, up=False
    )
    conditions["null_gate_identity"], _ = _patched_stage_paths(
        model, null_paths, null, block=block, gate=True, up=False
    )
    identity = max(
        identity,
        float(np.max(np.abs(conditions["full_gate_identity"] - conditions["baseline_full"]))),
        float(np.max(np.abs(conditions["null_gate_identity"] - conditions["baseline_null"]))),
    )
    conditions["null_gate_from_full"], _ = _patched_stage_paths(
        model, null_paths, full, block=block, gate=True, up=False
    )
    conditions["null_up_from_full"], _ = _patched_stage_paths(
        model, null_paths, full, block=block, gate=False, up=True
    )
    conditions["null_joint_from_full"], _ = _patched_stage_paths(
        model, null_paths, full, block=block, gate=True, up=True
    )
    conditions["full_gate_from_null"], _ = _patched_stage_paths(
        model, full_paths, null, block=block, gate=True, up=False
    )
    conditions["full_up_from_null"], _ = _patched_stage_paths(
        model, full_paths, null, block=block, gate=False, up=True
    )
    conditions["full_joint_from_null"], _ = _patched_stage_paths(
        model, full_paths, null, block=block, gate=True, up=True
    )
    conditions["null_gate_sign_flip"], _ = _patched_stage_paths(
        model, null_paths, null, block=block, gate=True, up=False, gate_scale=-1.0
    )
    conditions["null_up_sign_flip"], _ = _patched_stage_paths(
        model, null_paths, null, block=block, gate=False, up=True, up_scale=-1.0
    )
    for name, values in conditions.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise FloatingPointError(f"MLP stage condition is invalid: {name}")
        conditions[name] = values
    return {
        "conditions": conditions,
        "maximum_identity_delta": float(identity),
        "capture_product_identity_max_error": max(
            float(value) for value in (*full["product_errors"], *null["product_errors"])
        ),
    }


def _capture_stage_paths(
    model: Any, paths: Sequence[Mapping[str, Any]], *, block: int
) -> dict[str, Any]:
    scores = []
    gates = []
    ups = []
    product_errors = []
    with QwenMLPFeatureObserver(model, block) as observer:
        for path in paths:
            observer.arm(path["positions"], sequence_length=path["ids"].shape[1])
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            capture = observer.disarm()
            scores.append(_path_scores(output, path))
            gates.append(capture["captures"]["gate_pre"])
            ups.append(capture["captures"]["up"])
            product_errors.append(capture["product_recomposition_max_abs_error"])
    return {"scores": scores, "gates": gates, "ups": ups, "product_errors": product_errors}


def _patched_stage_paths(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    donor: Mapping[str, Any],
    *,
    block: int,
    gate: bool,
    up: bool,
    gate_scale: float = 1.0,
    up_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    if len(paths) != len(donor["gates"]) or len(paths) != len(donor["ups"]):
        raise ValueError("MLP stage paths and donors differ")
    values = []
    maximum_identity = 0.0
    with QwenMLPStagePatch(model, block) as patch:
        for path, gate_donor, up_donor in zip(paths, donor["gates"], donor["ups"]):
            gate_value = gate_donor * float(gate_scale) if gate else None
            up_value = up_donor * float(up_scale) if up else None
            patch.arm(path["positions"], gate_donor=gate_value, up_donor=up_value)
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            patch.disarm()
            values.append(_path_scores(output, path))
            if gate and up is False:
                # identity callers compare against the native path outside this helper
                pass
    return _aggregate_paths(paths, values), maximum_identity
