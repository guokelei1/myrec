"""Q2/Q3 native scoring kernels for registered layer-local RoPE conditions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.attention_edge_scoring import (
    _aggregate_paths,
    _build_paths,
    _path_scores,
)
from myrec.mechanism.rope_interventions import QwenRoPEPhaseIntervention


ROPE_SCORE_CONDITIONS = (
    "baseline_full",
    "zero_phase_identity",
    "common_offset_plus_17_identity",
    "readout_q_distance_compression",
    "readout_q_distance_expansion",
    "history_k_distance_compression",
    "history_k_distance_expansion",
    "paired_qk_distance_compression",
    "paired_qk_distance_expansion",
)
COMMON_OFFSET_BOUND_RATIO_TOLERANCE = 1.0e-4


def score_rope_chunk(
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
    """Score all registered RoPE phase conditions for one candidate chunk."""

    if not candidates:
        raise ValueError("RoPE scoring candidate chunk is empty")
    if content_control.get("eligible") is not True:
        raise ValueError("RoPE scoring requires frozen anchor eligibility")
    paths = _build_paths(
        tokenizer,
        record,
        candidates,
        content_control,
        config,
        device=device,
    )
    baseline_values = [_plain_path(model, path) for path in paths]
    conditions = {"baseline_full": _aggregate_paths(paths, baseline_values)}
    summaries: dict[str, list[dict[str, Any]]] = {}
    mode_to_condition = {
        "zero_phase_delta": "zero_phase_identity",
        "common_offset_plus_17": "common_offset_plus_17_identity",
        **{name: name for name in ROPE_SCORE_CONDITIONS[3:]},
    }
    for mode, condition in mode_to_condition.items():
        values = []
        path_summaries = []
        with QwenRoPEPhaseIntervention(model, block, mode) as intervention:
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
                path_summaries.append(intervention.disarm())
                values.append(_path_scores(output, path))
        conditions[condition] = _aggregate_paths(paths, values)
        summaries[condition] = path_summaries
    for name, values in conditions.items():
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise FloatingPointError(f"RoPE condition is invalid: {name}")
    identity_deltas = {
        name: float(np.max(np.abs(conditions[name] - conditions["baseline_full"])))
        for name in ("zero_phase_identity", "common_offset_plus_17_identity")
    }
    common_delta = np.abs(
        conditions["common_offset_plus_17_identity"]
        - conditions["baseline_full"]
    )
    # The frozen 4*eps rule applies to the RoPE vector-norm algebra audit, not
    # to native scores.  Both no-op conditions remain subject to the strict
    # 1e-5 score identity gate.
    common_ratio = max(
        max(
            float(summary["maximum_query_norm_low_precision_ratio"]),
            float(summary["maximum_key_norm_low_precision_ratio"]),
        )
        for summary in summaries["common_offset_plus_17_identity"]
    )
    return {
        "conditions": conditions,
        "summaries": summaries,
        "identity_deltas": identity_deltas,
        "maximum_identity_delta": max(identity_deltas.values()),
        "common_offset_score_identity_passed": (
            float(np.max(common_delta)) <= 1.0e-5
        ),
        "common_offset_low_precision_max_ratio": common_ratio,
        "common_offset_low_precision_passed": (
            common_ratio <= 1.0 + COMMON_OFFSET_BOUND_RATIO_TOLERANCE
        ),
    }


def _plain_path(model: Any, path: Mapping[str, Any]) -> np.ndarray:
    output = model(
        input_ids=path["ids"],
        attention_mask=path["mask"],
        use_cache=False,
        logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
    )
    return _path_scores(output, path)
