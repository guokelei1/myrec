"""Q2/Q3 scoring kernels for fixed-length D5 contextual history controls."""

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


CONTEXTUAL_SCORE_CONDITIONS = (
    "baseline_full",
    "unmodified_identity",
    "history_content_neutral",
    "history_attention_null",
)


def score_contextual_control_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    content_control: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("contextual-control candidate chunk is empty")
    if content_control.get("eligible") is not True:
        raise ValueError("contextual-control scoring requires frozen eligibility")
    paths = _build_paths(
        tokenizer,
        record,
        candidates,
        content_control,
        config,
        device=device,
    )
    baseline = [_score_path(model, path) for path in paths]
    unmodified = [_score_path(model, path) for path in paths]
    neutral = [_score_path(model, path) for path in _neutralize_paths(paths)]
    attention_null = [
        _score_path(model, _attention_null_path(path)) for path in paths
    ]
    conditions = {
        "baseline_full": _aggregate_paths(paths, baseline),
        "unmodified_identity": _aggregate_paths(paths, unmodified),
        "history_content_neutral": _aggregate_paths(paths, neutral),
        "history_attention_null": _aggregate_paths(paths, attention_null),
    }
    for name, values in conditions.items():
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise FloatingPointError(f"contextual score is invalid: {name}")
    return {
        "conditions": conditions,
        "maximum_identity_delta": float(
            np.max(
                np.abs(
                    conditions["unmodified_identity"] - conditions["baseline_full"]
                )
            )
        ),
    }


def _attention_null_path(path: Mapping[str, Any]) -> dict[str, Any]:
    mask = path["mask"].clone()
    for row in range(mask.shape[0]):
        mask[row, int(path["starts"][row]) : int(path["ends"][row])] = 0
    # Token IDs and tensor length are intentionally unchanged.  Qwen's frozen
    # full-sequence path uses cache_position/arange for RoPE positions, so the
    # internal zero span does not compact later token positions.
    return {**path, "mask": mask}


def _score_path(model: Any, path: Mapping[str, Any]) -> np.ndarray:
    output = model(
        input_ids=path["ids"],
        attention_mask=path["mask"],
        use_cache=False,
        logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
    )
    return _path_scores(output, path)
