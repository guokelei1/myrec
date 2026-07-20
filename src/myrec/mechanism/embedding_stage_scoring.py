"""Qrels-blind history-embedding-stage scoring for Q2/Q3."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.attention_edge_scoring import _aggregate_paths, _build_paths, _path_scores
from myrec.mechanism.attention_logit_scoring import _assert_shared_prompt_paths, _neutralize_ids
from myrec.mechanism.embedding_stage_interventions import (
    EMBEDDING_MODES,
    QwenHistoryEmbeddingIntervention,
)


EMBEDDING_STAGE_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_embedding_identity",
    "null_embedding_identity",
    "full_embedding_scale_half",
    "null_embedding_scale_half",
    "full_embedding_scale_double",
    "null_embedding_scale_double",
    "full_embedding_sign_flip",
    "null_embedding_sign_flip",
    "full_embedding_zero",
    "null_embedding_zero",
)


def score_embedding_stage_chunk(
    model: Any,
    tokenizer: Any,
    record: Any,
    candidates: Sequence[Mapping[str, Any]],
    content_control: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("embedding-stage candidate chunk is empty")
    if content_control.get("eligible") is not True:
        raise ValueError("embedding-stage scoring requires frozen content-neutral eligibility")
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
    conditions: dict[str, np.ndarray] = {
        "baseline_full": _run_native(model, full_paths),
        "baseline_null": _run_native(model, null_paths),
    }
    identity = 0.0
    summaries: dict[str, Any] = {}
    for mode in EMBEDDING_MODES:
        for path_kind, paths in (("full", full_paths), ("null", null_paths)):
            name = f"{path_kind}_embedding_{mode}"
            values, summary = _run_embedding(model, paths, mode=mode)
            conditions[name] = values
            summaries[name] = summary
            if mode == "identity":
                baseline = conditions[f"baseline_{path_kind}"]
                identity = max(identity, float(np.max(np.abs(values - baseline))))
    for name in EMBEDDING_STAGE_CONDITIONS:
        values = np.asarray(conditions[name], dtype=np.float32)
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise FloatingPointError(f"embedding-stage condition is invalid: {name}")
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


def _run_embedding(
    model: Any, paths: Sequence[Mapping[str, Any]], *, mode: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    summaries = []
    with QwenHistoryEmbeddingIntervention(model, mode) as intervention:
        for path in paths:
            intervention.arm(
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
