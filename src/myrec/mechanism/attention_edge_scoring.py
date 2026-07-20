"""Q2/Q3 native scoring kernels for registered attention-edge conditions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.baselines.motivation_v12_ranker import (
    _answer_target_tokens,
)
from myrec.mechanism.attention_edge_interventions import (
    QwenAttentionEdgeIntervention,
)
from myrec.mechanism.deep_dive_assignments import CONTENT_NEUTRAL_TOKEN_ID
from myrec.mechanism.history_kv_interventions import QwenHistoryKVIntervention
from myrec.mechanism.patch_scorer import _left_pad_sequences
from myrec.mechanism.representation_probe import instrument_pointwise_prompt


ATTENTION_SCORE_CONDITIONS = (
    "baseline_full",
    "zero_delta_identity",
    "same_kv_identity",
    "mask_then_restore_output_identity",
    "history_logits_mask",
    "history_value_edge_zero",
    "neutral_history_kv",
)


def score_attention_edge_chunk(
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
    """Score all registered aggregate attention conditions for one chunk."""

    import torch

    if not candidates:
        raise ValueError("attention scoring candidate chunk is empty")
    if content_control.get("eligible") is not True:
        raise ValueError("attention scoring requires frozen content-neutral eligibility")
    paths = _build_paths(
        tokenizer,
        record,
        candidates,
        content_control,
        config,
        device=device,
    )
    with QwenHistoryKVIntervention(model, block) as kv:
        baseline_values, full_donors = _capture_paths(model, kv, paths)
    with QwenAttentionEdgeIntervention(
        model, block, "zero_additive_delta"
    ) as edge:
        zero_values, zero_summaries = _edge_paths(model, edge, paths)
    with QwenAttentionEdgeIntervention(
        model, block, "history_logits_mask"
    ) as edge:
        mask_values, mask_summaries = _edge_paths(model, edge, paths)
    with QwenAttentionEdgeIntervention(
        model, block, "mask_then_restore_output"
    ) as edge:
        restore_values, restore_summaries = _edge_paths(model, edge, paths)
    with QwenAttentionEdgeIntervention(
        model, block, "history_value_edge_zero"
    ) as edge:
        value_values, value_summaries = _edge_paths(model, edge, paths)
    with QwenHistoryKVIntervention(model, block) as kv:
        same_values = _patch_paths(model, kv, paths, full_donors)
    neutral_paths = _neutralize_paths(paths)
    with QwenHistoryKVIntervention(model, block) as kv:
        _neutral_scores, neutral_donors = _capture_paths(model, kv, neutral_paths)
    with QwenHistoryKVIntervention(model, block) as kv:
        neutral_kv_values = _patch_paths(model, kv, paths, neutral_donors)
    conditions = {
        "baseline_full": _aggregate_paths(paths, baseline_values),
        "zero_delta_identity": _aggregate_paths(paths, zero_values),
        "same_kv_identity": _aggregate_paths(paths, same_values),
        "mask_then_restore_output_identity": _aggregate_paths(
            paths, restore_values
        ),
        "history_logits_mask": _aggregate_paths(paths, mask_values),
        "history_value_edge_zero": _aggregate_paths(paths, value_values),
        "neutral_history_kv": _aggregate_paths(paths, neutral_kv_values),
    }
    for name, values in conditions.items():
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise FloatingPointError(f"attention condition is invalid: {name}")
    return {
        "conditions": conditions,
        "summaries": {
            "zero_delta_identity": zero_summaries,
            "mask_then_restore_output_identity": restore_summaries,
            "history_logits_mask": mask_summaries,
            "history_value_edge_zero": value_summaries,
        },
        "maximum_identity_delta": max(
            float(np.max(np.abs(conditions["zero_delta_identity"] - conditions["baseline_full"]))),
            float(np.max(np.abs(conditions["same_kv_identity"] - conditions["baseline_full"]))),
            float(
                np.max(
                    np.abs(
                        conditions["mask_then_restore_output_identity"]
                        - conditions["baseline_full"]
                    )
                )
            ),
        ),
    }


def _build_paths(
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    content_control: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    method_id = str(config["method_id"])
    if method_id not in {"q2_recranker_generalqwen", "q3_tallrec_generalqwen"}:
        raise ValueError("attention scoring supports only Q2/Q3")
    targets: list[tuple[str, list[int], float]]
    if method_id == "q3_tallrec_generalqwen":
        targets = [
            ("yes", _answer_target_tokens(tokenizer, "Yes"), 1.0),
            ("no", _answer_target_tokens(tokenizer, "No"), -1.0),
        ]
        reserve = max(len(value[1]) for value in targets)
    else:
        targets = [("prompt", [], 1.0)]
        reserve = 0
    prompts = [
        instrument_pointwise_prompt(
            tokenizer,
            method_id,
            record,
            candidate,
            history=record.history,
            history_budget=int(config["training"]["history_budget"]),
            max_length=int(config["training"]["max_length"]) - reserve,
        )
        for candidate in candidates
    ]
    start = int(content_control["history_span_start"])
    end = int(content_control["history_span_end_exclusive"])
    if end - start != int(content_control["history_span_tokens"]):
        raise ValueError("attention content-neutral span length drifted")
    paths: list[dict[str, Any]] = []
    for name, target, weight in targets:
        sequences = [list(prompt.token_ids) + target for prompt in prompts]
        ids, mask, padding = _left_pad_sequences(
            sequences, tokenizer.pad_token_id, device
        )
        positions = []
        for left, prompt in zip(padding, prompts):
            if target:
                positions.append(
                    [left + prompt.candidate_readout + offset for offset in range(len(target))]
                )
            else:
                positions.append([left + prompt.candidate_readout])
        starts = torch.tensor(
            [left + start for left in padding], dtype=torch.long, device=device
        )
        ends = torch.tensor(
            [left + end for left in padding], dtype=torch.long, device=device
        )
        path_positions = torch.tensor(positions, dtype=torch.long, device=device)
        if bool((ends > path_positions.min(dim=1).values).any()):
            raise ValueError("attention frozen history span is not before native readout")
        paths.append(
            {
                "name": name,
                "weight": weight,
                "target": target,
                "ids": ids,
                "mask": mask,
                "positions": path_positions,
                "starts": starts,
                "ends": ends,
            }
        )
    return paths


def _capture_paths(
    model: Any,
    intervention: QwenHistoryKVIntervention,
    paths: Sequence[Mapping[str, Any]],
) -> tuple[list[np.ndarray], list[tuple[tuple[Any, ...], tuple[Any, ...]]]]:
    values: list[np.ndarray] = []
    donors = []
    for path in paths:
        intervention.arm_capture(path["starts"], path["ends"])
        output = model(
            input_ids=path["ids"],
            attention_mask=path["mask"],
            use_cache=False,
            logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
        )
        donors.append(intervention.disarm_capture())
        values.append(_path_scores(output, path))
    return values, donors


def _patch_paths(
    model: Any,
    intervention: QwenHistoryKVIntervention,
    paths: Sequence[Mapping[str, Any]],
    donors: Sequence[tuple[tuple[Any, ...], tuple[Any, ...]]],
) -> list[np.ndarray]:
    values = []
    for path, (keys, donor_values) in zip(paths, donors):
        intervention.arm_patch(path["starts"], path["ends"], keys, donor_values)
        output = model(
            input_ids=path["ids"],
            attention_mask=path["mask"],
            use_cache=False,
            logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
        )
        intervention.disarm_patch()
        values.append(_path_scores(output, path))
    return values


def _edge_paths(
    model: Any,
    intervention: QwenAttentionEdgeIntervention,
    paths: Sequence[Mapping[str, Any]],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    values = []
    summaries = []
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
    return values, summaries


def _neutralize_paths(paths: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        ids = path["ids"].clone()
        for row in range(ids.shape[0]):
            ids[row, int(path["starts"][row]) : int(path["ends"][row])] = (
                CONTENT_NEUTRAL_TOKEN_ID
            )
        result.append({**path, "ids": ids})
    return result


def _path_scores(output: Any, path: Mapping[str, Any]) -> np.ndarray:
    import torch
    from torch.nn import functional as F

    target = path["target"]
    if not target:
        logits = output.logits[:, -1]
        # IDs are attached by the caller-independent model tokenizer contract;
        # Q2 frozen IDs are asserted in the deep-dive manifest.
        yes_id, no_id = 9693, 2152
        return (logits[:, yes_id] - logits[:, no_id]).float().cpu().numpy()
    logits = output.logits[:, -(len(target) + 1) : -1].float()
    expected = torch.tensor(target, dtype=torch.long, device=logits.device)
    expected = expected[None, :, None].expand(logits.shape[0], -1, -1)
    return (
        F.log_softmax(logits, dim=-1)
        .gather(2, expected)
        .squeeze(2)
        .mean(dim=1)
        .cpu()
        .numpy()
    )


def _aggregate_paths(
    paths: Sequence[Mapping[str, Any]], values: Sequence[np.ndarray]
) -> np.ndarray:
    result = np.zeros_like(values[0], dtype=np.float32)
    for path, path_values in zip(paths, values):
        result += float(path["weight"]) * np.asarray(path_values, dtype=np.float32)
    return result
