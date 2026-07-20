"""Frozen-sample causal localization of GQA groups and attention formation edges."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.attention_edge_interventions import QwenAttentionEdgeIntervention
from myrec.mechanism.attention_edge_scoring import _aggregate_paths, _path_scores
from myrec.mechanism.attention_group_interventions import (
    GQA_GROUPS,
    QwenAttentionGQAIntervention,
    QwenHistoryKVGroupIntervention,
)
from myrec.mechanism.attention_observation_runtime import _build_observation_paths
from myrec.mechanism.deep_dive_assignments import CONTENT_NEUTRAL_TOKEN_ID
from myrec.mechanism.history_kv_interventions import QwenHistoryKVIntervention


GROUP_CONDITIONS = (
    "same_kv_identity",
    "mask_then_restore_identity",
    "history_to_readout_logits_mask",
    "history_to_readout_value_zero",
    "neutral_history_kv",
)
SUPPLEMENTAL_CONDITIONS = (
    "baseline_full",
    "query_to_history_logits_mask",
    "query_to_history_value_zero",
    "query_to_history_mask_restore_identity",
    "cross_request_history_summary_kv",
)


def score_attention_group_sample_row(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidate_ordinal: int,
    donor_record: ModelRecord,
    donor_candidate_ordinal: int,
    content_control: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    """Score all eight groups plus fixed formation/cross-summary controls."""

    paths = _build_observation_paths(
        tokenizer, record, candidate_ordinal, config, device=device
    )
    donor_paths = _build_observation_paths(
        tokenizer,
        donor_record,
        donor_candidate_ordinal,
        config,
        device=device,
    )
    baseline = _baseline_paths(model, paths)
    full_kv = _capture_kv_paths(model, paths, block, span="history")
    neutral_eligible = content_control.get("eligible") is True
    neutral_kv = None
    if neutral_eligible:
        _audit_frozen_neutral_span(paths, content_control)
        neutral_paths = _neutral_history_paths(paths, content_control)
        neutral_kv = _capture_kv_paths(model, neutral_paths, block, span="history")

    groups = []
    maximum_identity = 0.0
    for group in range(GQA_GROUPS):
        values = {
            "same_kv_identity": _patch_kv_group_paths(
                model, paths, block, group, full_kv, span="history"
            ),
            "mask_then_restore_identity": _edge_paths(
                model,
                paths,
                block,
                mode="mask_then_restore_output",
                query_scope="native_readout",
                span="history",
                group=group,
            )[0],
            "history_to_readout_logits_mask": _edge_paths(
                model,
                paths,
                block,
                mode="history_logits_mask",
                query_scope="native_readout",
                span="history",
                group=group,
            )[0],
            "history_to_readout_value_zero": _edge_paths(
                model,
                paths,
                block,
                mode="history_value_edge_zero",
                query_scope="native_readout",
                span="history",
                group=group,
            )[0],
            "neutral_history_kv": (
                _patch_kv_group_paths(
                    model, paths, block, group, neutral_kv, span="history"
                )
                if neutral_kv is not None
                else baseline
            ),
        }
        identity = max(
            abs(values["same_kv_identity"] - baseline),
            abs(values["mask_then_restore_identity"] - baseline),
        )
        maximum_identity = max(maximum_identity, identity)
        groups.append(
            {
                "gqa_group": group,
                "query_heads": [2 * group, 2 * group + 1],
                "conditions": values,
                "maximum_identity_delta": identity,
            }
        )

    formation = {}
    formation_summaries = {}
    for name, mode in (
        ("query_to_history_logits_mask", "history_logits_mask"),
        ("query_to_history_value_zero", "history_value_edge_zero"),
        ("query_to_history_mask_restore_identity", "mask_then_restore_output"),
    ):
        formation[name], formation_summaries[name] = _edge_paths(
            model,
            paths,
            block,
            mode=mode,
            query_scope="history_summary",
            span="query",
            group=None,
        )
    recipient_summary_kv = _capture_kv_paths(
        model, paths, block, span="history_summary"
    )
    donor_summary_kv = _capture_kv_paths(
        model, donor_paths, block, span="history_summary"
    )
    cross_kv = _selected_cross_donors(
        recipient_summary_kv,
        donor_summary_kv,
        recipient_row=_selected_batch_row(paths),
        donor_row=_selected_batch_row(donor_paths),
    )
    cross = _patch_kv_paths(
        model, paths, block, cross_kv, span="history_summary"
    )
    supplemental = {
        "baseline_full": baseline,
        **formation,
        "cross_request_history_summary_kv": cross,
    }
    maximum_identity = max(
        maximum_identity,
        abs(formation["query_to_history_mask_restore_identity"] - baseline),
    )
    if (
        len(groups) != GQA_GROUPS
        or any(set(row["conditions"]) != set(GROUP_CONDITIONS) for row in groups)
        or set(supplemental) != set(SUPPLEMENTAL_CONDITIONS)
        or not _all_finite({"groups": groups, "supplemental": supplemental})
    ):
        raise FloatingPointError("attention GQA condition coverage is invalid")
    return {
        "groups": groups,
        "supplemental": supplemental,
        "formation_summaries": formation_summaries,
        "neutral_history_eligible": neutral_eligible,
        "maximum_identity_delta": maximum_identity,
    }


def _baseline_paths(model, paths):
    descriptors = []
    values = []
    for pair in paths:
        path = pair["full"]
        output = model(
            input_ids=path["ids"],
            attention_mask=path["mask"],
            use_cache=False,
            logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
        )
        descriptors.append(path)
        values.append(_path_scores(output, path))
    return float(
        _aggregate_paths(descriptors, values)[_selected_batch_row(paths)]
    )


def _edge_paths(model, paths, block, *, mode, query_scope, span, group):
    intervention_type = (
        QwenAttentionEdgeIntervention
        if group is None
        else QwenAttentionGQAIntervention
    )
    arguments = (model, block, mode) if group is None else (model, block, mode, group)
    descriptors = []
    values = []
    summaries = []
    with intervention_type(*arguments) as intervention:
        for pair in paths:
            path = pair["full"]
            starts, ends = path["spans"][span]
            intervention.arm(
                path["query_positions"][query_scope],
                starts,
                ends,
                sequence_length=path["ids"].shape[1],
            )
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            summaries.append(intervention.disarm())
            descriptors.append(path)
            values.append(_path_scores(output, path))
    return (
        float(
            _aggregate_paths(descriptors, values)[_selected_batch_row(paths)]
        ),
        summaries,
    )


def _capture_kv_paths(model, paths, block, *, span):
    donors = []
    with QwenHistoryKVIntervention(model, block) as intervention:
        for pair in paths:
            path = pair["full"]
            starts, ends = _kv_span(path, span)
            intervention.arm_capture(starts, ends)
            model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            donors.append(intervention.disarm_capture())
    return donors


def _patch_kv_group_paths(model, paths, block, group, donors, *, span):
    descriptors = []
    values = []
    with QwenHistoryKVGroupIntervention(model, block, group) as intervention:
        for pair, (keys, donor_values) in zip(paths, donors):
            path = pair["full"]
            starts, ends = _kv_span(path, span)
            intervention.arm_patch(starts, ends, keys, donor_values)
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            intervention.disarm_patch()
            descriptors.append(path)
            values.append(_path_scores(output, path))
    return float(
        _aggregate_paths(descriptors, values)[_selected_batch_row(paths)]
    )


def _patch_kv_paths(model, paths, block, donors, *, span):
    descriptors = []
    values = []
    with QwenHistoryKVIntervention(model, block) as intervention:
        for pair, (keys, donor_values) in zip(paths, donors):
            path = pair["full"]
            starts, ends = _kv_span(path, span)
            intervention.arm_patch(starts, ends, keys, donor_values)
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            intervention.disarm_patch()
            descriptors.append(path)
            values.append(_path_scores(output, path))
    return float(
        _aggregate_paths(descriptors, values)[_selected_batch_row(paths)]
    )


def _neutral_history_paths(paths, content_control):
    result = []
    for pair in paths:
        path = pair["full"]
        ids = path["ids"].clone()
        starts, ends = path["spans"]["history"]
        for row in range(ids.shape[0]):
            ids[row, int(starts[row]) : int(ends[row])] = CONTENT_NEUTRAL_TOKEN_ID
        result.append(
            {
                "name": pair["name"],
                "selected_batch_row": pair["selected_batch_row"],
                "full": {**path, "ids": ids},
            }
        )
    return result


def _audit_frozen_neutral_span(paths, content_control):
    registered = (
        int(content_control["history_span_start"]),
        int(content_control["history_span_end_exclusive"]),
    )
    if registered[1] - registered[0] != int(content_control["history_span_tokens"]):
        raise ValueError("frozen content-neutral span length differs")
    for pair in paths:
        path = pair["full"]
        starts, ends = path["spans"]["history"]
        padding = path["left_padding"]
        observed = [
            (int(start - left), int(end - left))
            for start, end, left in zip(starts, ends, padding)
        ]
        if any(bounds != registered for bounds in observed):
            raise ValueError("attention GQA semantic history span differs from frozen control")


def _selected_batch_row(paths):
    rows = {int(pair["selected_batch_row"]) for pair in paths}
    if len(rows) != 1:
        raise ValueError("attention GQA selected batch row differs across native paths")
    return next(iter(rows))


def _selected_cross_donors(
    recipient_donors,
    donor_donors,
    *,
    recipient_row,
    donor_row,
):
    if len(recipient_donors) != len(donor_donors):
        raise ValueError("attention GQA cross native path count differs")
    result = []
    for (recipient_keys, recipient_values), (donor_keys, donor_values) in zip(
        recipient_donors,
        donor_donors,
    ):
        if not 0 <= recipient_row < len(recipient_keys):
            raise ValueError("attention GQA recipient batch row is outside capture")
        if not 0 <= donor_row < len(donor_keys):
            raise ValueError("attention GQA donor batch row is outside capture")
        keys = list(recipient_keys)
        values = list(recipient_values)
        keys[recipient_row] = donor_keys[donor_row]
        values[recipient_row] = donor_values[donor_row]
        result.append((tuple(keys), tuple(values)))
    return result


def _kv_span(path, span):
    if span == "history_summary":
        starts = path["query_positions"]["history_summary"][:, 0]
        return starts, starts + 1
    return path["spans"][span]


def _all_finite(value):
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    return True
