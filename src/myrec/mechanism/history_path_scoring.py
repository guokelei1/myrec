"""Qrels-blind history formation/transport path interventions for Q2/Q3."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.baselines.motivation_v12_ranker import _answer_target_tokens
from myrec.mechanism.attention_edge_interventions import QwenAttentionEdgeIntervention
from myrec.mechanism.attention_edge_scoring import _aggregate_paths, _path_scores
from myrec.mechanism.attention_observation_runtime import _semantic_spans
from myrec.mechanism.patch_scorer import _left_pad_sequences
from myrec.mechanism.representation_probe import instrument_pointwise_prompt


N9_SCORE_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "formation_logits_mask",
    "formation_value_zero",
    "transport_logits_mask",
    "transport_value_zero",
    "formation_transport_joint",
    "no_op_identity",
)


def score_history_path_chunk(
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
    """Score one candidate chunk for formation, transport, and their joint path."""

    if not candidates:
        raise ValueError("history path candidate chunk is empty")
    if content_control.get("eligible") is not True:
        raise ValueError("history path scoring requires frozen content eligibility")
    full_paths = _build_paths(
        tokenizer,
        record,
        candidates,
        config,
        history=record.history,
        content_control=content_control,
        device=device,
    )
    null_paths = _build_paths(
        tokenizer,
        record,
        candidates,
        config,
        history=[],
        content_control=None,
        device=device,
    )

    baseline_full = _native_paths(model, full_paths)
    baseline_null = _native_paths(model, null_paths)
    formation_logits = _scope_paths(
        model, full_paths, block, scope="formation", mode="history_logits_mask"
    )[0]
    formation_value = _scope_paths(
        model, full_paths, block, scope="formation", mode="history_value_edge_zero"
    )[0]
    transport_logits = _scope_paths(
        model, full_paths, block, scope="transport", mode="history_logits_mask"
    )[0]
    transport_value = _scope_paths(
        model, full_paths, block, scope="transport", mode="history_value_edge_zero"
    )[0]
    joint, joint_summary = _joint_paths(
        model, full_paths, block, formation_mode="history_logits_mask", transport_mode="history_logits_mask"
    )
    no_op, no_op_summary = _joint_paths(
        model, full_paths, block, formation_mode="zero_additive_delta", transport_mode="zero_additive_delta"
    )
    conditions = {
        "baseline_full": baseline_full,
        "baseline_null": baseline_null,
        "formation_logits_mask": formation_logits,
        "formation_value_zero": formation_value,
        "transport_logits_mask": transport_logits,
        "transport_value_zero": transport_value,
        "formation_transport_joint": joint,
        "no_op_identity": no_op,
    }
    for name, values in conditions.items():
        values = np.asarray(values)
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise FloatingPointError(f"history path condition is invalid: {name}")
    return {
        "conditions": conditions,
        "maximum_identity_delta": float(np.max(np.abs(no_op - baseline_full))),
        "maximum_manual_error": float(
            max(joint_summary["maximum_manual_error"], no_op_summary["maximum_manual_error"])
        ),
        "maximum_applied_delta": float(
            max(
                joint_summary["maximum_applied_delta"],
                no_op_summary["maximum_applied_delta"],
            )
        ),
    }


def _build_paths(
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]],
    content_control: Mapping[str, Any] | None,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    method_id = str(config["method_id"])
    if method_id == "q3_tallrec_generalqwen":
        targets = [
            ("yes", _answer_target_tokens(tokenizer, "Yes"), 1.0),
            ("no", _answer_target_tokens(tokenizer, "No"), -1.0),
        ]
        reserve = max(len(target) for _, target, _ in targets)
    elif method_id == "q2_recranker_generalqwen":
        targets = [("prompt", [], 1.0)]
        reserve = 0
    else:
        raise ValueError("history path scoring supports only Q2/Q3")
    prompts = [
        instrument_pointwise_prompt(
            tokenizer,
            method_id,
            record,
            candidate,
            history=list(history),
            history_budget=int(config["training"]["history_budget"]),
            max_length=int(config["training"]["max_length"]) - reserve,
        )
        for candidate in candidates
    ]
    semantic = [
        _semantic_spans(
            tokenizer,
            method_id,
            record,
            candidate,
            list(history),
            prompt,
            config,
        )
        for candidate, prompt in zip(candidates, prompts)
    ]
    if content_control is not None:
        registered = (
            int(content_control["history_span_start"]),
            int(content_control["history_span_end_exclusive"]),
        )
        if registered[1] - registered[0] != int(content_control["history_span_tokens"]):
            raise ValueError("history path frozen history span length drifted")
        for spans in semantic:
            if tuple(spans["history"]) != registered:
                raise ValueError("history path semantic history span differs from frozen control")

    # Build one tensor per native answer path, preserving the Q3 shared prompt and
    # teacher-forced yes/no paths exactly as the existing attention scorer does.
    result: list[dict[str, Any]] = []
    for name, target, weight in targets:
        ids, mask, padding = _left_pad_sequences(
            [list(prompt.token_ids) + list(target) for prompt in prompts],
            tokenizer.pad_token_id,
            device,
        )
        positions = torch.tensor(
            [
                [
                    left + prompt.candidate_readout + offset
                    for offset in range(max(1, len(target)))
                ]
                for left, prompt in zip(padding, prompts)
            ],
            dtype=torch.long,
            device=device,
        )
        formation_positions = torch.tensor(
            [[left + prompt.history_summary_end] for left, prompt in zip(padding, prompts)],
            dtype=torch.long,
            device=device,
        )
        formation_starts = torch.tensor(
            [left + spans["query"][0] for left, spans in zip(padding, semantic)],
            dtype=torch.long,
            device=device,
        )
        formation_ends = torch.tensor(
            [left + spans["query"][1] for left, spans in zip(padding, semantic)],
            dtype=torch.long,
            device=device,
        )
        transport_starts = torch.tensor(
            [left + spans["history"][0] for left, spans in zip(padding, semantic)],
            dtype=torch.long,
            device=device,
        )
        transport_ends = torch.tensor(
            [left + spans["history"][1] for left, spans in zip(padding, semantic)],
            dtype=torch.long,
            device=device,
        )
        if bool((formation_ends > formation_positions[:, 0]).any()):
            raise ValueError("formation query span is not before history summary readout")
        if bool((transport_ends > positions.min(dim=1).values).any()):
            raise ValueError("transport history span is not before candidate readout")
        result.append(
            {
                "name": name,
                "weight": float(weight),
                "target": list(target),
                "ids": ids,
                "mask": mask,
                "positions": positions,
                "formation_positions": formation_positions,
                "formation_starts": formation_starts,
                "formation_ends": formation_ends,
                "transport_starts": transport_starts,
                "transport_ends": transport_ends,
            }
        )
    return result


def _native_paths(model: Any, paths: Sequence[Mapping[str, Any]]) -> np.ndarray:
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


def _scope_paths(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    block: int,
    *,
    scope: str,
    mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    if scope not in {"formation", "transport"}:
        raise ValueError("history path scope is invalid")
    descriptors = []
    values = []
    maximum_applied = 0.0
    maximum_error = 0.0
    with QwenAttentionEdgeIntervention(model, block, mode) as intervention:
        for path in paths:
            if scope == "formation":
                positions = path["formation_positions"]
                starts, ends = path["formation_starts"], path["formation_ends"]
            else:
                positions = path["positions"]
                starts, ends = path["transport_starts"], path["transport_ends"]
            intervention.arm(positions, starts, ends, sequence_length=path["ids"].shape[1])
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            summary = intervention.disarm()
            maximum_applied = max(maximum_applied, float(summary.get("maximum_applied_delta", 0.0)))
            maximum_error = max(maximum_error, float(summary.get("manual_baseline_native_max_abs_error") or 0.0))
            descriptors.append(path)
            values.append(_path_scores(output, path))
    return _aggregate_paths(descriptors, values), {
        "maximum_applied_delta": maximum_applied,
        "maximum_manual_error": maximum_error,
    }


def _joint_paths(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    block: int,
    *,
    formation_mode: str,
    transport_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    descriptors = []
    values = []
    maximum_applied = 0.0
    maximum_error = 0.0
    with QwenAttentionEdgeIntervention(model, block, formation_mode) as formation:
        with QwenAttentionEdgeIntervention(model, block, transport_mode) as transport:
            for path in paths:
                formation.arm(
                    path["formation_positions"],
                    path["formation_starts"],
                    path["formation_ends"],
                    sequence_length=path["ids"].shape[1],
                )
                transport.arm(
                    path["positions"],
                    path["transport_starts"],
                    path["transport_ends"],
                    sequence_length=path["ids"].shape[1],
                )
                output = model(
                    input_ids=path["ids"],
                    attention_mask=path["mask"],
                    use_cache=False,
                    logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
                )
                transport_summary = transport.disarm()
                formation_summary = formation.disarm()
                for summary in (formation_summary, transport_summary):
                    maximum_applied = max(maximum_applied, float(summary.get("maximum_applied_delta", 0.0)))
                    maximum_error = max(maximum_error, float(summary.get("manual_baseline_native_max_abs_error") or 0.0))
                descriptors.append(path)
                values.append(_path_scores(output, path))
    return _aggregate_paths(descriptors, values), {
        "maximum_applied_delta": maximum_applied,
        "maximum_manual_error": maximum_error,
    }
