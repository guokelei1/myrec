"""Q2/Q3 qrels-blind kernels for the registered N27 mask/softmax family."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.baselines.motivation_v12_ranker import _answer_target_tokens
from myrec.mechanism.attention_edge_scoring import (
    _aggregate_paths,
    _path_scores,
)
from myrec.mechanism.mask_softmax_interventions import (
    MASK_SOFTMAX_MODES,
    QwenMaskSoftmaxIntervention,
)
from myrec.mechanism.patch_scorer import _left_pad_sequences
from myrec.mechanism.representation_probe import instrument_pointwise_prompt


MASK_SOFTMAX_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_visibility_identity",
    "null_visibility_identity",
    "full_prefix_history_swap",
    "null_prefix_history_swap",
    "full_candidate_visibility_swap",
    "null_candidate_visibility_swap",
    "full_temperature_half",
    "null_temperature_half",
    "full_temperature_double",
    "null_temperature_double",
)


def build_mask_paths(
    tokenizer: Any,
    method_id: str,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    content_control: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: str,
) -> list[dict[str, Any]]:
    """Build full/null paths plus explicit candidate spans for N27 masks."""

    import torch

    if method_id not in {"q2_recranker_generalqwen", "q3_tallrec_generalqwen"}:
        raise ValueError("N27 supports only Q2/Q3")
    if content_control.get("eligible") is not True:
        raise ValueError("N27 requires content-neutral eligibility")
    if method_id == "q3_tallrec_generalqwen":
        targets = [
            ("yes", _answer_target_tokens(tokenizer, "Yes"), 1.0),
            ("no", _answer_target_tokens(tokenizer, "No"), -1.0),
        ]
        reserve = max(len(target) for _, target, _ in targets)
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
        raise ValueError("N27 history span length drifted")
    paths: list[dict[str, Any]] = []
    for name, target, weight in targets:
        sequences = [list(prompt.token_ids) + target for prompt in prompts]
        ids, mask, padding = _left_pad_sequences(sequences, tokenizer.pad_token_id, device)
        positions = []
        starts, ends, candidate_starts, candidate_ends = [], [], [], []
        for left, prompt in zip(padding, prompts):
            if target:
                positions.append([
                    left + prompt.candidate_readout + offset
                    for offset in range(len(target))
                ])
            else:
                positions.append([left + prompt.candidate_readout])
            starts.append(left + start)
            ends.append(left + end)
            candidate_starts.append(left + prompt.candidate_start)
            candidate_ends.append(left + prompt.candidate_start + prompt.candidate_tokens)
        path_positions = torch.tensor(positions, dtype=torch.long, device=device)
        path = {
            "name": name,
            "weight": weight,
            "target": target,
            "ids": ids,
            "mask": mask,
            "positions": path_positions,
            "starts": torch.tensor(starts, dtype=torch.long, device=device),
            "ends": torch.tensor(ends, dtype=torch.long, device=device),
            "candidate_starts": torch.tensor(candidate_starts, dtype=torch.long, device=device),
            "candidate_ends": torch.tensor(candidate_ends, dtype=torch.long, device=device),
        }
        if bool((path["ends"] > path_positions.min(dim=1).values).any()):
            raise ValueError("N27 history span is not before native readout")
        if bool((path["candidate_ends"] <= path["candidate_starts"]).any()):
            raise ValueError("N27 candidate span is empty")
        paths.append(path)
    return paths


def build_visibility_masks(path: Mapping[str, Any]) -> dict[str, Any]:
    """Construct native-safe prefix/history and candidate visibility masks."""

    import torch

    positions = path["positions"]
    mask = path["mask"].to(dtype=torch.bool)
    starts = path["starts"]
    ends = path["ends"]
    candidate_starts = path["candidate_starts"]
    candidate_ends = path["candidate_ends"]
    if bool((candidate_ends <= candidate_starts).any()):
        raise ValueError("N27 candidate span is empty")
    sequence_length = int(path["ids"].shape[1])
    keys = torch.arange(sequence_length, device=positions.device)[None, None, :]
    native = []
    prefix = []
    candidate = []
    for row in range(int(positions.shape[0])):
        row_native, row_prefix, row_candidate = [], [], []
        for query_position in positions[row].tolist():
            allowed = mask[row] & (keys[0, 0] <= int(query_position))
            history_allowed = allowed.clone()
            history_allowed[int(starts[row]) : int(ends[row])] = False
            history_allowed[int(query_position)] = True
            candidate_allowed = allowed.clone()
            candidate_allowed[int(candidate_starts[row]) : int(candidate_ends[row])] = False
            candidate_allowed[int(query_position)] = True
            if not bool(allowed.any() and history_allowed.any() and candidate_allowed.any()):
                raise ValueError("N27 visibility alternative removes every causal key")
            row_native.append(allowed)
            row_prefix.append(history_allowed)
            row_candidate.append(candidate_allowed)
        native.append(torch.stack(row_native))
        prefix.append(torch.stack(row_prefix))
        candidate.append(torch.stack(row_candidate))
    return {
        "native": torch.stack(native),
        "prefix_history_swap": torch.stack(prefix),
        "candidate_visibility_swap": torch.stack(candidate),
    }


def score_mask_softmax_chunk(
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
    """Score all registered N27 conditions for one candidate chunk."""

    if not candidates:
        raise ValueError("N27 candidate chunk is empty")
    method_id = str(config["method_id"])
    full_paths = build_mask_paths(
        tokenizer, method_id, record, candidates, content_control, config, device=device
    )
    null_paths = [
        {**path, "ids": _neutralize_history(path["ids"], path["starts"], path["ends"])}
        for path in full_paths
    ]
    conditions: dict[str, np.ndarray] = {
        "baseline_full": _run_native(model, full_paths),
        "baseline_null": _run_native(model, null_paths),
    }
    summaries: dict[str, Any] = {}
    for mode in MASK_SOFTMAX_MODES:
        if mode == "identity":
            full_values = conditions["baseline_full"]
            null_values = conditions["baseline_null"]
            full_summary = {"mode": mode, "maximum_applied_delta": 0.0}
            null_summary = full_summary
        else:
            full_values, full_summary = _run_mode(model, full_paths, block, mode)
            null_values, null_summary = _run_mode(model, null_paths, block, mode)
        if mode == "identity":
            full_name, null_name = "full_visibility_identity", "null_visibility_identity"
        else:
            full_name, null_name = f"full_{mode}", f"null_{mode}"
        conditions[full_name] = full_values
        conditions[null_name] = null_values
        summaries[full_name] = full_summary
        summaries[null_name] = null_summary
    _validate_conditions(conditions, len(candidates))
    identity = max(
        float(np.max(np.abs(conditions["full_visibility_identity"] - conditions["baseline_full"]))),
        float(np.max(np.abs(conditions["null_visibility_identity"] - conditions["baseline_null"]))),
    )
    return {
        "conditions": conditions,
        "summaries": summaries,
        "maximum_identity_delta": identity,
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


def _run_mode(model: Any, paths: Sequence[Mapping[str, Any]], block: int, mode: str):
    values, summaries = [], []
    with QwenMaskSoftmaxIntervention(model, block, mode) as intervention:
        for path in paths:
            masks = build_visibility_masks(path)
            allowed = masks[mode] if mode in masks else None
            intervention.arm(
                path["positions"],
                allowed_keys=allowed,
                sequence_length=int(path["ids"].shape[1]),
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


def _neutralize_history(ids: Any, starts: Any, ends: Any) -> Any:
    from myrec.mechanism.deep_dive_assignments import CONTENT_NEUTRAL_TOKEN_ID

    result = ids.clone()
    for row in range(int(ids.shape[0])):
        result[row, int(starts[row]) : int(ends[row])] = CONTENT_NEUTRAL_TOKEN_ID
    return result


def _path_scores(output: Any, path: Mapping[str, Any]) -> np.ndarray:
    from myrec.mechanism.attention_edge_scoring import _path_scores as score

    return score(output, path)


def _validate_conditions(conditions: Mapping[str, np.ndarray], expected_count: int) -> None:
    if set(conditions) != set(MASK_SOFTMAX_CONDITIONS):
        raise ValueError("N27 condition set differs from registered manifest")
    for name, values in conditions.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (expected_count,) or not np.isfinite(values).all():
            raise FloatingPointError(f"N27 condition is invalid: {name}")
