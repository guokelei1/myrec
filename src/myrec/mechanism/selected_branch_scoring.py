"""Composition-safe selected-block branch interventions for Q2/Q3.

This module contains model-forward primitives only.  It reads no records,
manifests, scores, or qrels and performs no statistical selection.
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.deep_dive_native_patch import (
    _combine_terms,
    _path_terms,
    _q3_context,
)
from myrec.mechanism.native_readout_scoring import build_q2_pointwise_batch
from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
    QwenNodePatch,
    QwenPostAttentionStatePatch,
    rms_matched_random_direction,
)


SELECTED_NODES = (
    "block_input_residual",
    "input_rmsnorm_output",
    "attention_o_projection",
    "post_attention_residual",
    "post_attention_rmsnorm_output",
    "mlp_down_projection",
    "block_output_residual",
)
NODE_INTERVENTIONS = (
    "full_to_full_identity",
    "null_to_null_identity",
    "same_full_to_null",
    "cross_full_to_null",
    "wrong_history_to_null",
    "donor_direction_at_recipient_rms",
    "recipient_direction_at_donor_rms",
    "random_direction_at_recipient_rms",
)
RANDOM_DIRECTION_SEED = 20_260_715


def selected_branch_conditions() -> tuple[str, ...]:
    return (
        "baseline_full",
        "baseline_null",
        *(
            f"{node}.{condition}"
            for node in SELECTED_NODES
            for condition in NODE_INTERVENTIONS
        ),
    )


def score_selected_branch_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    cross_record: ModelRecord,
    cross_candidates: Sequence[Mapping[str, Any]],
    wrong_history: Sequence[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    """Score every registered selected-branch condition for one chunk."""

    if not candidates or len(candidates) != len(cross_candidates):
        raise ValueError("selected-branch recipient/cross chunk is empty or misaligned")
    block = int(block)
    if not 13 <= block <= 27:
        raise ValueError("selected-branch block must be in [13,27]")
    method_id = str(config.get("method_id"))
    if method_id == "q2_recranker_generalqwen":
        return _score_q2(
            model,
            tokenizer,
            record,
            candidates,
            cross_record,
            cross_candidates,
            wrong_history,
            config,
            block=block,
            device=device,
        )
    if method_id == "q3_tallrec_generalqwen":
        return _score_q3(
            model,
            tokenizer,
            record,
            candidates,
            cross_record,
            cross_candidates,
            wrong_history,
            config,
            block=block,
            device=device,
        )
    raise ValueError("selected-branch scoring supports only Q2/Q3")


def _score_q2(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    cross_record: ModelRecord,
    cross_candidates: Sequence[Mapping[str, Any]],
    wrong_history: Sequence[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    specs = tuple(NodeSpec(node, block) for node in SELECTED_NODES)
    full_batch = build_q2_pointwise_batch(
        tokenizer, record, candidates, record.history, config, device=device
    )
    null_batch = build_q2_pointwise_batch(
        tokenizer, record, candidates, [], config, device=device
    )
    cross_batch = build_q2_pointwise_batch(
        tokenizer,
        cross_record,
        cross_candidates,
        cross_record.history,
        config,
        device=device,
    )
    wrong_batch = (
        build_q2_pointwise_batch(
            tokenizer, record, candidates, wrong_history, config, device=device
        )
        if wrong_history is not None
        else None
    )
    with QwenNodeCapture(model, specs) as capture:
        full = _capture_q2(model, capture, full_batch)
        null = _capture_q2(model, capture, null_batch)
        cross = _capture_q2(model, capture, cross_batch)
        wrong = _capture_q2(model, capture, wrong_batch) if wrong_batch is not None else None

    conditions: dict[str, Any] = {
        "baseline_full": full["score"],
        "baseline_null": null["score"],
    }
    identity = 0.0
    direction_geometry = {}
    for spec in specs:
        node = spec.node_id
        full_state = full["states"][spec.key]
        null_state = null["states"][spec.key]
        controls, geometry = _direction_controls(
            full_state,
            null_state,
            identity_keys=_identity_keys(record, candidates, node, block, ("native_readout",)),
        )
        direction_geometry[node] = geometry
        donors = {
            "full_to_full_identity": (full_batch, full_state),
            "null_to_null_identity": (null_batch, null_state),
            "same_full_to_null": (null_batch, full_state),
            "cross_full_to_null": (null_batch, cross["states"][spec.key]),
            "wrong_history_to_null": (
                null_batch,
                null_state if wrong is None else wrong["states"][spec.key],
            ),
            **{name: (null_batch, value) for name, value in controls.items()},
        }
        for condition, (batch, donor_state) in donors.items():
            if condition == "wrong_history_to_null" and wrong is None:
                score = null["score"]
            else:
                score = _patch_q2(model, spec, batch, donor_state)
            conditions[f"{node}.{condition}"] = score
        identity = max(
            identity,
            float(
                (
                    conditions[f"{node}.full_to_full_identity"] - full["score"]
                ).abs().max().item()
            ),
            float(
                (
                    conditions[f"{node}.null_to_null_identity"] - null["score"]
                ).abs().max().item()
            ),
        )
    _validate_conditions(conditions, len(candidates))
    return {
        "conditions": conditions,
        "maximum_identity_delta": identity,
        "direction_geometry": direction_geometry,
        "wrong_history_eligible": wrong is not None,
    }


def _score_q3(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    cross_record: ModelRecord,
    cross_candidates: Sequence[Mapping[str, Any]],
    wrong_history: Sequence[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    specs = tuple(NodeSpec(node, block) for node in SELECTED_NODES)
    full_context = _q3_context(
        tokenizer, record, candidates, record.history, config, device
    )
    null_context = _q3_context(tokenizer, record, candidates, [], config, device)
    cross_context = _q3_context(
        tokenizer,
        cross_record,
        cross_candidates,
        cross_record.history,
        config,
        device,
    )
    wrong_context = (
        _q3_context(tokenizer, record, candidates, wrong_history, config, device)
        if wrong_history is not None
        else None
    )
    with QwenNodeCapture(model, specs) as capture:
        full = _capture_q3(model, capture, full_context, specs)
        null = _capture_q3(model, capture, null_context, specs)
        cross = _capture_q3(model, capture, cross_context, specs)
        wrong = (
            _capture_q3(model, capture, wrong_context, specs)
            if wrong_context is not None
            else None
        )
    conditions: dict[str, Any] = {
        "baseline_full": full["score"],
        "baseline_null": null["score"],
    }
    identity = 0.0
    direction_geometry = {}
    shared_prompt_delta = 0.0
    for spec in specs:
        node = spec.node_id
        for capture_result in (full, null, cross, wrong):
            if capture_result is None:
                continue
            states = capture_result["states"][spec.key]
            shared_prompt_delta = max(
                shared_prompt_delta,
                float(
                    (states["yes"][:, 0] - states["no"][:, 0])
                    .abs()
                    .max()
                    .item()
                ),
            )
        controls, geometry = _q3_direction_controls(
            full["states"][spec.key],
            null["states"][spec.key],
            record,
            candidates,
            node=node,
            block=block,
        )
        direction_geometry[node] = geometry
        donors = {
            "full_to_full_identity": (full_context, full["states"][spec.key]),
            "null_to_null_identity": (null_context, null["states"][spec.key]),
            "same_full_to_null": (null_context, full["states"][spec.key]),
            "cross_full_to_null": (null_context, cross["states"][spec.key]),
            "wrong_history_to_null": (
                null_context,
                null["states"][spec.key]
                if wrong is None
                else wrong["states"][spec.key],
            ),
            **{name: (null_context, value) for name, value in controls.items()},
        }
        for condition, (context, donor_state) in donors.items():
            if condition == "wrong_history_to_null" and wrong is None:
                score = null["score"]
            else:
                score = _patch_q3(model, spec, context, donor_state)
            conditions[f"{node}.{condition}"] = score
        identity = max(
            identity,
            float(
                (
                    conditions[f"{node}.full_to_full_identity"] - full["score"]
                ).abs().max().item()
            ),
            float(
                (
                    conditions[f"{node}.null_to_null_identity"] - null["score"]
                ).abs().max().item()
            ),
        )
    if shared_prompt_delta != 0.0:
        raise RuntimeError(
            "selected-branch Q3 shared prompt state differs across native paths: "
            f"{shared_prompt_delta}"
        )
    _validate_conditions(conditions, len(candidates))
    return {
        "conditions": conditions,
        "maximum_identity_delta": identity,
        "shared_prompt_path_max_abs_delta": shared_prompt_delta,
        "direction_geometry": direction_geometry,
        "wrong_history_eligible": wrong is not None,
    }


def _capture_q2(model: Any, capture: QwenNodeCapture, batch: Any) -> dict[str, Any]:
    ids, mask, positions = batch
    output, states = capture.capture_forward(
        input_ids=ids,
        attention_mask=mask,
        positions=positions,
        model_kwargs={"logits_to_keep": 1},
    )
    logits = output.logits[:, -1].float()
    return {"score": logits[:, 9693] - logits[:, 2152], "states": states}


def _capture_q3(
    model: Any,
    capture: QwenNodeCapture,
    context: Mapping[str, Any],
    specs: Sequence[NodeSpec],
) -> dict[str, Any]:
    branch_states = {spec.key: {} for spec in specs}
    terms = []
    for branch in ("yes", "no"):
        path = context["paths"][branch]
        output, states = capture.capture_forward(
            input_ids=path["ids"],
            attention_mask=path["mask"],
            positions=path["positions"],
            model_kwargs={"logits_to_keep": 3},
        )
        for spec in specs:
            branch_states[spec.key][branch] = states[spec.key]
        terms.append(_path_terms(output, path))
    term_matrix, score = _combine_terms(terms[0], terms[1])
    torch = _torch()
    return {
        "terms": torch.as_tensor(term_matrix, dtype=torch.float32, device=path["ids"].device),
        "score": torch.as_tensor(score, dtype=torch.float32, device=path["ids"].device),
        "states": branch_states,
    }


def _patch_q2(model: Any, spec: NodeSpec, batch: Any, donor: Any) -> Any:
    ids, mask, positions = batch
    with _patcher(model, spec) as patch:
        patch.arm(positions, donor, sequence_length=int(ids.shape[1]))
        output = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=1,
        )
        patch.disarm()
    logits = output.logits[:, -1].float()
    return logits[:, 9693] - logits[:, 2152]


def _patch_q3(
    model: Any,
    spec: NodeSpec,
    context: Mapping[str, Any],
    donor: Mapping[str, Any],
) -> Any:
    terms = []
    with _patcher(model, spec) as patch:
        for branch in ("yes", "no"):
            path = context["paths"][branch]
            patch.arm(
                path["positions"],
                donor[branch],
                sequence_length=int(path["ids"].shape[1]),
            )
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=3,
            )
            patch.disarm()
            terms.append(_path_terms(output, path))
    _matrix, score = _combine_terms(terms[0], terms[1])
    return _torch().as_tensor(score, dtype=_torch().float32, device=path["ids"].device)


def _patcher(model: Any, spec: NodeSpec) -> AbstractContextManager[Any]:
    if spec.node_id == "post_attention_residual":
        assert spec.block is not None
        return QwenPostAttentionStatePatch(model, spec.block)
    return QwenNodePatch(model, spec)


def _direction_controls(
    donor: Any,
    recipient: Any,
    *,
    identity_keys: Sequence[Sequence[str]],
) -> tuple[dict[str, Any], dict[str, float]]:
    if donor.shape != recipient.shape or donor.ndim != 3:
        raise ValueError("direction controls require aligned [batch,position,hidden] states")
    donor_rms = donor.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    recipient_rms = recipient.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    minimum = min(float(donor_rms.min().item()), float(recipient_rms.min().item()))
    if minimum <= 0.0 or not math.isfinite(minimum):
        raise FloatingPointError("direction control encountered a zero/non-finite RMS")
    d_at_r = (donor.float() / donor_rms * recipient_rms).to(donor.dtype)
    r_at_d = (recipient.float() / recipient_rms * donor_rms).to(recipient.dtype)
    z_at_r = rms_matched_random_direction(
        recipient,
        seed=RANDOM_DIRECTION_SEED,
        identity_keys=identity_keys,
    )
    controls = {
        "donor_direction_at_recipient_rms": d_at_r,
        "recipient_direction_at_donor_rms": r_at_d,
        "random_direction_at_recipient_rms": z_at_r,
    }
    geometry = {
        "donor_rms_min": float(donor_rms.min().item()),
        "donor_rms_max": float(donor_rms.max().item()),
        "recipient_rms_min": float(recipient_rms.min().item()),
        "recipient_rms_max": float(recipient_rms.max().item()),
        "d_at_r_rms_max_abs_error": _rms_error(d_at_r, recipient_rms),
        "r_at_d_rms_max_abs_error": _rms_error(r_at_d, donor_rms),
        "z_at_r_rms_max_abs_error": _rms_error(z_at_r, recipient_rms),
    }
    return controls, geometry


def _q3_direction_controls(
    donor: Mapping[str, Any],
    recipient: Mapping[str, Any],
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    *,
    node: str,
    block: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    controls = {name: {} for name in NODE_INTERVENTIONS[-3:]}
    geometry = {}
    position_names = {
        "yes": ("shared_prompt", "yes_context"),
        "no": ("shared_prompt", "no_context"),
    }
    for branch in ("yes", "no"):
        branch_controls, branch_geometry = _direction_controls(
            donor[branch],
            recipient[branch],
            identity_keys=_identity_keys(
                record,
                candidates,
                node,
                block,
                position_names[branch],
            ),
        )
        for name, value in branch_controls.items():
            controls[name][branch] = value
        geometry[branch] = branch_geometry
    for name, value in controls.items():
        delta = float((value["yes"][:, 0] - value["no"][:, 0]).abs().max().item())
        if delta != 0.0:
            raise RuntimeError(
                f"Q3 direction control {name} changed shared prompt across paths: {delta}"
            )
    return controls, geometry


def _identity_keys(
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    node: str,
    block: int,
    position_names: Sequence[str],
) -> list[list[str]]:
    return [
        [
            "\x1f".join(
                (
                    node,
                    str(block),
                    record.request_id,
                    str(candidate["item_id"]),
                    position_name,
                )
            )
            for position_name in position_names
        ]
        for candidate in candidates
    ]


def _rms_error(value: Any, target_rms: Any) -> float:
    observed = value.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    return float((observed - target_rms).abs().max().item())


def _validate_conditions(conditions: Mapping[str, Any], rows: int) -> None:
    if set(conditions) != set(selected_branch_conditions()):
        raise ValueError("selected-branch condition coverage drift")
    torch = _torch()
    for name, value in conditions.items():
        if value.ndim != 1 or value.shape[0] != rows:
            raise ValueError(f"selected-branch score shape drift: {name}")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"selected-branch score is non-finite: {name}")


def _torch() -> Any:
    import torch

    return torch
