"""Native Q2/Q3 post-block causal sweep primitives for D2."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.deep_dive_native_patch import (
    _capture_context,
    _patch_context,
    _q3_context,
)
from myrec.mechanism.native_readout_scoring import build_q2_pointwise_batch
from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
    QwenNodePatch,
)


POSTBLOCK_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_to_full_identity",
    "null_to_null_identity",
    "same_full_to_null",
    "cross_full_to_null",
)


def score_postblock_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    donor_record: ModelRecord,
    donor_candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    """Return six native-score conditions for one registered post-block."""

    if not candidates or len(candidates) != len(donor_candidates):
        raise ValueError("post-block recipient/donor chunks are empty or misaligned")
    block = int(block)
    if not 13 <= block <= 27:
        raise ValueError("post-block sweep block must be in [13,27]")
    method_id = str(config["method_id"])
    if method_id == "q2_recranker_generalqwen":
        return _score_q2(
            model,
            tokenizer,
            record,
            candidates,
            donor_record,
            donor_candidates,
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
            donor_record,
            donor_candidates,
            config,
            block=block,
            device=device,
        )
    raise ValueError("post-block sweep supports only Q2/Q3")


def score_q2_null_identity_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    """Recompute only Q2's missing null->null identity control.

    The frozen first-round Q2 block-13/27 bundles already contain the
    full->full, same-request, and cross-request interventions.  This narrow
    path is used only after an independent new-vs-old equivalence audit has
    admitted those bundles for byte-addressed reuse.
    """

    if str(config.get("method_id")) != "q2_recranker_generalqwen":
        raise ValueError("null-identity complement is Q2-only")
    block = int(block)
    if block not in (13, 27):
        raise ValueError("Q2 reuse complement is restricted to blocks 13/27")
    if not candidates:
        raise ValueError("Q2 null-identity candidate chunk is empty")
    spec = NodeSpec("block_output_residual", block)
    null_batch = build_q2_pointwise_batch(
        tokenizer, record, candidates, [], config, device=device
    )
    with QwenNodeCapture(model, [spec]) as capture:
        null = _capture_q2(model, capture, spec, null_batch)
    with QwenNodePatch(model, spec) as patch:
        null_identity = _patch_q2(model, patch, null_batch, null["states"])
    identity = float((null_identity - null["score"]).abs().max().item())
    return {
        "baseline_null": null["score"],
        "null_to_null_identity": null_identity,
        "maximum_identity_delta": identity,
    }


def _score_q2(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    donor_record: ModelRecord,
    donor_candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    spec = NodeSpec("block_output_residual", block)
    full_batch = build_q2_pointwise_batch(
        tokenizer, record, candidates, record.history, config, device=device
    )
    null_batch = build_q2_pointwise_batch(
        tokenizer, record, candidates, [], config, device=device
    )
    cross_batch = build_q2_pointwise_batch(
        tokenizer,
        donor_record,
        donor_candidates,
        donor_record.history,
        config,
        device=device,
    )
    with QwenNodeCapture(model, [spec]) as capture:
        full = _capture_q2(model, capture, spec, full_batch)
        null = _capture_q2(model, capture, spec, null_batch)
        cross = _capture_q2(model, capture, spec, cross_batch)
    with QwenNodePatch(model, spec) as patch:
        full_identity = _patch_q2(model, patch, full_batch, full["states"])
        null_identity = _patch_q2(model, patch, null_batch, null["states"])
        same = _patch_q2(model, patch, null_batch, full["states"])
        cross_score = _patch_q2(model, patch, null_batch, cross["states"])
    conditions = {
        "baseline_full": full["score"],
        "baseline_null": null["score"],
        "full_to_full_identity": full_identity,
        "null_to_null_identity": null_identity,
        "same_full_to_null": same,
        "cross_full_to_null": cross_score,
    }
    identity = max(
        float((full_identity - full["score"]).abs().max().item()),
        float((null_identity - null["score"]).abs().max().item()),
    )
    return {"conditions": conditions, "maximum_identity_delta": identity}


def _capture_q2(model: Any, capture: QwenNodeCapture, spec: NodeSpec, batch: Any) -> dict[str, Any]:
    ids, mask, positions = batch
    output, states = capture.capture_forward(
        input_ids=ids,
        attention_mask=mask,
        positions=positions,
        model_kwargs={"logits_to_keep": 1},
    )
    return {"score": _q2_score(output), "states": states[spec.key]}


def _patch_q2(model: Any, patch: QwenNodePatch, batch: Any, donor: Any) -> Any:
    ids, mask, positions = batch
    patch.arm(positions, donor, sequence_length=ids.shape[1])
    output = model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
        logits_to_keep=1,
    )
    patch.disarm()
    return _q2_score(output)


def _q2_score(output: Any) -> Any:
    logits = output.logits[:, -1].float()
    score = logits[:, 9693] - logits[:, 2152]
    if not bool(_torch().isfinite(score).all().item()):
        raise FloatingPointError("Q2 post-block score is non-finite")
    return score


def _score_q3(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    donor_record: ModelRecord,
    donor_candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    spec = NodeSpec("block_output_residual", block)
    full_context = _q3_context(
        tokenizer, record, candidates, record.history, config, device
    )
    null_context = _q3_context(tokenizer, record, candidates, [], config, device)
    cross_context = _q3_context(
        tokenizer,
        donor_record,
        donor_candidates,
        donor_record.history,
        config,
        device,
    )
    with QwenNodeCapture(model, [spec]) as capture:
        full = _capture_context(model, capture, spec, full_context)
        null = _capture_context(model, capture, spec, null_context)
        cross = _capture_context(model, capture, spec, cross_context)
    shared = max(
        float((full["yes_states"][:, 0] - full["no_states"][:, 0]).abs().max().item()),
        float((null["yes_states"][:, 0] - null["no_states"][:, 0]).abs().max().item()),
        float((cross["yes_states"][:, 0] - cross["no_states"][:, 0]).abs().max().item()),
    )
    if shared != 0.0:
        raise RuntimeError("Q3 post-block shared prompt state differs across paths")
    with QwenNodePatch(model, spec) as patch:
        full_identity = _patch_context(
            model, patch, full_context, full, scope="all_native_positions"
        )
        null_identity = _patch_context(
            model, patch, null_context, null, scope="all_native_positions"
        )
        same = _patch_context(
            model, patch, null_context, full, scope="all_native_positions"
        )
        cross_score = _patch_context(
            model, patch, null_context, cross, scope="all_native_positions"
        )
    conditions = {
        "baseline_full": _numpy_to_tensor(full["score"], device),
        "baseline_null": _numpy_to_tensor(null["score"], device),
        "full_to_full_identity": _numpy_to_tensor(full_identity["score"], device),
        "null_to_null_identity": _numpy_to_tensor(null_identity["score"], device),
        "same_full_to_null": _numpy_to_tensor(same["score"], device),
        "cross_full_to_null": _numpy_to_tensor(cross_score["score"], device),
    }
    identity = max(
        float((conditions["full_to_full_identity"] - conditions["baseline_full"]).abs().max().item()),
        float((conditions["null_to_null_identity"] - conditions["baseline_null"]).abs().max().item()),
    )
    return {
        "conditions": conditions,
        "maximum_identity_delta": identity,
        "shared_prompt_path_max_abs_delta": shared,
    }


def _numpy_to_tensor(value: Any, device: str) -> Any:
    return _torch().as_tensor(value, dtype=_torch().float32, device=device)


def _torch() -> Any:
    import torch

    return torch
