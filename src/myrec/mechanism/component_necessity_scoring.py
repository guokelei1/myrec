"""Reverse component-state removal primitives for Q2/Q3.

The registered D2 selected-branch scorer writes full-context states into a
null-context recipient and therefore establishes, at most, state sufficiency.
This independent extension performs the reverse intervention: a same-request
null-context state is written into a full-context recipient.  It deliberately
does not modify the parent D2 scorer, conditions, or implementation digest.

This module reads no records, manifests, scores, qrels, or scientific effects.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.attention_edge_scoring import _build_paths, _neutralize_paths
from myrec.mechanism.deep_dive_native_patch import _q3_context
from myrec.mechanism.native_readout_scoring import build_q2_pointwise_batch
from myrec.mechanism.selected_branch_scoring import (
    _capture_q2,
    _capture_q3,
    _patch_q2,
    _patch_q3,
)
from myrec.mechanism.transformer_instrumentation import NodeSpec, QwenNodeCapture


NECESSITY_NODES = (
    "block_input_residual",
    "attention_o_projection",
    "mlp_down_projection",
    "block_output_residual",
)
NECESSITY_INTERVENTIONS = (
    "full_to_full_identity",
    "null_to_full_removal",
    "neutral_to_full_removal",
)


def component_necessity_conditions() -> tuple[str, ...]:
    """Return the frozen condition order for scalar score bundles."""

    return (
        "baseline_full",
        "baseline_null",
        *(
            f"{node}.{condition}"
            for node in NECESSITY_NODES
            for condition in NECESSITY_INTERVENTIONS
        ),
    )


def score_component_necessity_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    content_control: Mapping[str, Any],
    block: int,
    device: str,
) -> dict[str, Any]:
    """Score every frozen reverse-removal condition for one candidate chunk."""

    if not candidates:
        raise ValueError("component-necessity candidate chunk is empty")
    block = int(block)
    if not 13 <= block <= 27:
        raise ValueError("component-necessity block must be in [13,27]")
    method_id = str(config.get("method_id"))
    if method_id == "q2_recranker_generalqwen":
        result = _score_q2_necessity(
            model,
            tokenizer,
            record,
            candidates,
            config,
            content_control=content_control,
            block=block,
            device=device,
        )
    elif method_id == "q3_tallrec_generalqwen":
        result = _score_q3_necessity(
            model,
            tokenizer,
            record,
            candidates,
            config,
            content_control=content_control,
            block=block,
            device=device,
        )
    else:
        raise ValueError("component-necessity scoring supports only Q2/Q3")
    _validate_conditions(result["conditions"], len(candidates))
    return result


def _score_q2_necessity(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    content_control: Mapping[str, Any],
    block: int,
    device: str,
) -> dict[str, Any]:
    specs = tuple(NodeSpec(node, block) for node in NECESSITY_NODES)
    full_batch = build_q2_pointwise_batch(
        tokenizer, record, candidates, record.history, config, device=device
    )
    null_batch = build_q2_pointwise_batch(
        tokenizer, record, candidates, [], config, device=device
    )
    neutral = None
    neutral_path_identity_passed = False
    with QwenNodeCapture(model, specs) as capture:
        full = _capture_q2(model, capture, full_batch)
        null = _capture_q2(model, capture, null_batch)
        if content_control.get("eligible") is True:
            full_paths = _build_paths(
                tokenizer,
                record,
                candidates,
                content_control,
                config,
                device=device,
            )
            if len(full_paths) != 1 or full_paths[0].get("name") != "prompt":
                raise RuntimeError("component-necessity Q2 full path contract drift")
            _assert_q2_full_path_identity(full_batch, full_paths[0])
            neutral_path = _neutralize_paths(full_paths)[0]
            _assert_neutral_path_identity(full_paths[0], neutral_path)
            neutral = _capture_q2(
                model,
                capture,
                (
                    neutral_path["ids"],
                    neutral_path["mask"],
                    neutral_path["positions"],
                ),
            )
            neutral_path_identity_passed = True
    conditions: dict[str, Any] = {
        "baseline_full": full["score"],
        "baseline_null": null["score"],
    }
    maximum_identity_delta = 0.0
    for spec in specs:
        node = spec.node_id
        identity = _patch_q2(model, spec, full_batch, full["states"][spec.key])
        removal = _patch_q2(model, spec, full_batch, null["states"][spec.key])
        neutral_removal = (
            identity
            if neutral is None
            else _patch_q2(model, spec, full_batch, neutral["states"][spec.key])
        )
        conditions[f"{node}.full_to_full_identity"] = identity
        conditions[f"{node}.null_to_full_removal"] = removal
        conditions[f"{node}.neutral_to_full_removal"] = neutral_removal
        maximum_identity_delta = max(
            maximum_identity_delta,
            float((identity - full["score"]).abs().max().item()),
        )
    return {
        "conditions": conditions,
        "maximum_identity_delta": maximum_identity_delta,
        "content_neutral_eligible": neutral is not None,
        "neutral_path_identity_passed": neutral_path_identity_passed,
    }


def _score_q3_necessity(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    content_control: Mapping[str, Any],
    block: int,
    device: str,
) -> dict[str, Any]:
    specs = tuple(NodeSpec(node, block) for node in NECESSITY_NODES)
    full_context = _q3_context(
        tokenizer, record, candidates, record.history, config, device
    )
    null_context = _q3_context(tokenizer, record, candidates, [], config, device)
    neutral = None
    neutral_path_identity_passed = False
    with QwenNodeCapture(model, specs) as capture:
        full = _capture_q3(model, capture, full_context, specs)
        null = _capture_q3(model, capture, null_context, specs)
        if content_control.get("eligible") is True:
            full_paths = _build_paths(
                tokenizer,
                record,
                candidates,
                content_control,
                config,
                device=device,
            )
            full_from_control = {
                "paths": {str(path["name"]): path for path in full_paths}
            }
            if set(full_from_control["paths"]) != {"yes", "no"}:
                raise RuntimeError("component-necessity Q3 full path contract drift")
            _assert_q3_full_path_identity(full_context, full_from_control)
            neutral_paths = _neutralize_paths(full_paths)
            for full_path, neutral_path in zip(full_paths, neutral_paths):
                _assert_neutral_path_identity(full_path, neutral_path)
            neutral_context = {
                "paths": {str(path["name"]): path for path in neutral_paths}
            }
            neutral = _capture_q3(model, capture, neutral_context, specs)
            neutral_path_identity_passed = True
    conditions: dict[str, Any] = {
        "baseline_full": full["score"],
        "baseline_null": null["score"],
    }
    maximum_identity_delta = 0.0
    shared_prompt_path_max_abs_delta = 0.0
    for spec in specs:
        node = spec.node_id
        for captured in (full, null, neutral):
            if captured is None:
                continue
            states = captured["states"][spec.key]
            shared_prompt_path_max_abs_delta = max(
                shared_prompt_path_max_abs_delta,
                float((states["yes"][:, 0] - states["no"][:, 0]).abs().max().item()),
            )
        identity = _patch_q3(
            model, spec, full_context, full["states"][spec.key]
        )
        removal = _patch_q3(
            model, spec, full_context, null["states"][spec.key]
        )
        neutral_removal = (
            identity
            if neutral is None
            else _patch_q3(
                model, spec, full_context, neutral["states"][spec.key]
            )
        )
        conditions[f"{node}.full_to_full_identity"] = identity
        conditions[f"{node}.null_to_full_removal"] = removal
        conditions[f"{node}.neutral_to_full_removal"] = neutral_removal
        maximum_identity_delta = max(
            maximum_identity_delta,
            float((identity - full["score"]).abs().max().item()),
        )
    if shared_prompt_path_max_abs_delta != 0.0:
        raise RuntimeError(
            "component-necessity Q3 shared prompt differs across native paths: "
            f"{shared_prompt_path_max_abs_delta}"
        )
    return {
        "conditions": conditions,
        "maximum_identity_delta": maximum_identity_delta,
        "shared_prompt_path_max_abs_delta": shared_prompt_path_max_abs_delta,
        "content_neutral_eligible": neutral is not None,
        "neutral_path_identity_passed": neutral_path_identity_passed,
    }


def _assert_q2_full_path_identity(batch: Any, path: Mapping[str, Any]) -> None:
    ids, mask, positions = batch
    torch = _torch()
    if not (
        torch.equal(ids, path["ids"])
        and torch.equal(mask, path["mask"])
        and torch.equal(positions, path["positions"])
    ):
        raise RuntimeError(
            "component-necessity Q2 full/content-control path identity failed"
        )


def _assert_q3_full_path_identity(
    context: Mapping[str, Any], control_context: Mapping[str, Any]
) -> None:
    torch = _torch()
    for branch in ("yes", "no"):
        expected = context["paths"][branch]
        observed = control_context["paths"][branch]
        if not (
            torch.equal(expected["ids"], observed["ids"])
            and torch.equal(expected["mask"], observed["mask"])
            and torch.equal(expected["positions"], observed["positions"])
            and list(expected["target"]) == list(observed["target"])
        ):
            raise RuntimeError(
                "component-necessity Q3 full/content-control path identity failed"
            )


def _assert_neutral_path_identity(
    full_path: Mapping[str, Any], neutral_path: Mapping[str, Any]
) -> None:
    torch = _torch()
    if not (
        full_path["ids"].shape == neutral_path["ids"].shape
        and torch.equal(full_path["mask"], neutral_path["mask"])
        and torch.equal(full_path["positions"], neutral_path["positions"])
        and torch.equal(full_path["starts"], neutral_path["starts"])
        and torch.equal(full_path["ends"], neutral_path["ends"])
    ):
        raise RuntimeError("component-necessity neutral path geometry drift")
    for row in range(full_path["ids"].shape[0]):
        start = int(full_path["starts"][row])
        end = int(full_path["ends"][row])
        if end <= start:
            raise RuntimeError("component-necessity neutral span is empty")
        outside = torch.ones(
            full_path["ids"].shape[1],
            dtype=torch.bool,
            device=full_path["ids"].device,
        )
        outside[start:end] = False
        if not torch.equal(
            full_path["ids"][row, outside], neutral_path["ids"][row, outside]
        ):
            raise RuntimeError("component-necessity neutral path changed span exterior")
        if not bool((neutral_path["ids"][row, start:end] == 151_643).all().item()):
            raise RuntimeError("component-necessity neutral span token drift")


def _validate_conditions(conditions: Mapping[str, Any], rows: int) -> None:
    expected = component_necessity_conditions()
    if tuple(conditions) != expected:
        raise ValueError("component-necessity condition order or coverage drift")
    torch = _torch()
    for name, value in conditions.items():
        if value.ndim != 1 or int(value.shape[0]) != rows:
            raise ValueError(f"component-necessity score shape drift: {name}")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(
                f"component-necessity score is non-finite: {name}"
            )


def _torch() -> Any:
    import torch

    return torch
