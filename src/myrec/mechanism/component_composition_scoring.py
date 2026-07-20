"""Forward kernels for the registered joint attention/MLP composition probe.

The probe replaces the attention ``o_proj`` and MLP ``down_proj`` states in a
full-context recipient with states captured from the same request's
position-preserving content-neutral path.  It is deliberately qrels-blind and
does not choose a block or a component from observed effects.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.attention_edge_scoring import (
    _build_paths,
    _neutralize_paths,
)
from myrec.mechanism.deep_dive_native_patch import (
    _combine_terms,
    _path_terms,
    _q3_context,
)
from myrec.mechanism.native_readout_scoring import build_q2_pointwise_batch
from myrec.mechanism.selected_branch_scoring import (
    _capture_q2,
    _capture_q3,
)
from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
    QwenNodePatch,
)


COMPOSITION_NODES = (
    "attention_o_projection",
    "mlp_down_projection",
)

COMPOSITION_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_identity",
    "neutral_identity",
    "attention_neutral_removal",
    "mlp_neutral_removal",
    "joint_attention_mlp_neutral_removal",
)


class QwenMultiNodePatch(AbstractContextManager["QwenMultiNodePatch"]):
    """Patch several explicit nodes during one forward pass.

    The existing single-node patch primitive is retained as the only module
    resolution and token-row replacement implementation.  This wrapper only
    arms two independent patchers together, so a joint condition cannot drift
    from the single-node identity semantics.
    """

    def __init__(self, model: Any, specs: Sequence[NodeSpec]) -> None:
        if not specs or len({spec.key for spec in specs}) != len(specs):
            raise ValueError("multi-node patch specs must be nonempty and unique")
        self.patchers = [QwenNodePatch(model, spec) for spec in specs]
        self._armed = False

    def __enter__(self) -> "QwenMultiNodePatch":
        for patcher in self.patchers:
            patcher.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for patcher in reversed(self.patchers):
            patcher.__exit__(exc_type, exc, traceback)
        self._armed = False

    def arm(
        self,
        positions: Any,
        donors: Mapping[str, Any],
        *,
        sequence_length: int,
    ) -> None:
        if self._armed:
            raise RuntimeError("multi-node patch is already armed")
        if set(donors) != {patcher.spec.key for patcher in self.patchers}:
            raise ValueError("multi-node donor keys do not match patch specs")
        for patcher in self.patchers:
            patcher.arm(
                positions,
                donors[patcher.spec.key],
                sequence_length=sequence_length,
            )
        self._armed = True

    def disarm(self) -> None:
        if not self._armed:
            raise RuntimeError("multi-node patch is not armed")
        for patcher in reversed(self.patchers):
            patcher.disarm()
        self._armed = False


def composition_conditions() -> tuple[str, ...]:
    """Return the frozen scalar condition order."""

    return COMPOSITION_CONDITIONS


def score_component_composition_chunk(
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
    """Score one candidate chunk for Q2 or Q3 without reading qrels."""

    if not candidates:
        raise ValueError("component-composition candidate chunk is empty")
    block = int(block)
    if not 13 <= block <= 27:
        raise ValueError("component-composition block must be in [13,27]")
    if content_control.get("eligible") is not True:
        raise ValueError("component-composition requires content-neutral eligibility")
    method_id = str(config.get("method_id"))
    specs = tuple(NodeSpec(node, block) for node in COMPOSITION_NODES)
    full_paths = _build_paths(
        tokenizer,
        record,
        candidates,
        content_control,
        config,
        device=device,
    )
    neutral_paths = _neutralize_paths(full_paths)
    if method_id == "q2_recranker_generalqwen":
        if len(full_paths) != 1 or full_paths[0].get("name") != "prompt":
            raise RuntimeError("Q2 component-composition path contract drift")
        _assert_path_geometry(full_paths[0], neutral_paths[0])
        full_batch = _path_batch(full_paths[0])
        neutral_batch = _path_batch(neutral_paths[0])
        return _score_q2(
            model,
            tokenizer,
            record,
            candidates,
            config,
            block=block,
            device=device,
            specs=specs,
            full_batch=full_batch,
            neutral_batch=neutral_batch,
        )
    if method_id == "q3_tallrec_generalqwen":
        if {str(path.get("name")) for path in full_paths} != {"yes", "no"}:
            raise RuntimeError("Q3 component-composition path contract drift")
        for full_path, neutral_path in zip(full_paths, neutral_paths):
            _assert_path_geometry(full_path, neutral_path)
        return _score_q3(
            model,
            tokenizer,
            record,
            candidates,
            config,
            block=block,
            device=device,
            specs=specs,
            full_context={"paths": {str(p["name"]): p for p in full_paths}},
            neutral_context={"paths": {str(p["name"]): p for p in neutral_paths}},
        )
    raise ValueError("component-composition supports only Q2/Q3")


def _score_q2(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
    specs: Sequence[NodeSpec],
    full_batch: Any,
    neutral_batch: Any,
) -> dict[str, Any]:
    null_batch = build_q2_pointwise_batch(
        tokenizer, record, candidates, [], config, device=device
    )
    with QwenNodeCapture(model, specs) as capture:
        full = _capture_q2(model, capture, full_batch)
        null = _capture_q2(model, capture, null_batch)
        neutral = _capture_q2(model, capture, neutral_batch)
    full_donors = {spec.key: full["states"][spec.key] for spec in specs}
    neutral_donors = {spec.key: neutral["states"][spec.key] for spec in specs}
    conditions = {
        "baseline_full": full["score"],
        "baseline_null": null["score"],
        "full_identity": _patch_q2(model, specs, full_batch, full_donors),
        "neutral_identity": _patch_q2(model, specs, neutral_batch, neutral_donors),
        "attention_neutral_removal": _patch_q2(
            model,
            specs,
            full_batch,
            {spec.key: (neutral_donors[spec.key] if spec.node_id == "attention_o_projection" else full_donors[spec.key]) for spec in specs},
        ),
        "mlp_neutral_removal": _patch_q2(
            model,
            specs,
            full_batch,
            {spec.key: (neutral_donors[spec.key] if spec.node_id == "mlp_down_projection" else full_donors[spec.key]) for spec in specs},
        ),
        "joint_attention_mlp_neutral_removal": _patch_q2(
            model, specs, full_batch, neutral_donors
        ),
    }
    _validate_scores(conditions, len(candidates))
    identity = max(
        float((conditions["full_identity"] - conditions["baseline_full"]).abs().max().item()),
        float((conditions["neutral_identity"] - neutral["score"]).abs().max().item()),
    )
    return {
        "conditions": conditions,
        "maximum_identity_delta": identity,
        "neutral_path_identity_passed": True,
    }


def _score_q3(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
    specs: Sequence[NodeSpec],
    full_context: Mapping[str, Any],
    neutral_context: Mapping[str, Any],
) -> dict[str, Any]:
    null_context = _q3_context(tokenizer, record, candidates, [], config, device)
    with QwenNodeCapture(model, specs) as capture:
        full = _capture_q3(model, capture, full_context, specs)
        null = _capture_q3(model, capture, null_context, specs)
        neutral = _capture_q3(model, capture, neutral_context, specs)
    full_donors = {spec.key: full["states"][spec.key] for spec in specs}
    neutral_donors = {spec.key: neutral["states"][spec.key] for spec in specs}
    conditions = {
        "baseline_full": full["score"],
        "baseline_null": null["score"],
        "full_identity": _patch_q3(model, specs, full_context, full_donors),
        "neutral_identity": _patch_q3(model, specs, neutral_context, neutral_donors),
        "attention_neutral_removal": _patch_q3(
            model,
            specs,
            full_context,
            {spec.key: (neutral_donors[spec.key] if spec.node_id == "attention_o_projection" else full_donors[spec.key]) for spec in specs},
        ),
        "mlp_neutral_removal": _patch_q3(
            model,
            specs,
            full_context,
            {spec.key: (neutral_donors[spec.key] if spec.node_id == "mlp_down_projection" else full_donors[spec.key]) for spec in specs},
        ),
        "joint_attention_mlp_neutral_removal": _patch_q3(
            model, specs, full_context, neutral_donors
        ),
    }
    _validate_scores(conditions, len(candidates))
    identity = max(
        float((conditions["full_identity"] - conditions["baseline_full"]).abs().max().item()),
        float((conditions["neutral_identity"] - neutral["score"]).abs().max().item()),
    )
    shared_delta = 0.0
    for result in (full, null, neutral):
        for spec in specs:
            shared_delta = max(
                shared_delta,
                float(
                    (result["states"][spec.key]["yes"][:, 0] - result["states"][spec.key]["no"][:, 0])
                    .abs()
                    .max()
                    .item()
                ),
            )
    if shared_delta != 0.0:
        raise RuntimeError(
            f"Q3 component-composition shared prompt state differs: {shared_delta}"
        )
    return {
        "conditions": conditions,
        "maximum_identity_delta": identity,
        "shared_prompt_path_max_abs_delta": shared_delta,
        "neutral_path_identity_passed": True,
    }


def _patch_q2(
    model: Any,
    specs: Sequence[NodeSpec],
    batch: Any,
    donors: Mapping[str, Any],
) -> Any:
    ids, mask, positions = batch
    with QwenMultiNodePatch(model, specs) as patcher:
        patcher.arm(positions, donors, sequence_length=int(ids.shape[1]))
        output = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=1,
        )
        patcher.disarm()
    logits = output.logits[:, -1].float()
    return logits[:, 9693] - logits[:, 2152]


def _patch_q3(
    model: Any,
    specs: Sequence[NodeSpec],
    context: Mapping[str, Any],
    donors: Mapping[str, Mapping[str, Any]],
) -> Any:
    terms = []
    with QwenMultiNodePatch(model, specs) as patcher:
        for branch in ("yes", "no"):
            path = context["paths"][branch]
            branch_donors = {
                spec.key: donors[spec.key][branch] for spec in specs
            }
            patcher.arm(
                path["positions"],
                branch_donors,
                sequence_length=int(path["ids"].shape[1]),
            )
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=3,
            )
            patcher.disarm()
            terms.append(_path_terms(output, path))
    _matrix, score = _combine_terms(terms[0], terms[1])
    torch = _torch()
    return torch.as_tensor(score, dtype=torch.float32, device=path["ids"].device)


def _path_batch(path: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return path["ids"], path["mask"], path["positions"]


def _assert_path_geometry(full_path: Mapping[str, Any], neutral_path: Mapping[str, Any]) -> None:
    torch = _torch()
    for key in ("mask", "positions", "starts", "ends"):
        if not torch.equal(full_path[key], neutral_path[key]):
            raise RuntimeError(f"component-composition neutral path changed {key}")
    if full_path["ids"].shape != neutral_path["ids"].shape:
        raise RuntimeError("component-composition neutral path changed shape")
    for row in range(full_path["ids"].shape[0]):
        start = int(full_path["starts"][row])
        end = int(full_path["ends"][row])
        if end <= start:
            raise RuntimeError("component-composition neutral span is empty")
        outside = torch.ones(
            full_path["ids"].shape[1], dtype=torch.bool, device=full_path["ids"].device
        )
        outside[start:end] = False
        if not torch.equal(full_path["ids"][row, outside], neutral_path["ids"][row, outside]):
            raise RuntimeError("component-composition neutral path changed span exterior")


def _validate_scores(conditions: Mapping[str, Any], rows: int) -> None:
    import torch

    if tuple(conditions) != COMPOSITION_CONDITIONS:
        raise RuntimeError("component-composition condition order drift")
    for name, value in conditions.items():
        if tuple(value.shape) != (rows,) or not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"component-composition score is invalid: {name}")


def _torch() -> Any:
    import torch

    return torch
