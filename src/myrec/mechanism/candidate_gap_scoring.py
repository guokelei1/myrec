"""Native final-readout candidate-gap perturbations for the N10 diagnostic wave.

The perturbations are deliberately qrels-blind.  For each request/candidate row
we compare the frozen full and null final-RMSNorm states and construct three
norm-matched directions: the full-minus-null direction, its candidate-common
slate mean, and a deterministic orthogonal direction.  The perturbation is
applied only at the registered final-RMSNorm input or output node; it is not a
training control or a proposed transfer architecture.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.breadth_readout_scoring import (
    _build_q0_path,
    _capture_q1_paths,
    _q0_scores,
    _score_q1_paths,
)
from myrec.mechanism.q1_kv_trajectory import instrument_q1_selection_prompt
from myrec.mechanism.native_readout_scoring import (
    build_q2_pointwise_batch,
    capture_q2_native_readout,
    score_q2_with_final_node_patch,
)
from myrec.mechanism.deep_dive_native_patch import _q3_context
from myrec.mechanism.q3_native_readout_scoring import (
    _combine_path_terms,
    _path_terms,
    capture_q3_native_readout,
    q3_native_score_from_terms,
)
from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
    QwenNodePatch,
)


CANDIDATE_GAP_NODES = ("final_rmsnorm_input", "final_rmsnorm_output")
CANDIDATE_GAP_MODES = (
    "full_null_direction",
    "candidate_common_direction",
    "orthogonal_direction",
)
CANDIDATE_GAP_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    *(f"{node}_{mode}" for node in CANDIDATE_GAP_NODES for mode in CANDIDATE_GAP_MODES),
    *(f"{node}_full_identity" for node in CANDIDATE_GAP_NODES),
)
PERTURBATION_FRACTION = 0.10
ORTHOGONAL_EPS = 1.0e-8


def candidate_gap_direction_tensors(full: Any, null: Any) -> dict[str, Any]:
    """Return full-minus-null, common, and deterministic orthogonal directions.

    ``full`` and ``null`` may have any leading dimensions; the last dimension
    is interpreted as hidden size.  The candidate-common mean is taken over the
    first leading dimension.  For Q1 continuation tensors this is the
    candidate dimension within each token position.
    """

    torch = _torch()
    if full.shape != null.shape or full.ndim < 2:
        raise ValueError("candidate-gap states must be aligned hidden tensors")
    delta = full.float() - null.float()
    common = delta.mean(dim=0, keepdim=True).expand_as(delta)
    relative = delta - common
    denominator = (delta * delta).sum(dim=-1, keepdim=True)
    projection = (relative * delta).sum(dim=-1, keepdim=True) / denominator.clamp_min(
        ORTHOGONAL_EPS
    )
    orthogonal = relative - projection * delta

    # A slate can occasionally have a zero candidate-relative component.  Use
    # a fixed coordinate basis in that case, then Gram--Schmidt it against the
    # full-minus-null vector.  The fallback is deterministic and independent of
    # labels, qrels, ranking outcomes, and layer selection.
    orth_norm = orthogonal.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(delta)
    fallback[..., 0] = 1.0
    if delta.shape[-1] > 1:
        alternate = torch.zeros_like(delta)
        alternate[..., 1] = 1.0
        use_alternate = (
            delta[..., 0:1].abs() / delta.norm(dim=-1, keepdim=True).clamp_min(ORTHOGONAL_EPS)
            > 0.9
        )
        fallback = torch.where(use_alternate, alternate, fallback)
    fallback = fallback - (
        (fallback * delta).sum(dim=-1, keepdim=True)
        / denominator.clamp_min(ORTHOGONAL_EPS)
    ) * delta
    orthogonal = torch.where(orth_norm > ORTHOGONAL_EPS, orthogonal, fallback)
    return {
        "full_null_direction": _norm_match(delta, delta),
        "candidate_common_direction": _norm_match(common, delta),
        "orthogonal_direction": _norm_match(orthogonal, delta),
    }


def perturb_state(full: Any, direction: Any) -> Any:
    """Apply the frozen small norm-matched perturbation in native dtype."""

    if full.shape != direction.shape:
        raise ValueError("candidate-gap perturbation shape differs from full state")
    return (full.float() + PERTURBATION_FRACTION * direction.float()).to(
        dtype=full.dtype, device=full.device
    )


def score_q0_candidate_gap_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    """Score Q0 candidate-gap conditions for one candidate chunk."""

    if not candidates:
        raise ValueError("Q0 candidate-gap chunk is empty")
    full = _build_q0_path(tokenizer, record, candidates, record.history, config, device)
    null = _build_q0_path(tokenizer, record, candidates, [], config, device)
    specs = tuple(NodeSpec(node_id=node, block=None) for node in CANDIDATE_GAP_NODES)
    with QwenNodeCapture(model, specs) as capture:
        full_output, full_states = capture.capture_forward(
            input_ids=full["ids"], attention_mask=full["mask"],
            positions=full["positions"], model_kwargs={"logits_to_keep": 1},
        )
    baseline_full = _q0_scores(full_output)
    baseline_null = _q0_scores(model(
        input_ids=null["ids"], attention_mask=null["mask"],
        use_cache=False, logits_to_keep=1,
    ))
    with QwenNodeCapture(model, specs) as capture:
        _, null_states = capture.capture_forward(
            input_ids=null["ids"], attention_mask=null["mask"],
            positions=null["positions"], model_kwargs={"logits_to_keep": 1},
        )
    conditions: dict[str, np.ndarray] = {
        "baseline_full": baseline_full,
        "baseline_null": baseline_null,
    }
    identity_delta = 0.0
    direction_norms: dict[str, float] = {}
    for node in CANDIDATE_GAP_NODES:
        spec = NodeSpec(node_id=node, block=None)
        key = spec.key
        full_state = full_states[key]
        null_state = null_states[key]
        directions = candidate_gap_direction_tensors(full_state, null_state)
        for mode, direction in directions.items():
            score = _score_q0_patch(model, spec, full, perturb_state(full_state, direction))
            conditions[f"{node}_{mode}"] = score
            direction_norms[f"{node}_{mode}"] = float(direction.float().norm(dim=-1).mean().item())
        identity = _score_q0_patch(model, spec, full, full_state)
        conditions[f"{node}_full_identity"] = identity
        identity_delta = max(identity_delta, float(np.max(np.abs(identity - baseline_full))))
    _validate_conditions(conditions, len(candidates))
    return {
        "conditions": conditions,
        "maximum_identity_delta": identity_delta,
        "direction_norms": direction_norms,
    }


def score_q2_candidate_gap_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    """Score Q2 candidate-gap conditions for one candidate chunk."""

    ids, mask, positions = build_q2_pointwise_batch(
        tokenizer, record, candidates, record.history, config, device=device
    )
    null_ids, null_mask, null_positions = build_q2_pointwise_batch(
        tokenizer, record, candidates, [], config, device=device
    )
    specs = tuple(NodeSpec(node_id=node, block=None) for node in CANDIDATE_GAP_NODES)
    with QwenNodeCapture(model, specs) as capture:
        full_output, full_states = capture.capture_forward(
            input_ids=ids, attention_mask=mask, positions=positions,
            model_kwargs={"logits_to_keep": 1},
        )
    full_capture = capture_q2_native_readout(model, ids, mask, positions)
    # The second capture above is intentional: it verifies the native tied-row
    # algebra while the first hook retains exact states for intervention.
    baseline_full = full_capture["native_score"].detach().float().cpu().numpy()
    baseline_null = capture_q2_native_readout(model, null_ids, null_mask, null_positions)[
        "native_score"
    ].detach().float().cpu().numpy()
    with QwenNodeCapture(model, specs) as capture:
        _, null_states = capture.capture_forward(
            input_ids=null_ids, attention_mask=null_mask, positions=null_positions,
            model_kwargs={"logits_to_keep": 1},
        )
    conditions: dict[str, np.ndarray] = {
        "baseline_full": baseline_full,
        "baseline_null": baseline_null,
    }
    identity_delta = 0.0
    direction_norms: dict[str, float] = {}
    for node in CANDIDATE_GAP_NODES:
        spec = NodeSpec(node_id=node, block=None)
        full_state = full_states[spec.key]
        directions = candidate_gap_direction_tensors(full_state, null_states[spec.key])
        for mode, direction in directions.items():
            score = score_q2_with_final_node_patch(
                model, ids, mask, positions, perturb_state(full_state, direction),
                node_id=node,
            ).detach().float().cpu().numpy()
            conditions[f"{node}_{mode}"] = score
            direction_norms[f"{node}_{mode}"] = float(direction.float().norm(dim=-1).mean().item())
        identity = score_q2_with_final_node_patch(
            model, ids, mask, positions, full_state, node_id=node,
        ).detach().float().cpu().numpy()
        conditions[f"{node}_full_identity"] = identity
        identity_delta = max(identity_delta, float(np.max(np.abs(identity - baseline_full))))
    _validate_conditions(conditions, len(candidates))
    return {
        "conditions": conditions,
        "maximum_identity_delta": identity_delta,
        "direction_norms": direction_norms,
    }


def score_q3_candidate_gap_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]],
    device: str,
) -> dict[str, Any]:
    """Score Q3 candidate-gap conditions on both Yes/No native paths."""

    if not candidates:
        raise ValueError("Q3 candidate-gap chunk is empty")
    full_context = _q3_context(tokenizer, record, candidates, history, config, device)
    null_context = _q3_context(tokenizer, record, candidates, [], config, device)
    full_capture = capture_q3_native_readout(model, full_context)
    null_capture = capture_q3_native_readout(model, null_context)
    shared_prompt_delta = max(
        max(float(full_capture["shared_prompt_path_max_abs_delta"][node]) for node in CANDIDATE_GAP_NODES),
        max(float(null_capture["shared_prompt_path_max_abs_delta"][node]) for node in CANDIDATE_GAP_NODES),
    )
    if shared_prompt_delta > 1.0e-5:
        raise RuntimeError(
            "candidate-gap Q3 shared prompt differs across native paths: "
            f"{shared_prompt_delta}"
        )
    conditions: dict[str, np.ndarray] = {
        "baseline_full": full_capture["score"].detach().float().cpu().numpy(),
        "baseline_null": null_capture["score"].detach().float().cpu().numpy(),
    }
    identity_delta = 0.0
    direction_norms: dict[str, float] = {}
    for node in CANDIDATE_GAP_NODES:
        per_path_directions = {
            path_name: candidate_gap_direction_tensors(
                full_capture["branches"][path_name][node],
                null_capture["branches"][path_name][node],
            )
            for path_name in ("yes", "no")
        }
        identity = _score_q3_patch(model, full_context, node, {
            path_name: full_capture["branches"][path_name][node]
            for path_name in ("yes", "no")
        })
        conditions[f"{node}_full_identity"] = identity
        identity_delta = max(identity_delta, float(np.max(np.abs(identity - conditions["baseline_full"]))))
        for mode in CANDIDATE_GAP_MODES:
            donors = {
                path_name: perturb_state(
                    full_capture["branches"][path_name][node],
                    per_path_directions[path_name][mode],
                )
                for path_name in ("yes", "no")
            }
            score = _score_q3_patch(model, full_context, node, donors)
            conditions[f"{node}_{mode}"] = score
            direction_norms[f"{node}_{mode}"] = float(
                torch_norm_mean(
                    [per_path_directions[path_name][mode] for path_name in ("yes", "no")]
                )
            )
    _validate_conditions(conditions, len(candidates))
    return {
        "conditions": conditions,
        "maximum_identity_delta": identity_delta,
        "direction_norms": direction_norms,
        "shared_prompt_path_max_abs_delta": shared_prompt_delta,
    }


def score_q1_candidate_gap_record(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    config: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Score Q1 candidate-gap conditions over prefix and all response tokens."""

    specs = tuple(NodeSpec(node_id=node, block=None) for node in CANDIDATE_GAP_NODES)
    full_prompt = instrument_q1_selection_prompt(tokenizer, record, record.history, config)
    null_prompt = instrument_q1_selection_prompt(tokenizer, record, [], config)
    full = _capture_q1_paths(model, tokenizer, record, full_prompt, specs, device=device, batch_size=batch_size)
    null = _capture_q1_paths(model, tokenizer, record, null_prompt, specs, device=device, batch_size=batch_size)
    conditions: dict[str, np.ndarray] = {
        "baseline_full": full["scores"],
        "baseline_null": null["scores"],
    }
    identity_delta = 0.0
    direction_norms: dict[str, float] = {}
    for node in CANDIDATE_GAP_NODES:
        spec = NodeSpec(node_id=node, block=None)
        full_donors = full["donors"][spec.key]
        null_donors = null["donors"][spec.key]
        donor_modes = _q1_donor_modes(full_donors, null_donors)
        identity = _score_q1_with_donors(
            model, tokenizer, record, full_prompt, spec, full_donors,
            device=device, batch_size=batch_size,
        )
        conditions[f"{node}_full_identity"] = identity["scores"]
        identity_delta = max(identity_delta, float(np.max(np.abs(identity["scores"] - full["scores"]))))
        for mode in CANDIDATE_GAP_MODES:
            result = _score_q1_with_donors(
                model, tokenizer, record, full_prompt, spec,
                donor_modes[mode], device=device, batch_size=batch_size,
            )
            conditions[f"{node}_{mode}"] = result["scores"]
            direction_norms[f"{node}_{mode}"] = _q1_donor_norm_mean(
                donor_modes[mode], full_donors, mode
            )
    _validate_conditions(conditions, len(record.candidates))
    return {
        "conditions": conditions,
        "maximum_identity_delta": identity_delta,
        "direction_norms": direction_norms,
        "response_tokens": int(full["response_tokens"]),
        "call_audit": {node: full["call_audit"] for node in CANDIDATE_GAP_NODES},
    }


def _score_q0_patch(model: Any, spec: NodeSpec, path: Mapping[str, Any], donor: Any) -> np.ndarray:
    with QwenNodePatch(model, spec) as patch:
        patch.arm(path["positions"], donor, sequence_length=path["ids"].shape[1])
        output = model(input_ids=path["ids"], attention_mask=path["mask"], use_cache=False, logits_to_keep=1)
        patch.disarm()
    return _q0_scores(output)


def _score_q3_patch(model: Any, context: Mapping[str, Any], node: str, donors: Mapping[str, Any]) -> np.ndarray:
    torch = _torch()
    terms = []
    spec = NodeSpec(node_id=node, block=None)
    with QwenNodePatch(model, spec) as patch:
        for path_name in ("yes", "no"):
            path = context["paths"][path_name]
            patch.arm(path["positions"], donors[path_name], sequence_length=path["ids"].shape[1])
            output = model(
                input_ids=path["ids"], attention_mask=path["mask"],
                use_cache=False, logits_to_keep=3,
            )
            patch.disarm()
            terms.append(_path_terms(output, path))
    return q3_native_score_from_terms(_combine_path_terms(terms[0], terms[1])).detach().float().cpu().numpy()


def _q1_donor_modes(full: Mapping[str, Any], null: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if set(full) != {"prefix", "continuations"} or set(null) != set(full):
        raise ValueError("Q1 final-readout donor structure differs")
    modes = {mode: {"prefix": None, "continuations": []} for mode in CANDIDATE_GAP_MODES}
    modes["full_null_direction"]["prefix"] = perturb_state(
        full["prefix"], candidate_gap_direction_tensors(full["prefix"], null["prefix"])["full_null_direction"]
    )
    for mode in CANDIDATE_GAP_MODES:
        directions = candidate_gap_direction_tensors(full["prefix"], null["prefix"])
        modes[mode]["prefix"] = perturb_state(full["prefix"], directions[mode])
    full_chunks = list(full["continuations"])
    null_chunks = list(null["continuations"])
    if len(full_chunks) != len(null_chunks):
        raise ValueError("Q1 final-readout continuation chunk count differs")
    for full_chunk, null_chunk in zip(full_chunks, null_chunks):
        directions = candidate_gap_direction_tensors(full_chunk, null_chunk)
        for mode in CANDIDATE_GAP_MODES:
            modes[mode]["continuations"].append(perturb_state(full_chunk, directions[mode]))
    return modes


def _score_q1_with_donors(model, tokenizer, record, prompt, spec, donors, *, device, batch_size):
    return _score_q1_paths(
        model, tokenizer, record, prompt, patch_spec=spec, donors=donors,
        device=device, batch_size=batch_size,
    )


def _q1_donor_norm_mean(donors: Mapping[str, Any], full: Mapping[str, Any], mode: str) -> float:
    values = []
    for donor, reference in zip(donors["continuations"], full["continuations"]):
        values.append((donor.float() - reference.float()).norm(dim=-1).mean())
    values.append((donors["prefix"].float() - full["prefix"].float()).norm(dim=-1).mean())
    return float(_torch().stack(values).mean().item())


def torch_norm_mean(values: Sequence[Any]) -> float:
    torch = _torch()
    return float(torch.stack([value.float().norm(dim=-1).mean() for value in values]).mean().item())


def _norm_match(direction: Any, reference: Any) -> Any:
    torch = _torch()
    direction_norm = direction.norm(dim=-1, keepdim=True)
    reference_norm = reference.norm(dim=-1, keepdim=True)
    return direction / direction_norm.clamp_min(ORTHOGONAL_EPS) * reference_norm


def _validate_conditions(conditions: Mapping[str, Any], expected_rows: int) -> None:
    if tuple(conditions) != CANDIDATE_GAP_CONDITIONS:
        raise ValueError("candidate-gap condition coverage differs")
    if any(
        value.shape != (expected_rows,) or not np.isfinite(value).all()
        for value in conditions.values()
    ):
        raise FloatingPointError("candidate-gap score condition is incomplete or non-finite")


def _torch() -> Any:
    import torch

    return torch
