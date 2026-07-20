"""Native final-RMSNorm/readout patches for breadth models Q0 and Q1."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.patch_scorer import _left_pad_sequences
from myrec.mechanism.q0_representation_prompt import instrument_q0_pointwise_prompt
from myrec.mechanism.q1_branch_scoring import _capture_q1_paths, _score_q1_paths
from myrec.mechanism.q1_kv_trajectory import instrument_q1_selection_prompt
from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
    QwenNodePatch,
)


BREADTH_READOUT_NODES = ("final_rmsnorm_input", "final_rmsnorm_output")
BREADTH_READOUT_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    *tuple(f"{node}_full_identity" for node in BREADTH_READOUT_NODES),
    *tuple(f"{node}_same_to_null" for node in BREADTH_READOUT_NODES),
)


def score_q0_readout_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    """Patch both final-norm nodes on Q0's native yes/no scoring path."""

    if not candidates:
        raise ValueError("Q0 readout candidate chunk is empty")
    if config["method_id"] != "q0_qwen3_reranker_06b":
        raise ValueError("Q0 readout scorer received another model")
    full = _build_q0_path(tokenizer, record, candidates, record.history, config, device)
    null = _build_q0_path(tokenizer, record, candidates, [], config, device)
    specs = tuple(NodeSpec(node_id=node, block=None) for node in BREADTH_READOUT_NODES)
    with QwenNodeCapture(model, specs) as capture:
        full_output, donors = capture.capture_forward(
            input_ids=full["ids"],
            attention_mask=full["mask"],
            positions=full["positions"],
            model_kwargs={"logits_to_keep": 1},
        )
    baseline_full = _q0_scores(full_output)
    baseline_null = _q0_scores(
        model(
            input_ids=null["ids"],
            attention_mask=null["mask"],
            use_cache=False,
            logits_to_keep=1,
        )
    )
    conditions = {"baseline_full": baseline_full, "baseline_null": baseline_null}
    maximum_identity = 0.0
    for node in BREADTH_READOUT_NODES:
        spec = NodeSpec(node_id=node, block=None)
        donor = donors[spec.key]
        identity = _patched_q0_scores(model, spec, full, donor)
        same = _patched_q0_scores(model, spec, null, donor)
        conditions[f"{node}_full_identity"] = identity
        conditions[f"{node}_same_to_null"] = same
        maximum_identity = max(
            maximum_identity, float(np.max(np.abs(identity - baseline_full)))
        )
    if set(conditions) != set(BREADTH_READOUT_CONDITIONS) or any(
        values.shape != (len(candidates),) or not np.isfinite(values).all()
        for values in conditions.values()
    ):
        raise FloatingPointError("Q0 readout score condition coverage differs")
    return {"conditions": conditions, "maximum_identity_delta": maximum_identity}


def score_q1_readout_record(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    config: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Patch final-norm nodes in Q1's prefix and every response-token call."""

    if config["method_id"] != "q1_instructrec_generalqwen":
        raise ValueError("Q1 readout scorer received another model")
    if int(batch_size) <= 0:
        raise ValueError("Q1 readout batch size must be positive")
    specs = tuple(NodeSpec(node_id=node, block=None) for node in BREADTH_READOUT_NODES)
    full_prompt = instrument_q1_selection_prompt(tokenizer, record, record.history, config)
    null_prompt = instrument_q1_selection_prompt(tokenizer, record, [], config)
    full = _capture_q1_paths(
        model,
        tokenizer,
        record,
        full_prompt,
        specs,
        device=device,
        batch_size=int(batch_size),
    )
    null = _score_q1_paths(
        model,
        tokenizer,
        record,
        null_prompt,
        patch_spec=None,
        donors=None,
        device=device,
        batch_size=int(batch_size),
    )
    conditions = {"baseline_full": full["scores"], "baseline_null": null["scores"]}
    maximum_identity = 0.0
    call_audit = {
        "full_capture": full["call_audit"],
        "null_baseline": null["call_audit"],
        "patched": {},
    }
    for node in BREADTH_READOUT_NODES:
        spec = NodeSpec(node_id=node, block=None)
        identity = _score_q1_paths(
            model,
            tokenizer,
            record,
            full_prompt,
            patch_spec=spec,
            donors=full["donors"][spec.key],
            device=device,
            batch_size=int(batch_size),
        )
        same = _score_q1_paths(
            model,
            tokenizer,
            record,
            null_prompt,
            patch_spec=spec,
            donors=full["donors"][spec.key],
            device=device,
            batch_size=int(batch_size),
        )
        conditions[f"{node}_full_identity"] = identity["scores"]
        conditions[f"{node}_same_to_null"] = same["scores"]
        maximum_identity = max(
            maximum_identity,
            float(np.max(np.abs(identity["scores"] - full["scores"]))),
        )
        call_audit["patched"][node] = {
            "full_identity": identity["call_audit"],
            "same_to_null": same["call_audit"],
        }
    candidates = len(record.candidates)
    if set(conditions) != set(BREADTH_READOUT_CONDITIONS) or any(
        values.shape != (candidates,) or not np.isfinite(values).all()
        for values in conditions.values()
    ):
        raise FloatingPointError("Q1 readout score condition coverage differs")
    return {
        "conditions": conditions,
        "maximum_identity_delta": maximum_identity,
        "call_audit": call_audit,
        "response_tokens": int(full["response_tokens"]),
    }


def _build_q0_path(tokenizer, record, candidates, history, config, device):
    prompts = [
        instrument_q0_pointwise_prompt(
            tokenizer,
            config["method_id"],
            record,
            candidate,
            history=history,
            history_budget=int(config["training"]["history_budget"]),
            max_length=int(config["training"]["max_length"]),
        )
        for candidate in candidates
    ]
    ids, mask, padding = _left_pad_sequences(
        [list(prompt.token_ids) for prompt in prompts], tokenizer.pad_token_id, device
    )
    import torch

    positions = torch.tensor(
        [[left + prompt.candidate_readout] for left, prompt in zip(padding, prompts)],
        dtype=torch.long,
        device=device,
    )
    if any(int(position) != ids.shape[1] - 1 for position in positions[:, 0]):
        raise ValueError("Q0 native readout is not the final prompt token")
    return {"ids": ids, "mask": mask, "positions": positions}


def _patched_q0_scores(
    model, spec, path, donor, *, yes_token_id=9693, no_token_id=2152
):
    with QwenNodePatch(model, spec) as patch:
        patch.arm(path["positions"], donor, sequence_length=path["ids"].shape[1])
        output = model(
            input_ids=path["ids"],
            attention_mask=path["mask"],
            use_cache=False,
            logits_to_keep=1,
        )
        patch.disarm()
    return _q0_scores(
        output, yes_token_id=yes_token_id, no_token_id=no_token_id
    )


def _q0_scores(output, *, yes_token_id=9693, no_token_id=2152):
    return (
        output.logits[:, -1, int(yes_token_id)]
        - output.logits[:, -1, int(no_token_id)]
    ).float().cpu().numpy()
