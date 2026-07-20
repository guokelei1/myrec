"""Native Q0 branch-aggregate patches at fixed Transformer blocks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.patch_scorer import _left_pad_sequences
from myrec.mechanism.q0_representation_prompt import instrument_q0_pointwise_prompt
from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
    QwenNodePatch,
)


Q0_BRANCH_NODES = (
    "block_input_residual",
    "attention_o_projection",
    "mlp_down_projection",
    "block_output_residual",
)
Q0_BRANCH_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    *tuple(f"{node}_full_identity" for node in Q0_BRANCH_NODES),
    *tuple(f"{node}_same_to_null" for node in Q0_BRANCH_NODES),
)


def score_q0_branch_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    """Patch all four registered aggregate nodes for one Q0 candidate chunk."""

    if not candidates:
        raise ValueError("Q0 branch candidate chunk is empty")
    if config["method_id"] != "q0_qwen3_reranker_06b":
        raise ValueError("Q0 branch scorer received another model")
    block = int(block)
    if block not in (13, 20, 27):
        raise ValueError("Q0 branch block must be 13,20,or27")
    full = _build_q0_path(tokenizer, record, candidates, record.history, config, device)
    null = _build_q0_path(tokenizer, record, candidates, [], config, device)
    specs = tuple(NodeSpec(node_id=node, block=block) for node in Q0_BRANCH_NODES)
    with QwenNodeCapture(model, specs) as capture:
        full_output, donors = capture.capture_forward(
            input_ids=full["ids"], attention_mask=full["mask"], positions=full["positions"],
            model_kwargs={"logits_to_keep": 1},
        )
    baseline_full = _q0_scores(full_output)
    baseline_null = _q0_scores(
        model(
            input_ids=null["ids"], attention_mask=null["mask"],
            use_cache=False, logits_to_keep=1,
        )
    )
    conditions = {
        "baseline_full": baseline_full,
        "baseline_null": baseline_null,
    }
    maximum_identity = 0.0
    for node in Q0_BRANCH_NODES:
        spec = NodeSpec(node_id=node, block=block)
        donor = donors[spec.key]
        identity = _patched_scores(model, spec, full, donor)
        same = _patched_scores(model, spec, null, donor)
        conditions[f"{node}_full_identity"] = identity
        conditions[f"{node}_same_to_null"] = same
        maximum_identity = max(
            maximum_identity,
            float(np.max(np.abs(identity - baseline_full))),
        )
    if set(conditions) != set(Q0_BRANCH_CONDITIONS) or any(
        values.shape != (len(candidates),) or not np.isfinite(values).all()
        for values in conditions.values()
    ):
        raise FloatingPointError("Q0 branch score condition coverage differs")
    return {"conditions": conditions, "maximum_identity_delta": maximum_identity}


def _build_q0_path(tokenizer, record, candidates, history, config, device):
    prompts = [
        instrument_q0_pointwise_prompt(
            tokenizer, config["method_id"], record, candidate,
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
    return {"ids": ids, "mask": mask, "positions": positions}


def _patched_scores(model, spec, path, donor):
    with QwenNodePatch(model, spec) as patch:
        patch.arm(path["positions"], donor, sequence_length=path["ids"].shape[1])
        output = model(
            input_ids=path["ids"], attention_mask=path["mask"],
            use_cache=False, logits_to_keep=1,
        )
        patch.disarm()
    return _q0_scores(output)


def _q0_scores(output):
    return (
        output.logits[:, -1, 9693] - output.logits[:, -1, 2152]
    ).float().cpu().numpy()
