"""Exact Q1 listwise KV-cache branch patches over every response token."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.q1_kv_trajectory import instrument_q1_selection_prompt
from myrec.mechanism.q0_branch_scoring import Q0_BRANCH_NODES
from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
    QwenNodePatch,
)


Q1_BRANCH_NODES = Q0_BRANCH_NODES
Q1_BRANCH_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    *tuple(f"{node}_full_identity" for node in Q1_BRANCH_NODES),
    *tuple(f"{node}_same_to_null" for node in Q1_BRANCH_NODES),
)


def score_q1_branch_record(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Patch four nodes in prefix and all cached candidate continuations."""

    if config["method_id"] != "q1_instructrec_generalqwen":
        raise ValueError("Q1 branch scorer received another model")
    block = int(block)
    if block not in (13, 20, 27):
        raise ValueError("Q1 branch block must be 13,20,or27")
    if int(batch_size) <= 0:
        raise ValueError("Q1 branch batch size must be positive")
    specs = tuple(NodeSpec(node_id=node, block=block) for node in Q1_BRANCH_NODES)
    full_prompt = instrument_q1_selection_prompt(tokenizer, record, record.history, config)
    null_prompt = instrument_q1_selection_prompt(tokenizer, record, [], config)
    full = _capture_q1_paths(
        model, tokenizer, record, full_prompt, specs,
        device=device, batch_size=int(batch_size),
    )
    null_scores = _score_q1_paths(
        model, tokenizer, record, null_prompt, patch_spec=None, donors=None,
        device=device, batch_size=int(batch_size),
    )
    conditions = {
        "baseline_full": full["scores"],
        "baseline_null": null_scores["scores"],
    }
    maximum_identity = 0.0
    call_audit = {
        "full_capture": full["call_audit"],
        "null_baseline": null_scores["call_audit"],
        "patched": {},
    }
    for node in Q1_BRANCH_NODES:
        spec = NodeSpec(node_id=node, block=block)
        identity = _score_q1_paths(
            model, tokenizer, record, full_prompt, patch_spec=spec,
            donors=full["donors"][spec.key], device=device,
            batch_size=int(batch_size),
        )
        same = _score_q1_paths(
            model, tokenizer, record, null_prompt, patch_spec=spec,
            donors=full["donors"][spec.key], device=device,
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
    if set(conditions) != set(Q1_BRANCH_CONDITIONS) or any(
        values.shape != (candidates,) or not np.isfinite(values).all()
        for values in conditions.values()
    ):
        raise FloatingPointError("Q1 branch score condition coverage differs")
    return {
        "conditions": conditions,
        "maximum_identity_delta": maximum_identity,
        "call_audit": call_audit,
        "response_tokens": int(full["response_tokens"]),
    }


def _capture_q1_paths(model, tokenizer, record, prompt, specs, *, device, batch_size):
    torch = _torch()
    ids = torch.tensor([prompt.token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    position = torch.tensor([[prompt.prompt_readout]], dtype=torch.long, device=device)
    donors = {spec.key: {"prefix": None, "continuations": []} for spec in specs}
    with QwenNodeCapture(model, specs) as capture:
        capture.arm(position, sequence_length=ids.shape[1])
        prefix = model(
            input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1
        )
        prefix_nodes = capture.disarm()
        for spec in specs:
            donors[spec.key]["prefix"] = prefix_nodes[spec.key]
        scores, response_tokens, continuation_calls = _continuation_chunks(
            model, tokenizer, record, prompt, prefix,
            device=device, batch_size=batch_size,
            capture=capture, specs=specs, donors=donors,
        )
    return {
        "scores": scores,
        "donors": donors,
        "response_tokens": response_tokens,
        "call_audit": {
            "prefix_calls": 1,
            "continuation_calls": continuation_calls,
            "response_tokens": response_tokens,
            "all_response_tokens_captured": True,
        },
    }


def _score_q1_paths(
    model, tokenizer, record, prompt, *, patch_spec, donors, device, batch_size
):
    torch = _torch()
    ids = torch.tensor([prompt.token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    position = torch.tensor([[prompt.prompt_readout]], dtype=torch.long, device=device)
    patch_context = (
        _NullContext()
        if patch_spec is None
        else QwenNodePatch(model, patch_spec)
    )
    with patch_context as patch:
        if patch_spec is not None:
            patch.arm(position, donors["prefix"], sequence_length=ids.shape[1])
        prefix = model(
            input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1
        )
        if patch_spec is not None:
            patch.disarm()
        scores, response_tokens, continuation_calls = _continuation_chunks(
            model, tokenizer, record, prompt, prefix,
            device=device, batch_size=batch_size,
            patch=patch if patch_spec is not None else None,
            patch_donors=(donors["continuations"] if donors is not None else None),
        )
    return {
        "scores": scores,
        "response_tokens": response_tokens,
        "call_audit": {
            "prefix_calls": 1,
            "continuation_calls": continuation_calls,
            "response_tokens": response_tokens,
            "all_response_tokens_patched": patch_spec is not None,
        },
    }


def _continuation_chunks(
    model,
    tokenizer,
    record,
    prompt,
    prefix,
    *,
    device,
    batch_size,
    capture=None,
    specs: Sequence[NodeSpec] = (),
    donors=None,
    patch=None,
    patch_donors=None,
):
    torch = _torch()
    first_log_probs = torch.nn.functional.log_softmax(prefix.logits[0, -1].float(), dim=-1)
    prefix_cache = prefix.past_key_values
    if not hasattr(prefix_cache, "batch_repeat_interleave"):
        raise TypeError("Q1 branch prefix cache lacks batch_repeat_interleave")
    scores = []
    response_tokens = 0
    continuation_calls = 0
    candidates = list(record.candidates)
    for chunk_index, start in enumerate(range(0, len(candidates), batch_size)):
        chunk = candidates[start : start + batch_size]
        targets = [prompt.response_by_item[str(row["item_id"])] for row in chunk]
        lengths = [len(target) - 1 for target in targets]
        width = max(lengths)
        if width <= 0:
            raise ValueError("Q1 branch response lacks continuation tokens")
        continuation_ids = torch.full(
            (len(targets), width), int(tokenizer.pad_token_id),
            dtype=torch.long, device=device,
        )
        continuation_mask = torch.zeros_like(continuation_ids)
        for row, target in enumerate(targets):
            values = torch.tensor(target[:-1], dtype=torch.long, device=device)
            continuation_ids[row, :len(values)] = values
            continuation_mask[row, :len(values)] = 1
        attention_mask = torch.cat(
            [
                torch.ones((len(targets), len(prompt.token_ids)), dtype=torch.long, device=device),
                continuation_mask,
            ],
            dim=1,
        )
        cache = copy.deepcopy(prefix_cache)
        cache.batch_repeat_interleave(len(targets))
        positions = torch.arange(width, device=device)[None, :].expand(len(targets), -1)
        if capture is not None:
            capture.arm(positions, sequence_length=width)
        if patch is not None:
            if patch_donors is None or chunk_index >= len(patch_donors):
                raise ValueError("Q1 branch continuation donor coverage differs")
            patch.arm(positions, patch_donors[chunk_index], sequence_length=width)
        output = model(
            input_ids=continuation_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=False,
        )
        if capture is not None:
            captured = capture.disarm()
            for spec in specs:
                donors[spec.key]["continuations"].append(captured[spec.key])
        if patch is not None:
            patch.disarm()
        for row, (target, length) in enumerate(zip(targets, lengths)):
            expected = torch.tensor(target[1:], dtype=torch.long, device=device)
            continuation = torch.nn.functional.log_softmax(
                output.logits[row, :length].float(), dim=-1
            ).gather(1, expected[:, None]).squeeze(1)
            token_values = torch.cat((first_log_probs[int(target[0])][None], continuation))
            scores.append(float(token_values.mean().item()))
            response_tokens += len(target)
        continuation_calls += 1
    if len(scores) != len(candidates):
        raise ValueError("Q1 branch candidate score coverage differs")
    return np.asarray(scores, dtype=np.float32), response_tokens, continuation_calls


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


def _torch():
    import torch
    return torch
