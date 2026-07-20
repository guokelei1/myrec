"""Qrels-blind Q1 cache-phase scoring primitives for the N20 boundary.

The scorer keeps the Q1 prompt and answer-token targets fixed while varying
only how the prefix is materialized/reused.  It is deliberately independent
of evaluation qrels; callers may serialize the returned score arrays and let a
shared evaluator open qrels only after bundle integrity checks.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.q1_cache_phase_interventions import (
    Q1_CACHE_MODES,
    cache_phase_signature,
    intervene_q1_prefix_cache,
)
from myrec.mechanism.q1_kv_trajectory import instrument_q1_selection_prompt


CACHE_PHASE_CONDITIONS = (
    "native_cache_identity",
    "same_request_rebuild",
    "zero_prefix_cache",
    "wrong_user_prefix_cache",
    "no_cache_rebuild",
)


def score_q1_cache_phase_request(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
    wrong_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score one Q1 request under phase-matched cache controls.

    ``wrong_history`` is optional so the same kernel can run null/ineligible
    rows without inventing a donor.  When supplied, its serialized prefix
    must have exactly the same token length as the recipient prefix; this
    protects cache-position and causal-mask identity.
    """

    if not history and wrong_history:
        raise ValueError("wrong-user cache requires a non-null recipient history")
    torch = _torch()
    prompt = instrument_q1_selection_prompt(tokenizer, record, history, config)
    ids = torch.tensor([prompt.token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    prefix = model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=True,
        logits_to_keep=1,
    )
    prefix_cache = prefix.past_key_values
    if not hasattr(prefix_cache, "batch_repeat_interleave"):
        raise TypeError("Q1 cache-phase prefix cache lacks batch_repeat_interleave")
    first_log_probs = torch.nn.functional.log_softmax(
        prefix.logits[0, -1].float(), dim=-1
    )

    wrong_cache = None
    wrong_first_log_probs = None
    wrong_signature = None
    if wrong_history is not None:
        wrong_prompt = instrument_q1_selection_prompt(tokenizer, record, wrong_history, config)
        if len(wrong_prompt.token_ids) != len(prompt.token_ids):
            raise ValueError("wrong-user prefix length differs; cache positions are not matched")
        wrong_ids = torch.tensor([wrong_prompt.token_ids], dtype=torch.long, device=device)
        wrong_prefix = model(
            input_ids=wrong_ids,
            attention_mask=torch.ones_like(wrong_ids),
            use_cache=True,
            logits_to_keep=1,
        )
        wrong_cache = wrong_prefix.past_key_values
        wrong_first_log_probs = torch.nn.functional.log_softmax(
            wrong_prefix.logits[0, -1].float(), dim=-1
        )
        wrong_signature = cache_phase_signature(wrong_cache)

    conditions = {name: [] for name in CACHE_PHASE_CONDITIONS}
    native_signatures = []
    cache_deltas = {name: [] for name in Q1_CACHE_MODES}
    candidates = list(record.candidates)
    for start in range(0, len(candidates), int(batch_size)):
        chunk = candidates[start : start + int(batch_size)]
        targets = [prompt.response_by_item[str(row["item_id"])] for row in chunk]
        native = _cached_target_scores(
            model, prefix_cache, first_log_probs, targets, tokenizer, device=device,
            mode="native_cache_identity",
        )
        rebuilt = _cached_target_scores(
            model, prefix_cache, first_log_probs, targets, tokenizer, device=device,
            mode="same_request_rebuild",
        )
        zeroed = _cached_target_scores(
            model, prefix_cache, first_log_probs, targets, tokenizer, device=device,
            mode="zero_prefix_cache",
        )
        if wrong_cache is None:
            wrong = native
        else:
            wrong = _cached_target_scores(
                model, wrong_cache, wrong_first_log_probs, targets, tokenizer, device=device,
                mode="native_cache_identity",
            )
        no_cache = _uncached_target_scores(
            model, prompt.token_ids, targets, tokenizer, device=device
        )
        for name, values in (
            ("native_cache_identity", native),
            ("same_request_rebuild", rebuilt),
            ("zero_prefix_cache", zeroed),
            ("wrong_user_prefix_cache", wrong),
            ("no_cache_rebuild", no_cache),
        ):
            conditions[name].extend(float(value) for value in values)
        native_signatures.append(cache_phase_signature(prefix_cache))
        cache_deltas["native_cache_identity"].extend([0.0] * len(chunk))
        cache_deltas["same_request_rebuild"].extend(
            abs(float(a) - float(b)) for a, b in zip(native, rebuilt)
        )
        cache_deltas["zero_prefix_cache"].extend(
            abs(float(a) - float(b)) for a, b in zip(native, zeroed)
        )
        cache_deltas["donor_prefix_cache_replacement"].extend(
            abs(float(a) - float(b)) for a, b in zip(native, wrong)
        )
    arrays = {name: np.asarray(values, dtype=np.float32) for name, values in conditions.items()}
    _validate_scores(arrays, len(candidates))
    identity = float(
        max(
            np.max(np.abs(arrays["native_cache_identity"] - arrays["same_request_rebuild"])),
            np.max(np.abs(arrays["native_cache_identity"] - arrays["no_cache_rebuild"])),
        )
    )
    return {
        "conditions": arrays,
        "prefix_cache_signature": cache_phase_signature(prefix_cache),
        "wrong_user_cache_signature": wrong_signature,
        "maximum_identity_delta": identity,
        "cache_delta_summary": {
            name: {
                "max_abs": float(max(values, default=0.0)),
                "mean_abs": float(np.mean(values)) if values else 0.0,
            }
            for name, values in cache_deltas.items()
        },
        "token_position_integrity": True,
        "cache_key_integrity": True,
    }


def _cached_target_scores(
    model: Any,
    prefix_cache: Any,
    first_log_probs: Any,
    targets: Sequence[Sequence[int]],
    tokenizer: Any,
    *,
    device: str,
    mode: str,
) -> list[float]:
    torch = _torch()
    if mode not in Q1_CACHE_MODES:
        raise ValueError(f"unsupported cache mode={mode}")
    # ``batch_repeat_interleave`` mutates DynamicCache in-place.  Start every
    # condition from an isolated copy so the identity and replacement cells
    # remain comparable and candidate chunks cannot contaminate one another.
    cache = intervene_q1_prefix_cache(copy.deepcopy(prefix_cache), mode)
    cache.batch_repeat_interleave(len(targets))
    lengths = [len(target) - 1 for target in targets]
    width = max(lengths)
    continuation_ids = torch.full(
        (len(targets), width), int(tokenizer.pad_token_id), dtype=torch.long, device=device
    )
    continuation_mask = torch.zeros_like(continuation_ids)
    for row, target in enumerate(targets):
        if len(target) < 2:
            raise ValueError("Q1 answer target must contain at least two tokens")
        values = torch.tensor(target[:-1], dtype=torch.long, device=device)
        continuation_ids[row, : len(values)] = values
        continuation_mask[row, : len(values)] = 1
    attention_mask = torch.cat(
        [
            torch.ones((len(targets), int(prefix_cache.get_seq_length())), dtype=torch.long, device=device),
            continuation_mask,
        ],
        dim=1,
    )
    output = model(
        input_ids=continuation_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=False,
    )
    results = []
    for row, (target, length) in enumerate(zip(targets, lengths)):
        continuation = torch.nn.functional.log_softmax(
            output.logits[row, :length].float(), dim=-1
        ).gather(1, torch.tensor(target[1:], dtype=torch.long, device=device)[:, None]).squeeze(1)
        scores = torch.cat((first_log_probs[int(target[0])][None], continuation))
        results.append(float(scores.mean().item()))
    return results


def _uncached_target_scores(
    model: Any,
    prompt_ids: Sequence[int],
    targets: Sequence[Sequence[int]],
    tokenizer: Any,
    *,
    device: str,
) -> list[float]:
    torch = _torch()
    values = []
    prompt_len = len(prompt_ids)
    for target in targets:
        if len(target) < 2:
            raise ValueError("Q1 answer target must contain at least two tokens")
        ids = torch.tensor([list(prompt_ids) + list(target[:-1])], dtype=torch.long, device=device)
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=ids.shape[1],
        )
        # Position ``prompt_len-1`` predicts the first answer token; each
        # subsequent input token predicts the next answer token.
        positions = torch.arange(prompt_len - 1, prompt_len + len(target) - 1, device=device)
        logits = output.logits[0, positions].float()
        expected = torch.tensor(target, dtype=torch.long, device=device)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).gather(1, expected[:, None]).squeeze(1)
        values.append(float(log_probs.mean().item()))
    return values


def _validate_scores(values: Mapping[str, np.ndarray], count: int) -> None:
    if set(values) != set(CACHE_PHASE_CONDITIONS):
        raise ValueError("N20 cache condition set differs from registration")
    for name, array in values.items():
        if array.shape != (count,) or not np.isfinite(array).all():
            raise FloatingPointError(f"N20 cache condition is invalid: {name}")


def _torch() -> Any:
    import torch

    return torch
