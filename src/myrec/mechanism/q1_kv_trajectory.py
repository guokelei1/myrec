"""Phase-aware Q1 listwise KV-cache trajectory instrumentation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import (
    ModelRecord,
    build_instructrec_selection_sections,
    serialize_history,
)
from myrec.baselines.motivation_v12_ranker import (
    encode_instructrec_selection_prompt,
    instructrec_template_index,
)
from myrec.mechanism.q0_trajectory_evaluator import trajectory_geometry
from myrec.mechanism.representation_probe import (
    MechanicalPositionError,
    _encode,
    _encode_with_offsets,
    _token_covering_span_end,
)
from myrec.mechanism.representation_runtime import resolve_transformer_layers


Q1_METHOD_ID = "q1_instructrec_generalqwen"
Q1_STATE_INDICES = tuple(range(29))
Q1_REQUEST_POSITIONS = ("query_end", "history_summary_end", "prompt_readout")


@dataclass(frozen=True)
class InstrumentedQ1Prompt:
    token_ids: tuple[int, ...]
    query_end: int
    history_summary_end: int
    prompt_readout: int
    response_by_item: Mapping[str, tuple[int, ...]]
    template_index: int
    audit: Mapping[str, Any]


class Q1AllStateCapture:
    """Capture selected rows across every residual state for each Q1 phase."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.torch = _torch()
        self.layers = resolve_transformer_layers(model)
        self.embedding = model.get_input_embeddings()
        if self.embedding is None:
            raise TypeError("Q1 trajectory model has no input embedding")
        self.positions: Any = None
        self.values: dict[int, Any] = {}
        self.counts: dict[int, int] = {}
        self.handles: list[Any] = []
        self.call_shapes: list[dict[str, Any]] = []

    def __enter__(self) -> "Q1AllStateCapture":
        self.handles.append(self.embedding.register_forward_hook(self._hook(0)))
        for state in Q1_STATE_INDICES[1:]:
            self.handles.append(
                self.layers[state - 1].register_forward_hook(self._hook(state))
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.positions = None
        self.values.clear()
        self.counts.clear()

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        positions: Any,
        *,
        phase: str,
        model_kwargs: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        if self.positions is not None or positions.ndim != 2:
            raise ValueError("Q1 capture positions must be unarmed [batch,count]")
        if positions.shape[0] != input_ids.shape[0]:
            raise ValueError("Q1 capture position batch differs")
        positions = positions.to(input_ids.device)
        if int(positions.min()) < 0 or int(positions.max()) >= input_ids.shape[1]:
            raise ValueError("Q1 capture position is outside current phase")
        self.positions = positions
        self.values = {}
        self.counts = {}
        try:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **dict(model_kwargs),
            )
        except Exception:
            self.positions = None
            self.values = {}
            self.counts = {}
            raise
        if set(self.values) != set(Q1_STATE_INDICES) or any(
            self.counts.get(state) != 1 for state in Q1_STATE_INDICES
        ):
            raise RuntimeError("Q1 all-state hooks did not fire exactly once")
        states = self.torch.stack(
            [self.values[state] for state in Q1_STATE_INDICES], dim=2
        )
        self.call_shapes.append(
            {
                "phase": str(phase),
                "input_shape": list(input_ids.shape),
                "positions_shape": list(positions.shape),
                "states_shape": list(states.shape),
            }
        )
        self.positions = None
        self.values = {}
        self.counts = {}
        return output, states

    def _hook(self, state: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if self.positions is None:
                raise RuntimeError("Q1 all-state hook fired while unarmed")
            hidden = output[0] if isinstance(output, tuple) else output
            rows = self.torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            self.values[state] = hidden[rows, self.positions].detach()
            self.counts[state] = self.counts.get(state, 0) + 1

        return hook


def instrument_q1_selection_prompt(
    tokenizer: Any,
    record: ModelRecord,
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> InstrumentedQ1Prompt:
    """Locate Q1 request/readout positions in its unchanged listwise prompt."""

    training = config["training"]
    template = instructrec_template_index(
        record.request_id, seed=int(training["seed"])
    )
    max_target = int(training.get("max_target_length", 96))
    prompt, response_by_item, audit = encode_instructrec_selection_prompt(
        tokenizer,
        record,
        record.candidates,
        history=list(history),
        history_budget=int(training["history_budget"]),
        template_index=template,
        max_length=int(training["max_length"]) - max_target,
        context_token_budget=int(training["context_token_budget"]),
        max_target_length=max_target,
    )
    sections = build_instructrec_selection_sections(
        record,
        record.candidates,
        history=list(history),
        history_budget=int(training["history_budget"]),
        template_index=template,
    )
    prefix = (
        f"<|im_start|>system\n{sections.system}<|im_end|>\n"
        "<|im_start|>user\n"
    )
    prefix_ids = _encode(tokenizer, prefix)
    context_ids, offsets = _encode_with_offsets(tokenizer, sections.context)
    visible_context = min(len(context_ids), int(training["context_token_budget"]))
    history_text = serialize_history(
        history, history_budget=int(training["history_budget"])
    )
    query_span, history_span = q1_context_spans(
        sections.context,
        query=record.query,
        history_text=history_text,
        template_index=template,
    )
    query_token = _token_covering_span_end(offsets, query_span[1], name="query_end")
    history_token = _token_covering_span_end(
        offsets, history_span[1], name="history_summary_end"
    )
    if query_token >= visible_context or history_token >= visible_context:
        raise MechanicalPositionError(
            "q1_request_endpoint_truncated",
            "Q1 query/history endpoint exceeds the fixed context budget",
        )
    query_position = len(prefix_ids) + query_token
    history_position = len(prefix_ids) + history_token
    candidate_start = len(prefix_ids) + visible_context
    readout = len(prompt) - 1
    if not (0 <= query_position < history_position < candidate_start <= readout):
        raise MechanicalPositionError(
            "q1_noncausal_position_order",
            "expected Q1 query < history < slate <= prompt readout",
        )
    return InstrumentedQ1Prompt(
        token_ids=tuple(map(int, prompt)),
        query_end=query_position,
        history_summary_end=history_position,
        prompt_readout=readout,
        response_by_item={
            str(item_id): tuple(map(int, target))
            for item_id, target in response_by_item.items()
        },
        template_index=template,
        audit=dict(audit),
    )


def q1_context_spans(
    context: str,
    *,
    query: str,
    history_text: str,
    template_index: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if template_index == 0:
        query_marker = "Current user intention (query): "
        history_marker = "\nImplicit preference evidence (newest first):\n"
    elif template_index == 1:
        query_marker = "Search query: "
        history_marker = "\nInteraction history:\n"
    else:
        raise ValueError("Q1 template index must be zero or one")
    if context.count(query_marker) != 1 or context.count(history_marker) != 1:
        raise MechanicalPositionError(
            "q1_context_template_drift", "Q1 query/history marker uniqueness changed"
        )
    query_start = context.index(query_marker) + len(query_marker)
    query_end = query_start + len(query)
    if context[query_start:query_end] != query:
        raise MechanicalPositionError("q1_query_span_mismatch", "Q1 query bytes changed")
    if not context.startswith(history_marker, query_end):
        raise MechanicalPositionError(
            "q1_context_template_drift", "Q1 history marker changed"
        )
    history_start = query_end + len(history_marker)
    history_end = history_start + len(history_text)
    if context[history_start:history_end] != history_text:
        raise MechanicalPositionError(
            "q1_history_span_mismatch", "Q1 serialized history bytes changed"
        )
    if not query or not history_text:
        raise MechanicalPositionError(
            "q1_empty_span", "Q1 query/history span cannot be empty"
        )
    return (query_start, query_end), (history_start, history_end)


def capture_q1_request_trajectory(
    model: Any,
    capture: Q1AllStateCapture,
    tokenizer: Any,
    record: ModelRecord,
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Run Q1's exact prefix-cache scorer and retain every response state."""

    torch = _torch()
    prompt = instrument_q1_selection_prompt(tokenizer, record, history, config)
    ids = torch.tensor([prompt.token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    positions = torch.tensor(
        [[prompt.query_end, prompt.history_summary_end, prompt.prompt_readout]],
        dtype=torch.long,
        device=device,
    )
    prefix, prefix_states = capture.forward(
        ids,
        mask,
        positions,
        phase="prefix",
        model_kwargs={"use_cache": True, "logits_to_keep": 1},
    )
    first_log_probs = torch.nn.functional.log_softmax(
        prefix.logits[0, -1].float(), dim=-1
    )
    prefix_cache = prefix.past_key_values
    if not hasattr(prefix_cache, "batch_repeat_interleave"):
        raise TypeError("Q1 trajectory prefix cache lacks batch_repeat_interleave")
    results = []
    candidates = list(record.candidates)
    for start in range(0, len(candidates), int(batch_size)):
        chunk = candidates[start : start + int(batch_size)]
        targets = [prompt.response_by_item[str(row["item_id"])] for row in chunk]
        lengths = [len(target) - 1 for target in targets]
        width = max(lengths)
        if width <= 0:
            raise ValueError("Q1 registered response target lacks continuation tokens")
        continuation_ids = torch.full(
            (len(targets), width),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        continuation_mask = torch.zeros_like(continuation_ids)
        for row, target in enumerate(targets):
            values = torch.tensor(target[:-1], dtype=torch.long, device=device)
            continuation_ids[row, : len(values)] = values
            continuation_mask[row, : len(values)] = 1
        attention_mask = torch.cat(
            [
                torch.ones(
                    (len(targets), len(prompt.token_ids)),
                    dtype=torch.long,
                    device=device,
                ),
                continuation_mask,
            ],
            dim=1,
        )
        cache = copy.deepcopy(prefix_cache)
        cache.batch_repeat_interleave(len(targets))
        continuation_positions = torch.arange(width, device=device)[None, :].expand(
            len(targets), -1
        )
        output, states = capture.forward(
            continuation_ids,
            attention_mask,
            continuation_positions,
            phase="cached_continuation",
            model_kwargs={"past_key_values": cache, "use_cache": False},
        )
        for row, (candidate, target, length) in enumerate(zip(chunk, targets, lengths)):
            continuation_logits = output.logits[row, :length].float()
            expected = torch.tensor(target[1:], dtype=torch.long, device=device)
            continuation_log_probs = torch.nn.functional.log_softmax(
                continuation_logits, dim=-1
            ).gather(1, expected[:, None]).squeeze(1)
            token_log_probs = torch.cat(
                (first_log_probs[int(target[0])][None], continuation_log_probs)
            )
            response_states = torch.cat(
                (prefix_states[:, 2:3], states[row : row + 1, :length]), dim=1
            )[0]
            if response_states.shape[:2] != (len(target), 29):
                raise ValueError("Q1 response-state/token coverage differs")
            results.append(
                {
                    "candidate_item_id": str(candidate["item_id"]),
                    "candidate_ordinal": start + row,
                    "target_length": len(target),
                    "score": float(token_log_probs.mean().item()),
                    "token_log_probs": token_log_probs.detach(),
                    "response_states": response_states.detach(),
                }
            )
    if [row["candidate_item_id"] for row in results] != [
        str(row["item_id"]) for row in candidates
    ]:
        raise ValueError("Q1 trajectory candidate identity/order drift")
    return {
        "prompt": prompt,
        "request_states": prefix_states[0, :2].detach(),
        "prompt_readout_states": prefix_states[0, 2].detach(),
        "candidates": results,
    }


def compare_q1_full_null(full: Mapping[str, Any], null: Mapping[str, Any]) -> dict[str, Any]:
    """Pair aligned Q1 phases and return scalar all-layer geometry."""

    if len(full["candidates"]) != len(null["candidates"]):
        raise ValueError("Q1 full/null candidate coverage differs")
    request = trajectory_geometry(
        full["request_states"].float().cpu().numpy(),
        null["request_states"].float().cpu().numpy(),
    )
    # trajectory_geometry expects [row,29,hidden]; request rows are query/history.
    prompt = trajectory_geometry(
        full["prompt_readout_states"][None].float().cpu().numpy(),
        null["prompt_readout_states"][None].float().cpu().numpy(),
    )
    candidate_rows = []
    for full_row, null_row in zip(full["candidates"], null["candidates"]):
        if (
            full_row["candidate_item_id"] != null_row["candidate_item_id"]
            or full_row["target_length"] != null_row["target_length"]
        ):
            raise ValueError("Q1 full/null response identity/length differs")
        token_geometry = trajectory_geometry(
            full_row["response_states"].float().cpu().numpy(),
            null_row["response_states"].float().cpu().numpy(),
        )
        candidate_rows.append(
            {
                "candidate_item_id": full_row["candidate_item_id"],
                "candidate_ordinal": full_row["candidate_ordinal"],
                "target_length": full_row["target_length"],
                "full_score": full_row["score"],
                "null_score": null_row["score"],
                "mean_token_geometry": {
                    metric: values.mean(axis=0) for metric, values in token_geometry.items()
                },
                "first_token_geometry": {
                    metric: values[0] for metric, values in token_geometry.items()
                },
                "continuation_geometry": {
                    metric: values[1:].mean(axis=0) for metric, values in token_geometry.items()
                },
            }
        )
    result = {
        "request_geometry": request,
        "prompt_readout_geometry": {metric: value[0] for metric, value in prompt.items()},
        "candidates": candidate_rows,
    }
    if not _all_finite(result):
        raise FloatingPointError("Q1 full/null trajectory is non-finite")
    return result


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _torch() -> Any:
    import torch

    return torch
