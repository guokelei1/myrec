"""Exact Q0 specialized-reranker positions for deep-dive breadth capture."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import (
    ModelRecord,
    build_prompt_sections,
    serialize_history,
)
from myrec.baselines.motivation_v12_ranker import encode_prompt_sections
from myrec.mechanism.representation_probe import (
    ANSWER_SUFFIX,
    PREFIX_TEMPLATE,
    InstrumentedPrompt,
    MechanicalPositionError,
    _encode,
    _encode_with_offsets,
    _token_covering_span_end,
)


Q0_METHOD_ID = "q0_qwen3_reranker_06b"


def instrument_q0_pointwise_prompt(
    tokenizer: Any,
    method_id: str,
    record: ModelRecord,
    candidate: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]],
    history_budget: int,
    max_length: int,
) -> InstrumentedPrompt:
    """Locate Q0 query/history/readout positions without changing its prompt."""

    if method_id != Q0_METHOD_ID:
        raise ValueError("Q0 prompt instrument received another method")
    if max_length < 64:
        raise ValueError("max_length must be at least 64")
    sections = build_prompt_sections(
        method_id,
        record,
        dict(candidate),
        history=list(history),
        history_budget=history_budget,
    )
    prefix_ids = _encode(tokenizer, PREFIX_TEMPLATE.format(system=sections.system))
    context_ids, context_offsets = _encode_with_offsets(tokenizer, sections.context)
    candidate_ids = _encode(tokenizer, sections.candidate)
    suffix_ids = _encode(tokenizer, ANSWER_SUFFIX)
    body_budget = max_length - len(prefix_ids) - len(suffix_ids)
    if body_budget < 16:
        raise ValueError("max_length leaves no room for Q0 prompt body")
    candidate_budget = min(len(candidate_ids), max(8, body_budget // 2))
    context_budget = body_budget - candidate_budget
    if len(context_ids) < context_budget:
        candidate_budget = min(len(candidate_ids), body_budget - len(context_ids))
        context_budget = body_budget - candidate_budget
    elif len(candidate_ids) < candidate_budget:
        context_budget = body_budget - len(candidate_ids)
        candidate_budget = len(candidate_ids)

    history_text = serialize_history(history, history_budget=history_budget)
    query_span, history_span = q0_context_spans(
        sections.context, query=record.query, history_text=history_text
    )
    query_token = _token_covering_span_end(
        context_offsets, query_span[1], name="query_end"
    )
    history_token = _token_covering_span_end(
        context_offsets, history_span[1], name="history_summary_end"
    )
    if query_token >= context_budget:
        raise MechanicalPositionError(
            "query_endpoint_truncated",
            f"query token {query_token} outside context budget {context_budget}",
        )
    if history_token >= context_budget:
        raise MechanicalPositionError(
            "history_endpoint_truncated",
            f"history token {history_token} outside context budget {context_budget}",
        )
    if candidate_budget != len(candidate_ids):
        raise MechanicalPositionError(
            "candidate_text_truncated",
            f"visible candidate tokens {candidate_budget}/{len(candidate_ids)}",
        )
    token_ids = (
        prefix_ids
        + context_ids[:context_budget]
        + candidate_ids[:candidate_budget]
        + suffix_ids
    )
    frozen = encode_prompt_sections(tokenizer, sections, max_length=max_length)
    if token_ids != frozen:
        raise MechanicalPositionError(
            "frozen_encoder_mismatch",
            "Q0 instrumented token IDs differ from encode_prompt_sections",
        )
    query_position = len(prefix_ids) + query_token
    history_position = len(prefix_ids) + history_token
    candidate_start = len(prefix_ids) + min(len(context_ids), context_budget)
    readout = len(token_ids) - 1
    if not (0 <= query_position < history_position < candidate_start <= readout):
        raise MechanicalPositionError(
            "noncausal_position_order",
            "expected Q0 query < history < candidate block <= readout",
        )
    return InstrumentedPrompt(
        token_ids=tuple(token_ids),
        query_end=query_position,
        history_summary_end=history_position,
        candidate_readout=readout,
        candidate_start=candidate_start,
        context_tokens=min(len(context_ids), context_budget),
        candidate_tokens=candidate_budget,
        prompt_at_max_boundary=len(token_ids) == max_length,
    )


def q0_context_spans(
    context: str, *, query: str, history_text: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return unique byte-stable Q0 query/history character spans."""

    query_marker = "<Query>: "
    history_marker = "\n<Prior user history>:\n"
    if context.count(query_marker) != 1 or context.count(history_marker) != 1:
        raise MechanicalPositionError(
            "context_template_drift", "Q0 query/history marker uniqueness changed"
        )
    query_start = context.index(query_marker) + len(query_marker)
    query_end = query_start + len(query)
    if context[query_start:query_end] != query:
        raise MechanicalPositionError("query_span_mismatch", "Q0 query bytes changed")
    if not context.startswith(history_marker, query_end):
        raise MechanicalPositionError(
            "context_template_drift", "Q0 history marker changed"
        )
    history_start = query_end + len(history_marker)
    history_end = history_start + len(history_text)
    if context[history_start:history_end] != history_text:
        raise MechanicalPositionError(
            "history_span_mismatch", "Q0 serialized history bytes changed"
        )
    if not query or not history_text:
        raise MechanicalPositionError(
            "empty_span", "Q0 query/history span cannot be empty"
        )
    return (query_start, query_end), (history_start, history_end)
