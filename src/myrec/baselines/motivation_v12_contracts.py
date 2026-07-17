"""Shared, label-isolated contracts for Motivation V1.2 Qwen rankers.

This module intentionally contains no model or evaluator imports.  Q0--Q3 use
the same record sanitizer, deterministic training groups, candidate identity
checks, prompt token boundary, and ranking-loss conversions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from myrec.utils.jsonl import iter_jsonl


METHOD_IDS = (
    "q0_qwen3_reranker_06b",
    "q1_instructrec_generalqwen",
    "q2_recranker_generalqwen",
    "q3_tallrec_generalqwen",
)

FORBIDDEN_MODEL_INPUT_FIELDS = frozenset(
    {
        "clicked",
        "is_clicked",
        "is_purchased",
        "label",
        "labels",
        "purchased",
        "relevance",
        "target",
    }
)
HISTORY_INPUT_FIELDS = ("item_id", "title", "brand", "cat", "event", "query", "ts")
CANDIDATE_INPUT_FIELDS = ("item_id", "title", "brand", "cat")
SERIALIZED_INPUT_FIELDS = (
    "query",
    "history.title",
    "history.brand",
    "history.cat",
    "history.event",
    "history.query",
    "candidates.title",
    "candidates.brand",
    "candidates.cat",
)
NO_HISTORY_MARKER = "[NO_HISTORY]"


@dataclass(frozen=True)
class ModelRecord:
    """The only representation passed from unified JSONL into prompt code."""

    request_id: str
    query: str
    history: tuple[dict[str, Any], ...]
    candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TrainingGroup:
    """One request-level ranking group; gains are targets, never prompt fields."""

    record: ModelRecord
    candidates: tuple[dict[str, Any], ...]
    gains: tuple[float, ...]


@dataclass(frozen=True)
class PromptSections:
    system: str
    context: str
    candidate: str


def sanitize_record_for_model(record: dict[str, Any]) -> ModelRecord:
    """Project a unified record through the one explicit V1.2 input whitelist."""

    request_id = str(record.get("request_id") or "")
    query = str(record.get("query") or "").strip()
    if not request_id or not query:
        raise ValueError("record must contain non-empty request_id and query")
    raw_history = record.get("history")
    raw_candidates = record.get("candidates")
    if not isinstance(raw_history, list):
        raise ValueError(f"request_id={request_id}: history must be a list")
    if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
        raise ValueError(
            f"request_id={request_id}: candidates must contain at least two rows"
        )
    history = tuple(
        _whitelist_object(row, HISTORY_INPUT_FIELDS, f"history[{index}]")
        for index, row in enumerate(raw_history)
    )
    candidates = tuple(
        _whitelist_object(row, CANDIDATE_INPUT_FIELDS, f"candidates[{index}]")
        for index, row in enumerate(raw_candidates)
    )
    candidate_ids = [str(row["item_id"]) for row in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"request_id={request_id}: duplicate candidate item_id")
    return ModelRecord(
        request_id=request_id,
        query=query,
        history=history,
        candidates=candidates,
    )


def load_training_groups(
    records_path: str | Path,
    qrels_path: str | Path,
    *,
    seed: int,
    negatives_per_positive: int,
    max_group_size: int = 8,
) -> tuple[list[TrainingGroup], dict[str, Any]]:
    """Load train-only labels separately and build deterministic shared groups."""

    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive")
    if max_group_size < 2:
        raise ValueError("max_group_size must be at least two")
    qrels = _load_training_qrels(Path(qrels_path))
    groups: list[TrainingGroup] = []
    seen: set[str] = set()
    skipped_no_positive = 0
    skipped_no_negative = 0
    selected_candidate_rows = 0
    for raw_record in iter_jsonl(records_path):
        record = sanitize_record_for_model(raw_record)
        if record.request_id in seen:
            raise ValueError(f"duplicate record request_id={record.request_id}")
        seen.add(record.request_id)
        if record.request_id not in qrels:
            raise ValueError(f"missing train qrels request_id={record.request_id}")
        gain_by_item = qrels[record.request_id]
        candidate_ids = {str(row["item_id"]) for row in record.candidates}
        unknown = set(gain_by_item) - candidate_ids
        if unknown:
            raise ValueError(
                f"positive labels outside candidate slate for {record.request_id}: "
                f"{sorted(unknown)[:5]}"
            )
        positives = [
            row
            for row in record.candidates
            if float(gain_by_item.get(str(row["item_id"]), 0.0)) > 0.0
        ]
        negatives = [
            row
            for row in record.candidates
            if float(gain_by_item.get(str(row["item_id"]), 0.0)) <= 0.0
        ]
        if not positives:
            skipped_no_positive += 1
            continue
        if not negatives:
            skipped_no_negative += 1
            continue
        selected = _select_group_candidates(
            record.request_id,
            positives,
            negatives,
            seed=seed,
            negatives_per_positive=negatives_per_positive,
            max_group_size=max_group_size,
        )
        gains = tuple(
            float(gain_by_item.get(str(row["item_id"]), 0.0)) for row in selected
        )
        if not any(value > 0 for value in gains) or not any(value == 0 for value in gains):
            raise AssertionError("selected training group lost a positive or negative")
        groups.append(TrainingGroup(record=record, candidates=selected, gains=gains))
        selected_candidate_rows += len(selected)
    if seen != set(qrels):
        raise ValueError("train records and qrels have different request coverage")
    if not groups:
        raise ValueError("no V1.2 training groups were constructed")
    return groups, {
        "groups": len(groups),
        "max_group_size": max_group_size,
        "negatives_per_positive": negatives_per_positive,
        "record_requests": len(seen),
        "selected_candidate_rows": selected_candidate_rows,
        "skipped_no_negative": skipped_no_negative,
        "skipped_no_positive": skipped_no_positive,
    }


def pairwise_index_pairs(gains: Sequence[float]) -> list[tuple[int, int]]:
    """Convert one graded list into every strict (higher, lower) pair."""

    _validate_gains(gains)
    pairs: list[tuple[int, int]] = []
    for left in range(len(gains)):
        for right in range(left + 1, len(gains)):
            if gains[left] == gains[right]:
                continue
            pairs.append((left, right) if gains[left] > gains[right] else (right, left))
    return pairs


def listwise_target_distribution(gains: Sequence[float]) -> list[float]:
    """Normalize ``2**relevance - 1`` for a tie-aware ListNet target."""

    _validate_gains(gains)
    transformed = [2.0 ** float(value) - 1.0 for value in gains]
    total = sum(transformed)
    if total <= 0:
        raise ValueError("listwise target requires at least one positive gain")
    return [value / total for value in transformed]


def complete_candidate_chunks(
    candidates: Sequence[dict[str, Any]], chunk_size: int
) -> list[tuple[dict[str, Any], ...]]:
    """Partition a slate without dropping, duplicating, or reidentifying items."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    item_ids = [str(row["item_id"]) for row in candidates]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("candidate chunking requires unique item_id values")
    chunks = [
        tuple(candidates[start : start + chunk_size])
        for start in range(0, len(candidates), chunk_size)
    ]
    flattened = [str(row["item_id"]) for chunk in chunks for row in chunk]
    if flattened != item_ids:
        raise AssertionError("candidate chunking changed identity or order")
    return chunks


def build_prompt_sections(
    method_id: str,
    record: ModelRecord,
    candidate: dict[str, Any],
    *,
    history: Sequence[dict[str, Any]] | None = None,
    history_budget: int,
    slate: Sequence[dict[str, Any]] | None = None,
) -> PromptSections:
    """Build method-specific text using only sanitized input objects."""

    if method_id not in METHOD_IDS:
        raise ValueError(f"unsupported V1.2 method_id={method_id}")
    selected_history = tuple(record.history if history is None else history)
    history_text = serialize_history(selected_history, history_budget=history_budget)
    document = serialize_candidate(candidate)
    if method_id == "q0_qwen3_reranker_06b":
        system = (
            "Judge whether the Document meets the requirements based on the Query "
            'and the Instruct provided. The answer can only be "yes" or "no".'
        )
        context = (
            "<Instruct>: Rank the product for the current e-commerce search query "
            "and the user's prior behavior.\n"
            f"<Query>: {record.query}\n<Prior user history>:\n{history_text}\n"
        )
        candidate_text = f"<Document>: {document}"
    elif method_id == "q1_instructrec_generalqwen":
        raise ValueError(
            "Q1 uses build_instructrec_selection_sections and normalized candidate "
            "response likelihood, not the Yes/No point scorer"
        )
    elif method_id == "q2_recranker_generalqwen":
        system = (
            "You are a top-k recommendation ranker jointly trained with pairwise "
            'and listwise ranking objectives. Answer only "yes" or "no".'
        )
        context = (
            f"Query: {record.query}\nUser history (newest first):\n{history_text}\n"
            "Instruction: score this product for top-k ranking. Its log-odds will "
            "be compared with every other candidate under the same request.\n"
        )
        candidate_text = f"Candidate product: {document}"
    else:
        system = (
            "You are a recommendation-aligned language model. Decide whether the "
            'candidate matches the user and current query; answer only "Yes" or "No".'
        )
        context = (
            f"Current query: {record.query}\n"
            f"Past interactions (newest first):\n{history_text}\n"
            "Recommendation alignment task: predict whether this product should be "
            "preferred in the current candidate set.\n"
        )
        candidate_text = f"Candidate product: {document}"
    return PromptSections(system=system, context=context, candidate=candidate_text)


def build_instructrec_selection_sections(
    record: ModelRecord,
    slate: Sequence[dict[str, Any]],
    *,
    history: Sequence[dict[str, Any]] | None = None,
    history_budget: int,
    template_index: int,
) -> PromptSections:
    """InstructRec-style T3 selection prompt with two frozen P/I phrasings."""

    if template_index not in {0, 1}:
        raise ValueError("InstructRec template_index must be zero or one")
    if len(slate) < 2:
        raise ValueError("InstructRec selection requires at least two candidates")
    item_ids = [str(row["item_id"]) for row in slate]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("InstructRec slate contains duplicate candidate identity")
    selected_history = tuple(record.history if history is None else history)
    history_text = serialize_history(selected_history, history_budget=history_budget)
    slate_text = "\n".join(
        f"{index}. {serialize_candidate(row)}"
        for index, row in enumerate(slate, start=1)
    )
    system = (
        "You are an instruction-following personalized product-search engine. "
        "Return only the exact marked candidate line for the single best product."
    )
    if template_index == 0:
        context = (
            "Task form: personalized product reranking.\n"
            f"Current user intention (query): {record.query}\n"
            f"Implicit preference evidence (newest first):\n{history_text}\n"
            "Instruction: select the candidate that best satisfies the current "
            "query and the user's implicit preferences.\n"
        )
    else:
        context = (
            "You must execute a personalized search instruction. Infer preferences "
            "only from the causal history, then apply them to the current query.\n"
            f"Search query: {record.query}\nInteraction history:\n{history_text}\n"
            "Instruction: choose one product from the fixed candidate set.\n"
        )
    return PromptSections(
        system=system,
        context=context,
        candidate=f"Candidate products:\n{slate_text}\n\nSelected product:",
    )


def instructrec_template_index(request_id: str, *, seed: int) -> int:
    return _stable_seed(seed, "instructrec_template", request_id) % 2


def encode_prompt_sections(
    tokenizer: Any,
    sections: PromptSections,
    *,
    max_length: int,
) -> list[int]:
    """Encode a prompt while reserving the answer suffix and candidate identity."""

    if max_length < 64:
        raise ValueError("max_length must be at least 64")
    prefix = (
        f"<|im_start|>system\n{sections.system}<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    context_ids = tokenizer.encode(sections.context, add_special_tokens=False)
    candidate_ids = tokenizer.encode(sections.candidate, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    body_budget = max_length - len(prefix_ids) - len(suffix_ids)
    if body_budget < 16:
        raise ValueError("max_length leaves no room for query and candidate text")

    # At least half of the body remains available to the marked candidate/slate.
    candidate_budget = min(len(candidate_ids), max(8, body_budget // 2))
    context_budget = body_budget - candidate_budget
    if len(context_ids) < context_budget:
        candidate_budget = min(len(candidate_ids), body_budget - len(context_ids))
        context_budget = body_budget - candidate_budget
    elif len(candidate_ids) < candidate_budget:
        context_budget = body_budget - len(candidate_ids)
        candidate_budget = len(candidate_ids)
    token_ids = (
        prefix_ids
        + context_ids[:context_budget]
        + candidate_ids[:candidate_budget]
        + suffix_ids
    )
    if len(token_ids) > max_length:
        raise AssertionError("prompt truncation exceeded max_length")
    return token_ids


def encode_instructrec_selection_prompt(
    tokenizer: Any,
    record: ModelRecord,
    slate: Sequence[dict[str, Any]],
    *,
    history: Sequence[dict[str, Any]] | None = None,
    history_budget: int,
    template_index: int,
    max_length: int,
    context_token_budget: int,
    max_target_length: int,
) -> tuple[list[int], dict[str, list[int]], dict[str, Any]]:
    """Encode Q1 with history-invariant, complete candidate exposure.

    The context and slate receive separate fixed budgets.  Every candidate gets
    a unique ordinal marker and a balanced visible text prefix; the likelihood
    target is exactly the marked representation present in the prompt.  Thus a
    full/null history swap cannot change candidate coverage or target tokens.
    """

    if context_token_budget <= 0 or max_target_length < 4:
        raise ValueError("invalid InstructRec token budgets")
    sections = build_instructrec_selection_sections(
        record,
        slate,
        history=history,
        history_budget=history_budget,
        template_index=template_index,
    )
    prefix = (
        f"<|im_start|>system\n{sections.system}<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    context_ids = tokenizer.encode(sections.context, add_special_tokens=False)
    header_ids = tokenizer.encode("Candidate products:\n", add_special_tokens=False)
    footer_ids = tokenizer.encode("\nSelected product:", add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    answer_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if not answer_end_ids:
        raise ValueError("InstructRec answer terminator tokenization is empty")
    markers = [
        tokenizer.encode(f"{index}. ", add_special_tokens=False)
        for index in range(1, len(slate) + 1)
    ]
    contents = [
        tokenizer.encode(serialize_candidate(candidate), add_special_tokens=False)
        for candidate in slate
    ]
    if any(not values for values in contents):
        raise ValueError("InstructRec candidate produced an empty representation")
    fixed_overhead = (
        len(prefix_ids)
        + context_token_budget
        + len(header_ids)
        + len(footer_ids)
        + len(suffix_ids)
        + len(newline_ids) * len(slate)
        + sum(len(marker) for marker in markers)
    )
    content_budget = max_length - fixed_overhead
    if content_budget < len(slate):
        raise ValueError(
            "max_length cannot preserve one visible token for every Q1 candidate"
        )
    caps = [
        min(
            len(content),
            max_target_length - len(marker) - len(answer_end_ids),
        )
        for marker, content in zip(markers, contents)
    ]
    if any(cap <= 0 for cap in caps):
        raise ValueError("max_target_length cannot preserve candidate markers")
    allocations = _balanced_token_allocations(caps, content_budget)
    responses: dict[str, list[int]] = {}
    candidate_block: list[int] = [*header_ids]
    for candidate, marker, content, allocation in zip(
        slate, markers, contents, allocations
    ):
        visible_response = [*marker, *content[:allocation]]
        response = [*visible_response, *answer_end_ids]
        item_id = str(candidate["item_id"])
        if item_id in responses:
            raise ValueError("duplicate Q1 candidate item_id")
        responses[item_id] = response
        candidate_block.extend(visible_response)
        candidate_block.extend(newline_ids)
    candidate_block.extend(footer_ids)
    token_ids = (
        prefix_ids
        + context_ids[:context_token_budget]
        + candidate_block
        + suffix_ids
    )
    if len(token_ids) > max_length:
        raise AssertionError("fixed-budget Q1 encoding exceeded max_length")
    response_values = list(responses.values())
    collisions = len(response_values) - len({tuple(value) for value in response_values})
    if collisions:
        raise AssertionError("ordinal-marked Q1 candidate responses must be unique")
    return token_ids, responses, {
        "all_candidate_markers_preserved": True,
        "candidate_count": len(slate),
        "candidate_content_tokens": allocations,
        "candidate_response_collisions": 0,
        "candidate_targets_include_answer_end": True,
        "context_tokens_observed": min(len(context_ids), context_token_budget),
        "context_token_budget": context_token_budget,
        "prompt_tokens": len(token_ids),
    }


def _balanced_token_allocations(caps: Sequence[int], budget: int) -> list[int]:
    """Water-fill a fixed budget while giving every candidate one token."""

    if not caps or any(cap <= 0 for cap in caps) or budget < len(caps):
        raise ValueError("invalid balanced candidate token allocation")
    allocations = [1] * len(caps)
    remaining = min(budget - len(caps), sum(caps) - len(caps))
    active = {index for index, cap in enumerate(caps) if cap > 1}
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = 0
        for index in sorted(active):
            addition = min(share, caps[index] - allocations[index], remaining)
            allocations[index] += addition
            remaining -= addition
            progressed += addition
            if remaining == 0:
                break
        active = {
            index for index in active if allocations[index] < caps[index]
        }
        if progressed == 0:
            break
    return allocations


def serialize_history(
    history: Sequence[dict[str, Any]], *, history_budget: int
) -> str:
    if history_budget < 0:
        raise ValueError("history_budget must be non-negative")
    selected = list(history[-history_budget:]) if history_budget else []
    if not selected:
        return NO_HISTORY_MARKER
    rows = []
    for index, event in enumerate(reversed(selected), start=1):
        categories = "/".join(str(value) for value in event.get("cat", []) if value)
        visible = [
            str(event.get("event") or "interaction"),
            f"prior query={event['query']}"
            if str(event.get("query") or "").strip()
            else "",
            str(event.get("title") or ""),
            str(event.get("brand") or ""),
            categories,
        ]
        rows.append(f"{index}. " + " | ".join(value for value in visible if value))
    return "\n".join(rows)


def serialize_candidate(candidate: dict[str, Any]) -> str:
    categories = "/".join(str(value) for value in candidate.get("cat", []) if value)
    visible = [
        str(candidate.get("title") or ""),
        str(candidate.get("brand") or ""),
        categories,
    ]
    text = " | ".join(value for value in visible if value)
    return text or "[MISSING_PRODUCT_TEXT]"


def epoch_batch_order(length: int, *, seed: int, epoch: int) -> list[int]:
    if length <= 0 or epoch < 0:
        raise ValueError("invalid epoch order arguments")
    order = list(range(length))
    random.Random(_stable_seed(seed, "epoch", str(epoch))).shuffle(order)
    return order


def batched_indices(order: Sequence[int], batch_size: int) -> list[tuple[int, ...]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [tuple(order[start : start + batch_size]) for start in range(0, len(order), batch_size)]


def canonical_scoring_signature(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _whitelist_object(
    row: Any, fields: Iterable[str], location: str
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{location} must be an object")
    result = {field: row[field] for field in fields if field in row}
    item_id = str(result.get("item_id") or "")
    if not item_id:
        raise ValueError(f"{location}.item_id must be non-empty")
    result["item_id"] = item_id
    cat = result.get("cat", [])
    if cat is None:
        cat = []
    if not isinstance(cat, list):
        raise ValueError(f"{location}.cat must be a list")
    result["cat"] = [str(value) for value in cat]
    return result


def _load_training_qrels(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate qrels request_id={request_id}")
        relevance = row.get("relevance", {})
        if relevance and not isinstance(relevance, dict):
            raise ValueError("training relevance must be an item-to-gain object")
        gains = {
            str(item_id): float(gain)
            for item_id, gain in (relevance or {}).items()
            if float(gain) > 0
        }
        if not gains:
            gains = {
                **{str(item_id): 1.0 for item_id in row.get("clicked", [])},
                **{str(item_id): 2.0 for item_id in row.get("purchased", [])},
            }
        if any(not math.isfinite(value) or value <= 0 for value in gains.values()):
            raise ValueError(f"invalid positive training gain for {request_id}")
        result[request_id] = gains
    if not result:
        raise ValueError(f"empty qrels file: {path}")
    return result


def _select_group_candidates(
    request_id: str,
    positives: Sequence[dict[str, Any]],
    negatives: Sequence[dict[str, Any]],
    *,
    seed: int,
    negatives_per_positive: int,
    max_group_size: int,
) -> tuple[dict[str, Any], ...]:
    selected_positives = list(positives[: max_group_size - 1])
    negative_limit = min(
        len(negatives),
        max_group_size - len(selected_positives),
        max(1, negatives_per_positive * len(selected_positives)),
    )
    shuffled = list(negatives)
    random.Random(_stable_seed(seed, "negatives", request_id)).shuffle(shuffled)
    selected = [*selected_positives, *shuffled[:negative_limit]]
    # A stable final shuffle avoids making the first position a label proxy.
    random.Random(_stable_seed(seed, "group_order", request_id)).shuffle(selected)
    return tuple(selected)


def _validate_gains(gains: Sequence[float]) -> None:
    if len(gains) < 2:
        raise ValueError("ranking conversion requires at least two candidates")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in gains):
        raise ValueError("gains must be finite and non-negative")


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
