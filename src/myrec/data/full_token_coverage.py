"""Label-free token-budget coverage audit for full-token ranker inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from myrec.baselines.core import document_text
from myrec.baselines.full_token_cross_encoder import serialize_query_history
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json, write_jsonl


def fit_history_assignments_to_context_budget(
    records_path: str | Path,
    assignments_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    model_name: str,
    cache_folder: str | Path = "models/huggingface/cross_encoders",
    max_length: int = 512,
    history_budget: int = 10,
    min_candidate_tokens: int = 1,
    local_files_only: bool = True,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Materialize the largest recent assigned history that leaves candidate room.

    The operation is label-free and deterministic. It is intended for
    ``only_second`` scoring, where an overlong first sequence cannot legally be
    repaired by truncating the candidate. Histories that already fit are left
    effective-input equivalent; histories longer than ``history_budget`` are
    materialized to the exact suffix that the scorer would consume.
    """

    if max_length <= 0 or history_budget < 0 or min_candidate_tokens <= 0:
        raise ValueError("invalid context-budget arguments")
    records_path = Path(records_path)
    assignments_path = Path(assignments_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    queries: dict[str, str] = {}
    for record in iter_jsonl(records_path):
        request_id = str(record["request_id"])
        if request_id in queries:
            raise ValueError(f"duplicate request_id={request_id}")
        queries[request_id] = str(record["query"])
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_folder),
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    pair_special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
    max_context_tokens = max_length - pair_special_tokens - min_candidate_tokens
    if max_context_tokens < 1:
        raise ValueError("max_length leaves no context capacity")

    output_rows = []
    trimmed_requests = 0
    dropped_events = 0
    materialized_suffix_requests = 0
    max_context_tokens_after = 0
    for row in iter_jsonl(assignments_path):
        request_id = str(row["request_id"])
        query = queries.pop(request_id, None)
        if query is None:
            raise ValueError(f"unknown or duplicate assignment request_id={request_id}")
        original_history = list(row.get("history", []))
        visible_history = (
            original_history[-history_budget:] if history_budget else []
        )
        materialized_suffix_requests += int(len(original_history) > len(visible_history))
        fitted_history = visible_history
        while fitted_history:
            context = serialize_query_history(
                query,
                fitted_history,
                history_budget=history_budget,
                serialization_version="query_history_event_text_v1",
            )
            context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
            if context_tokens <= max_context_tokens:
                break
            fitted_history = fitted_history[1:]
        context = serialize_query_history(
            query,
            fitted_history,
            history_budget=history_budget,
            serialization_version="query_history_event_text_v1",
        )
        context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
        if context_tokens > max_context_tokens:
            raise ValueError(
                f"query alone exceeds context budget for request_id={request_id}"
            )
        removed = len(visible_history) - len(fitted_history)
        trimmed_requests += int(removed > 0)
        dropped_events += removed
        max_context_tokens_after = max(max_context_tokens_after, context_tokens)
        output_row = dict(row)
        output_row["history"] = fitted_history
        output_rows.append(output_row)
    if queries:
        raise ValueError(f"missing assignments for request_ids={sorted(queries)[:5]}")

    write_jsonl(output_path, output_rows)
    result = {
        "analysis_type": "history_assignment_context_budget_fit",
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "input_assignments_path": str(assignments_path),
        "input_assignments_sha256": sha256_file(assignments_path),
        "output_assignments_path": str(output_path),
        "output_assignments_sha256": sha256_file(output_path),
        "requests": len(output_rows),
        "history_budget": history_budget,
        "max_length": max_length,
        "min_candidate_tokens": min_candidate_tokens,
        "pair_special_tokens": pair_special_tokens,
        "max_context_tokens_allowed": max_context_tokens,
        "max_context_tokens_after": max_context_tokens_after,
        "trimmed_requests": trimmed_requests,
        "history_events_dropped_for_context": dropped_events,
        "materialized_suffix_requests": materialized_suffix_requests,
        "model_name": model_name,
        "local_files_only": local_files_only,
        "qrels_read": False,
        "model_scores_read": False,
    }
    write_json(report_path, result)
    return result


def audit_full_token_coverage(
    records_path: str | Path,
    report_path: str | Path,
    *,
    model_name: str,
    cache_folder: str | Path = "models/huggingface/cross_encoders",
    max_length: int = 512,
    history_budget: int = 10,
    truncation_strategy: str = "longest_first",
    max_candidates_per_request: int | None = None,
    history_assignments_path: str | Path | None = None,
    local_files_only: bool = True,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Measure pair-length overflow without reading outcomes or qrels."""

    if max_length <= 0 or history_budget < 0:
        raise ValueError("max_length must be positive and history_budget non-negative")
    if truncation_strategy not in {"longest_first", "only_second"}:
        raise ValueError(f"unsupported truncation_strategy={truncation_strategy}")
    if max_candidates_per_request is not None and max_candidates_per_request <= 0:
        raise ValueError("max_candidates_per_request must be positive or None")
    records_path = Path(records_path)
    assignments = None
    assignments_sha256 = None
    if history_assignments_path is not None:
        history_assignments_path = Path(history_assignments_path)
        assignments = {}
        for row in iter_jsonl(history_assignments_path):
            request_id = str(row["request_id"])
            if request_id in assignments:
                raise ValueError(f"duplicate history assignment for {request_id}")
            assignments[request_id] = list(row.get("history", []))
        assignments_sha256 = sha256_file(history_assignments_path)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_folder),
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    pair_special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
    context_lengths = []
    query_lengths = []
    history_increment_lengths = []
    candidate_lengths = []
    pair_lengths = []
    overflow_tokens = []
    candidate_capacities = []
    history_present_pairs = 0
    history_present_overflow_pairs = 0
    requests = 0
    history_present_requests = 0
    unique_documents: dict[str, int] = {}

    for record in iter_jsonl(records_path):
        requests += 1
        request_id = str(record["request_id"])
        history = (
            list(record.get("history", []))
            if assignments is None
            else assignments.pop(request_id, None)
        )
        if history is None:
            raise ValueError(f"history assignment missing request_id={request_id}")
        history_present = bool(history)
        history_present_requests += int(history_present)
        query = str(record["query"])
        context = serialize_query_history(
            query,
            history,
            history_budget=history_budget,
            serialization_version="query_history_event_text_v1",
        )
        query_length = len(tokenizer.encode(query, add_special_tokens=False))
        context_length = len(tokenizer.encode(context, add_special_tokens=False))
        query_lengths.append(query_length)
        context_lengths.append(context_length)
        history_increment_lengths.append(max(0, context_length - query_length))
        candidate_capacities.append(max_length - context_length - pair_special_tokens)
        candidates = record["candidates"]
        if max_candidates_per_request is not None:
            candidates = candidates[:max_candidates_per_request]
        for candidate in candidates:
            item_id = str(candidate["item_id"])
            if item_id not in unique_documents:
                unique_documents[item_id] = len(
                    tokenizer.encode(document_text(candidate), add_special_tokens=False)
                )
            candidate_length = unique_documents[item_id]
            total = context_length + candidate_length + pair_special_tokens
            overflow = max(0, total - max_length)
            candidate_lengths.append(candidate_length)
            pair_lengths.append(total)
            overflow_tokens.append(overflow)
            if history_present:
                history_present_pairs += 1
                history_present_overflow_pairs += int(overflow > 0)

    if not pair_lengths:
        raise ValueError(f"no candidate pairs in {records_path}")
    if assignments:
        raise ValueError(f"unknown history assignment request_ids: {sorted(assignments)[:5]}")
    overflow_pairs = sum(value > 0 for value in overflow_tokens)
    result = {
        "analysis_type": "full_token_input_coverage",
        "candidate_pairs": len(pair_lengths),
        "candidate_token_length": _summary(candidate_lengths),
        "context_token_length": _summary(context_lengths),
        "history_budget": history_budget,
        "history_assignments_path": (
            str(history_assignments_path) if history_assignments_path is not None else None
        ),
        "history_assignments_sha256": assignments_sha256,
        "history_increment_token_length": _summary(history_increment_lengths),
        "history_present_overflow_pairs": history_present_overflow_pairs,
        "history_present_overflow_rate": (
            history_present_overflow_pairs / history_present_pairs
            if history_present_pairs
            else None
        ),
        "history_present_requests": history_present_requests,
        "candidate_capacity_if_context_preserved": _summary(candidate_capacities),
        "context_preserved_under_configured_truncation": (
            truncation_strategy == "only_second" and min(candidate_capacities) >= 1
        ),
        "label_fields_read": False,
        "local_files_only": local_files_only,
        "max_length": max_length,
        "max_candidates_per_request": max_candidates_per_request,
        "model_name": model_name,
        "overflow_pair_rate": overflow_pairs / len(pair_lengths),
        "overflow_pairs": overflow_pairs,
        "overflow_token_count": _summary(overflow_tokens),
        "pair_special_tokens": pair_special_tokens,
        "pair_token_length_before_truncation": _summary(pair_lengths),
        "qrels_read": False,
        "query_token_length": _summary(query_lengths),
        "truncation_strategy": truncation_strategy,
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "requests": requests,
        "unique_candidate_documents": len(unique_documents),
    }
    write_json(report_path, result)
    return result


def audit_cross_encoder_preprocess_coverage(
    records_path: str | Path,
    report_path: str | Path,
    *,
    model_name: str,
    max_length: int = 512,
    history_budget: int = 6,
    audit_max_length: int = 32768,
    batch_size: int = 128,
    device: str = "cpu",
    dtype: str = "bfloat16",
    local_files_only: bool = True,
    predictor: Any | None = None,
) -> dict[str, Any]:
    """Measure lengths after the model's real prompt/chat preprocessing."""

    if max_length <= 0 or audit_max_length < max_length:
        raise ValueError("audit_max_length must be at least max_length")
    if history_budget < 0 or batch_size <= 0:
        raise ValueError("history_budget must be non-negative and batch_size positive")
    if predictor is None:
        import torch
        from sentence_transformers import CrossEncoder

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        predictor = CrossEncoder(
            model_name,
            device=device,
            max_length=audit_max_length,
            local_files_only=local_files_only,
            model_kwargs={"dtype": torch_dtype},
        )
    prompt = predictor._resolve_prompt(None, None)
    records_path = Path(records_path)
    pair_buffer = []
    history_flags = []
    pair_lengths = []
    history_present_pair_lengths = []
    requests = 0

    def flush() -> None:
        if not pair_buffer:
            return
        processed = predictor.preprocess(pair_buffer, prompt=prompt)
        attention_mask = processed.get("attention_mask")
        if attention_mask is None:
            raise ValueError("CrossEncoder preprocess did not return attention_mask")
        lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
        if len(lengths) != len(history_flags):
            raise ValueError("preprocess output length mismatch")
        pair_lengths.extend(int(value) for value in lengths)
        history_present_pair_lengths.extend(
            int(value)
            for value, present in zip(lengths, history_flags)
            if present
        )
        pair_buffer.clear()
        history_flags.clear()

    for record in iter_jsonl(records_path):
        requests += 1
        history = list(record.get("history", []))
        context = serialize_query_history(
            str(record["query"]),
            history,
            history_budget=history_budget,
            serialization_version="query_history_event_text_v1",
        )
        for candidate in record["candidates"]:
            pair_buffer.append((context, document_text(candidate)))
            history_flags.append(bool(history))
            if len(pair_buffer) >= batch_size:
                flush()
    flush()
    if not pair_lengths:
        raise ValueError(f"no candidate pairs in {records_path}")
    overflow_pairs = sum(value > max_length for value in pair_lengths)
    history_overflow = sum(
        value > max_length for value in history_present_pair_lengths
    )
    result = {
        "analysis_type": "cross_encoder_real_preprocess_coverage",
        "audit_max_length": audit_max_length,
        "batch_size": batch_size,
        "candidate_pairs": len(pair_lengths),
        "dtype": dtype,
        "history_budget": history_budget,
        "history_present_overflow_pairs": history_overflow,
        "history_present_overflow_rate": (
            history_overflow / len(history_present_pair_lengths)
            if history_present_pair_lengths
            else None
        ),
        "history_present_pairs": len(history_present_pair_lengths),
        "label_fields_read": False,
        "local_files_only": local_files_only,
        "max_length": max_length,
        "model_name": model_name,
        "overflow_pair_rate": overflow_pairs / len(pair_lengths),
        "overflow_pairs": overflow_pairs,
        "pair_token_length_before_truncation": _summary(pair_lengths),
        "preprocess_prompt": prompt,
        "qrels_read": False,
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "requests": requests,
    }
    write_json(report_path, result)
    return result


def _summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p50": _quantile(ordered, 0.50),
        "p90": _quantile(ordered, 0.90),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
    }


def _quantile(ordered: list[int], fraction: float) -> int:
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]
