"""Qrels-free control assignments for the Transformer deep-dive.

The wrong-user mapping is intentionally stricter than the first-round pilot
control: it preserves the visible event count, excludes every recipient item,
and matches the exact frozen-Qwen serialized-history token length.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import (
    build_prompt_sections,
    sanitize_record_for_model,
    serialize_history,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json, write_jsonl


WRONG_USER_NAMESPACE = "deep-dive-wrong-user-v1"
HISTORY_BUDGET = 6
CONTENT_NEUTRAL_TOKEN_ID = 151_643
CONTENT_NEUTRAL_METHODS = (
    "q2_recranker_generalqwen",
    "q3_tallrec_generalqwen",
)
FIXED_SAMPLE_NAMESPACE = "deep-dive-fixed-candidate-rows-v1"
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def materialize_wrong_user_mapping(
    target_records_path: str | Path,
    donor_records_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    expected_tokenizer_sha256: str,
) -> dict[str, Any]:
    """Materialize the registered train-to-dev wrong-user mapping.

    This function reads unified records and tokenizer files only. It never
    accepts a qrels or model-score path.
    """

    target_records_path = Path(target_records_path)
    donor_records_path = Path(donor_records_path)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"deep-dive control directory is not empty: {output_dir}")
    tokenizer_json = tokenizer_path / "tokenizer.json"
    if sha256_file(tokenizer_json) != expected_tokenizer_sha256:
        raise ValueError("deep-dive tokenizer differs from the frozen tokenizer")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    targets = _load_records(target_records_path, role="target")
    donors = _load_records(donor_records_path, role="donor")
    donor_buckets, donor_lengths = _index_donors(donors, tokenizer)

    rows: list[dict[str, Any]] = []
    match_counts: Counter[str] = Counter()
    token_differences: list[int] = []
    for target in targets:
        row = _match_target(
            target,
            donor_buckets=donor_buckets,
            donor_lengths=donor_lengths,
            tokenizer=tokenizer,
        )
        rows.append(row)
        match_counts[str(row["match_type"])] += 1
        if row["eligible"]:
            token_differences.append(int(row["token_length_absolute_difference"]))

    _audit_rows(rows, targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "wrong_user_mapping.jsonl"
    write_jsonl(mapping_path, rows)
    eligible_ids = [str(row["request_id"]) for row in rows if row["eligible"]]
    eligible_ids_sha256 = sha256_text(
        json.dumps(eligible_ids, ensure_ascii=False, separators=(",", ":"))
    )
    report = {
        "schema_version": 1,
        "control_id": WRONG_USER_NAMESPACE,
        "status": "frozen_qrels_blind_assignment",
        "target_records_path": str(target_records_path),
        "target_records_sha256": sha256_file(target_records_path),
        "donor_records_path": str(donor_records_path),
        "donor_records_sha256": sha256_file(donor_records_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "history_budget": HISTORY_BUDGET,
        "requests": len(rows),
        "eligible_requests": len(eligible_ids),
        "ineligible_requests": len(rows) - len(eligible_ids),
        "eligible_request_ids_sha256": eligible_ids_sha256,
        "match_counts": dict(sorted(match_counts.items())),
        "token_length_absolute_difference": _summary(token_differences),
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "recipe": {
            "target_population": "internal_dev",
            "donor_population": "train_only",
            "visible_count": "min(6,len(recipient_original_history))",
            "donor_visible_events": "last_H_events_in_source_time_order",
            "hard_constraints": [
                "different_user",
                "all_selected_donor_event_ts_strictly_before_recipient_ts",
                "selected_donor_item_ids_disjoint_recipient_candidates",
                "selected_donor_item_ids_disjoint_all_recipient_original_history",
                "exact_same_visible_event_count",
            ],
            "distance": "absolute_Qwen_serialized_history_token_length_difference",
            "tie_break": (
                "minimum_sha256(namespace,recipient_request_id,donor_request_id)"
            ),
            "ineligible_scoring": "copy_frozen_null_score_for_full_coverage",
        },
        "qrels_read": False,
        "model_scores_read": False,
        "source_test_opened": False,
    }
    write_json(output_dir / "manifest.json", report)
    return report


def materialize_content_neutral_eligibility(
    target_records_path: str | Path,
    config_paths: Mapping[str, str | Path],
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    expected_tokenizer_sha256: str,
) -> dict[str, Any]:
    """Freeze exact same-length history-token neutralization spans."""

    target_records_path = Path(target_records_path)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"content-neutral directory is not empty: {output_dir}")
    tokenizer_json = tokenizer_path / "tokenizer.json"
    if sha256_file(tokenizer_json) != expected_tokenizer_sha256:
        raise ValueError("content-neutral tokenizer differs from frozen tokenizer")
    if set(config_paths) != set(CONTENT_NEUTRAL_METHODS):
        raise ValueError("content-neutral eligibility requires exact Q2/Q3 configs")

    from transformers import AutoTokenizer
    from myrec.baselines.motivation_v12_ranker import (
        _answer_target_tokens,
        load_v12_ranker_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    configs = {
        method_id: load_v12_ranker_config(Path(config_paths[method_id]))
        for method_id in CONTENT_NEUTRAL_METHODS
    }
    for method_id, config in configs.items():
        if config["method_id"] != method_id:
            raise ValueError(f"content-neutral config/method mismatch: {method_id}")
        if config["model"]["tokenizer_sha256"] != expected_tokenizer_sha256:
            raise ValueError("content-neutral config tokenizer hash drifted")

    records = [
        sanitize_record_for_model(dict(row)) for row in iter_jsonl(target_records_path)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, Any] = {}
    for method_id in CONTENT_NEUTRAL_METHODS:
        config = configs[method_id]
        reserve = 0
        if method_id == "q3_tallrec_generalqwen":
            reserve = max(
                len(_answer_target_tokens(tokenizer, "Yes")),
                len(_answer_target_tokens(tokenizer, "No")),
            )
        max_length = int(config["training"]["max_length"]) - reserve
        rows = [
            _content_neutral_row(
                method_id,
                record,
                tokenizer,
                max_length=max_length,
                history_budget=int(config["training"]["history_budget"]),
            )
            for record in records
        ]
        path = output_dir / f"{method_id}.jsonl"
        write_jsonl(path, rows)
        eligible_ids = [row["request_id"] for row in rows if row["eligible"]]
        ineligible = Counter(
            str(row["reason"]) for row in rows if not row["eligible"]
        )
        methods[method_id] = {
            "config_path": str(config_paths[method_id]),
            "config_sha256": config["_config_sha256"],
            "max_prompt_length_after_target_reserve": max_length,
            "requests": len(rows),
            "eligible_requests": len(eligible_ids),
            "ineligible_requests": len(rows) - len(eligible_ids),
            "ineligible_reason_counts": dict(sorted(ineligible.items())),
            "eligible_request_ids_sha256": sha256_text(
                json.dumps(eligible_ids, ensure_ascii=False, separators=(",", ":"))
            ),
            "path": str(path),
            "sha256": sha256_file(path),
        }
    report = {
        "schema_version": 1,
        "control_id": "deep-dive-content-neutral-v1",
        "status": "frozen_qrels_blind_eligibility",
        "target_records_path": str(target_records_path),
        "target_records_sha256": sha256_file(target_records_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "neutral_token_id": CONTENT_NEUTRAL_TOKEN_ID,
        "methods": methods,
        "recipe": {
            "scope": "exact_retained_history_text_substring_tokens",
            "replacement": "replace_each_history_span_token_with_151643",
            "preserved": [
                "token_count",
                "attention_mask",
                "position_ids",
                "prefix_query_candidate_suffix_tokens",
                "history_external_delimiters",
            ],
            "request_eligibility": (
                "visible_history_and_exact_token_boundaries_and_span_retained_for_all_candidates"
            ),
            "ineligible_scoring": "copy_frozen_baseline_score_for_full_coverage",
        },
        "qrels_read": False,
        "model_scores_read": False,
        "source_test_opened": False,
    }
    write_json(output_dir / "manifest.json", report)
    return report


def materialize_fixed_candidate_sample(
    target_records_path: str | Path,
    output_dir: str | Path,
    *,
    sample_rows: int = 512,
) -> dict[str, Any]:
    """Freeze a qrels-blind candidate-row sample for high-dimensional probes."""

    target_records_path = Path(target_records_path)
    output_dir = Path(output_dir)
    if sample_rows <= 0:
        raise ValueError("fixed sample_rows must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"fixed-sample directory is not empty: {output_dir}")
    candidates: list[tuple[str, dict[str, Any]]] = []
    request_count = 0
    history_present_requests = 0
    for raw in iter_jsonl(target_records_path):
        record = sanitize_record_for_model(dict(raw))
        request_count += 1
        history_ids = {str(event["item_id"]) for event in record.history}
        if not history_ids:
            continue
        history_present_requests += 1
        for ordinal, candidate in enumerate(record.candidates):
            candidate_id = str(candidate["item_id"])
            if candidate_id in history_ids:
                continue
            key = hashlib.sha256(
                "\x1f".join(
                    (
                        FIXED_SAMPLE_NAMESPACE,
                        record.request_id,
                        candidate_id,
                        str(ordinal),
                    )
                ).encode("utf-8")
            ).hexdigest()
            candidates.append(
                (
                    key,
                    {
                        "request_id": record.request_id,
                        "candidate_item_id": candidate_id,
                        "candidate_ordinal": ordinal,
                        "selection_sha256": key,
                    },
                )
            )
    candidates.sort(
        key=lambda row: (
            row[0],
            row[1]["request_id"],
            row[1]["candidate_ordinal"],
        )
    )
    rows = [row for _key, row in candidates[:sample_rows]]
    if len(rows) != sample_rows:
        raise ValueError("fixed sample does not have enough eligible candidate rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "candidate_rows.jsonl"
    write_jsonl(path, rows)
    report = {
        "schema_version": 1,
        "sample_id": FIXED_SAMPLE_NAMESPACE,
        "status": "frozen_qrels_blind_sample",
        "target_records_path": str(target_records_path),
        "target_records_sha256": sha256_file(target_records_path),
        "source_requests": request_count,
        "history_present_requests": history_present_requests,
        "eligible_candidate_rows": len(candidates),
        "selected_candidate_rows": len(rows),
        "selection": (
            "history_present_and_candidate_not_in_original_history_then_sha256_first"
        ),
        "path": str(path),
        "sha256": sha256_file(path),
        "qrels_read": False,
        "model_scores_read": False,
        "source_test_opened": False,
    }
    write_json(output_dir / "manifest.json", report)
    return report


def _content_neutral_row(
    method_id: str,
    record: Any,
    tokenizer: Any,
    *,
    max_length: int,
    history_budget: int,
) -> dict[str, Any]:
    base = {
        "request_id": record.request_id,
        "candidate_count": len(record.candidates),
        "candidate_ids_sha256": sha256_text(
            json.dumps(
                [str(row["item_id"]) for row in record.candidates],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
    }
    if not record.history:
        return {
            **base,
            "eligible": False,
            "reason": "no_visible_history",
            "history_span_start": None,
            "history_span_end_exclusive": None,
            "history_span_tokens": 0,
        }
    first = record.candidates[0]
    sections = build_prompt_sections(
        method_id,
        record,
        first,
        history=record.history,
        history_budget=history_budget,
    )
    prefix = (
        f"<|im_start|>system\n{sections.system}<|im_end|>\n"
        "<|im_start|>user\n"
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(_PROMPT_SUFFIX, add_special_tokens=False)
    encoded = tokenizer(
        sections.context,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    context_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
    history_text = serialize_history(record.history, history_budget=history_budget)
    occurrences = _substring_occurrences(sections.context, history_text)
    if len(occurrences) != 1:
        return {
            **base,
            "eligible": False,
            "reason": "history_substring_not_unique",
            "history_span_start": None,
            "history_span_end_exclusive": None,
            "history_span_tokens": 0,
        }
    char_start = occurrences[0]
    char_end = char_start + len(history_text)
    span = _exact_token_span(offsets, char_start, char_end)
    if span is None:
        return {
            **base,
            "eligible": False,
            "reason": "history_not_exact_token_boundary",
            "history_span_start": None,
            "history_span_end_exclusive": None,
            "history_span_tokens": 0,
        }
    context_start, context_end = span
    body_budget = max_length - len(prefix_ids) - len(suffix_ids)
    minimum_context_budget = len(context_ids)
    for candidate in record.candidates:
        candidate_sections = build_prompt_sections(
            method_id,
            record,
            candidate,
            history=record.history,
            history_budget=history_budget,
        )
        if (
            candidate_sections.system != sections.system
            or candidate_sections.context != sections.context
        ):
            raise AssertionError("candidate changed content-neutral prefix/context")
        candidate_ids = tokenizer.encode(
            candidate_sections.candidate, add_special_tokens=False
        )
        context_budget = _context_budget(
            len(context_ids), len(candidate_ids), body_budget
        )
        minimum_context_budget = min(minimum_context_budget, context_budget)
    if context_end > minimum_context_budget:
        return {
            **base,
            "eligible": False,
            "reason": "history_span_truncated_for_candidate",
            "history_span_start": None,
            "history_span_end_exclusive": None,
            "history_span_tokens": 0,
        }
    global_start = len(prefix_ids) + context_start
    global_end = len(prefix_ids) + context_end
    if any(value == CONTENT_NEUTRAL_TOKEN_ID for value in context_ids[context_start:context_end]):
        reason = "eligible_existing_neutral_id"
    else:
        reason = "eligible"
    return {
        **base,
        "eligible": True,
        "reason": reason,
        "history_span_start": global_start,
        "history_span_end_exclusive": global_end,
        "history_span_tokens": global_end - global_start,
        "minimum_context_budget": minimum_context_budget,
        "neutralized_prompt_length_delta": 0,
    }


def _context_budget(context_length: int, candidate_length: int, body_budget: int) -> int:
    if body_budget < 16:
        raise ValueError("max length leaves no prompt body")
    candidate_budget = min(candidate_length, max(8, body_budget // 2))
    context_budget = body_budget - candidate_budget
    if context_length < context_budget:
        candidate_budget = min(candidate_length, body_budget - context_length)
        context_budget = body_budget - candidate_budget
    elif candidate_length < candidate_budget:
        context_budget = body_budget - candidate_length
    return context_budget


def _substring_occurrences(text: str, substring: str) -> list[int]:
    result: list[int] = []
    start = 0
    while True:
        index = text.find(substring, start)
        if index < 0:
            return result
        result.append(index)
        start = index + 1


def _exact_token_span(
    offsets: Sequence[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int] | None:
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not indices:
        return None
    first, last = indices[0], indices[-1]
    if offsets[first][0] != char_start or offsets[last][1] != char_end:
        return None
    if indices != list(range(first, last + 1)):
        return None
    return first, last + 1


def _load_records(path: Path, *, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in iter_jsonl(path):
        request_id = str(raw.get("request_id") or "")
        user_id = str(raw.get("user_id") or "")
        if not request_id or not user_id or request_id in seen:
            raise ValueError(f"invalid or duplicate {role} request: {request_id}")
        seen.add(request_id)
        request_ts = int(raw["ts"])
        history = [dict(event) for event in raw.get("history", [])]
        if any(int(event["ts"]) >= request_ts for event in history):
            raise ValueError(f"noncausal {role} history: {request_id}")
        candidates = raw.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"empty {role} candidate slate: {request_id}")
        rows.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "ts": request_ts,
                "history": history,
                "candidate_ids": {str(row["item_id"]) for row in candidates},
            }
        )
    if not rows:
        raise ValueError(f"empty {role} records: {path}")
    return rows


def _index_donors(
    donors: Sequence[Mapping[str, Any]], tokenizer: Any
) -> tuple[
    dict[tuple[int, int], list[dict[str, Any]]],
    dict[int, tuple[int, ...]],
]:
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    lengths: dict[int, set[int]] = defaultdict(set)
    for donor in donors:
        history = list(donor["history"])
        for count in range(1, HISTORY_BUDGET + 1):
            if len(history) < count:
                continue
            selected = history[-count:]
            serialized = serialize_history(selected, history_budget=count)
            token_length = len(
                tokenizer.encode(serialized, add_special_tokens=False)
            )
            entry = {
                "request_id": str(donor["request_id"]),
                "user_id": str(donor["user_id"]),
                "history": selected,
                "item_ids": frozenset(str(event["item_id"]) for event in selected),
                "maximum_ts": max(int(event["ts"]) for event in selected),
                "token_length": token_length,
            }
            buckets[(count, token_length)].append(entry)
            lengths[count].add(token_length)
    return buckets, {key: tuple(sorted(value)) for key, value in lengths.items()}


def _match_target(
    target: Mapping[str, Any],
    *,
    donor_buckets: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    donor_lengths: Mapping[int, Sequence[int]],
    tokenizer: Any,
) -> dict[str, Any]:
    count = min(HISTORY_BUDGET, len(target["history"]))
    base = {
        "request_id": str(target["request_id"]),
        "visible_history_count": count,
    }
    if count == 0:
        return {
            **base,
            "eligible": False,
            "match_type": "recipient_no_visible_history",
            "donor_request_id": None,
            "donor_user_id": None,
            "recipient_token_length": 0,
            "donor_token_length": None,
            "token_length_absolute_difference": None,
            "history": [],
        }
    target_history = list(target["history"])[-count:]
    target_text = serialize_history(target_history, history_budget=count)
    target_length = len(tokenizer.encode(target_text, add_special_tokens=False))
    forbidden = set(target["candidate_ids"])
    forbidden.update(str(event["item_id"]) for event in target["history"])
    for token_length in sorted(
        donor_lengths.get(count, ()), key=lambda value: (abs(value - target_length), value)
    ):
        eligible = [
            donor
            for donor in donor_buckets[(count, int(token_length))]
            if donor["user_id"] != target["user_id"]
            and int(donor["maximum_ts"]) < int(target["ts"])
            and not donor["item_ids"].intersection(forbidden)
        ]
        if not eligible:
            continue
        donor = min(
            eligible,
            key=lambda row: _tie_hash(
                str(target["request_id"]), str(row["request_id"])
            ),
        )
        return {
            **base,
            "eligible": True,
            "match_type": "exact_count_token_length_nearest_wrong_user",
            "donor_request_id": str(donor["request_id"]),
            "donor_user_id": str(donor["user_id"]),
            "recipient_token_length": target_length,
            "donor_token_length": int(donor["token_length"]),
            "token_length_absolute_difference": abs(
                int(donor["token_length"]) - target_length
            ),
            "history": list(donor["history"]),
        }
    return {
        **base,
        "eligible": False,
        "match_type": "no_eligible_train_donor",
        "donor_request_id": None,
        "donor_user_id": None,
        "recipient_token_length": target_length,
        "donor_token_length": None,
        "token_length_absolute_difference": None,
        "history": [],
    }


def _tie_hash(recipient_request_id: str, donor_request_id: str) -> str:
    payload = "\x1f".join(
        (WRONG_USER_NAMESPACE, recipient_request_id, donor_request_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audit_rows(
    rows: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]
) -> None:
    if len(rows) != len(targets):
        raise AssertionError("wrong-user mapping lost target coverage")
    for row, target in zip(rows, targets, strict=True):
        if row["request_id"] != target["request_id"]:
            raise AssertionError("wrong-user mapping changed target order")
        if not row["eligible"]:
            continue
        history = row["history"]
        if len(history) != row["visible_history_count"]:
            raise AssertionError("wrong-user mapping changed visible event count")
        if row["donor_user_id"] == target["user_id"]:
            raise AssertionError("wrong-user mapping contains same-user donor")
        forbidden = set(target["candidate_ids"])
        forbidden.update(str(event["item_id"]) for event in target["history"])
        if any(str(event["item_id"]) in forbidden for event in history):
            raise AssertionError("wrong-user mapping contains recipient item")
        if any(int(event["ts"]) >= int(target["ts"]) for event in history):
            raise AssertionError("wrong-user mapping contains noncausal event")


def _summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }
