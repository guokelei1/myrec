"""Read-only tokenizer length audit for frozen M1 history interventions.

This diagnostic reuses the production scorer's assignment admission and the
frozen Q0--Q3 prompt/history builders.  It loads tokenizer artifacts only: no
model weights, qrels, model scores, confirmation records, or source test are
opened.  The six intervention assignments are immutable inputs.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import (
    ModelRecord,
    build_instructrec_selection_sections,
    build_prompt_sections,
    encode_instructrec_selection_prompt,
    instructrec_template_index,
    serialize_candidate,
    serialize_history,
)
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _answer_target_tokens,
)
from myrec.mechanism.history_interventions import CONDITION_IDS, HISTORY_BUDGET
from myrec.mechanism.scorer import (
    _load_assignments,
    _load_frozen_records,
    _validate_assignment_manifest,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import write_json


PROBE_MANIFEST_SHA256 = (
    "adedf0e662b9d8529162b8abffedcf6b10962913f28580af6119d807cc5d929c"
)
ASSIGNMENT_MANIFEST_SHA256 = (
    "ced5b21aeb350b64a2ce317fddb189fe18d8d5f02beb715d17487fe19aa606c6"
)
METHOD_IDS = (
    "q0_qwen3_reranker_06b",
    "q1_instructrec_generalqwen",
    "q2_recranker_generalqwen",
    "q3_tallrec_generalqwen",
)

_PREFIX_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
)
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class RequestTokenMeasurement:
    """One request's exact scorer-input length accounting."""

    total_prompt_tokens: int
    raw_total_prompt_tokens: int
    visible_history_tokens: int
    prompt_instances: int
    truncated_prompt_instances: int
    raw_overlength_prompt_instances: int
    at_max_boundary_prompt_instances: int

    @property
    def request_truncated(self) -> bool:
        return self.truncated_prompt_instances > 0

    @property
    def request_raw_overlength(self) -> bool:
        return self.raw_overlength_prompt_instances > 0

    @property
    def request_at_max_boundary(self) -> bool:
        return self.at_max_boundary_prompt_instances > 0


class _CachingTokenizer:
    """Small LRU around the exact local tokenizer for repeated request segments."""

    def __init__(self, tokenizer: Any, *, max_entries: int = 8_192) -> None:
        self.tokenizer = tokenizer
        self.max_entries = max_entries
        self.cache: OrderedDict[tuple[str, bool], list[int]] = OrderedDict()

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        key = (text, bool(add_special_tokens))
        value = self.cache.pop(key, None)
        if value is None:
            value = self.tokenizer.encode(
                text, add_special_tokens=add_special_tokens
            )
        self.cache[key] = value
        if len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def __len__(self) -> int:
        return len(self.tokenizer)


def measure_request_tokens(
    method_id: str,
    config: Mapping[str, Any],
    tokenizer: Any,
    record: ModelRecord,
    history: Sequence[dict[str, Any]],
) -> RequestTokenMeasurement:
    """Measure exactly the prompt(s) the formal scorer would construct."""

    if method_id not in METHOD_IDS:
        raise ValueError(f"unsupported token-audit method_id={method_id}")
    training = config["training"]
    history_budget = int(training["history_budget"])
    if history_budget != HISTORY_BUDGET:
        raise ValueError("token audit requires the frozen visible history budget=6")
    visible_history_text = serialize_history(history, history_budget=history_budget)
    visible_history_tokens = len(
        tokenizer.encode(visible_history_text, add_special_tokens=False)
    )
    if method_id == "q1_instructrec_generalqwen":
        return _measure_q1(
            config,
            tokenizer,
            record,
            history,
            visible_history_tokens=visible_history_tokens,
        )
    return _measure_pointwise(
        method_id,
        config,
        tokenizer,
        record,
        history,
        visible_history_tokens=visible_history_tokens,
    )


def summarize_measurement_comparisons(
    baseline: Sequence[RequestTokenMeasurement],
    intervention: Sequence[RequestTokenMeasurement],
) -> dict[str, Any]:
    """Aggregate paired request measurements using fixed nearest-rank tails."""

    if not baseline or len(baseline) != len(intervention):
        raise ValueError("baseline/intervention measurements must be non-empty pairs")
    total = _metric_comparison(
        [row.total_prompt_tokens for row in baseline],
        [row.total_prompt_tokens for row in intervention],
    )
    raw_total = _metric_comparison(
        [row.raw_total_prompt_tokens for row in baseline],
        [row.raw_total_prompt_tokens for row in intervention],
    )
    visible = _metric_comparison(
        [row.visible_history_tokens for row in baseline],
        [row.visible_history_tokens for row in intervention],
    )
    return {
        "requests": len(baseline),
        "total_prompt_tokens": total,
        "raw_total_prompt_tokens": raw_total,
        "visible_history_tokens": visible,
        "truncation_and_overlength": _rate_comparison(baseline, intervention),
    }


def run_token_length_audit(
    standardized_dir: str | Path,
    probe_manifest_path: str | Path,
    assignment_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the frozen 8k Q0--Q3 tokenizer audit and write run-local evidence."""

    started = time.monotonic()
    standardized = Path(standardized_dir).resolve()
    probe_path = Path(probe_manifest_path).resolve()
    assignment_manifest_path = Path(assignment_manifest_path).resolve()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"token audit output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    probe_sha256 = sha256_file(probe_path)
    assignment_manifest_sha256 = sha256_file(assignment_manifest_path)
    if probe_sha256 != PROBE_MANIFEST_SHA256:
        raise ValueError("frozen mechanism probe manifest SHA-256 differs")
    if assignment_manifest_sha256 != ASSIGNMENT_MANIFEST_SHA256:
        raise ValueError("frozen M1 assignment manifest SHA-256 differs")
    probe = _read_yaml_object(probe_path)
    assignment_manifest = _read_json_object(assignment_manifest_path)
    _validate_probe_scope(probe)

    records_path = standardized / "records_dev.jsonl"
    dataset_manifest_path = standardized / "manifest.json"
    candidate_manifest_path = standardized / "candidate_manifest.json"
    request_manifest_path = standardized / "request_manifest.json"
    frozen_inputs = probe["frozen_inputs"]
    source_hashes = {
        "records_sha256": sha256_file(records_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
    }
    expected_source_hashes = {
        "records_sha256": str(frozen_inputs["records_dev_sha256"]),
        "dataset_manifest_sha256": str(frozen_inputs["dataset_manifest_sha256"]),
        "candidate_manifest_sha256": str(
            frozen_inputs["candidate_manifest_sha256"]
        ),
        "request_manifest_sha256": str(frozen_inputs["request_manifest_sha256"]),
    }
    if source_hashes != expected_source_hashes:
        raise ValueError("token audit source hashes differ from probe manifest")
    dataset_manifest = _read_json_object(dataset_manifest_path)
    dataset_id = str(dataset_manifest.get("dataset_id"))
    dataset_version = str(dataset_manifest.get("dataset_version"))
    if dataset_id != "kuaisearch" or dataset_version != str(
        probe["scope"]["development_population"]
    ):
        raise ValueError("token audit is restricted to frozen KuaiSearch internal-dev")

    records = _load_frozen_records(records_path)
    if len(records) != 8_000:
        raise ValueError("token audit requires all 8000 frozen internal-dev requests")
    condition_assignments: dict[str, dict[str, list[dict[str, Any]]]] = {}
    condition_inputs: dict[str, Any] = {}
    for condition_id in CONDITION_IDS:
        entry = assignment_manifest.get("conditions", {}).get(condition_id)
        if not isinstance(entry, Mapping):
            raise ValueError(f"assignment manifest lacks condition={condition_id}")
        assignment_path = _resolve_assignment_path(
            assignment_manifest_path, str(entry.get("path") or "")
        )
        hashes = {
            **source_hashes,
            "assignment_sha256": sha256_file(assignment_path),
            "assignment_manifest_sha256": assignment_manifest_sha256,
        }
        _validate_assignment_manifest(
            assignment_manifest_path,
            assignment_path=assignment_path,
            condition_id=condition_id,
            hashes=hashes,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            split="dev",
            request_count=len(records),
        )
        condition_assignments[condition_id] = _load_assignments(
            assignment_path,
            condition_id=condition_id,
            records=records,
        )
        condition_inputs[condition_id] = {
            "path": str(assignment_path),
            "sha256": hashes["assignment_sha256"],
            "request_count": len(condition_assignments[condition_id]),
        }

    request_rows_path = output / "request_token_lengths.jsonl"
    summaries: dict[str, Any] = {}
    model_metadata: dict[str, Any] = {}
    with request_rows_path.open("x", encoding="utf-8") as request_handle:
        for method_id in METHOD_IDS:
            config, checkpoint_root, model_audit = _load_frozen_method_inputs(
                probe, method_id
            )
            raw_tokenizer, tokenizer_audit = _load_local_tokenizer(checkpoint_root)
            tokenizer = _CachingTokenizer(raw_tokenizer)
            model_metadata[method_id] = {**model_audit, **tokenizer_audit}
            paired: dict[
                str, tuple[list[RequestTokenMeasurement], list[RequestTokenMeasurement]]
            ] = {
                condition_id: ([], []) for condition_id in CONDITION_IDS
            }
            for frozen in records:
                record = frozen["record"]
                baseline = measure_request_tokens(
                    method_id, config, tokenizer, record, record.history
                )
                for condition_id in CONDITION_IDS:
                    intervention = measure_request_tokens(
                        method_id,
                        config,
                        tokenizer,
                        record,
                        condition_assignments[condition_id][record.request_id],
                    )
                    paired[condition_id][0].append(baseline)
                    paired[condition_id][1].append(intervention)
                    request_handle.write(
                        _canonical_json(
                            {
                                "request_id": record.request_id,
                                "method_id": method_id,
                                "condition_id": condition_id,
                                "baseline": asdict(baseline),
                                "intervention": asdict(intervention),
                                "delta": {
                                    "total_prompt_tokens": (
                                        intervention.total_prompt_tokens
                                        - baseline.total_prompt_tokens
                                    ),
                                    "raw_total_prompt_tokens": (
                                        intervention.raw_total_prompt_tokens
                                        - baseline.raw_total_prompt_tokens
                                    ),
                                    "visible_history_tokens": (
                                        intervention.visible_history_tokens
                                        - baseline.visible_history_tokens
                                    ),
                                },
                            }
                        )
                        + "\n"
                    )
            summaries[method_id] = {
                condition_id: summarize_measurement_comparisons(*paired[condition_id])
                for condition_id in CONDITION_IDS
            }

    summary = {
        "schema_version": 1,
        "audit_id": "m1_token_length_audit_v1",
        "population": "kuaisearch_internal_dev_8000",
        "request_count": len(records),
        "method_order": list(METHOD_IDS),
        "condition_order": list(CONDITION_IDS),
        "measurement_contract": {
            "total_prompt_tokens": (
                "per-request sum of finalized formal-scorer prompt tokens; one slate "
                "prompt for Q1 and one prompt per candidate for Q0/Q2/Q3"
            ),
            "raw_total_prompt_tokens": (
                "same request-level sum before formal context/candidate clipping"
            ),
            "visible_history_tokens": (
                "tokens in formal serialize_history output before prompt clipping"
            ),
            "percentiles": "nearest-rank; median is exact sample median",
            "truncation": "any formal prompt segment clipped before model input",
            "raw_overlength": "unclipped prompt length exceeds scorer input limit",
        },
        "methods": summaries,
        "qrels_read": False,
        "model_scores_read": False,
        "model_weights_read": False,
        "confirmation_records_read": False,
        "source_test_opened": False,
    }
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    implementation = _implementation_identity()
    metadata = {
        "schema_version": 1,
        "run_id": output.name,
        "status": "completed",
        "audit_id": summary["audit_id"],
        "command": list(command) if command is not None else None,
        "elapsed_seconds": time.monotonic() - started,
        "probe_manifest_path": str(probe_path),
        "probe_manifest_sha256": probe_sha256,
        "assignment_manifest_path": str(assignment_manifest_path),
        "assignment_manifest_sha256": assignment_manifest_sha256,
        "source": {
            "standardized_dir": str(standardized),
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            **source_hashes,
        },
        "conditions": condition_inputs,
        "models": model_metadata,
        "outputs": {
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "request_token_lengths": {
                "path": str(request_rows_path),
                "sha256": sha256_file(request_rows_path),
                "rows": len(records) * len(METHOD_IDS) * len(CONDITION_IDS),
            },
        },
        "implementation": implementation,
        "qrels_read": False,
        "model_scores_read": False,
        "model_weights_read": False,
        "confirmation_records_read": False,
        "source_test_opened": False,
        "assignment_files_modified": False,
        "assignment_manifest_modified": False,
    }
    metadata_path = output / "metadata.json"
    write_json(metadata_path, metadata)
    return metadata


def _measure_pointwise(
    method_id: str,
    config: Mapping[str, Any],
    tokenizer: Any,
    record: ModelRecord,
    history: Sequence[dict[str, Any]],
    *,
    visible_history_tokens: int,
) -> RequestTokenMeasurement:
    training = config["training"]
    target_reserve = 0
    if method_id == "q3_tallrec_generalqwen":
        target_reserve = max(
            len(_answer_target_tokens(tokenizer, "Yes")),
            len(_answer_target_tokens(tokenizer, "No")),
        )
    limit = int(training["max_length"]) - target_reserve
    total = 0
    raw_total = 0
    truncated = 0
    overlength = 0
    boundary = 0
    for candidate in record.candidates:
        sections = build_prompt_sections(
            method_id,
            record,
            candidate,
            history=history,
            history_budget=int(training["history_budget"]),
        )
        prefix_tokens = len(
            tokenizer.encode(
                _PREFIX_TEMPLATE.format(system=sections.system),
                add_special_tokens=False,
            )
        )
        context_tokens = len(
            tokenizer.encode(sections.context, add_special_tokens=False)
        )
        candidate_tokens = len(
            tokenizer.encode(sections.candidate, add_special_tokens=False)
        )
        suffix_tokens = len(
            tokenizer.encode(_PROMPT_SUFFIX, add_special_tokens=False)
        )
        raw = prefix_tokens + context_tokens + candidate_tokens + suffix_tokens
        final = _pointwise_encoded_length(
            prefix_tokens,
            context_tokens,
            candidate_tokens,
            suffix_tokens,
            max_length=limit,
        )
        total += final
        raw_total += raw
        truncated += int(final < raw)
        overlength += int(raw > limit)
        boundary += int(final == limit)
    return RequestTokenMeasurement(
        total_prompt_tokens=total,
        raw_total_prompt_tokens=raw_total,
        visible_history_tokens=visible_history_tokens,
        prompt_instances=len(record.candidates),
        truncated_prompt_instances=truncated,
        raw_overlength_prompt_instances=overlength,
        at_max_boundary_prompt_instances=boundary,
    )


def _measure_q1(
    config: Mapping[str, Any],
    tokenizer: Any,
    record: ModelRecord,
    history: Sequence[dict[str, Any]],
    *,
    visible_history_tokens: int,
) -> RequestTokenMeasurement:
    training = config["training"]
    max_target = int(training.get("max_target_length", 96))
    limit = int(training["max_length"]) - max_target
    template_index = instructrec_template_index(
        record.request_id, seed=int(training["seed"])
    )
    final, _responses, encoding_audit = encode_instructrec_selection_prompt(
        tokenizer,
        record,
        record.candidates,
        history=history,
        history_budget=int(training["history_budget"]),
        template_index=template_index,
        max_length=limit,
        context_token_budget=int(training["context_token_budget"]),
        max_target_length=max_target,
    )
    raw = _raw_q1_prompt_length(
        tokenizer,
        record,
        history,
        history_budget=int(training["history_budget"]),
        template_index=template_index,
    )
    context_was_clipped = int(encoding_audit["context_tokens_observed"]) < len(
        tokenizer.encode(
            build_instructrec_selection_sections(
                record,
                record.candidates,
                history=history,
                history_budget=int(training["history_budget"]),
                template_index=template_index,
            ).context,
            add_special_tokens=False,
        )
    )
    truncated = int(len(final) < raw or context_was_clipped)
    return RequestTokenMeasurement(
        total_prompt_tokens=len(final),
        raw_total_prompt_tokens=raw,
        visible_history_tokens=visible_history_tokens,
        prompt_instances=1,
        truncated_prompt_instances=truncated,
        raw_overlength_prompt_instances=int(raw > limit),
        at_max_boundary_prompt_instances=int(len(final) == limit),
    )


def _pointwise_encoded_length(
    prefix_tokens: int,
    context_tokens: int,
    candidate_tokens: int,
    suffix_tokens: int,
    *,
    max_length: int,
) -> int:
    """Length-only equivalent of the frozen ``encode_prompt_sections``."""

    if max_length < 64:
        raise ValueError("max_length must be at least 64")
    body_budget = max_length - prefix_tokens - suffix_tokens
    if body_budget < 16:
        raise ValueError("max_length leaves no room for query and candidate text")
    candidate_budget = min(candidate_tokens, max(8, body_budget // 2))
    context_budget = body_budget - candidate_budget
    if context_tokens < context_budget:
        candidate_budget = min(candidate_tokens, body_budget - context_tokens)
        context_budget = body_budget - candidate_budget
    elif candidate_tokens < candidate_budget:
        context_budget = body_budget - candidate_tokens
        candidate_budget = candidate_tokens
    return (
        prefix_tokens
        + min(context_tokens, context_budget)
        + min(candidate_tokens, candidate_budget)
        + suffix_tokens
    )


def _raw_q1_prompt_length(
    tokenizer: Any,
    record: ModelRecord,
    history: Sequence[dict[str, Any]],
    *,
    history_budget: int,
    template_index: int,
) -> int:
    sections = build_instructrec_selection_sections(
        record,
        record.candidates,
        history=history,
        history_budget=history_budget,
        template_index=template_index,
    )
    prefix = _PREFIX_TEMPLATE.format(system=sections.system)
    # Use the same separately encoded segments as the production Q1 encoder.
    header = "Candidate products:\n"
    footer = "\nSelected product:"
    return (
        len(tokenizer.encode(prefix, add_special_tokens=False))
        + len(tokenizer.encode(sections.context, add_special_tokens=False))
        + len(tokenizer.encode(header, add_special_tokens=False))
        + sum(
            len(tokenizer.encode(f"{index}. ", add_special_tokens=False))
            + len(tokenizer.encode(serialize_candidate(candidate), add_special_tokens=False))
            + len(tokenizer.encode("\n", add_special_tokens=False))
            for index, candidate in enumerate(record.candidates, start=1)
        )
        + len(tokenizer.encode(footer, add_special_tokens=False))
        + len(tokenizer.encode(_PROMPT_SUFFIX, add_special_tokens=False))
    )


def _metric_comparison(baseline: Sequence[int], intervention: Sequence[int]) -> dict[str, Any]:
    if len(baseline) != len(intervention):
        raise ValueError("paired metric lengths differ")
    delta = [right - left for left, right in zip(baseline, intervention)]
    absolute = [abs(value) for value in delta]
    return {
        "baseline": _distribution(baseline),
        "intervention": _distribution(intervention),
        "delta": _distribution(delta),
        "absolute_delta": _distribution(absolute),
        "changed_requests": sum(value != 0 for value in delta),
        "changed_rate": sum(value != 0 for value in delta) / len(delta),
        "positive_delta_requests": sum(value > 0 for value in delta),
        "negative_delta_requests": sum(value < 0 for value in delta),
    }


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": statistics.median(ordered),
        "p90": _nearest_rank(ordered, 0.90),
        "p99": _nearest_rank(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _nearest_rank(ordered: Sequence[int], probability: float) -> int:
    if not 0.0 < probability <= 1.0:
        raise ValueError("nearest-rank probability must be in (0, 1]")
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return int(ordered[index])


def _rate_comparison(
    baseline: Sequence[RequestTokenMeasurement],
    intervention: Sequence[RequestTokenMeasurement],
) -> dict[str, Any]:
    def rates(rows: Sequence[RequestTokenMeasurement]) -> dict[str, Any]:
        requests = len(rows)
        prompts = sum(row.prompt_instances for row in rows)
        truncated_prompts = sum(row.truncated_prompt_instances for row in rows)
        overlength_prompts = sum(
            row.raw_overlength_prompt_instances for row in rows
        )
        boundary_prompts = sum(row.at_max_boundary_prompt_instances for row in rows)
        return {
            "request_truncation_count": sum(row.request_truncated for row in rows),
            "request_truncation_rate": sum(row.request_truncated for row in rows)
            / requests,
            "prompt_truncation_count": truncated_prompts,
            "prompt_truncation_rate": truncated_prompts / prompts,
            "request_raw_overlength_count": sum(
                row.request_raw_overlength for row in rows
            ),
            "request_raw_overlength_rate": sum(
                row.request_raw_overlength for row in rows
            )
            / requests,
            "prompt_raw_overlength_count": overlength_prompts,
            "prompt_raw_overlength_rate": overlength_prompts / prompts,
            "request_at_max_boundary_count": sum(
                row.request_at_max_boundary for row in rows
            ),
            "request_at_max_boundary_rate": sum(
                row.request_at_max_boundary for row in rows
            )
            / requests,
            "prompt_at_max_boundary_count": boundary_prompts,
            "prompt_at_max_boundary_rate": boundary_prompts / prompts,
            "prompt_instances": prompts,
        }

    left = rates(baseline)
    right = rates(intervention)
    rate_delta = {
        key: right[key] - left[key]
        for key in left
        if key.endswith("_rate")
    }
    return {
        "baseline": left,
        "intervention": right,
        "rate_delta": rate_delta,
        "request_truncation_transitions": {
            "unchanged_not_truncated": sum(
                not left_row.request_truncated and not right_row.request_truncated
                for left_row, right_row in zip(baseline, intervention)
            ),
            "newly_truncated": sum(
                not left_row.request_truncated and right_row.request_truncated
                for left_row, right_row in zip(baseline, intervention)
            ),
            "no_longer_truncated": sum(
                left_row.request_truncated and not right_row.request_truncated
                for left_row, right_row in zip(baseline, intervention)
            ),
            "unchanged_truncated": sum(
                left_row.request_truncated and right_row.request_truncated
                for left_row, right_row in zip(baseline, intervention)
            ),
        },
    }


def _load_frozen_method_inputs(
    probe: Mapping[str, Any], method_id: str
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    import yaml

    declared = probe["frozen_inputs"]["models"].get(method_id)
    if not isinstance(declared, Mapping):
        raise ValueError(f"probe manifest lacks frozen model={method_id}")
    config_path = Path(str(declared["config"])).resolve()
    config_sha256 = sha256_file(config_path)
    if config_sha256 != str(declared["config_sha256"]):
        raise ValueError(f"frozen config hash differs for method={method_id}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or str(config.get("method_id")) != method_id:
        raise ValueError(f"invalid frozen config for method={method_id}")
    if int(config["training"]["history_budget"]) != HISTORY_BUDGET:
        raise ValueError(f"history budget drift for method={method_id}")
    protocol_path = Path(str(config["protocol"]["path"])).resolve()
    if sha256_file(protocol_path) != str(config["protocol"]["sha256"]):
        raise ValueError(f"protocol hash differs for method={method_id}")
    checkpoint_root = Path(str(declared["checkpoint"])).resolve()
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json_object(training_metadata_path)
    checkpoint_id = str(declared["checkpoint_id"])
    if str(training_metadata.get("checkpoint_id")) != checkpoint_id:
        raise ValueError(f"checkpoint metadata differs for method={method_id}")
    return config, checkpoint_root, {
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_id": checkpoint_id,
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "prompt_max_length": int(config["training"]["max_length"]),
        "history_budget": int(config["training"]["history_budget"]),
    }


def _load_local_tokenizer(checkpoint_root: Path) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    if not tokenizer_path.is_file() or not tokenizer_config_path.is_file():
        raise FileNotFoundError(f"checkpoint tokenizer is incomplete: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, {
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
        "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "local_files_only": True,
    }


def _validate_probe_scope(probe: Mapping[str, Any]) -> None:
    if int(probe.get("schema_version", -1)) != 1:
        raise ValueError("invalid mechanism probe manifest schema")
    scope = probe.get("scope", {})
    if scope.get("dataset_id") != "kuaisearch":
        raise ValueError("token audit probe scope is not KuaiSearch")
    if scope.get("evaluation_population") != "internal_dev_only":
        raise ValueError("token audit probe scope is not internal-dev only")
    if scope.get("source_test_opened") is not False:
        raise ValueError("mechanism probe manifest crossed source-test boundary")
    m1 = probe.get("m1_input_interventions", {})
    if tuple(m1.get("models", ())) != METHOD_IDS:
        raise ValueError("mechanism probe model order differs")
    if tuple(m1.get("conditions", {}).keys()) != CONDITION_IDS:
        raise ValueError("mechanism probe intervention conditions differ")


def _resolve_assignment_path(manifest_path: Path, declared: str) -> Path:
    if not declared:
        raise ValueError("assignment condition path is empty")
    path = Path(declared)
    candidates = [path.resolve()]
    if not path.is_absolute():
        candidates.append((manifest_path.parent / path).resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"assignment path cannot be resolved: {declared}")


def _implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = {
        "scripts/audit_mechanism_token_lengths.py": (
            root / "scripts/audit_mechanism_token_lengths.py"
        ),
        "src/myrec/mechanism/token_length_audit.py": Path(__file__).resolve(),
    }
    files = [
        {"path": name, "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    ]
    return {
        "files": files,
        "digest": sha256_text(_canonical_json(files)),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_yaml_object(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
