"""Resumable qrels-blind D3 per-head observation on the frozen 512 rows."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import (
    build_prompt_sections,
    complete_candidate_chunks,
    sanitize_record_for_model,
    serialize_history,
)
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _answer_target_tokens,
    _checkpoint_identity,
    _git_revision,
    _load_model_and_tokenizer,
    _runtime_metadata,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.mechanism.attention_edge_runtime import (
    BASELINE_SCORE_DIRS,
    DEEP_DIVE_MANIFEST_PATH,
    MAX_WALL_SECONDS,
    _assert_native_targets,
    _canonical_sha256,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.attention_head_observation import QwenAttentionHeadObserver
from myrec.mechanism.patch_scorer import _left_pad_sequences
from myrec.mechanism.postblock_sweep_runtime import NULL_BASELINE_DIRS
from myrec.mechanism.representation_probe import (
    PREFIX_TEMPLATE,
    _context_spans,
    _encode,
    _encode_with_offsets,
    instrument_pointwise_prompt,
)
from myrec.mechanism.attention_edge_scoring import _aggregate_paths, _path_scores
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


SUPPORTED_METHODS = ("q2_recranker_generalqwen", "q3_tallrec_generalqwen")
FIXED_BLOCKS = (13, 20, 27)
SAMPLE_MANIFEST = Path(
    "artifacts/motivation_transformer_deep_dive/frozen_controls/"
    "fixed_candidate_rows_v1/manifest.json"
)
SAMPLE_MANIFEST_SHA256 = (
    "84cdf68a0fabefcab055806bb690adf96f2a36ad2921c2d10c5d0aae8310aa61"
)


def write_attention_observation_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    sample_manifest_path: str | Path = SAMPLE_MANIFEST,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_rows: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Observe every registered head/KV group without reading labels."""

    _validate_run_id(run_id)
    block = int(block)
    if block not in FIXED_BLOCKS:
        raise ValueError("attention observation block must be 13,20,or27")
    if not str(device).strip():
        raise ValueError("an explicit attention observation device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("attention observation wall time must be in (0,13500]")
    if max_rows is not None and not 0 < int(max_rows) < 512:
        raise ValueError("attention observation smoke rows must be in [1,511]")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    sample_manifest_path = Path(sample_manifest_path)
    if sha256_file(sample_manifest_path) != SAMPLE_MANIFEST_SHA256:
        raise ValueError("attention high-dimensional sample manifest differs")
    sample_manifest = _read_json(sample_manifest_path)
    sample_path = Path(sample_manifest["path"])
    if (
        sample_manifest.get("qrels_read") is not False
        or sample_manifest.get("model_scores_read") is not False
        or sample_manifest.get("selected_candidate_rows") != 512
        or sample_manifest.get("sha256") != sha256_file(sample_path)
    ):
        raise ValueError("attention high-dimensional sample is not frozen qrels-blind")
    records_path = standardized_dir / "records_dev.jsonl"
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("attention observation requires frozen 8000-request dev")
    records_by_id = {record.request_id: record for record in records}
    samples = list(iter_jsonl(sample_path))
    _audit_sample(samples, records_by_id)
    target_samples = samples if max_rows is None else samples[: int(max_rows)]

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    native_scoring_batch_size = int(config.get("scoring", {}).get("batch_size", 8))
    if native_scoring_batch_size <= 0:
        raise ValueError("attention observation scoring batch size must be positive")
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("attention observation supports only Q2/Q3")
    frozen = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("attention observation config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(model_dir, method_id)
    if checkpoint_id != frozen["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("attention observation checkpoint differs from frozen binding")
    baseline_full, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    implementation = attention_observation_implementation_identity()
    evidence_mode = "registered_qrels_blind_high_dimensional_observation" if max_rows is None else "mechanical_smoke_non_result"
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "sample_rows_sha256": sha256_file(sample_path),
        "target_rows": len(target_samples),
        "native_scoring_batch_size": native_scoring_batch_size,
        "selected_row_batch_policy": "original_request_candidate_chunk",
        "full_scores_sha256": full_identity["scores_sha256"],
        "null_scores_sha256": null_identity["scores_sha256"],
        "implementation_digest": implementation["digest"],
        "device": str(device),
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d3_attention_head_observation",
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_sha256": sha256_file(records_path),
        "sample_manifest_path": str(sample_manifest_path),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "sample_rows_path": str(sample_path),
        "sample_rows_sha256": sha256_file(sample_path),
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "query_heads": 16,
        "kv_heads": 8,
        "gqa_heads_per_kv": 2,
        "native_scoring_batch_size": native_scoring_batch_size,
        "selected_row_batch_policy": "original_request_candidate_chunk",
        "observation_scopes": ["history_summary", "native_readout"],
        "observation_spans": ["query", "history", "candidate"],
        "semantic_span_token_policy": "minimal contiguous tokenizer rows overlapping the exact character span; permits Qwen UTF-8 fallback offset overlap",
        "qk_stages": ["pre_norm", "post_norm", "post_rope"],
        "manual_attention_reconstruction_dtype": "float32",
        "implementation_identity": implementation,
        "qrels_read": False,
        "source_test_opened": False,
        "evidence_mode": evidence_mode,
        "result_eligible": max_rows is None,
        "run_contract": contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    metadata, progress = _prepare(
        run_dir, metadata, contract_sha256, target_samples, resume=resume
    )
    completed = int(progress["completed_rows"])
    if completed >= len(target_samples):
        return _finalize(run_dir, metadata, progress, target_samples)

    started = time.monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(
            config, device=str(device), training=False, checkpoint_model_dir=model_dir
        )
        model.eval()
        _assert_native_targets(tokenizer, method_id, manifest)
        metadata.update(_runtime_metadata(method_id, torch, transformers))
        metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", metadata)
        max_identity = float(metadata.get("maximum_score_identity_delta", 0.0))
        max_manual = float(metadata.get("maximum_manual_attention_error", 0.0))
        max_manual_ratio = float(
            metadata.get("maximum_manual_attention_low_precision_ratio", 0.0)
        )
        with torch.inference_mode(), QwenAttentionHeadObserver(model, block) as observer:
            for row_index in range(completed, len(target_samples)):
                if time.monotonic() - started >= max_wall_seconds:
                    metadata.update(
                        {
                            "status": "wall_time_exhausted",
                            "resumable": True,
                            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started,
                            "maximum_score_identity_delta": max_identity,
                            "maximum_manual_attention_error": max_manual,
                            "maximum_manual_attention_low_precision_ratio": (
                                max_manual_ratio
                            ),
                        }
                    )
                    _write_json(run_dir / "metadata.json", metadata)
                    return metadata
                sample = target_samples[row_index]
                record = records_by_id[str(sample["request_id"])]
                candidate_ordinal = int(sample["candidate_ordinal"])
                candidate = record.candidates[candidate_ordinal]
                paths = _build_observation_paths(
                    tokenizer,
                    record,
                    candidate_ordinal,
                    config,
                    device=str(device),
                )
                path_rows = {}
                full_values = []
                null_values = []
                path_descriptors = []
                selected_batch_row = None
                for path in paths:
                    full_output, full_observed = _observe_forward(model, observer, path["full"])
                    null_output, null_observed = _observe_forward(model, observer, path["null"])
                    full_values.append(_path_scores(full_output, path["full"]))
                    null_values.append(_path_scores(null_output, path["null"]))
                    path_descriptors.append(path["full"])
                    path_selected_row = int(path["selected_batch_row"])
                    if selected_batch_row is None:
                        selected_batch_row = path_selected_row
                    elif selected_batch_row != path_selected_row:
                        raise ValueError("attention observation selected batch row drifted")
                    summarized = _summarize_path(
                        full_observed,
                        null_observed,
                        selected_batch_row=path_selected_row,
                    )
                    max_manual = max(max_manual, summarized["manual_attention_error"])
                    max_manual_ratio = max(
                        max_manual_ratio,
                        summarized["manual_attention_low_precision_ratio"],
                    )
                    if max_manual_ratio > 1.0:
                        raise ValueError(
                            "attention observation manual selected-row reconstruction "
                            f"exceeded low-precision bound: {max_manual_ratio}"
                        )
                    path_rows[path["name"]] = summarized
                if selected_batch_row is None:
                    raise ValueError("attention observation produced no native paths")
                observed_full = float(
                    _aggregate_paths(path_descriptors, full_values)[selected_batch_row]
                )
                observed_null = float(
                    _aggregate_paths(path_descriptors, null_values)[selected_batch_row]
                )
                key = (record.request_id, str(candidate["item_id"]))
                identity_delta = max(
                    abs(observed_full - baseline_full[key]),
                    abs(observed_null - baseline_null[key]),
                )
                max_identity = max(max_identity, identity_delta)
                if identity_delta > 1.0e-5:
                    raise ValueError(f"attention observation no-op score identity failed: {identity_delta}")
                row = {
                    "row_index": row_index,
                    "request_id": record.request_id,
                    "candidate_item_id": str(candidate["item_id"]),
                    "candidate_ordinal": int(sample["candidate_ordinal"]),
                    "selection_sha256": str(sample["selection_sha256"]),
                    "baseline_full_score": observed_full,
                    "baseline_null_score": observed_null,
                    "maximum_score_identity_delta": identity_delta,
                    "paths": path_rows,
                }
                _append_sync(run_dir / "observations.partial.jsonl", row)
                progress.update(
                    {
                        "completed_rows": row_index + 1,
                        "last_selection_sha256": str(sample["selection_sha256"]),
                        "partial_sha256": sha256_file(run_dir / "observations.partial.jsonl"),
                        "status": "running",
                    }
                )
                _write_json(run_dir / "progress.json", progress)
    except Exception as exc:
        metadata.update(
            {
                "status": "mechanical_failure",
                "resumable": True,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "maximum_score_identity_delta": max_identity,
                "maximum_manual_attention_error": max_manual,
                "maximum_manual_attention_low_precision_ratio": max_manual_ratio,
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        raise
    metadata.update(
        {
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started,
            "maximum_score_identity_delta": max_identity,
            "maximum_manual_attention_error": max_manual,
            "maximum_manual_attention_low_precision_ratio": max_manual_ratio,
        }
    )
    return _finalize(run_dir, metadata, progress, target_samples)


def _build_observation_paths(tokenizer, record, candidate_ordinal, config, *, device):
    import torch

    method_id = str(config["method_id"])
    candidate_ordinal = int(candidate_ordinal)
    if not 0 <= candidate_ordinal < len(record.candidates):
        raise ValueError("attention observation candidate ordinal is outside slate")
    batch_size = int(config.get("scoring", {}).get("batch_size", 8))
    chunks = complete_candidate_chunks(record.candidates, batch_size)
    chunk_index = candidate_ordinal // batch_size
    candidates = list(chunks[chunk_index])
    selected_batch_row = candidate_ordinal - chunk_index * batch_size
    if str(candidates[selected_batch_row]["item_id"]) != str(
        record.candidates[candidate_ordinal]["item_id"]
    ):
        raise ValueError("attention observation frozen candidate chunk drifted")
    if method_id == "q3_tallrec_generalqwen":
        targets = [
            ("yes", _answer_target_tokens(tokenizer, "Yes"), 1.0),
            ("no", _answer_target_tokens(tokenizer, "No"), -1.0),
        ]
        reserve = max(len(target) for _, target, _ in targets)
    else:
        targets = [("prompt", [], 1.0)]
        reserve = 0
    result = []
    prompts = {}
    for condition, history in (("full", record.history), ("null", [])):
        condition_prompts = [
            instrument_pointwise_prompt(
                tokenizer,
                method_id,
                record,
                candidate,
                history=history,
                history_budget=int(config["training"]["history_budget"]),
                max_length=int(config["training"]["max_length"]) - reserve,
            )
            for candidate in candidates
        ]
        condition_spans = [
            _semantic_spans(
                tokenizer,
                method_id,
                record,
                candidate,
                history,
                prompt,
                config,
            )
            for candidate, prompt in zip(candidates, condition_prompts)
        ]
        prompts[condition] = (condition_prompts, condition_spans)
    for name, target, weight in targets:
        pair = {}
        for condition in ("full", "null"):
            condition_prompts, condition_spans = prompts[condition]
            sequences = [
                list(prompt.token_ids) + list(target)
                for prompt in condition_prompts
            ]
            ids, mask, padding = _left_pad_sequences(
                sequences,
                tokenizer.pad_token_id,
                device,
            )
            native = [
                [
                    left + prompt.candidate_readout + offset
                    for offset in range(max(1, len(target)))
                ]
                for left, prompt in zip(padding, condition_prompts)
            ]
            capture = [
                [
                    left + prompt.query_end,
                    left + prompt.history_summary_end,
                    *native_rows,
                ]
                for left, prompt, native_rows in zip(
                    padding,
                    condition_prompts,
                    native,
                )
            ]
            pair[condition] = {
                "name": name,
                "weight": weight,
                "target": list(target),
                "ids": ids,
                "mask": mask,
                "left_padding": torch.tensor(
                    padding,
                    dtype=torch.long,
                    device=device,
                ),
                "positions": torch.tensor(native, dtype=torch.long, device=device),
                "capture_positions": torch.tensor(
                    capture,
                    dtype=torch.long,
                    device=device,
                ),
                "query_positions": {
                    "history_summary": torch.tensor(
                        [
                            [left + prompt.history_summary_end]
                            for left, prompt in zip(padding, condition_prompts)
                        ],
                        dtype=torch.long,
                        device=device,
                    ),
                    "native_readout": torch.tensor(
                        native,
                        dtype=torch.long,
                        device=device,
                    ),
                },
                "spans": {
                    span: (
                        torch.tensor(
                            [
                                left + bounds[span][0]
                                for left, bounds in zip(padding, condition_spans)
                            ],
                            dtype=torch.long,
                            device=device,
                        ),
                        torch.tensor(
                            [
                                left + bounds[span][1]
                                for left, bounds in zip(padding, condition_spans)
                            ],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                    for span in ("query", "history", "candidate")
                },
            }
        result.append(
            {
                "name": name,
                "selected_batch_row": selected_batch_row,
                **pair,
            }
        )
    return result


def _semantic_spans(tokenizer, method_id, record, candidate, history, prompt, config):
    sections = build_prompt_sections(
        method_id, record, dict(candidate), history=list(history),
        history_budget=int(config["training"]["history_budget"]),
    )
    prefix_ids = _encode(tokenizer, PREFIX_TEMPLATE.format(system=sections.system))
    _context_ids, offsets = _encode_with_offsets(tokenizer, sections.context)
    query_chars, history_chars = _context_spans(
        method_id, sections.context, query=record.query,
        history_text=serialize_history(history, history_budget=int(config["training"]["history_budget"])),
    )
    query_tokens = covering_token_span(offsets, *query_chars)
    history_tokens = covering_token_span(offsets, *history_chars)
    spans = {
        "query": (len(prefix_ids) + query_tokens[0], len(prefix_ids) + query_tokens[1]),
        "history": (len(prefix_ids) + history_tokens[0], len(prefix_ids) + history_tokens[1]),
        "candidate": (prompt.candidate_start, prompt.candidate_start + prompt.candidate_tokens),
    }
    if any(not 0 <= start < end <= len(prompt.token_ids) for start, end in spans.values()):
        raise ValueError("attention observation semantic span is outside prompt")
    return spans


def covering_token_span(offsets, char_start: int, char_end: int) -> tuple[int, int]:
    """Return the minimal contiguous token rows overlapping one exact char span."""

    char_start, char_end = int(char_start), int(char_end)
    if not 0 <= char_start < char_end:
        raise ValueError("semantic character span is empty or reversed")
    indices = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > char_start and left < char_end and right > left
    ]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("semantic character span is not covered contiguously")
    covered_start = min(int(offsets[index][0]) for index in indices)
    covered_end = max(int(offsets[index][1]) for index in indices)
    if covered_start > char_start or covered_end < char_end:
        raise ValueError("semantic token rows do not cover the exact character span")
    return indices[0], indices[-1] + 1


def _observe_forward(model, observer, path):
    observer.arm(
        path["capture_positions"], path["query_positions"], path["spans"],
        sequence_length=path["ids"].shape[1],
    )
    output = model(
        input_ids=path["ids"], attention_mask=path["mask"], use_cache=False,
        logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
    )
    return output, observer.disarm()


def _summarize_path(full, null, *, selected_batch_row=0):
    import torch

    selected_batch_row = int(selected_batch_row)
    observations = {}
    for scope, spans in full["observations"]["scopes"].items():
        observations[scope] = {}
        for span, metrics in spans.items():
            observations[scope][span] = {}
            for metric, tensor in metrics.items():
                values = tensor.detach().float().cpu()[selected_batch_row]
                head = values.mean(dim=0)
                gqa = head.reshape(8, 2).mean(dim=1)
                observations[scope][span][metric] = {
                    "query_head": head.tolist(),
                    "gqa_group": gqa.tolist(),
                }
    geometry = {}
    for kind in ("q", "k"):
        geometry[kind] = {}
        for stage in ("pre_norm", "post_norm", "post_rope"):
            full_tensor = full["captures"][f"{kind}_{stage}"].detach().float().cpu()[
                selected_batch_row
            ]
            null_tensor = null["captures"][f"{kind}_{stage}"].detach().float().cpu()[
                selected_batch_row
            ]
            if full_tensor.shape != null_tensor.shape:
                raise ValueError("attention full/null QK capture shape differs")
            full_norm = full_tensor.norm(dim=-1)
            null_norm = null_tensor.norm(dim=-1)
            delta_norm = (full_tensor - null_tensor).norm(dim=-1)
            denominator = full_norm * null_norm
            cosine = torch.where(
                denominator > 0,
                (full_tensor * null_tensor).sum(dim=-1) / denominator,
                torch.zeros_like(denominator),
            ).clamp(-1, 1)
            geometry[kind][stage] = {
                "full_norm": full_norm.tolist(),
                "null_norm": null_norm.tolist(),
                "full_null_delta_norm": delta_norm.tolist(),
                "full_null_cosine": cosine.tolist(),
            }
    return {
        "observations": observations,
        "qk_geometry": geometry,
        "capture_position_order": ["query_end", "history_summary_end", "native_readout..."],
        "manual_attention_error": max(
            float(
                full["observations"][
                    "manual_selected_row_native_max_abs_error"
                ]
            ),
            float(
                null["observations"][
                    "manual_selected_row_native_max_abs_error"
                ]
            ),
        ),
        "manual_attention_low_precision_ratio": max(
            float(
                full["observations"][
                    "manual_selected_row_native_low_precision_ratio"
                ]
            ),
            float(
                null["observations"][
                    "manual_selected_row_native_low_precision_ratio"
                ]
            ),
        ),
    }


def _audit_sample(samples, records_by_id):
    if len(samples) != 512 or len({row["selection_sha256"] for row in samples}) != 512:
        raise ValueError("attention observation sample coverage differs")
    for row in samples:
        record = records_by_id.get(str(row.get("request_id") or ""))
        ordinal = int(row.get("candidate_ordinal", -1))
        if record is None or not 0 <= ordinal < len(record.candidates):
            raise ValueError("attention observation sample identity differs")
        if str(record.candidates[ordinal]["item_id"]) != str(row.get("candidate_item_id")):
            raise ValueError("attention observation sample candidate differs")


def attention_observation_implementation_identity():
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/attention_observation_runtime.py",
        "src/myrec/mechanism/attention_head_observation.py",
        "src/myrec/mechanism/attention_edge_interventions.py",
        "src/myrec/mechanism/representation_probe.py",
        "scripts/observe_deep_dive_attention_heads.py",
    )
    files = [{"path": path, "sha256": sha256_file(root / path), "size_bytes": (root / path).stat().st_size} for path in paths]
    return {"files": files, "digest": _canonical_sha256(files)}


def _prepare(run_dir, metadata, contract_sha256, samples, *, resume):
    partial = run_dir / "observations.partial.jsonl"
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"attention observation run is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial.touch(exist_ok=False)
        metadata.update({"elapsed_seconds": 0.0, "resumable": True, "resume_lineage": []})
        progress = {"schema_version": 1, "run_contract_sha256": contract_sha256, "completed_rows": 0, "last_selection_sha256": None, "partial_sha256": sha256_file(partial), "status": "initializing"}
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "progress.json", progress)
        return metadata, progress
    stored = _read_json(run_dir / "metadata.json")
    progress = _read_json(run_dir / "progress.json")
    if stored.get("run_contract_sha256") != contract_sha256 or progress.get("run_contract_sha256") != contract_sha256:
        raise ValueError("attention observation resume contract drift")
    observed = _audit_partial(partial, samples)
    if progress.get("completed_rows") != observed["completed_rows"] or progress.get("partial_sha256") != observed["partial_sha256"]:
        raise ValueError("attention observation progress differs")
    lineage = list(stored.get("resume_lineage", []))
    lineage.append({"completed_rows": observed["completed_rows"], "partial_sha256": observed["partial_sha256"]})
    stored["resume_lineage"] = lineage
    _write_json(run_dir / "metadata.json", stored)
    return stored, progress


def _audit_partial(path, samples):
    count = 0
    for row in iter_jsonl(path):
        if count >= len(samples) or row.get("row_index") != count or row.get("selection_sha256") != samples[count]["selection_sha256"]:
            raise ValueError("attention observation partial identity differs")
        count += 1
    return {"completed_rows": count, "partial_sha256": sha256_file(path)}


def _finalize(run_dir, metadata, progress, samples):
    observed = _audit_partial(run_dir / "observations.partial.jsonl", samples)
    if observed["completed_rows"] != len(samples):
        raise ValueError("cannot finalize incomplete attention observations")
    path = run_dir / "observations.jsonl"
    os.replace(run_dir / "observations.partial.jsonl", path)
    progress["status"] = "completed"
    metadata.update({"status": "completed", "resumable": False, "observation_rows": len(samples), "observations_path": str(path), "observations_sha256": sha256_file(path), "complete_finite_observation_coverage": True})
    _write_json(run_dir / "progress.json", progress)
    _write_json(run_dir / "metadata.json", metadata)
    return metadata


def _append_sync(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
