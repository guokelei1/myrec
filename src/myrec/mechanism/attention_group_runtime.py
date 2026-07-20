"""Resumable D3 frozen-sample GQA causal localization runtime."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
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
from myrec.mechanism.attention_group_scoring import (
    GROUP_CONDITIONS,
    SUPPLEMENTAL_CONDITIONS,
    score_attention_group_sample_row,
)
from myrec.mechanism.attention_observation_runtime import (
    SAMPLE_MANIFEST,
    SAMPLE_MANIFEST_SHA256,
    _audit_sample,
)
from myrec.mechanism.patch_scorer import _cross_request_mapping
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


SUPPORTED_METHODS = ("q2_recranker_generalqwen", "q3_tallrec_generalqwen")
FIXED_BLOCKS = (13, 20, 27)
CONTENT_MANIFEST = Path(
    "artifacts/motivation_transformer_deep_dive/frozen_controls/"
    "content_neutral_v1/manifest.json"
)


def write_attention_group_bundle(
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
    """Localize all eight fixed GQA groups without reading qrels."""

    _validate_run_id(run_id)
    block = int(block)
    if block not in FIXED_BLOCKS:
        raise ValueError("attention GQA block must be 13,20,or27")
    if not str(device).strip():
        raise ValueError("an explicit attention GQA device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("attention GQA wall time must be in (0,13500]")
    if max_rows is not None and not 0 < int(max_rows) < 512:
        raise ValueError("attention GQA smoke rows must be in [1,511]")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    records_path = standardized_dir / "records_dev.jsonl"
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000 or sha256_file(records_path) != manifest["frozen_inputs"][
        "records_dev_sha256"
    ]:
        raise ValueError("attention GQA frozen records differ")
    records_by_id = {record.request_id: record for record in records}
    cross_mapping = _cross_request_mapping(records)

    sample_manifest_path = Path(sample_manifest_path)
    if sha256_file(sample_manifest_path) != SAMPLE_MANIFEST_SHA256:
        raise ValueError("attention GQA sample manifest differs")
    sample_manifest = _read_json(sample_manifest_path)
    sample_path = Path(sample_manifest["path"])
    if (
        sample_manifest.get("selected_candidate_rows") != 512
        or sample_manifest.get("sha256") != sha256_file(sample_path)
        or sample_manifest.get("qrels_read") is not False
        or sample_manifest.get("model_scores_read") is not False
    ):
        raise ValueError("attention GQA sample is not frozen qrels-blind")
    samples = list(iter_jsonl(sample_path))
    _audit_sample(samples, records_by_id)
    target_samples = samples if max_rows is None else samples[: int(max_rows)]

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    native_scoring_batch_size = int(config.get("scoring", {}).get("batch_size", 8))
    if native_scoring_batch_size <= 0:
        raise ValueError("attention GQA scoring batch size must be positive")
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("attention GQA supports only Q2/Q3")
    frozen = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("attention GQA config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(model_dir, method_id)
    if checkpoint_id != frozen["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("attention GQA checkpoint differs from frozen binding")
    baseline_full, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    content_manifest = _read_json(CONTENT_MANIFEST)
    registered_content = manifest["frozen_qrels_blind_controls"]["content_neutral"]
    if (
        sha256_file(CONTENT_MANIFEST) != registered_content["manifest_sha256"]
        or content_manifest.get("qrels_read") is not False
        or content_manifest.get("model_scores_read") is not False
    ):
        raise ValueError("attention GQA content-neutral manifest differs")
    content_identity = content_manifest["methods"][method_id]
    content_path = Path(content_identity["path"])
    if content_identity["sha256"] != sha256_file(content_path):
        raise ValueError("attention GQA content-neutral rows differ")
    content_controls = {
        str(row["request_id"]): row for row in iter_jsonl(content_path)
    }
    if len(content_controls) != len(records):
        raise ValueError("attention GQA content-neutral coverage differs")

    implementation = attention_group_implementation_identity()
    evidence_mode = (
        "exploratory_localization"
        if max_rows is None
        else "mechanical_smoke_non_result"
    )
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "sample_rows_sha256": sha256_file(sample_path),
        "content_manifest_sha256": sha256_file(CONTENT_MANIFEST),
        "content_rows_sha256": sha256_file(content_path),
        "cross_mapping_sha256": _canonical_sha256(cross_mapping),
        "target_rows": len(target_samples),
        "native_scoring_batch_size": native_scoring_batch_size,
        "selected_row_batch_policy": "original_request_candidate_chunk",
        "groups": 8,
        "group_conditions": list(GROUP_CONDITIONS),
        "supplemental_conditions": list(SUPPLEMENTAL_CONDITIONS),
        "implementation_digest": implementation["digest"],
        "device": str(device),
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d3_attention_gqa_causal_localization",
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
        "content_neutral_manifest_sha256": sha256_file(CONTENT_MANIFEST),
        "content_neutral_rows_sha256": sha256_file(content_path),
        "cross_request_mapping_sha256": _canonical_sha256(cross_mapping),
        "frozen_full_baseline": full_identity,
        "query_heads": 16,
        "kv_heads": 8,
        "gqa_groups": 8,
        "query_heads_per_group": 2,
        "native_scoring_batch_size": native_scoring_batch_size,
        "selected_row_batch_policy": "original_request_candidate_chunk",
        "group_conditions": list(GROUP_CONDITIONS),
        "supplemental_conditions": list(SUPPLEMENTAL_CONDITIONS),
        "selection": "all eight fixed GQA groups; no outcome selection",
        "formation_edge": "query_span_to_history_summary_query",
        "transport_edge": "history_span_to_all_native_readout_queries",
        "cross_stress": "one history_summary_end post-RoPE K/V token",
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
        maximum_identity = float(metadata.get("maximum_identity_delta", 0.0))
        maximum_baseline = float(metadata.get("maximum_baseline_delta", 0.0))
        with torch.inference_mode():
            for row_index in range(completed, len(target_samples)):
                if time.monotonic() - started >= max_wall_seconds:
                    _progress(
                        metadata,
                        started,
                        maximum_identity,
                        maximum_baseline,
                        "wall_time_exhausted",
                    )
                    _write_json(run_dir / "metadata.json", metadata)
                    return metadata
                sample = target_samples[row_index]
                record = records_by_id[str(sample["request_id"])]
                candidate_ordinal = int(sample["candidate_ordinal"])
                candidate = record.candidates[candidate_ordinal]
                donor_record = records_by_id[cross_mapping[record.request_id]]
                donor_candidate_ordinal = candidate_ordinal % len(
                    donor_record.candidates
                )
                donor_candidate = donor_record.candidates[donor_candidate_ordinal]
                content_control = content_controls[record.request_id]
                if (
                    int(content_control.get("candidate_count", -1))
                    != len(record.candidates)
                    or content_control.get("candidate_ids_sha256")
                    != _canonical_sha256(
                        [str(row["item_id"]) for row in record.candidates]
                    )
                ):
                    raise ValueError(
                        "attention GQA content-neutral candidate binding differs"
                    )
                result = score_attention_group_sample_row(
                    model,
                    tokenizer,
                    record,
                    candidate_ordinal,
                    donor_record,
                    donor_candidate_ordinal,
                    content_control,
                    config,
                    block=block,
                    device=str(device),
                )
                maximum_identity = max(
                    maximum_identity, float(result["maximum_identity_delta"])
                )
                key = (record.request_id, str(candidate["item_id"]))
                baseline_delta = abs(
                    float(result["supplemental"]["baseline_full"])
                    - float(baseline_full[key])
                )
                maximum_baseline = max(maximum_baseline, baseline_delta)
                if baseline_delta > 1.0e-5:
                    raise ValueError(
                        f"attention GQA native baseline identity failed: {baseline_delta}"
                    )
                row = {
                    "row_index": row_index,
                    "selection_sha256": sample["selection_sha256"],
                    "request_id": record.request_id,
                    "candidate_item_id": str(candidate["item_id"]),
                    "candidate_ordinal": int(sample["candidate_ordinal"]),
                    "donor_request_id": donor_record.request_id,
                    "donor_candidate_item_id": str(donor_candidate["item_id"]),
                    "result": result,
                }
                _append_sync(run_dir / "groups.partial.jsonl", row)
                progress.update(
                    {
                        "completed_rows": row_index + 1,
                        "last_selection_sha256": sample["selection_sha256"],
                        "partial_sha256": sha256_file(
                            run_dir / "groups.partial.jsonl"
                        ),
                        "status": "running",
                    }
                )
                _write_json(run_dir / "progress.json", progress)
    except Exception as exc:
        _progress(
            metadata,
            started,
            maximum_identity,
            maximum_baseline,
            "mechanical_failure",
        )
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(run_dir / "metadata.json", metadata)
        raise
    _progress(
        metadata, started, maximum_identity, maximum_baseline, "running"
    )
    return _finalize(run_dir, metadata, progress, target_samples)


def _prepare(run_dir, metadata, contract_sha256, samples, *, resume):
    partial = run_dir / "groups.partial.jsonl"
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"attention GQA run is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial.touch(exist_ok=False)
        metadata.update(
            {"elapsed_seconds": 0.0, "resumable": True, "resume_lineage": []}
        )
        progress = {
            "schema_version": 1,
            "run_contract_sha256": contract_sha256,
            "completed_rows": 0,
            "last_selection_sha256": None,
            "partial_sha256": sha256_file(partial),
            "status": "initializing",
        }
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "progress.json", progress)
        return metadata, progress
    stored = _read_json(run_dir / "metadata.json")
    progress = _read_json(run_dir / "progress.json")
    if stored.get("run_contract_sha256") != contract_sha256 or progress.get(
        "run_contract_sha256"
    ) != contract_sha256:
        raise ValueError("attention GQA resume contract drift")
    observed = _audit_partial(partial, samples)
    if progress.get("completed_rows") != observed["completed_rows"] or progress.get(
        "partial_sha256"
    ) != observed["partial_sha256"]:
        raise ValueError("attention GQA progress differs")
    lineage = list(stored.get("resume_lineage", []))
    lineage.append(
        {
            "completed_rows": observed["completed_rows"],
            "partial_sha256": observed["partial_sha256"],
        }
    )
    stored["resume_lineage"] = lineage
    _write_json(run_dir / "metadata.json", stored)
    return stored, progress


def _audit_partial(path, samples):
    count = 0
    for row in iter_jsonl(path):
        if (
            count >= len(samples)
            or row.get("row_index") != count
            or row.get("selection_sha256") != samples[count]["selection_sha256"]
            or len(row.get("result", {}).get("groups", [])) != 8
            or set(row.get("result", {}).get("supplemental", {}))
            != set(SUPPLEMENTAL_CONDITIONS)
        ):
            raise ValueError("attention GQA partial identity/coverage differs")
        count += 1
    return {"completed_rows": count, "partial_sha256": sha256_file(path)}


def _finalize(run_dir, metadata, progress, samples):
    observed = _audit_partial(run_dir / "groups.partial.jsonl", samples)
    if observed["completed_rows"] != len(samples):
        raise ValueError("cannot finalize incomplete attention GQA rows")
    if float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("attention GQA score identity gate failed")
    if float(metadata.get("maximum_baseline_delta", math.inf)) > 1.0e-5:
        raise ValueError("attention GQA baseline identity gate failed")
    path = run_dir / "groups.jsonl"
    os.replace(run_dir / "groups.partial.jsonl", path)
    progress["status"] = "completed"
    metadata.update(
        {
            "status": "completed",
            "resumable": False,
            "row_count": len(samples),
            "groups_path": str(path),
            "groups_sha256": sha256_file(path),
            "complete_finite_group_coverage": True,
            "identity_passed": True,
        }
    )
    _write_json(run_dir / "progress.json", progress)
    _write_json(run_dir / "metadata.json", metadata)
    return metadata


def _progress(metadata, started, maximum_identity, maximum_baseline, status):
    metadata.update(
        {
            "status": status,
            "resumable": status != "running",
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + time.monotonic()
            - started,
            "maximum_identity_delta": maximum_identity,
            "maximum_baseline_delta": maximum_baseline,
        }
    )


def _append_sync(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def attention_group_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/attention_group_runtime.py",
        "src/myrec/mechanism/attention_group_scoring.py",
        "src/myrec/mechanism/attention_group_interventions.py",
        "src/myrec/mechanism/attention_observation_runtime.py",
        "src/myrec/mechanism/attention_edge_interventions.py",
        "src/myrec/mechanism/history_kv_interventions.py",
        "scripts/score_deep_dive_attention_groups.py",
    )
    files = [
        {
            "path": path,
            "sha256": sha256_file(root / path),
            "size_bytes": (root / path).stat().st_size,
        }
        for path in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}
