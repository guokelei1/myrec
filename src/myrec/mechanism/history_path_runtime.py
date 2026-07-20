"""Resumable qrels-blind N9 history-path scorer for Q2/Q3."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _checkpoint_identity,
    _git_revision,
    _load_model_and_tokenizer,
    _runtime_metadata,
    _single_token_id,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.mechanism.attention_edge_runtime import (
    BASELINE_SCORE_DIRS,
    DEEP_DIVE_MANIFEST_PATH,
    FIXED_BLOCKS,
    SUPPORTED_METHODS,
    _assert_native_targets,
    _canonical_sha256,
    _load_content_controls,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.postblock_sweep_runtime import NULL_BASELINE_DIRS
from myrec.mechanism.history_path_scoring import N9_SCORE_CONDITIONS, score_history_path_chunk
from myrec.mechanism.representation_probe import normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


N9_MANIFEST_PATH = Path("experiments/motivation/transformer_n9_history_path_manifest_v1.yaml")
N9_MANIFEST_SHA256 = "a35c176b792feabc3b83c1a4bb83fb2fbfafc6966de2774c3f9f52475b74382b"
MAX_WALL_SECONDS = 13_500.0


def write_history_path_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = N9_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    block = int(block)
    if block not in FIXED_BLOCKS:
        raise ValueError("N9 history-path block must be one of 13,20,27")
    if not str(device).strip():
        raise ValueError("N9 history-path scorer requires an explicit device")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("N9 history-path max_wall_seconds must be in (0,13500]")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("N9 history-path max_requests must be positive")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    n9_manifest = _load_n9_manifest(manifest_path)
    deep_manifest = _load_manifest(DEEP_DIVE_MANIFEST_PATH)
    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, deep_manifest["frozen_inputs"]["records_dev_sha256"]),
        (standardized_dir / "manifest.json", deep_manifest["frozen_inputs"]["dataset_manifest_sha256"]),
        (standardized_dir / "request_manifest.json", deep_manifest["frozen_inputs"]["request_manifest_sha256"]),
        (standardized_dir / "candidate_manifest.json", deep_manifest["frozen_inputs"]["candidate_manifest_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"N9 frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("N9 history-path scorer requires all 8000 internal-dev requests")

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("N9 history-path scorer supports only Q2/Q3")
    frozen_model = deep_manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("N9 config differs from frozen deep-dive manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != frozen_model["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("N9 checkpoint differs from frozen binding")

    controls, control_identity = _load_content_controls(deep_manifest, method_id, records)
    baseline_full, baseline_full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    baseline_null, baseline_null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    fold1_records = [record for record in records if normalized_query_fold(record.query) == 1]
    target_records = fold1_records if max_requests is None else fold1_records[: int(max_requests)]
    evidence_mode = "registered_n9_path_diagnostic" if max_requests is None else "mechanical_smoke_non_result"
    implementation = history_path_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "n9_manifest_sha256": n9_manifest["_sha256"],
        "deep_dive_manifest_sha256": deep_manifest["_sha256"],
        "score_conditions": list(N9_SCORE_CONDITIONS),
        "target_requests": len(target_records),
        "evidence_mode": evidence_mode,
        "device": str(device),
        "implementation_digest": implementation["digest"],
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_n9_history_path",
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "normalized_query_fold": 1,
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "n9_manifest_path": str(manifest_path),
        "n9_manifest_sha256": n9_manifest["_sha256"],
        "deep_dive_manifest_path": str(DEEP_DIVE_MANIFEST_PATH),
        "deep_dive_manifest_sha256": deep_manifest["_sha256"],
        "content_neutral_control": control_identity,
        "frozen_full_baseline": baseline_full_identity,
        "frozen_null_baseline": baseline_null_identity,
        "score_conditions": list(N9_SCORE_CONDITIONS),
        "identity_tolerance": 1.0e-5,
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage_required": True,
        "ineligible_scoring": "copy_frozen_baseline_score",
        "evidence_mode": evidence_mode,
        "result_eligible": max_requests is None,
        "run_contract": contract,
        "run_contract_sha256": contract_sha256,
        "implementation_identity": implementation,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    prepared = prepare_scalar_bundle(
        run_dir,
        metadata=metadata,
        contract_sha256=contract_sha256,
        records=target_records,
        conditions=N9_SCORE_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    maximum_identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    maximum_full_baseline_delta = float(prepared.metadata.get("maximum_full_baseline_delta", 0.0))
    maximum_null_baseline_delta = float(prepared.metadata.get("maximum_null_baseline_delta", 0.0))
    eligible_completed = int(prepared.metadata.get("eligible_requests_completed", 0))
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir,
            prepared,
            target_records,
            N9_SCORE_CONDITIONS,
            maximum_identity_delta=maximum_identity_delta,
        )

    started = time.monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(
            config,
            device=str(device),
            training=False,
            checkpoint_model_dir=checkpoint_model_dir,
        )
        model.eval()
        _assert_native_targets(tokenizer, method_id, deep_manifest)
        prepared.metadata.update(_runtime_metadata(method_id, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    prepared.metadata.update(
                        {
                            "status": "wall_time_exhausted",
                            "resumable": True,
                            "completed_requests": ordinal,
                            "elapsed_seconds": float(prepared.metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started,
                            "maximum_identity_delta": maximum_identity_delta,
                            "maximum_full_baseline_delta": maximum_full_baseline_delta,
                            "maximum_null_baseline_delta": maximum_null_baseline_delta,
                            "eligible_requests_completed": eligible_completed,
                        }
                    )
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                control = controls[record.request_id]
                rows: list[dict[str, Any]] = []
                if control["eligible"] is True:
                    eligible_completed += 1
                    for start in range(0, len(record.candidates), batch_size):
                        candidates = list(record.candidates[start : start + batch_size])
                        result = score_history_path_chunk(
                            model,
                            tokenizer,
                            record,
                            candidates,
                            control,
                            config,
                            block=block,
                            device=str(device),
                        )
                        maximum_identity_delta = max(maximum_identity_delta, float(result["maximum_identity_delta"]))
                        for local, candidate in enumerate(candidates):
                            values = {name: float(result["conditions"][name][local]) for name in N9_SCORE_CONDITIONS}
                            if not all(math.isfinite(value) for value in values.values()):
                                raise FloatingPointError("N9 history-path bundle produced a non-finite score")
                            key = (record.request_id, str(candidate["item_id"]))
                            maximum_full_baseline_delta = max(maximum_full_baseline_delta, abs(values["baseline_full"] - baseline_full[key]))
                            maximum_null_baseline_delta = max(maximum_null_baseline_delta, abs(values["baseline_null"] - baseline_null[key]))
                            rows.append(
                                {
                                    "request_id": record.request_id,
                                    "candidate_item_id": str(candidate["item_id"]),
                                    "candidate_ordinal": start + local,
                                    "conditions": values,
                                }
                            )
                else:
                    for candidate_ordinal, candidate in enumerate(record.candidates):
                        full = baseline_full[(record.request_id, str(candidate["item_id"]))]
                        null = baseline_null[(record.request_id, str(candidate["item_id"]))]
                        values = {name: full for name in N9_SCORE_CONDITIONS}
                        values["baseline_null"] = null
                        rows.append(
                            {
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "candidate_ordinal": candidate_ordinal,
                                "conditions": values,
                            }
                        )
                append_scalar_request(
                    run_dir,
                    {
                        "ordinal": ordinal,
                        "request_id": record.request_id,
                        "block_zero_based": block,
                        "content_control_eligible": bool(control["eligible"]),
                        "rows": rows,
                        "rows_sha256": _canonical_sha256(rows),
                    },
                    prepared,
                )
    except Exception as exc:
        prepared.metadata.update(
            {
                "status": "mechanical_failure",
                "resumable": True,
                "qrels_read": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "maximum_identity_delta": maximum_identity_delta,
                "maximum_full_baseline_delta": maximum_full_baseline_delta,
                "maximum_null_baseline_delta": maximum_null_baseline_delta,
                "eligible_requests_completed": eligible_completed,
            }
        )
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    prepared.metadata.update(
        {
            "elapsed_seconds": float(prepared.metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started,
            "maximum_identity_delta": maximum_identity_delta,
            "maximum_full_baseline_delta": maximum_full_baseline_delta,
            "maximum_null_baseline_delta": maximum_null_baseline_delta,
            "eligible_requests_completed": eligible_completed,
        }
    )
    return finalize_scalar_bundle(
        run_dir,
        prepared,
        target_records,
        N9_SCORE_CONDITIONS,
        maximum_identity_delta=maximum_identity_delta,
    )


def _load_n9_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    observed = sha256_file(path)
    if observed != N9_MANIFEST_SHA256:
        raise ValueError("N9 history-path manifest differs from frozen digest")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("N9 history-path manifest is not a mapping")
    value["_sha256"] = observed
    return value


def history_path_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "src/myrec/mechanism/history_path_scoring.py",
        "src/myrec/mechanism/history_path_runtime.py",
        "src/myrec/mechanism/attention_edge_interventions.py",
        "src/myrec/mechanism/attention_edge_scoring.py",
        "src/myrec/mechanism/attention_observation_runtime.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "scripts/score_deep_dive_history_path.py",
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"N9 implementation file is missing: {path}")
        files.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"files": files, "digest": _canonical_sha256(files)}
