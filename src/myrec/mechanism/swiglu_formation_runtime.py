"""Resumable qrels-blind runtime for N25 SwiGLU formation bundles."""

from __future__ import annotations

import math
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
    FIXED_BLOCKS,
    MAX_WALL_SECONDS,
    SUPPORTED_METHODS,
    _canonical_sha256,
    _load_content_controls,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.postblock_sweep_runtime import NULL_BASELINE_DIRS
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.mechanism.swiglu_formation_interventions import SWIGLU_OPERATORS
from myrec.mechanism.swiglu_formation_scoring import SWIGLU_CONDITIONS, score_swiglu_formation_chunk
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def write_swiglu_formation_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    operator: str,
    device: str,
    runs_dir: str | Path = "runs",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    if int(block) not in FIXED_BLOCKS:
        raise ValueError("N25 requires fixed block 13/20/27")
    if str(operator) not in (*SWIGLU_OPERATORS, "all"):
        raise ValueError("N25 operator is not registered")
    if not str(device).strip():
        raise ValueError("N25 requires an explicit device")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("N25 max_wall_seconds is out of range")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("N25 max_requests must be positive")
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(DEEP_DIVE_MANIFEST_PATH)
    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, manifest["frozen_inputs"]["records_dev_sha256"]),
        (standardized_dir / "manifest.json", manifest["frozen_inputs"]["dataset_manifest_sha256"]),
        (standardized_dir / "request_manifest.json", manifest["frozen_inputs"]["request_manifest_sha256"]),
        (standardized_dir / "candidate_manifest.json", manifest["frozen_inputs"]["candidate_manifest_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"N25 frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("N25 requires all 8000 dev requests")
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("N25 supports only Q2/Q3")
    binding = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != binding["config_sha256"]:
        raise ValueError("N25 config differs from frozen model binding")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != binding["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("N25 checkpoint differs from frozen model binding")
    controls, control_identity = _load_content_controls(manifest, method_id, records)
    full_scores, full_identity = _load_frozen_baseline(BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records)
    null_scores, null_identity = _load_frozen_baseline(NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records)
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = "registered_mechanism_diagnostic" if max_requests is None else "mechanical_smoke_non_result"
    implementation = swiglu_formation_runtime_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "analysis_stage": "n25_swiglu_formation",
        "method_id": method_id,
        "block_zero_based": int(block),
        "operator": str(operator),
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "content_control_sha256": control_identity["rows_sha256"],
        "baseline_full_sha256": full_identity["scores_sha256"],
        "baseline_null_sha256": null_identity["scores_sha256"],
        "score_conditions": list(SWIGLU_CONDITIONS),
        "target_requests": len(target_records),
        "evidence_mode": evidence_mode,
        "device": str(device),
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "n25_swiglu_formation",
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": int(block),
        "operator": str(operator),
        "score_conditions": list(SWIGLU_CONDITIONS),
        "identity_conditions": [
            f"{path}_{name}_identity"
            for path in ("full", "null")
            for name in SWIGLU_OPERATORS
        ],
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "content_control": control_identity,
        "frozen_baseline_full": full_identity,
        "frozen_baseline_null": null_identity,
        "identity_tolerance": 1.0e-5,
        "implementation_identity": implementation,
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage_required": True,
        "result_eligible": max_requests is None,
        "evidence_mode": evidence_mode,
        "run_contract": contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    prepared = prepare_scalar_bundle(
        run_dir,
        metadata=metadata,
        contract_sha256=contract_sha256,
        records=target_records,
        conditions=SWIGLU_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    baseline_delta = float(prepared.metadata.get("maximum_frozen_baseline_delta", 0.0))
    if completed >= len(target_records):
        return finalize_scalar_bundle(run_dir, prepared, target_records, SWIGLU_CONDITIONS, maximum_identity_delta=identity_delta)
    started = time.monotonic()
    try:
        import torch
        import transformers
        tokenizer, model = _load_model_and_tokenizer(config, device=str(device), training=False, checkpoint_model_dir=checkpoint_model_dir)
        model.eval()
        prepared.metadata.update(_runtime_metadata(method_id, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    prepared.metadata.update({"status": "wall_time_exhausted", "resumable": True, "maximum_identity_delta": identity_delta, "maximum_frozen_baseline_delta": baseline_delta})
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                control = controls[record.request_id]
                rows = []
                if control.get("eligible") is True:
                    for start in range(0, len(record.candidates), batch_size):
                        candidates = list(record.candidates[start : start + batch_size])
                        result = score_swiglu_formation_chunk(model, tokenizer, record, candidates, config, block=int(block), device=str(device))
                        identity_delta = max(identity_delta, float(result["maximum_identity_delta"]))
                        for local, candidate in enumerate(candidates):
                            key = (record.request_id, str(candidate["item_id"]))
                            values = {name: float(result["conditions"][name][local]) for name in SWIGLU_CONDITIONS}
                            baseline_delta = max(baseline_delta, abs(values["baseline_full"] - full_scores[key]), abs(values["baseline_null"] - null_scores[key]))
                            rows.append({"request_id": record.request_id, "candidate_item_id": str(candidate["item_id"]), "candidate_ordinal": start + local, "conditions": values})
                else:
                    for candidate_ordinal, candidate in enumerate(record.candidates):
                        key = (record.request_id, str(candidate["item_id"]))
                        full, null = full_scores[key], null_scores[key]
                        rows.append({"request_id": record.request_id, "candidate_item_id": str(candidate["item_id"]), "candidate_ordinal": candidate_ordinal, "conditions": {name: (null if name == "baseline_null" or name.startswith("null_") else full) for name in SWIGLU_CONDITIONS}})
                append_scalar_request(run_dir, {"ordinal": ordinal, "request_id": record.request_id, "block_zero_based": int(block), "operator": str(operator), "content_control_eligible": bool(control.get("eligible") is True), "rows": rows, "rows_sha256": _canonical_sha256(rows)}, prepared)
    except Exception as exc:
        prepared.metadata.update({"status": "mechanical_failure", "resumable": True, "qrels_read": False, "error": {"type": type(exc).__name__, "message": str(exc)}, "maximum_identity_delta": identity_delta, "maximum_frozen_baseline_delta": baseline_delta})
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    prepared.metadata.update({"elapsed_seconds": float(prepared.metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started, "maximum_identity_delta": identity_delta, "maximum_frozen_baseline_delta": baseline_delta})
    return finalize_scalar_bundle(run_dir, prepared, target_records, SWIGLU_CONDITIONS, maximum_identity_delta=identity_delta)


def swiglu_formation_runtime_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "src/myrec/mechanism/swiglu_formation_interventions.py",
        "src/myrec/mechanism/swiglu_formation_scoring.py",
        "src/myrec/mechanism/swiglu_formation_runtime.py",
        "src/myrec/mechanism/mlp_group_scoring.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "scripts/score_deep_dive_swiglu_formation.py",
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"N25 implementation file is missing: {path}")
        files.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"files": files, "digest": _canonical_sha256(files)}
