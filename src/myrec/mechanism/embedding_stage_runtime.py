"""Resumable qrels-blind N14 history-embedding-stage bundles."""

from __future__ import annotations

import json
import math
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
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.mechanism.attention_edge_runtime import (
    BASELINE_SCORE_DIRS,
    DEEP_DIVE_MANIFEST_PATH,
    MAX_WALL_SECONDS,
    SUPPORTED_METHODS,
    _assert_native_targets,
    _canonical_sha256,
    _load_frozen_baseline,
    _read_json,
    _write_json,
)
from myrec.mechanism.attention_logit_runtime import _load_content_controls_from_n11
from myrec.mechanism.embedding_stage_scoring import (
    EMBEDDING_STAGE_CONDITIONS,
    score_embedding_stage_chunk,
)
from myrec.mechanism.postblock_sweep_runtime import NULL_BASELINE_DIRS
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


N14_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_n14_embedding_stage_manifest_v1.yaml"
)
N14_MANIFEST_SHA256 = "c24c92febef125adef09cdfacab3918cf4f4bd3b2aacd45b4e9e33f70de7cc02"


def write_embedding_stage_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = N14_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("N14 max_requests must be positive")
    if not str(device).strip():
        raise ValueError("N14 requires an explicit device")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("N14 max_wall_seconds must be in (0,13500]")
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_n14_manifest(manifest_path)
    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, manifest["frozen_inputs"]["records_sha256"]),
        (standardized_dir / "manifest.json", manifest["frozen_inputs"]["dataset_manifest_sha256"]),
        (standardized_dir / "request_manifest.json", manifest["frozen_inputs"]["request_manifest_sha256"]),
        (standardized_dir / "candidate_manifest.json", manifest["frozen_inputs"]["candidate_manifest_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"N14 frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("N14 requires all 8000 internal-dev requests")
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("N14 supports only Q2/Q3")
    binding = _load_n14_model_binding(manifest, method_id)
    if config["_config_sha256"] != binding["config_sha256"]:
        raise ValueError("N14 config differs from frozen binding")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != binding["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("N14 checkpoint differs from frozen binding")
    controls, control_identity = _load_content_controls_from_n11(manifest, method_id, records)
    baseline_full, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = "registered_mechanism_diagnostic" if max_requests is None else "mechanical_smoke_non_result"
    implementation = embedding_stage_runtime_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "n14_manifest_sha256": manifest["_sha256"],
        "content_control_sha256": control_identity["rows_sha256"],
        "baseline_full_sha256": full_identity["scores_sha256"],
        "baseline_null_sha256": null_identity["scores_sha256"],
        "score_conditions": list(EMBEDDING_STAGE_CONDITIONS),
        "target_requests": len(target_records),
        "evidence_mode": evidence_mode,
        "device": str(device),
        "implementation_digest": implementation["digest"],
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_n14_embedding_stage_operator",
        "run_id": run_id,
        "method_id": method_id,
        "score_conditions": list(EMBEDDING_STAGE_CONDITIONS),
        "active_conditions": [name for name in EMBEDDING_STAGE_CONDITIONS if name not in {"baseline_full", "baseline_null", "full_embedding_identity", "null_embedding_identity"}],
        "identity_conditions": ["full_embedding_identity", "null_embedding_identity"],
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
        "n14_manifest_path": str(manifest_path),
        "n14_manifest_sha256": manifest["_sha256"],
        "content_control": control_identity,
        "frozen_baseline_full": full_identity,
        "frozen_baseline_null": null_identity,
        "implementation_identity": implementation,
        "identity_tolerance": 1.0e-5,
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage_required": True,
        "ineligible_scoring": "copy_frozen_baseline_pair",
        "evidence_mode": evidence_mode,
        "result_eligible": max_requests is None,
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
        conditions=EMBEDDING_STAGE_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    maximum_identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    maximum_baseline_delta = float(prepared.metadata.get("maximum_frozen_baseline_delta", 0.0))
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir, prepared, target_records, EMBEDDING_STAGE_CONDITIONS,
            maximum_identity_delta=maximum_identity_delta,
        )
    started = time.monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(
            config, device=str(device), training=False, checkpoint_model_dir=checkpoint_model_dir
        )
        model.eval()
        _assert_native_targets(tokenizer, method_id, _load_parent_manifest())
        prepared.metadata.update(_runtime_metadata(method_id, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    prepared.metadata.update({
                        "status": "wall_time_exhausted",
                        "resumable": True,
                        "elapsed_seconds": float(prepared.metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started,
                        "maximum_identity_delta": maximum_identity_delta,
                        "maximum_frozen_baseline_delta": maximum_baseline_delta,
                    })
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                control = controls[record.request_id]
                rows: list[dict[str, Any]] = []
                if control["eligible"] is True:
                    for start in range(0, len(record.candidates), batch_size):
                        candidates = list(record.candidates[start : start + batch_size])
                        result = score_embedding_stage_chunk(
                            model, tokenizer, record, candidates, control, config,
                            device=str(device)
                        )
                        maximum_identity_delta = max(maximum_identity_delta, float(result["maximum_identity_delta"]))
                        for local, candidate in enumerate(candidates):
                            values = {name: float(result["conditions"][name][local]) for name in EMBEDDING_STAGE_CONDITIONS}
                            if not all(math.isfinite(value) for value in values.values()):
                                raise FloatingPointError("N14 produced a non-finite score")
                            key = (record.request_id, str(candidate["item_id"]))
                            maximum_baseline_delta = max(
                                maximum_baseline_delta,
                                abs(values["baseline_full"] - baseline_full[key]),
                                abs(values["baseline_null"] - baseline_null[key]),
                            )
                            rows.append({
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "candidate_ordinal": start + local,
                                "conditions": values,
                            })
                else:
                    for candidate_ordinal, candidate in enumerate(record.candidates):
                        key = (record.request_id, str(candidate["item_id"]))
                        full = baseline_full[key]
                        null = baseline_null[key]
                        values = {
                            name: (full if name == "baseline_full" or name.startswith("full_") else null)
                            for name in EMBEDDING_STAGE_CONDITIONS
                        }
                        rows.append({
                            "request_id": record.request_id,
                            "candidate_item_id": str(candidate["item_id"]),
                            "candidate_ordinal": candidate_ordinal,
                            "conditions": values,
                        })
                append_scalar_request(
                    run_dir,
                    {
                        "ordinal": ordinal,
                        "request_id": record.request_id,
                        "content_control_eligible": bool(control["eligible"]),
                        "rows": rows,
                        "rows_sha256": _canonical_sha256(rows),
                    },
                    prepared,
                )
    except Exception as exc:
        prepared.metadata.update({
            "status": "mechanical_failure",
            "resumable": True,
            "qrels_read": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "maximum_identity_delta": maximum_identity_delta,
            "maximum_frozen_baseline_delta": maximum_baseline_delta,
        })
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    prepared.metadata.update({
        "elapsed_seconds": float(prepared.metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started,
        "maximum_identity_delta": maximum_identity_delta,
        "maximum_frozen_baseline_delta": maximum_baseline_delta,
    })
    return finalize_scalar_bundle(
        run_dir, prepared, target_records, EMBEDDING_STAGE_CONDITIONS,
        maximum_identity_delta=maximum_identity_delta,
    )


def embedding_stage_runtime_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "src/myrec/mechanism/embedding_stage_interventions.py",
        "src/myrec/mechanism/embedding_stage_scoring.py",
        "src/myrec/mechanism/embedding_stage_runtime.py",
        "src/myrec/mechanism/attention_logit_scoring.py",
        "src/myrec/mechanism/attention_edge_interventions.py",
        "src/myrec/mechanism/attention_edge_scoring.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "scripts/score_deep_dive_embedding_stage.py",
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"N14 implementation file is missing: {path}")
        files.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"files": files, "digest": _canonical_sha256(files)}


def _load_n14_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    observed = sha256_file(path)
    if observed != N14_MANIFEST_SHA256:
        raise ValueError("N14 manifest differs from immutable digest")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("N14 manifest is not a mapping")
    value["_sha256"] = observed
    return value


def _load_n14_model_binding(manifest: Mapping[str, Any], method_id: str) -> Mapping[str, Any]:
    binding = manifest.get("model_bindings", {}).get(method_id)
    if not isinstance(binding, Mapping):
        raise ValueError(f"N14 model binding missing for {method_id}")
    return binding


def _load_parent_manifest() -> dict[str, Any]:
    value = yaml.safe_load(DEEP_DIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("parent deep-dive manifest is not a mapping")
    return value
