"""Resumable qrels-blind runtime for the preregistered N20 Q1 cache boundary."""

from __future__ import annotations

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
    DEEP_DIVE_MANIFEST_PATH,
    MAX_WALL_SECONDS,
    _canonical_sha256,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.q1_cache_phase_scoring import (
    CACHE_PHASE_CONDITIONS,
    score_q1_cache_phase_request,
)
from myrec.mechanism.q1_trajectory_runtime import (
    Q1_FULL_BASELINE_DIR,
    Q1_NULL_BASELINE_DIR,
)
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


N17_N20_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_n17_n20_boundary_manifest_v1.yaml"
)
N17_N20_MANIFEST_SHA256 = (
    "e13177bd48c422002359dd3ba1e98f21c9fe8be7c3c460a12860af0074f01760"
)

N20_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_cache_identity",
    "full_cache_rebuild",
    "full_zero_prefix",
    "full_wrong_user_prefix",
    "full_no_cache_rebuild",
    "null_cache_identity",
    "null_cache_rebuild",
    "null_zero_prefix",
    "null_no_cache_rebuild",
)


def write_q1_cache_phase_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run N20 cache controls while retaining qrels-blind scalar scores."""

    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("N20 requires an explicit device")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("N20 max_wall_seconds is out of range")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("N20 max_requests must be positive")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    boundary_manifest = _load_boundary_manifest()
    parent_manifest = _load_manifest(DEEP_DIVE_MANIFEST_PATH)
    config = load_v12_ranker_config(config_path)
    if config.get("method_id") != "q1_instructrec_generalqwen":
        raise ValueError("N20 cache phase is Q1-only")
    binding = parent_manifest["model_bindings"]["q1_instructrec_generalqwen"]
    if config["_config_sha256"] != binding["config_sha256"]:
        raise ValueError("N20 config differs from frozen Q1 binding")
    records_path = standardized_dir / "records_dev.jsonl"
    expected_hashes = {
        records_path: parent_manifest["frozen_inputs"]["records_dev_sha256"],
        standardized_dir / "manifest.json": parent_manifest["frozen_inputs"]["dataset_manifest_sha256"],
        standardized_dir / "request_manifest.json": parent_manifest["frozen_inputs"]["request_manifest_sha256"],
        standardized_dir / "candidate_manifest.json": parent_manifest["frozen_inputs"]["candidate_manifest_sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"N20 frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("N20 requires all 8000 internal-dev requests")

    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, "q1_instructrec_generalqwen"
    )
    if checkpoint_id != binding["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("N20 checkpoint differs from frozen Q1 binding")
    full_baseline, full_identity = _load_frozen_baseline(
        Q1_FULL_BASELINE_DIR, "q1_instructrec_generalqwen", checkpoint_id, records
    )
    null_baseline, null_identity = _load_frozen_baseline(
        Q1_NULL_BASELINE_DIR, "q1_instructrec_generalqwen", checkpoint_id, records
    )
    wrong_history, wrong_identity = _load_wrong_user_histories(parent_manifest, records)
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if max_requests is None
        else "mechanical_smoke_non_result"
    )
    implementation = q1_cache_phase_runtime_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "analysis_stage": "n20_q1_cache_phase",
        "method_id": "q1_instructrec_generalqwen",
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "boundary_manifest_sha256": boundary_manifest["_sha256"],
        "wrong_user_mapping_sha256": wrong_identity["mapping_sha256"],
        "baseline_full_sha256": full_identity["scores_sha256"],
        "baseline_null_sha256": null_identity["scores_sha256"],
        "score_conditions": list(N20_CONDITIONS),
        "target_requests": len(target_records),
        "evidence_mode": evidence_mode,
        "device": str(device),
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "n20_q1_cache_phase",
        "run_id": run_id,
        "method_id": "q1_instructrec_generalqwen",
        "score_conditions": list(N20_CONDITIONS),
        "identity_conditions": [
            "full_cache_identity",
            "full_cache_rebuild",
            "full_no_cache_rebuild",
            "null_cache_identity",
            "null_cache_rebuild",
            "null_no_cache_rebuild",
        ],
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
        "boundary_manifest_path": str(N17_N20_MANIFEST_PATH),
        "boundary_manifest_sha256": boundary_manifest["_sha256"],
        "wrong_user_control": wrong_identity,
        "frozen_baseline_full": full_identity,
        "frozen_baseline_null": null_identity,
        "identity_tolerance": 1.0e-5,
        "implementation_identity": implementation,
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage_required": True,
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
        conditions=N20_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    maximum_identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    maximum_baseline_delta = float(prepared.metadata.get("maximum_frozen_baseline_delta", 0.0))
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir, prepared, target_records, N20_CONDITIONS,
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
        prepared.metadata.update(_runtime_metadata("q1_instructrec_generalqwen", torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 2))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    prepared.metadata.update({
                        "status": "wall_time_exhausted",
                        "resumable": True,
                        "maximum_identity_delta": maximum_identity_delta,
                        "maximum_frozen_baseline_delta": maximum_baseline_delta,
                    })
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                wrong = wrong_history[record.request_id]
                if wrong["eligible"]:
                    full_result = score_q1_cache_phase_request(
                        model, tokenizer, record, record.history, config,
                        device=str(device), batch_size=batch_size,
                        wrong_history=wrong["history"],
                    )
                else:
                    full_result = None
                null_result = (
                    score_q1_cache_phase_request(
                        model, tokenizer, record, [], config,
                        device=str(device), batch_size=batch_size,
                    )
                    if wrong["eligible"]
                    else None
                )
                rows = []
                for candidate in record.candidates:
                    key = (record.request_id, str(candidate["item_id"]))
                    full_reference = float(full_baseline[key])
                    null_reference = float(null_baseline[key])
                    if full_result is None or null_result is None:
                        values = {
                            "baseline_full": full_reference,
                            "baseline_null": null_reference,
                            "full_cache_identity": full_reference,
                            "full_cache_rebuild": full_reference,
                            "full_zero_prefix": full_reference,
                            "full_wrong_user_prefix": full_reference,
                            "full_no_cache_rebuild": full_reference,
                            "null_cache_identity": null_reference,
                            "null_cache_rebuild": null_reference,
                            "null_zero_prefix": null_reference,
                            "null_no_cache_rebuild": null_reference,
                        }
                    else:
                        local = next(
                            i for i, row in enumerate(record.candidates)
                            if str(row["item_id"]) == str(candidate["item_id"])
                        )
                        values = {
                            "baseline_full": full_reference,
                            "baseline_null": null_reference,
                            "full_cache_identity": float(full_result["conditions"]["native_cache_identity"][local]),
                            "full_cache_rebuild": float(full_result["conditions"]["same_request_rebuild"][local]),
                            "full_zero_prefix": float(full_result["conditions"]["zero_prefix_cache"][local]),
                            "full_wrong_user_prefix": float(full_result["conditions"]["wrong_user_prefix_cache"][local]),
                            "full_no_cache_rebuild": float(full_result["conditions"]["no_cache_rebuild"][local]),
                            "null_cache_identity": float(null_result["conditions"]["native_cache_identity"][local]),
                            "null_cache_rebuild": float(null_result["conditions"]["same_request_rebuild"][local]),
                            "null_zero_prefix": float(null_result["conditions"]["zero_prefix_cache"][local]),
                            "null_no_cache_rebuild": float(null_result["conditions"]["no_cache_rebuild"][local]),
                        }
                    if any(not math.isfinite(float(value)) for value in values.values()):
                        raise FloatingPointError("N20 cache phase produced a non-finite score")
                    maximum_baseline_delta = max(
                        maximum_baseline_delta,
                        abs(values["full_cache_identity"] - full_reference),
                        abs(values["null_cache_identity"] - null_reference),
                    )
                    rows.append({
                        "request_id": record.request_id,
                        "candidate_item_id": str(candidate["item_id"]),
                        "candidate_ordinal": len(rows),
                        "conditions": values,
                    })
                maximum_identity_delta = max(
                    maximum_identity_delta,
                    _result_identity_delta(full_result),
                    _result_identity_delta(null_result),
                )
                append_scalar_request(
                    run_dir,
                    {
                        "ordinal": ordinal,
                        "request_id": record.request_id,
                        "content_control_eligible": bool(wrong["eligible"]),
                        "phase_audit": {
                            "full": full_result["prefix_cache_signature"] if full_result else None,
                            "full_wrong_user": full_result["wrong_user_cache_signature"] if full_result else None,
                            "null": null_result["prefix_cache_signature"] if null_result else None,
                            "token_position_integrity": bool(full_result and null_result) if wrong["eligible"] else True,
                            "cache_key_integrity": bool(full_result and null_result) if wrong["eligible"] else True,
                        },
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
        run_dir, prepared, target_records, N20_CONDITIONS,
        maximum_identity_delta=maximum_identity_delta,
    )


def _result_identity_delta(result: Mapping[str, Any] | None) -> float:
    if not result:
        return 0.0
    return float(result.get("maximum_identity_delta", 0.0))


def _load_wrong_user_histories(
    parent_manifest: Mapping[str, Any], records: Sequence[Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    entry = parent_manifest["frozen_qrels_blind_controls"]["wrong_user"]
    manifest_path = Path(entry["manifest_path"])
    if sha256_file(manifest_path) != entry["manifest_sha256"]:
        raise ValueError("N20 wrong-user control manifest drift")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("qrels_read") is not False
        or manifest.get("model_scores_read") is not False
        or manifest.get("source_test_opened") is not False
        or manifest.get("target_records_sha256")
        != parent_manifest["frozen_inputs"]["records_dev_sha256"]
    ):
        raise ValueError("N20 wrong-user mapping crossed the label/input boundary")
    mapping_path = Path(manifest["mapping_path"])
    if sha256_file(mapping_path) != manifest["mapping_sha256"]:
        raise ValueError("N20 wrong-user mapping drift")
    rows = list(iter_jsonl(mapping_path))
    if len(rows) != len(records):
        raise ValueError("N20 wrong-user mapping count drift")
    controls: dict[str, dict[str, Any]] = {}
    for record, row in zip(records, rows):
        if str(row.get("request_id")) != record.request_id:
            raise ValueError("N20 wrong-user request order drift")
        projected = sanitize_record_for_model({
            "request_id": record.request_id,
            "query": record.query,
            "history": list(row.get("history", [])),
            "candidates": list(record.candidates),
        })
        controls[record.request_id] = {
            "eligible": bool(row.get("eligible") is True),
            "history": list(projected.history),
        }
    eligible = sum(row["eligible"] for row in controls.values())
    if eligible != int(entry["eligible_requests"]):
        raise ValueError("N20 wrong-user eligible count drift")
    return controls, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": entry["manifest_sha256"],
        "mapping_path": str(mapping_path),
        "mapping_sha256": manifest["mapping_sha256"],
        "eligible_requests": eligible,
        "ineligible_requests": len(records) - eligible,
    }


def q1_cache_phase_runtime_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "src/myrec/mechanism/q1_cache_phase_interventions.py",
        "src/myrec/mechanism/q1_cache_phase_scoring.py",
        "src/myrec/mechanism/q1_cache_phase_runtime.py",
        "src/myrec/mechanism/q1_kv_trajectory.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "src/myrec/mechanism/representation_probe.py",
        "src/myrec/baselines/motivation_v12_ranker.py",
        "scripts/score_deep_dive_q1_cache_phase.py",
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"N20 implementation file is missing: {path}")
        files.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"files": files, "digest": _canonical_sha256(files)}


def _load_boundary_manifest() -> dict[str, Any]:
    observed = sha256_file(N17_N20_MANIFEST_PATH)
    if observed != N17_N20_MANIFEST_SHA256:
        raise ValueError("N17/N20 manifest differs from immutable digest")
    value = yaml.safe_load(N17_N20_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("N17/N20 manifest is not a mapping")
    value["_sha256"] = observed
    return value
