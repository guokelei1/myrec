"""Resumable qrels-blind N15/N16 operator-stage bundles."""

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
    BASELINE_SCORE_DIRS,
    DEEP_DIVE_MANIFEST_PATH,
    FIXED_BLOCKS,
    MAX_WALL_SECONDS,
    SUPPORTED_METHODS,
    _assert_native_targets,
    _canonical_sha256,
    _load_frozen_baseline,
    _read_json,
    _write_json,
)
from myrec.mechanism.attention_logit_runtime import _load_content_controls_from_n11
from myrec.mechanism.operator_stage_scoring import (
    RMSNORM_CONDITIONS,
    RESIDUAL_CONDITIONS,
    score_rmsnorm_chunk,
    score_residual_composition_chunk,
)
from myrec.mechanism.postblock_sweep_runtime import NULL_BASELINE_DIRS
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


N15_N16_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_n15_n16_operator_manifest_v1.yaml"
)
N15_N16_MANIFEST_SHA256 = "b6c47313c44e0dc4e51f04db6e0b46fef5099bb6fc4a2a84067f65f4f1469a71"


def write_residual_composition_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    branch: str,
    device: str,
    runs_dir: str | Path = "runs",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    return _write_operator_bundle(
        standardized_dir,
        config_path,
        checkpoint_root,
        run_id,
        kind="n15_residual_composition",
        block=int(block),
        branch=str(branch),
        scope=None,
        device=device,
        runs_dir=runs_dir,
        resume=resume,
        max_wall_seconds=max_wall_seconds,
        max_requests=max_requests,
        command=command,
    )


def write_rmsnorm_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    scope: str,
    block: int | None,
    device: str,
    runs_dir: str | Path = "runs",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    return _write_operator_bundle(
        standardized_dir,
        config_path,
        checkpoint_root,
        run_id,
        kind="n16_rmsnorm",
        block=None if block is None else int(block),
        branch=None,
        scope=str(scope),
        device=device,
        runs_dir=runs_dir,
        resume=resume,
        max_wall_seconds=max_wall_seconds,
        max_requests=max_requests,
        command=command,
    )


def _write_operator_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    kind: str,
    block: int | None,
    branch: str | None,
    scope: str | None,
    device: str,
    runs_dir: str | Path,
    resume: bool,
    max_wall_seconds: float,
    max_requests: int | None,
    command: Sequence[str] | None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("N15/N16 requires an explicit device")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("N15/N16 max_wall_seconds is out of range")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("N15/N16 max_requests must be positive")
    if kind == "n15_residual_composition":
        if block not in FIXED_BLOCKS or branch not in {"attention", "mlp"}:
            raise ValueError("N15 requires fixed block 13/20/27 and attention/mlp branch")
        conditions = RESIDUAL_CONDITIONS
    elif kind == "n16_rmsnorm":
        if scope not in {"input", "post_attention", "final"}:
            raise ValueError("N16 scope must be input, post_attention, or final")
        if scope == "final" and block is not None:
            raise ValueError("N16 final scope does not take a block")
        if scope != "final" and block not in FIXED_BLOCKS:
            raise ValueError("N16 block scope requires block 13/20/27")
        conditions = RMSNORM_CONDITIONS
    else:  # pragma: no cover - private dispatch guard
        raise ValueError(f"unsupported operator bundle kind={kind}")
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest()
    records_path = standardized_dir / "records_dev.jsonl"
    expected_hashes = {
        records_path: manifest["frozen_inputs"]["records_sha256"],
        standardized_dir / "manifest.json": manifest["frozen_inputs"]["dataset_manifest_sha256"],
        standardized_dir / "request_manifest.json": manifest["frozen_inputs"]["request_manifest_sha256"],
        standardized_dir / "candidate_manifest.json": manifest["frozen_inputs"]["candidate_manifest_sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"{kind} frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError(f"{kind} requires all 8000 internal-dev requests")
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError(f"{kind} supports only Q2/Q3")
    binding = manifest["model_bindings"].get(method_id)
    if not isinstance(binding, Mapping) or config["_config_sha256"] != binding["config_sha256"]:
        raise ValueError(f"{kind} config differs from frozen binding")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != binding["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError(f"{kind} checkpoint differs from frozen binding")
    controls, control_identity = _load_content_controls_from_n11(manifest, method_id, records)
    baseline_full, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = "registered_mechanism_diagnostic" if max_requests is None else "mechanical_smoke_non_result"
    implementation = operator_stage_runtime_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "analysis_stage": kind,
        "method_id": method_id,
        "block_zero_based": block,
        "branch": branch,
        "scope": scope,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "manifest_sha256": manifest["_sha256"],
        "content_control_sha256": control_identity["rows_sha256"],
        "baseline_full_sha256": full_identity["scores_sha256"],
        "baseline_null_sha256": null_identity["scores_sha256"],
        "score_conditions": list(conditions),
        "target_requests": len(target_records),
        "evidence_mode": evidence_mode,
        "device": str(device),
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": kind,
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "branch": branch,
        "scope": scope,
        "score_conditions": list(conditions),
        "identity_conditions": [name for name in conditions if name.endswith("identity")],
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
        "manifest_path": str(N15_N16_MANIFEST_PATH),
        "manifest_sha256": manifest["_sha256"],
        "content_control": control_identity,
        "frozen_baseline_full": full_identity,
        "frozen_baseline_null": null_identity,
        "identity_tolerance": 1.0e-5,
        "implementation_identity": implementation,
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
        conditions=conditions,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    maximum_identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    maximum_baseline_delta = float(prepared.metadata.get("maximum_frozen_baseline_delta", 0.0))
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir, prepared, target_records, conditions,
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
        prepared.metadata["implementation_identity"] = implementation
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
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
                control = controls[record.request_id]
                rows: list[dict[str, Any]] = []
                if control.get("eligible") is True:
                    for start in range(0, len(record.candidates), batch_size):
                        candidates = list(record.candidates[start : start + batch_size])
                        if kind == "n15_residual_composition":
                            result = score_residual_composition_chunk(
                                model, tokenizer, record, candidates, control, config,
                                block=int(block), branch=str(branch), device=str(device)
                            )
                        else:
                            result = score_rmsnorm_chunk(
                                model, tokenizer, record, candidates, control, config,
                                scope=str(scope), block=block, device=str(device)
                            )
                        maximum_identity_delta = max(
                            maximum_identity_delta, float(result["maximum_identity_delta"])
                        )
                        for local, candidate in enumerate(candidates):
                            values = {name: float(result["conditions"][name][local]) for name in conditions}
                            if not all(math.isfinite(value) for value in values.values()):
                                raise FloatingPointError(f"{kind} produced a non-finite score")
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
                        full, null = baseline_full[key], baseline_null[key]
                        values = {
                            name: (null if name == "baseline_null" or name.startswith("null_") else full)
                            for name in conditions
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
                        "content_control_eligible": bool(control.get("eligible") is True),
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
        run_dir, prepared, target_records, conditions,
        maximum_identity_delta=maximum_identity_delta,
    )


def operator_stage_runtime_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "src/myrec/mechanism/residual_composition_interventions.py",
        "src/myrec/mechanism/rmsnorm_interventions.py",
        "src/myrec/mechanism/operator_stage_scoring.py",
        "src/myrec/mechanism/operator_stage_runtime.py",
        "src/myrec/mechanism/attention_edge_scoring.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"N15/N16 implementation file is missing: {path}")
        files.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"files": files, "digest": _canonical_sha256(files)}


def _load_manifest() -> dict[str, Any]:
    observed = sha256_file(N15_N16_MANIFEST_PATH)
    if observed != N15_N16_MANIFEST_SHA256:
        raise ValueError("N15/N16 manifest differs from immutable digest")
    value = yaml.safe_load(N15_N16_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("N15/N16 manifest is not a mapping")
    value["_sha256"] = observed
    return value


def _load_parent_manifest() -> dict[str, Any]:
    value = yaml.safe_load(DEEP_DIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("parent deep-dive manifest is not a mapping")
    return value
