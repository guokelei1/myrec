"""Resumable qrels-blind N11 scaled-QK-logit bundles for Q2/Q3."""

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
    _answer_target_tokens,
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
    MAX_WALL_SECONDS,
    SUPPORTED_METHODS,
    _assert_native_targets,
    _canonical_sha256,
    _load_content_controls,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.attention_logit_scoring import (
    ATTENTION_LOGIT_CONDITIONS,
    score_attention_logit_chunk,
)
from myrec.mechanism.postblock_sweep_runtime import NULL_BASELINE_DIRS
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


N11_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_n11_attention_logit_manifest_v1.yaml"
)
N11_MANIFEST_SHA256 = "9c70ff17a3c5fbef3eb309b5bc0b44ddf9eac1740d2a1e1f43dd7a3120d89275"


def write_attention_logit_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = N11_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    block = int(block)
    if block not in FIXED_BLOCKS:
        raise ValueError("N11 attention-logit block must be one of 13,20,27")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("N11 max_requests must be positive")
    if not str(device).strip():
        raise ValueError("N11 requires an explicit device")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("N11 max_wall_seconds must be in (0,13500]")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_n11_manifest(manifest_path)
    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, manifest["frozen_inputs"]["records_sha256"]),
        (standardized_dir / "manifest.json", manifest["frozen_inputs"]["dataset_manifest_sha256"]),
        (standardized_dir / "request_manifest.json", manifest["frozen_inputs"]["request_manifest_sha256"]),
        (standardized_dir / "candidate_manifest.json", manifest["frozen_inputs"]["candidate_manifest_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"N11 frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("N11 requires all 8000 internal-dev requests")

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("N11 supports only Q2/Q3")
    frozen_model = _load_n11_model_binding(manifest, method_id)
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("N11 config differs from frozen binding")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != frozen_model["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("N11 checkpoint differs from frozen binding")

    controls, control_identity = _load_content_controls_from_n11(manifest, method_id, records)
    baseline_full, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = "registered_mechanism_diagnostic" if max_requests is None else "mechanical_smoke_non_result"
    implementation = attention_logit_runtime_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "n11_manifest_sha256": manifest["_sha256"],
        "content_control_sha256": control_identity["rows_sha256"],
        "baseline_full_sha256": full_identity["scores_sha256"],
        "baseline_null_sha256": null_identity["scores_sha256"],
        "score_conditions": list(ATTENTION_LOGIT_CONDITIONS),
        "target_requests": len(target_records),
        "evidence_mode": evidence_mode,
        "device": str(device),
        "implementation_digest": implementation["digest"],
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_n11_attention_logit_operator",
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "score_conditions": list(ATTENTION_LOGIT_CONDITIONS),
        "active_conditions": [
            "full_qk_scale_half", "null_qk_scale_half",
            "full_qk_scale_double", "null_qk_scale_double",
            "full_qk_sign_flip", "null_qk_sign_flip",
        ],
        "identity_conditions": ["full_qk_identity", "null_qk_identity"],
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
        "n11_manifest_path": str(manifest_path),
        "n11_manifest_sha256": manifest["_sha256"],
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
        conditions=ATTENTION_LOGIT_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    maximum_identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    maximum_baseline_delta = float(prepared.metadata.get("maximum_frozen_baseline_delta", 0.0))
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir, prepared, target_records, ATTENTION_LOGIT_CONDITIONS,
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
                        "status": "wall_time_exhausted", "resumable": True,
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
                        result = score_attention_logit_chunk(
                            model, tokenizer, record, candidates, control, config,
                            block=block, device=str(device)
                        )
                        if method_id == "q3_tallrec_generalqwen" and result.get(
                            "shared_prompt_path_identity"
                        ) is not True:
                            raise ValueError("N11 Q3 shared prompt path identity failed")
                        maximum_identity_delta = max(maximum_identity_delta, float(result["maximum_identity_delta"]))
                        for local, candidate in enumerate(candidates):
                            values = {name: float(result["conditions"][name][local]) for name in ATTENTION_LOGIT_CONDITIONS}
                            if not all(math.isfinite(value) for value in values.values()):
                                raise FloatingPointError("N11 produced a non-finite score")
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
                            "baseline_full": full, "baseline_null": null,
                            "full_qk_identity": full, "null_qk_identity": null,
                            "full_qk_scale_half": full, "null_qk_scale_half": null,
                            "full_qk_scale_double": full, "null_qk_scale_double": null,
                            "full_qk_sign_flip": full, "null_qk_sign_flip": null,
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
                        "block_zero_based": block,
                        "content_control_eligible": bool(control["eligible"]),
                        "rows": rows,
                        "rows_sha256": _canonical_sha256(rows),
                    },
                    prepared,
                )
    except Exception as exc:
        prepared.metadata.update({
            "status": "mechanical_failure", "resumable": True,
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
        run_dir, prepared, target_records, ATTENTION_LOGIT_CONDITIONS,
        maximum_identity_delta=maximum_identity_delta,
    )


def attention_logit_runtime_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "src/myrec/mechanism/attention_logit_interventions.py",
        "src/myrec/mechanism/attention_logit_scoring.py",
        "src/myrec/mechanism/attention_logit_runtime.py",
        "src/myrec/mechanism/attention_edge_interventions.py",
        "src/myrec/mechanism/attention_edge_scoring.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "scripts/score_deep_dive_attention_logits.py",
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"N11 implementation file is missing: {path}")
        files.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"files": files, "digest": _canonical_sha256(files)}


def _load_n11_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    observed = sha256_file(path)
    if observed != N11_MANIFEST_SHA256:
        raise ValueError("N11 manifest differs from immutable digest")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("N11 manifest is not a mapping")
    value["_sha256"] = observed
    return value


def _load_n11_model_binding(manifest: Mapping[str, Any], method_id: str) -> Mapping[str, Any]:
    bindings = manifest.get("model_bindings", {})
    binding = bindings.get(method_id)
    if not isinstance(binding, Mapping):
        raise ValueError(f"N11 model binding missing for {method_id}")
    return binding


def _load_content_controls_from_n11(
    manifest: Mapping[str, Any], method_id: str, records: Sequence[Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    parent = _load_parent_manifest()
    return _load_content_controls(parent, method_id, records)


def _load_parent_manifest() -> dict[str, Any]:
    parent = DEEP_DIVE_MANIFEST_PATH
    observed = sha256_file(parent)
    value = yaml.safe_load(parent.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("parent deep-dive manifest is not a mapping")
    value["_sha256"] = observed
    return value


def _canonical_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
