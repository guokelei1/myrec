"""Qrels-blind resumable D5 fixed-length contextual-control scorer."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

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
    _load_content_controls,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.contextual_control_scoring import (
    CONTEXTUAL_SCORE_CONDITIONS,
    score_contextual_control_chunk,
)
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def write_contextual_control_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("an explicit contextual-control device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("contextual-control max_wall_seconds must be in (0, 13500]")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("max_requests must be positive")
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, manifest["frozen_inputs"]["records_dev_sha256"]),
        (
            standardized_dir / "manifest.json",
            manifest["frozen_inputs"]["dataset_manifest_sha256"],
        ),
        (
            standardized_dir / "request_manifest.json",
            manifest["frozen_inputs"]["request_manifest_sha256"],
        ),
        (
            standardized_dir / "candidate_manifest.json",
            manifest["frozen_inputs"]["candidate_manifest_sha256"],
        ),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"contextual-control frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("contextual-control scorer requires all 8000 dev requests")
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("contextual-control scorer supports only Q2/Q3")
    frozen_model = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("contextual-control config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, method_id
    )
    if checkpoint_id != frozen_model["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("contextual-control checkpoint differs from frozen binding")
    controls, control_identity = _load_content_controls(manifest, method_id, records)
    baseline, baseline_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if max_requests is None
        else "mechanical_smoke_non_result"
    )
    implementation = contextual_runtime_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "content_control_sha256": control_identity["rows_sha256"],
        "baseline_scores_sha256": baseline_identity["scores_sha256"],
        "score_conditions": list(CONTEXTUAL_SCORE_CONDITIONS),
        "target_requests": len(target_records),
        "evidence_mode": evidence_mode,
        "device": str(device),
        "implementation_digest": implementation["digest"],
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d5_contextual_controls",
        "run_id": run_id,
        "method_id": method_id,
        "score_conditions": list(CONTEXTUAL_SCORE_CONDITIONS),
        "identity_conditions": ["unmodified_identity"],
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "request_manifest_sha256": sha256_file(
            standardized_dir / "request_manifest.json"
        ),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "content_control": control_identity,
        "frozen_baseline": baseline_identity,
        "implementation_identity": implementation,
        "identity_tolerance": 1.0e-5,
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage_required": True,
        "ineligible_scoring": "copy_frozen_baseline_score",
        "position_contract": "token_ids_and_lengths_fixed; Qwen cache_position arange unchanged",
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
        conditions=CONTEXTUAL_SCORE_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    maximum_identity_delta = float(
        prepared.metadata.get("maximum_identity_delta", 0.0)
    )
    maximum_baseline_delta = float(
        prepared.metadata.get("maximum_frozen_baseline_delta", 0.0)
    )
    eligible_completed = int(prepared.metadata.get("eligible_requests_completed", 0))
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir,
            prepared,
            target_records,
            CONTEXTUAL_SCORE_CONDITIONS,
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
        _assert_native_targets(tokenizer, method_id, manifest)
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
                            "elapsed_seconds": float(
                                prepared.metadata.get("elapsed_seconds", 0.0)
                            )
                            + time.monotonic()
                            - started,
                            "maximum_identity_delta": maximum_identity_delta,
                            "maximum_frozen_baseline_delta": maximum_baseline_delta,
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
                        result = score_contextual_control_chunk(
                            model,
                            tokenizer,
                            record,
                            candidates,
                            control,
                            config,
                            device=str(device),
                        )
                        maximum_identity_delta = max(
                            maximum_identity_delta,
                            float(result["maximum_identity_delta"]),
                        )
                        for local, candidate in enumerate(candidates):
                            values = {
                                name: float(result["conditions"][name][local])
                                for name in CONTEXTUAL_SCORE_CONDITIONS
                            }
                            if not all(math.isfinite(value) for value in values.values()):
                                raise FloatingPointError(
                                    "contextual-control bundle produced a non-finite score"
                                )
                            frozen = baseline[(record.request_id, str(candidate["item_id"]))]
                            maximum_baseline_delta = max(
                                maximum_baseline_delta,
                                abs(values["baseline_full"] - frozen),
                            )
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
                        frozen = baseline[(record.request_id, str(candidate["item_id"]))]
                        rows.append(
                            {
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "candidate_ordinal": candidate_ordinal,
                                "conditions": {
                                    name: frozen for name in CONTEXTUAL_SCORE_CONDITIONS
                                },
                            }
                        )
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
        prepared.metadata.update(
            {
                "status": "mechanical_failure",
                "resumable": True,
                "qrels_read": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "maximum_identity_delta": maximum_identity_delta,
                "maximum_frozen_baseline_delta": maximum_baseline_delta,
                "eligible_requests_completed": eligible_completed,
            }
        )
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    prepared.metadata.update(
        {
            "elapsed_seconds": float(prepared.metadata.get("elapsed_seconds", 0.0))
            + time.monotonic()
            - started,
            "maximum_identity_delta": maximum_identity_delta,
            "maximum_frozen_baseline_delta": maximum_baseline_delta,
            "eligible_requests_completed": eligible_completed,
        }
    )
    return finalize_scalar_bundle(
        run_dir,
        prepared,
        target_records,
        CONTEXTUAL_SCORE_CONDITIONS,
        maximum_identity_delta=maximum_identity_delta,
    )


def contextual_runtime_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/contextual_control_scoring.py",
        "src/myrec/mechanism/contextual_control_runtime.py",
        "src/myrec/mechanism/attention_edge_scoring.py",
        "src/myrec/mechanism/attention_edge_runtime.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "scripts/score_deep_dive_contextual_controls.py",
    )
    files = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}
