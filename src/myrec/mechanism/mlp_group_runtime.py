"""Resumable qrels-blind D4 fixed-sample MLP group localization runtime."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
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
    DEEP_DIVE_MANIFEST_PATH,
    FIXED_BLOCKS,
    MAX_WALL_SECONDS,
    SUPPORTED_METHODS,
    _assert_native_targets,
    _canonical_sha256,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.mlp_group_scoring import score_mlp_group_sample_row
from myrec.mechanism.patch_scorer import _cross_request_mapping
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def write_mlp_group_sample_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_rows: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    block = int(block)
    if block not in FIXED_BLOCKS:
        raise ValueError("D4 MLP block must be one of 13, 20, 27")
    if not str(device).strip():
        raise ValueError("an explicit D4 MLP device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("D4 MLP max_wall_seconds must be in (0, 13500]")
    if max_rows is not None and not 0 < int(max_rows) < 512:
        raise ValueError("D4 MLP smoke row cap must be in [1,511]")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    sample_contract = manifest["frozen_qrels_blind_controls"]["fixed_high_dimensional_sample"]
    sample_manifest_path = Path(sample_contract["manifest_path"])
    if sha256_file(sample_manifest_path) != sample_contract["manifest_sha256"]:
        raise ValueError("D4 frozen sample manifest hash mismatch")
    sample_manifest = _read_json(sample_manifest_path)
    sample_path = Path(sample_manifest["path"])
    if sha256_file(sample_path) != sample_contract["rows_sha256"]:
        raise ValueError("D4 frozen sample row hash mismatch")
    sample_rows = list(iter_jsonl(sample_path))
    if len(sample_rows) != 512:
        raise ValueError("D4 frozen sample must contain 512 candidate rows")
    target_rows = sample_rows if max_rows is None else sample_rows[: int(max_rows)]

    records_path = standardized_dir / "records_dev.jsonl"
    if sha256_file(records_path) != manifest["frozen_inputs"]["records_dev_sha256"]:
        raise ValueError("D4 records_dev hash mismatch")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    records_by_id = {record.request_id: record for record in records}
    cross_mapping = _cross_request_mapping(records)
    for row in sample_rows:
        record = records_by_id.get(str(row["request_id"]))
        ordinal = int(row["candidate_ordinal"])
        if (
            record is None
            or not 0 <= ordinal < len(record.candidates)
            or str(record.candidates[ordinal]["item_id"]) != str(row["candidate_item_id"])
        ):
            raise ValueError("D4 frozen candidate sample differs from dev records")

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("D4 MLP runtime supports only Q2/Q3")
    frozen_model = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("D4 MLP config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != frozen_model["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("D4 MLP checkpoint differs from frozen binding")

    evidence_mode = (
        "registered_descriptive_localization"
        if max_rows is None
        else "mechanical_smoke_non_result"
    )
    implementation = mlp_group_runtime_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "sample_rows_sha256": sample_contract["rows_sha256"],
        "target_rows": len(target_rows),
        "cross_mapping_sha256": _canonical_sha256(cross_mapping),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "implementation_digest": implementation["digest"],
        "device": str(device),
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d4_mlp_groups",
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "sample_manifest_path": str(sample_manifest_path),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "sample_rows_path": str(sample_path),
        "sample_rows_sha256": sha256_file(sample_path),
        "target_rows": len(target_rows),
        "groups": 16,
        "group_seed": 20260718,
        "permutation_recomposition_dtype": "float32",
        "permutation_bound_reference_dtype": "native_swiglu_product_dtype",
        "cross_mapping_rule": "frozen M2 sha256 ring successor; candidate ordinal modulo donor slate",
        "cross_mapping_sha256": _canonical_sha256(cross_mapping),
        "deep_dive_manifest_sha256": manifest["_sha256"],
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
        run_dir, metadata, contract_sha256, target_rows, resume=resume
    )
    completed = int(progress["completed_rows"])
    maximum_identity_delta = float(metadata.get("maximum_same_group_identity_delta", 0.0))
    maximum_permutation_ratio = float(metadata.get("maximum_permutation_low_precision_ratio", 0.0))
    if completed >= len(target_rows):
        return _finalize_or_record_failure(
            run_dir, metadata, progress, target_rows
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
        metadata.update(_runtime_metadata(method_id, torch, transformers))
        metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", metadata)
        with torch.inference_mode():
            for ordinal in range(completed, len(target_rows)):
                if time.monotonic() - started >= max_wall_seconds:
                    metadata.update(
                        {
                            "status": "wall_time_exhausted",
                            "resumable": True,
                            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
                            + time.monotonic()
                            - started,
                            "maximum_same_group_identity_delta": maximum_identity_delta,
                            "maximum_permutation_low_precision_ratio": maximum_permutation_ratio,
                        }
                    )
                    _write_json(run_dir / "metadata.json", metadata)
                    return metadata
                sample = target_rows[ordinal]
                record = records_by_id[str(sample["request_id"])]
                candidate_ordinal = int(sample["candidate_ordinal"])
                candidate = record.candidates[candidate_ordinal]
                donor_record = records_by_id[cross_mapping[record.request_id]]
                donor_ordinal = candidate_ordinal % len(donor_record.candidates)
                result = score_mlp_group_sample_row(
                    model,
                    tokenizer,
                    record,
                    candidate,
                    donor_record,
                    donor_record.candidates[donor_ordinal],
                    config,
                    block=block,
                    device=str(device),
                )
                maximum_identity_delta = max(
                    maximum_identity_delta,
                    float(result["maximum_same_group_identity_delta"]),
                )
                maximum_permutation_ratio = max(
                    maximum_permutation_ratio,
                    float(result["permutation_low_precision_max_ratio"]),
                )
                row = {
                    "ordinal": ordinal,
                    "request_id": record.request_id,
                    "candidate_ordinal": candidate_ordinal,
                    "candidate_item_id": str(candidate["item_id"]),
                    "donor_request_id": donor_record.request_id,
                    "donor_candidate_ordinal": donor_ordinal,
                    "donor_candidate_item_id": str(
                        donor_record.candidates[donor_ordinal]["item_id"]
                    ),
                    "block_zero_based": block,
                    "result": result,
                }
                _append_sync(run_dir / "rows.partial.jsonl", row)
                progress.update(
                    {
                        "completed_rows": ordinal + 1,
                        "last_request_id": record.request_id,
                        "partial_sha256": sha256_file(run_dir / "rows.partial.jsonl"),
                        "status": "running",
                        "updated_at": _utc_now(),
                    }
                )
                _write_json(run_dir / "progress.json", progress)
    except Exception as exc:
        metadata.update(
            {
                "status": "mechanical_failure",
                "resumable": True,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "maximum_same_group_identity_delta": maximum_identity_delta,
                "maximum_permutation_low_precision_ratio": maximum_permutation_ratio,
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        raise
    metadata.update(
        {
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + time.monotonic()
            - started,
            "maximum_same_group_identity_delta": maximum_identity_delta,
            "maximum_permutation_low_precision_ratio": maximum_permutation_ratio,
        }
    )
    return _finalize_or_record_failure(run_dir, metadata, progress, target_rows)


def mlp_group_runtime_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/mlp_group_runtime.py",
        "src/myrec/mechanism/mlp_group_scoring.py",
        "src/myrec/mechanism/mlp_group_interventions.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "scripts/score_deep_dive_mlp_groups.py",
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


def _prepare(
    run_dir: Path,
    metadata: dict[str, Any],
    contract_sha256: str,
    target_rows: Sequence[Mapping[str, Any]],
    *,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    partial = run_dir / "rows.partial.jsonl"
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"D4 MLP run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial.touch(exist_ok=False)
        metadata.update({"elapsed_seconds": 0.0, "resumable": True, "resume_lineage": []})
        progress = {
            "schema_version": 1,
            "run_contract_sha256": contract_sha256,
            "completed_rows": 0,
            "last_request_id": None,
            "partial_sha256": sha256_file(partial),
            "status": "initializing",
            "updated_at": _utc_now(),
        }
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "progress.json", progress)
        return metadata, progress
    stored = _read_json(run_dir / "metadata.json")
    progress = _read_json(run_dir / "progress.json")
    if stored.get("run_contract_sha256") != contract_sha256 or progress.get(
        "run_contract_sha256"
    ) != contract_sha256:
        raise ValueError("D4 MLP resume contract drift")
    observed = _audit_partial(partial, target_rows)
    if progress.get("completed_rows") != observed["completed_rows"] or progress.get(
        "partial_sha256"
    ) != observed["partial_sha256"]:
        raise ValueError("D4 MLP progress differs from partial")
    lineage = list(stored.get("resume_lineage", []))
    lineage.append({"resumed_at": _utc_now(), **observed})
    stored.update({"status": "initializing", "resume_lineage": lineage})
    _write_json(run_dir / "metadata.json", stored)
    return stored, progress


def _audit_partial(
    path: Path, target_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    count = 0
    for row in iter_jsonl(path):
        if count >= len(target_rows):
            raise ValueError("D4 MLP partial has excess rows")
        expected = target_rows[count]
        if (
            row.get("ordinal") != count
            or row.get("request_id") != expected["request_id"]
            or row.get("candidate_ordinal") != expected["candidate_ordinal"]
            or row.get("candidate_item_id") != expected["candidate_item_id"]
        ):
            raise ValueError("D4 MLP partial sample identity/order drift")
        if len(row.get("result", {}).get("groups", [])) != 16:
            raise ValueError("D4 MLP partial group coverage drift")
        count += 1
    return {"completed_rows": count, "partial_sha256": sha256_file(path)}


def _finalize(
    run_dir: Path,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    target_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = _audit_partial(run_dir / "rows.partial.jsonl", target_rows)
    if observed["completed_rows"] != len(target_rows):
        raise ValueError("cannot finalize incomplete D4 MLP bundle")
    if float(metadata.get("maximum_same_group_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("D4 MLP same-group identity gate failed")
    if float(metadata.get("maximum_permutation_low_precision_ratio", math.inf)) > 1.0:
        raise ValueError("D4 MLP permutation recomposition gate failed")
    os.replace(run_dir / "rows.partial.jsonl", run_dir / "rows.jsonl")
    progress.update({"status": "completed", "updated_at": _utc_now()})
    metadata.update(
        {
            "status": "completed",
            "resumable": False,
            "row_count": len(target_rows),
            "rows_path": str(run_dir / "rows.jsonl"),
            "rows_sha256": sha256_file(run_dir / "rows.jsonl"),
            "identity_passed": True,
            "permutation_recomposition_passed": True,
        }
    )
    _write_json(run_dir / "progress.json", progress)
    _write_json(run_dir / "metadata.json", metadata)
    return metadata


def _finalize_or_record_failure(
    run_dir: Path,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    target_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        return _finalize(run_dir, metadata, progress, target_rows)
    except Exception as exc:
        metadata.update(
            {
                "status": "mechanical_failure",
                "resumable": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        raise


def _append_sync(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
