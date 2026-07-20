"""Resumable qrels-blind D2 selected-block branch scoring runtime."""

from __future__ import annotations

import hashlib
import json
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
    MAX_WALL_SECONDS,
    _assert_native_targets,
    _canonical_sha256,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.patch_scorer import _cross_request_mapping
from myrec.mechanism.representation_probe import normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.mechanism.selected_branch_scoring import (
    SELECTED_NODES,
    score_selected_branch_chunk,
    selected_branch_conditions,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


SUPPORTED_METHODS = ("q2_recranker_generalqwen", "q3_tallrec_generalqwen")
NULL_BASELINE_DIRS = {
    "q2_recranker_generalqwen": Path(
        "runs/20260717_kuaisearch_q2_recranker_generalqwen_internal_dev_null_score"
    ),
    "q3_tallrec_generalqwen": Path(
        "runs/20260717_kuaisearch_q3_tallrec_generalqwen_internal_dev_null_score"
    ),
}


def write_selected_branch_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str,
    branch_contract_path: str | Path | None = None,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    request_shard_index: int = 0,
    request_shard_count: int = 1,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score all seven nodes and eight interventions at the frozen block."""

    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("an explicit selected-branch device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("selected-branch max_wall_seconds must be in (0,13500]")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("selected-branch max_requests must be positive")
    request_shard_index = int(request_shard_index)
    request_shard_count = int(request_shard_count)
    if request_shard_count <= 0 or not 0 <= request_shard_index < request_shard_count:
        raise ValueError("selected-branch request shard must satisfy 0 <= index < count")
    if max_requests is not None and request_shard_count != 1:
        raise ValueError("selected-branch smoke cannot be combined with request sharding")
    formal = max_requests is None
    if branch_contract_path is None:
        raise ValueError(
            "selected-branch scoring, including smoke, requires a frozen branch contract"
        )

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, manifest["frozen_inputs"]["records_dev_sha256"]),
        (standardized_dir / "manifest.json", manifest["frozen_inputs"]["dataset_manifest_sha256"]),
        (standardized_dir / "request_manifest.json", manifest["frozen_inputs"]["request_manifest_sha256"]),
        (standardized_dir / "candidate_manifest.json", manifest["frozen_inputs"]["candidate_manifest_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"selected-branch frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("selected-branch scoring requires frozen 8000-request dev")
    records_by_id = {record.request_id: record for record in records}
    cross_mapping = _cross_request_mapping(records)
    cross_mapping_sha256 = _canonical_sha256(cross_mapping)

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("selected-branch scoring supports only Q2/Q3")
    frozen = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("selected-branch config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != frozen["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("selected-branch checkpoint differs from frozen binding")

    branch_contract_path = Path(branch_contract_path)
    branch_contract = _read_json(branch_contract_path)
    if (
        branch_contract.get("contract_type")
        != "transformer_deep_dive_d2_selected_branch_contract"
        or branch_contract.get("status") != "completed"
        or branch_contract.get("branch_scoring_eligible") is not True
        or branch_contract.get("method_id") != method_id
        or branch_contract.get("checkpoint_id") != checkpoint_id
        or branch_contract.get("scoring_population")
        != "normalized_query_fold_1"
        or branch_contract.get("selected_nodes") != list(SELECTED_NODES)
        or branch_contract.get("qrels_values_exposed_to_scorer") is not False
    ):
        raise ValueError("selected-branch contract is not admissible")
    block = int(branch_contract["selected_block"])
    contract_identity = {
        "path": str(branch_contract_path),
        "sha256": sha256_file(branch_contract_path),
        "evidence_role": branch_contract["evidence_role"],
    }
    if not 13 <= block <= 27:
        raise ValueError("selected-branch block is outside [13,27]")

    wrong_mapping, wrong_identity = _load_wrong_user_mapping(manifest, records)
    baseline_full, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    fold1_records = [record for record in records if normalized_query_fold(record.query) == 1]
    target_records = (
        _request_shard_records(
            fold1_records,
            request_shard_index=request_shard_index,
            request_shard_count=request_shard_count,
        )
        if formal
        else _stable_smoke_records(fold1_records, int(max_requests))
    )
    conditions = selected_branch_conditions()
    implementation = selected_branch_implementation_identity()
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if formal and request_shard_count == 1
        else "registered_mechanism_diagnostic_request_shard"
        if formal
        else "mechanical_smoke_non_result"
    )
    request_shard = {
        "index": request_shard_index,
        "count": request_shard_count,
        "rule": "fold1_ordinal_mod_request_shard_count",
        "request_count": len(target_records),
    }
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "selected_block": block,
        "branch_contract": contract_identity,
        "normalized_query_fold": 1,
        "target_requests": len(target_records),
        "request_shard": request_shard,
        "score_conditions": list(conditions),
        "records_sha256": sha256_file(records_path),
        "cross_request_mapping_sha256": cross_mapping_sha256,
        "wrong_user_mapping_sha256": wrong_identity["mapping_sha256"],
        "full_scores_sha256": full_identity["scores_sha256"],
        "null_scores_sha256": null_identity["scores_sha256"],
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "device": str(device),
        "implementation_digest": implementation["digest"],
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d2_selected_branch",
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "selected_block": block,
        "selected_nodes": list(SELECTED_NODES),
        "branch_contract": contract_identity,
        "evidence_role": branch_contract["evidence_role"],
        "normalized_query_fold": 1,
        "full_population_request_count": len(records),
        "fold1_request_count": len(fold1_records),
        "request_shard": request_shard,
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "cross_request_mapping_sha256": cross_mapping_sha256,
        "wrong_user_control": wrong_identity,
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "score_conditions": list(conditions),
        "identity_tolerance": 1.0e-5,
        "random_direction_seed": 20_260_715,
        "wrong_user_ineligible_scoring": "copy_frozen_null_score",
        "implementation_identity": implementation,
        "qrels_read": False,
        "source_test_opened": False,
        "evidence_mode": evidence_mode,
        "result_eligible": formal and request_shard_count == 1,
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
    identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    full_delta = float(prepared.metadata.get("maximum_full_baseline_delta", 0.0))
    null_delta = float(prepared.metadata.get("maximum_null_baseline_delta", 0.0))
    baseline_ratio = float(
        prepared.metadata.get("maximum_baseline_low_precision_ratio", 0.0)
    )
    rms_error = float(
        prepared.metadata.get("maximum_direction_rms_reconstruction_error", 0.0)
    )
    shared_delta = float(
        prepared.metadata.get("shared_prompt_path_max_abs_delta", 0.0)
    )
    wrong_eligible = sum(
        wrong_mapping[record.request_id]["eligible"] is True
        for record in target_records[:completed]
    )
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir,
            prepared,
            target_records,
            conditions,
            maximum_identity_delta=identity_delta,
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
                    _store_progress(
                        prepared.metadata,
                        started,
                        identity_delta,
                        full_delta,
                        null_delta,
                        baseline_ratio,
                        rms_error,
                        shared_delta,
                        wrong_eligible,
                        status="wall_time_exhausted",
                    )
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                cross_record = records_by_id[cross_mapping[record.request_id]]
                wrong_row = wrong_mapping[record.request_id]
                request_wrong_history = (
                    list(wrong_row["history"]) if wrong_row["eligible"] else None
                )
                rows = []
                for start in range(0, len(record.candidates), batch_size):
                    candidates = list(record.candidates[start : start + batch_size])
                    cross_candidates = [
                        cross_record.candidates[(start + local) % len(cross_record.candidates)]
                        for local in range(len(candidates))
                    ]
                    result = score_selected_branch_chunk(
                        model,
                        tokenizer,
                        record,
                        candidates,
                        cross_record,
                        cross_candidates,
                        request_wrong_history,
                        config,
                        block=block,
                        device=str(device),
                    )
                    identity_delta = max(
                        identity_delta, float(result["maximum_identity_delta"])
                    )
                    shared_delta = max(
                        shared_delta,
                        float(result.get("shared_prompt_path_max_abs_delta", 0.0)),
                    )
                    rms_error = max(
                        rms_error,
                        max(
                            _extract_rms_errors(result["direction_geometry"]),
                            default=0.0,
                        ),
                    )
                    score_values = result["conditions"]
                    for local, candidate in enumerate(candidates):
                        values = {
                            name: float(score_values[name][local].item())
                            for name in conditions
                        }
                        if not all(math.isfinite(value) for value in values.values()):
                            raise FloatingPointError("selected-branch score is non-finite")
                        key = (record.request_id, str(candidate["item_id"]))
                        full_delta = max(
                            full_delta,
                            abs(values["baseline_full"] - baseline_full[key]),
                        )
                        null_delta = max(
                            null_delta,
                            abs(values["baseline_null"] - baseline_null[key]),
                        )
                        for condition, reference in (
                            ("baseline_full", baseline_full[key]),
                            ("baseline_null", baseline_null[key]),
                        ):
                            bound = 8.0 * (2.0**-7) * max(1.0, abs(reference))
                            baseline_ratio = max(
                                baseline_ratio,
                                abs(values[condition] - reference) / bound,
                            )
                        if request_wrong_history is None:
                            for node in SELECTED_NODES:
                                values[f"{node}.wrong_history_to_null"] = baseline_null[key]
                        rows.append(
                            {
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "candidate_ordinal": start + local,
                                "wrong_user_eligible": request_wrong_history is not None,
                                "conditions": values,
                            }
                        )
                append_scalar_request(
                    run_dir,
                    {
                        "ordinal": ordinal,
                        "request_id": record.request_id,
                        "cross_donor_request_id": cross_record.request_id,
                        "wrong_user_eligible": request_wrong_history is not None,
                        "wrong_user_donor_request_id": wrong_row.get("donor_request_id"),
                        "rows": rows,
                        "rows_sha256": _canonical_sha256(rows),
                    },
                    prepared,
                )
                if request_wrong_history is not None:
                    wrong_eligible += 1
    except Exception as exc:
        _store_progress(
            prepared.metadata,
            started,
            identity_delta,
            full_delta,
            null_delta,
            baseline_ratio,
            rms_error,
            shared_delta,
            wrong_eligible,
            status="mechanical_failure",
        )
        prepared.metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    if baseline_ratio > 1.0:
        message = (
            "selected-branch recomputed baseline exceeded path-local BF16 bound: "
            f"ratio={baseline_ratio}"
        )
        _store_progress(
            prepared.metadata,
            started,
            identity_delta,
            full_delta,
            null_delta,
            baseline_ratio,
            rms_error,
            shared_delta,
            wrong_eligible,
            status="mechanical_failure",
        )
        prepared.metadata["error"] = {"type": "ValueError", "message": message}
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise ValueError(message)
    _store_progress(
        prepared.metadata,
        started,
        identity_delta,
        full_delta,
        null_delta,
        baseline_ratio,
        rms_error,
        shared_delta,
        wrong_eligible,
        status="running",
    )
    return finalize_scalar_bundle(
        run_dir,
        prepared,
        target_records,
        conditions,
        maximum_identity_delta=identity_delta,
    )


def _load_wrong_user_mapping(
    manifest: Mapping[str, Any], records: Sequence[Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    frozen = manifest["frozen_qrels_blind_controls"]["wrong_user"]
    mapping_path = Path(frozen["manifest_path"]).parent / "wrong_user_mapping.jsonl"
    manifest_path = Path(frozen["manifest_path"])
    if (
        sha256_file(manifest_path) != frozen["manifest_sha256"]
        or sha256_file(mapping_path) != frozen["mapping_sha256"]
    ):
        raise ValueError("wrong-user frozen control bytes changed")
    rows = list(iter_jsonl(mapping_path))
    if len(rows) != len(records):
        raise ValueError("wrong-user mapping request coverage drift")
    result = {}
    eligible = 0
    for record, row in zip(records, rows):
        if row.get("request_id") != record.request_id:
            raise ValueError("wrong-user mapping request order drift")
        is_eligible = row.get("eligible") is True
        history = row.get("history")
        if is_eligible:
            eligible += 1
            expected = min(6, len(record.history))
            if not isinstance(history, list) or len(history) != expected or expected <= 0:
                raise ValueError("wrong-user eligible history count drift")
        elif history not in (None, []):
            raise ValueError("wrong-user ineligible row unexpectedly has history")
        result[record.request_id] = dict(row)
    if eligible != int(frozen["eligible_requests"]):
        raise ValueError("wrong-user eligible request count drift")
    return result, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "eligible_requests": eligible,
        "ineligible_requests": len(records) - eligible,
    }


def _stable_smoke_records(records: Sequence[Any], limit: int) -> list[Any]:
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"deep-dive-selected-branch-smoke-v1\0{record.request_id}".encode("utf-8")
        ).digest(),
    )[:limit]


def _request_shard_records(
    records: Sequence[Any],
    *,
    request_shard_index: int,
    request_shard_count: int,
) -> list[Any]:
    """Deterministically partition the frozen fold without reading outcomes."""

    request_shard_index = int(request_shard_index)
    request_shard_count = int(request_shard_count)
    if request_shard_count <= 0 or not 0 <= request_shard_index < request_shard_count:
        raise ValueError("selected-branch request shard must satisfy 0 <= index < count")
    return [
        record
        for ordinal, record in enumerate(records)
        if ordinal % request_shard_count == request_shard_index
    ]


def _extract_rms_errors(value: Any, key: str = "") -> list[float]:
    if isinstance(value, Mapping):
        result = []
        for name, item in value.items():
            result.extend(_extract_rms_errors(item, str(name)))
        return result
    if key.endswith("rms_max_abs_error") and isinstance(value, (int, float)):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise FloatingPointError("selected-branch RMS audit is non-finite")
        return [abs(scalar)]
    return []


def selected_branch_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/selected_branch_runtime.py",
        "src/myrec/mechanism/selected_branch_shard_merge.py",
        "src/myrec/mechanism/selected_branch_scoring.py",
        "src/myrec/mechanism/selected_branch_contract.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "src/myrec/mechanism/deep_dive_native_patch.py",
        "src/myrec/mechanism/native_readout_scoring.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "scripts/score_deep_dive_selected_branches.py",
        "scripts/merge_deep_dive_selected_branch_shards.py",
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


def _store_progress(
    metadata: dict[str, Any],
    started: float,
    identity: float,
    full_delta: float,
    null_delta: float,
    baseline_ratio: float,
    rms_error: float,
    shared_delta: float,
    wrong_eligible: int,
    *,
    status: str,
) -> None:
    metadata.update(
        {
            "status": status,
            "resumable": status != "running",
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + time.monotonic()
            - started,
            "maximum_identity_delta": identity,
            "maximum_full_baseline_delta": full_delta,
            "maximum_null_baseline_delta": null_delta,
            "maximum_baseline_low_precision_ratio": baseline_ratio,
            "maximum_direction_rms_reconstruction_error": rms_error,
            "shared_prompt_path_max_abs_delta": shared_delta,
            "wrong_user_eligible_requests": wrong_eligible,
        }
    )
