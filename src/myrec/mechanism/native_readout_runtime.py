"""Qrels-blind resumable D6 Q2 native final-norm/readout scorer."""

from __future__ import annotations

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
from myrec.mechanism.native_readout_scoring import (
    Q2_FINAL_NODES,
    build_q2_pointwise_batch,
    capture_q2_native_readout,
    score_q2_with_final_node_patch,
)
from myrec.mechanism.patch_scorer import _cross_request_mapping
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


Q2_METHOD_ID = "q2_recranker_generalqwen"
Q2_NULL_BASELINE_DIR = Path(
    "runs/20260717_kuaisearch_q2_recranker_generalqwen_internal_dev_null_score"
)
Q2_NATIVE_READOUT_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "input_same_full_to_null",
    "input_cross_full_to_null",
    "input_null_identity",
    "input_full_identity",
    "output_same_full_to_null",
    "output_cross_full_to_null",
    "output_null_identity",
    "output_full_identity",
)


def write_q2_native_readout_bundle(
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
    """Materialize every registered Q2 native-readout score without qrels."""

    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("an explicit Q2 native-readout device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("Q2 native-readout max_wall_seconds must be in (0, 13500]")
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
        (standardized_dir / "manifest.json", manifest["frozen_inputs"]["dataset_manifest_sha256"]),
        (standardized_dir / "request_manifest.json", manifest["frozen_inputs"]["request_manifest_sha256"]),
        (standardized_dir / "candidate_manifest.json", manifest["frozen_inputs"]["candidate_manifest_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"Q2 native-readout frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("Q2 native-readout scorer requires all 8000 dev requests")
    records_by_id = {row.request_id: row for row in records}
    cross_mapping = _cross_request_mapping(records)
    cross_mapping_sha256 = _canonical_sha256(cross_mapping)

    config = load_v12_ranker_config(config_path)
    if config["method_id"] != Q2_METHOD_ID:
        raise ValueError("native-readout runtime currently implements registered Q2 only")
    frozen_model = manifest["frozen_inputs"]["models"][Q2_METHOD_ID]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("Q2 native-readout config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, Q2_METHOD_ID
    )
    if checkpoint_id != frozen_model["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("Q2 native-readout checkpoint differs from frozen binding")
    baseline_full, baseline_full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[Q2_METHOD_ID], Q2_METHOD_ID, checkpoint_id, records
    )
    baseline_null, baseline_null_identity = _load_frozen_baseline(
        Q2_NULL_BASELINE_DIR, Q2_METHOD_ID, checkpoint_id, records
    )
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if max_requests is None
        else "mechanical_smoke_non_result"
    )
    implementation = q2_native_readout_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": Q2_METHOD_ID,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "cross_request_mapping_sha256": cross_mapping_sha256,
        "full_scores_sha256": baseline_full_identity["scores_sha256"],
        "null_scores_sha256": baseline_null_identity["scores_sha256"],
        "score_conditions": list(Q2_NATIVE_READOUT_CONDITIONS),
        "target_requests": len(target_records),
        "device": str(device),
        "evidence_mode": evidence_mode,
        "implementation_digest": implementation["digest"],
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d6_q2_native_readout",
        "run_id": run_id,
        "method_id": Q2_METHOD_ID,
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
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "cross_request_mapping_rule": "frozen M2 sha256 ring successor; candidate ordinal modulo donor slate",
        "cross_request_mapping_sha256": cross_mapping_sha256,
        "frozen_full_baseline": baseline_full_identity,
        "frozen_null_baseline": baseline_null_identity,
        "score_conditions": list(Q2_NATIVE_READOUT_CONDITIONS),
        "nodes": list(Q2_FINAL_NODES),
        "identity_conditions": [
            "input_null_identity",
            "input_full_identity",
            "output_null_identity",
            "output_full_identity",
        ],
        "native_token_ids": {"yes": 9693, "no": 2152},
        "readout_algebra": "h_finalnorm dot (E_yes-E_no)",
        "implementation_identity": implementation,
        "identity_tolerance": 1.0e-5,
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
        conditions=Q2_NATIVE_READOUT_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    maximum_identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    maximum_full_baseline_delta = float(prepared.metadata.get("maximum_full_baseline_delta", 0.0))
    maximum_null_baseline_delta = float(prepared.metadata.get("maximum_null_baseline_delta", 0.0))
    maximum_algebra_delta = float(prepared.metadata.get("maximum_algebra_delta", 0.0))
    maximum_algebra_ratio = float(
        prepared.metadata.get("maximum_algebra_low_precision_ratio", 0.0)
    )
    maximum_baseline_ratio = float(
        prepared.metadata.get("maximum_baseline_low_precision_ratio", 0.0)
    )
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir,
            prepared,
            target_records,
            Q2_NATIVE_READOUT_CONDITIONS,
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
        _assert_native_targets(tokenizer, Q2_METHOD_ID, manifest)
        prepared.metadata.update(_runtime_metadata(Q2_METHOD_ID, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 16))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    prepared.metadata.update(
                        _runtime_progress(
                            ordinal,
                            started,
                            prepared.metadata,
                            maximum_identity_delta,
                            maximum_full_baseline_delta,
                            maximum_null_baseline_delta,
                            maximum_algebra_delta,
                            maximum_baseline_ratio,
                            maximum_algebra_ratio,
                        )
                    )
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                donor_record = records_by_id[cross_mapping[record.request_id]]
                rows: list[dict[str, Any]] = []
                for start in range(0, len(record.candidates), batch_size):
                    candidates = list(record.candidates[start : start + batch_size])
                    donor_candidates = [
                        donor_record.candidates[(start + local) % len(donor_record.candidates)]
                        for local in range(len(candidates))
                    ]
                    full_batch = build_q2_pointwise_batch(
                        tokenizer, record, candidates, record.history, config, device=str(device)
                    )
                    null_batch = build_q2_pointwise_batch(
                        tokenizer, record, candidates, [], config, device=str(device)
                    )
                    cross_batch = build_q2_pointwise_batch(
                        tokenizer,
                        donor_record,
                        donor_candidates,
                        donor_record.history,
                        config,
                        device=str(device),
                    )
                    full = capture_q2_native_readout(model, *full_batch)
                    null = capture_q2_native_readout(model, *null_batch)
                    cross = capture_q2_native_readout(model, *cross_batch)
                    maximum_algebra_delta = max(
                        maximum_algebra_delta,
                        float(full["algebra_max_abs_error"]),
                        float(null["algebra_max_abs_error"]),
                        float(cross["algebra_max_abs_error"]),
                    )
                    maximum_algebra_ratio = max(
                        maximum_algebra_ratio,
                        float(full["algebra_low_precision_max_ratio"]),
                        float(null["algebra_low_precision_max_ratio"]),
                        float(cross["algebra_low_precision_max_ratio"]),
                    )
                    condition_tensors: dict[str, Any] = {
                        "baseline_full": full["native_score"],
                        "baseline_null": null["native_score"],
                    }
                    for prefix, node in (
                        ("input", "final_rmsnorm_input"),
                        ("output", "final_rmsnorm_output"),
                    ):
                        condition_tensors[f"{prefix}_same_full_to_null"] = score_q2_with_final_node_patch(
                            model, *null_batch, full[node], node_id=node
                        )
                        condition_tensors[f"{prefix}_cross_full_to_null"] = score_q2_with_final_node_patch(
                            model, *null_batch, cross[node], node_id=node
                        )
                        condition_tensors[f"{prefix}_null_identity"] = score_q2_with_final_node_patch(
                            model, *null_batch, null[node], node_id=node
                        )
                        condition_tensors[f"{prefix}_full_identity"] = score_q2_with_final_node_patch(
                            model, *full_batch, full[node], node_id=node
                        )
                        maximum_identity_delta = max(
                            maximum_identity_delta,
                            float((condition_tensors[f"{prefix}_null_identity"] - null["native_score"]).abs().max().item()),
                            float((condition_tensors[f"{prefix}_full_identity"] - full["native_score"]).abs().max().item()),
                        )
                    for local, candidate in enumerate(candidates):
                        values = {
                            name: float(condition_tensors[name][local].item())
                            for name in Q2_NATIVE_READOUT_CONDITIONS
                        }
                        if not all(math.isfinite(value) for value in values.values()):
                            raise FloatingPointError("Q2 native-readout produced a non-finite score")
                        key = (record.request_id, str(candidate["item_id"]))
                        maximum_full_baseline_delta = max(
                            maximum_full_baseline_delta,
                            abs(values["baseline_full"] - baseline_full[key]),
                        )
                        maximum_null_baseline_delta = max(
                            maximum_null_baseline_delta,
                            abs(values["baseline_null"] - baseline_null[key]),
                        )
                        for condition, reference in (
                            ("baseline_full", baseline_full[key]),
                            ("baseline_null", baseline_null[key]),
                        ):
                            bound = 4.0 * (2.0**-7) * max(1.0, abs(reference))
                            maximum_baseline_ratio = max(
                                maximum_baseline_ratio,
                                abs(values[condition] - reference) / bound,
                            )
                        rows.append(
                            {
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "candidate_ordinal": start + local,
                                "conditions": values,
                                "readout_diagnostics": {
                                    "full_common_offset": float(full["common_offset"][local].item()),
                                    "null_common_offset": float(null["common_offset"][local].item()),
                                    "full_input_norm": float(full["geometry"]["input_norm"][local].item()),
                                    "full_output_norm": float(full["geometry"]["output_norm"][local].item()),
                                    "full_input_output_cosine": float(full["geometry"]["input_output_cosine"][local].item()),
                                },
                            }
                        )
                append_scalar_request(
                    run_dir,
                    {
                        "ordinal": ordinal,
                        "request_id": record.request_id,
                        "cross_donor_request_id": donor_record.request_id,
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
                "maximum_algebra_delta": maximum_algebra_delta,
                "maximum_baseline_low_precision_ratio": maximum_baseline_ratio,
                "maximum_algebra_low_precision_ratio": maximum_algebra_ratio,
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
            "maximum_full_baseline_delta": maximum_full_baseline_delta,
            "maximum_null_baseline_delta": maximum_null_baseline_delta,
            "maximum_algebra_delta": maximum_algebra_delta,
            "maximum_baseline_low_precision_ratio": maximum_baseline_ratio,
            "maximum_algebra_low_precision_ratio": maximum_algebra_ratio,
            "low_precision_algebra_passed": (
                maximum_baseline_ratio <= 1.0 and maximum_algebra_ratio <= 1.0
            ),
        }
    )
    return finalize_scalar_bundle(
        run_dir,
        prepared,
        target_records,
        Q2_NATIVE_READOUT_CONDITIONS,
        maximum_identity_delta=maximum_identity_delta,
    )


def q2_native_readout_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/native_readout_runtime.py",
        "src/myrec/mechanism/native_readout_scoring.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "scripts/score_deep_dive_q2_native_readout.py",
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


def _runtime_progress(
    ordinal: int,
    started: float,
    metadata: Mapping[str, Any],
    identity: float,
    full_delta: float,
    null_delta: float,
    algebra: float,
    baseline_ratio: float,
    algebra_ratio: float,
) -> dict[str, Any]:
    return {
        "status": "wall_time_exhausted",
        "resumable": True,
        "completed_requests": int(ordinal),
        "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
        + time.monotonic()
        - started,
        "maximum_identity_delta": identity,
        "maximum_full_baseline_delta": full_delta,
        "maximum_null_baseline_delta": null_delta,
        "maximum_algebra_delta": algebra,
        "maximum_baseline_low_precision_ratio": baseline_ratio,
        "maximum_algebra_low_precision_ratio": algebra_ratio,
    }
