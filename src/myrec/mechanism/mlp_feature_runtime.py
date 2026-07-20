"""Resumable qrels-blind SwiGLU formation observation on the frozen 512 rows."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
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
    _assert_native_targets,
    _canonical_sha256,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.attention_edge_scoring import _aggregate_paths, _path_scores
from myrec.mechanism.attention_observation_runtime import (
    FIXED_BLOCKS,
    SAMPLE_MANIFEST,
    SAMPLE_MANIFEST_SHA256,
    SUPPORTED_METHODS,
    _audit_sample,
    _build_observation_paths,
)
from myrec.mechanism.mlp_feature_formation import (
    QwenMLPFeatureObserver,
    summarize_mlp_feature_pair,
)
from myrec.mechanism.postblock_sweep_runtime import NULL_BASELINE_DIRS
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def write_mlp_feature_observation_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    sample_manifest_path: str | Path = SAMPLE_MANIFEST,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_rows: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    block = int(block)
    if block not in FIXED_BLOCKS:
        raise ValueError("MLP formation block must be 13,20,or27")
    if not str(device).strip():
        raise ValueError("an explicit MLP formation device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("MLP formation wall time must be in (0,13500]")
    if max_rows is not None and not 0 < int(max_rows) < 512:
        raise ValueError("MLP formation smoke rows must be in [1,511]")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    sample_manifest_path = Path(sample_manifest_path)
    if sha256_file(sample_manifest_path) != SAMPLE_MANIFEST_SHA256:
        raise ValueError("MLP formation sample manifest differs")
    sample_manifest = _read_json(sample_manifest_path)
    sample_path = Path(sample_manifest["path"])
    if (
        sample_manifest.get("qrels_read") is not False
        or sample_manifest.get("model_scores_read") is not False
        or sample_manifest.get("selected_candidate_rows") != 512
        or sample_manifest.get("sha256") != sha256_file(sample_path)
    ):
        raise ValueError("MLP formation sample is not frozen qrels/score blind")
    samples = list(iter_jsonl(sample_path))
    records_path = standardized_dir / "records_dev.jsonl"
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("MLP formation requires frozen 8000-request dev")
    records_by_id = {record.request_id: record for record in records}
    _audit_sample(samples, records_by_id)
    target_samples = samples if max_rows is None else samples[: int(max_rows)]

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("MLP formation supports only Q2/Q3")
    frozen = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("MLP formation config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(model_dir, method_id)
    if checkpoint_id != frozen["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("MLP formation checkpoint differs from frozen binding")
    baseline_full, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    implementation = mlp_feature_runtime_implementation_identity()
    evidence_mode = (
        "exploratory_qrels_blind_fixed_grid_mlp_formation"
        if max_rows is None
        else "mechanical_smoke_non_result"
    )
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "sample_rows_sha256": sha256_file(sample_path),
        "target_rows": len(target_samples),
        "full_scores_sha256": full_identity["scores_sha256"],
        "null_scores_sha256": null_identity["scores_sha256"],
        "implementation_digest": implementation["digest"],
        "device": str(device),
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d4_mlp_feature_formation_extension",
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_sha256": sha256_file(records_path),
        "sample_manifest_path": str(sample_manifest_path),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "sample_rows_path": str(sample_path),
        "sample_rows_sha256": sha256_file(sample_path),
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "fixed_blocks": list(FIXED_BLOCKS),
        "feature_stages": ["gate_pre", "gate_activated", "up", "product"],
        "delta_decomposition": [
            "delta_gate_times_null_up",
            "null_gate_times_delta_up",
            "delta_gate_times_delta_up",
        ],
        "groups": 16,
        "group_seed": 20260718,
        "implementation_identity": implementation,
        "qrels_read": False,
        "source_test_opened": False,
        "evidence_mode": evidence_mode,
        "result_eligible": max_rows is None,
        "confirmatory_family_member": False,
        "layer_or_group_selection_authorized": False,
        "run_contract": contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    metadata, progress = _prepare(
        run_dir, metadata, contract_sha256, target_samples, resume=resume
    )
    completed = int(progress["completed_rows"])
    if completed >= len(target_samples):
        return _finalize(run_dir, metadata, progress, target_samples)

    started = time.monotonic()
    maximum_identity = float(metadata.get("maximum_score_identity_delta", 0.0))
    maximum_product_ratio = float(
        metadata.get("maximum_product_recomposition_low_precision_ratio", 0.0)
    )
    maximum_delta_error = float(
        metadata.get("maximum_delta_recomposition_abs_error", 0.0)
    )
    maximum_quantization_error = float(
        metadata.get("maximum_actual_product_quantization_abs_error", 0.0)
    )
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(
            config, device=str(device), training=False, checkpoint_model_dir=model_dir
        )
        model.eval()
        _assert_native_targets(tokenizer, method_id, manifest)
        metadata.update(_runtime_metadata(method_id, torch, transformers))
        metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", metadata)
        with torch.inference_mode(), QwenMLPFeatureObserver(model, block) as observer:
            for ordinal in range(completed, len(target_samples)):
                if time.monotonic() - started >= max_wall_seconds:
                    return _store_partial(
                        run_dir,
                        metadata,
                        started,
                        maximum_identity,
                        maximum_product_ratio,
                        maximum_delta_error,
                        maximum_quantization_error,
                    )
                sample = target_samples[ordinal]
                record = records_by_id[str(sample["request_id"])]
                candidate_ordinal = int(sample["candidate_ordinal"])
                candidate = record.candidates[candidate_ordinal]
                paths = _build_observation_paths(
                    tokenizer, record, candidate_ordinal, config, device=str(device)
                )
                full_scores, null_scores, descriptors = [], [], []
                path_rows = {}
                selected_batch_row = None
                for path in paths:
                    full_output, full_capture = _observe(model, observer, path["full"])
                    null_output, null_capture = _observe(model, observer, path["null"])
                    full_scores.append(_path_scores(full_output, path["full"]))
                    null_scores.append(_path_scores(null_output, path["null"]))
                    descriptors.append(path["full"])
                    selected = int(path["selected_batch_row"])
                    if selected_batch_row is None:
                        selected_batch_row = selected
                    elif selected_batch_row != selected:
                        raise ValueError("MLP formation selected row differs across paths")
                    maximum_product_ratio = max(
                        maximum_product_ratio,
                        float(full_capture["product_recomposition_low_precision_ratio"]),
                        float(null_capture["product_recomposition_low_precision_ratio"]),
                    )
                    summary = summarize_mlp_feature_pair(
                        {
                            name: value[selected]
                            for name, value in full_capture["captures"].items()
                        },
                        {
                            name: value[selected]
                            for name, value in null_capture["captures"].items()
                        },
                    )
                    maximum_delta_error = max(
                        maximum_delta_error,
                        float(summary["maximum_product_delta_recomposition_abs_error"]),
                    )
                    maximum_quantization_error = max(
                        maximum_quantization_error,
                        float(summary["maximum_actual_product_quantization_abs_error"]),
                    )
                    path_rows[path["name"]] = {
                        "capture_position_order": [
                            "query_end",
                            "history_summary_end",
                            "native_readout...",
                        ],
                        "summary": summary,
                    }
                if selected_batch_row is None:
                    raise ValueError("MLP formation produced no native path")
                observed_full = float(
                    _aggregate_paths(descriptors, full_scores)[selected_batch_row]
                )
                observed_null = float(
                    _aggregate_paths(descriptors, null_scores)[selected_batch_row]
                )
                key = (record.request_id, str(candidate["item_id"]))
                identity_delta = max(
                    abs(observed_full - baseline_full[key]),
                    abs(observed_null - baseline_null[key]),
                )
                maximum_identity = max(maximum_identity, identity_delta)
                if identity_delta > 1.0e-5:
                    raise ValueError("MLP formation no-op score identity failed")
                row = {
                    "ordinal": ordinal,
                    "request_id": record.request_id,
                    "candidate_item_id": str(candidate["item_id"]),
                    "candidate_ordinal": candidate_ordinal,
                    "selection_sha256": str(sample["selection_sha256"]),
                    "maximum_score_identity_delta": identity_delta,
                    "paths": path_rows,
                }
                _append_sync(run_dir / "rows.partial.jsonl", row)
                progress.update(
                    {
                        "completed_rows": ordinal + 1,
                        "last_selection_sha256": str(sample["selection_sha256"]),
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
                "maximum_score_identity_delta": maximum_identity,
                "maximum_product_recomposition_low_precision_ratio": maximum_product_ratio,
                "maximum_delta_recomposition_abs_error": maximum_delta_error,
                "maximum_actual_product_quantization_abs_error": (
                    maximum_quantization_error
                ),
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        raise
    metadata.update(
        {
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + time.monotonic()
            - started,
            "maximum_score_identity_delta": maximum_identity,
            "maximum_product_recomposition_low_precision_ratio": maximum_product_ratio,
            "maximum_delta_recomposition_abs_error": maximum_delta_error,
            "maximum_actual_product_quantization_abs_error": (
                maximum_quantization_error
            ),
        }
    )
    return _finalize(run_dir, metadata, progress, target_samples)


def _observe(model: Any, observer: QwenMLPFeatureObserver, path: dict[str, Any]):
    observer.arm(path["capture_positions"], sequence_length=path["ids"].shape[1])
    output = model(
        input_ids=path["ids"],
        attention_mask=path["mask"],
        use_cache=False,
        logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
    )
    return output, observer.disarm()


def mlp_feature_runtime_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/mlp_feature_formation.py",
        "src/myrec/mechanism/mlp_feature_runtime.py",
        "src/myrec/mechanism/attention_observation_runtime.py",
        "scripts/observe_deep_dive_mlp_features.py",
    )
    files = [
        {"path": path, "sha256": sha256_file(root / path), "size_bytes": (root / path).stat().st_size}
        for path in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}


def _prepare(run_dir, metadata, contract_sha256, samples, *, resume):
    partial = run_dir / "rows.partial.jsonl"
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"MLP formation run is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial.touch(exist_ok=False)
        metadata.update({"elapsed_seconds": 0.0, "resumable": True, "resume_lineage": []})
        progress = {
            "schema_version": 1,
            "run_contract_sha256": contract_sha256,
            "completed_rows": 0,
            "last_selection_sha256": None,
            "partial_sha256": sha256_file(partial),
            "status": "initializing",
            "updated_at": _utc_now(),
        }
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "progress.json", progress)
        return metadata, progress
    stored = _read_json(run_dir / "metadata.json")
    progress = _read_json(run_dir / "progress.json")
    if stored.get("run_contract_sha256") != contract_sha256 or progress.get("run_contract_sha256") != contract_sha256:
        raise ValueError("MLP formation resume contract drift")
    observed = _audit_partial(partial, samples)
    if progress.get("completed_rows") != observed["completed_rows"] or progress.get("partial_sha256") != observed["partial_sha256"]:
        raise ValueError("MLP formation progress differs")
    lineage = list(stored.get("resume_lineage", []))
    lineage.append(observed)
    stored["resume_lineage"] = lineage
    _write_json(run_dir / "metadata.json", stored)
    return stored, progress


def _audit_partial(path: Path, samples) -> dict[str, Any]:
    count = 0
    for row in iter_jsonl(path):
        if (
            count >= len(samples)
            or row.get("ordinal") != count
            or row.get("selection_sha256") != samples[count]["selection_sha256"]
        ):
            raise ValueError("MLP formation partial identity differs")
        count += 1
    return {"completed_rows": count, "partial_sha256": sha256_file(path)}


def _store_partial(
    run_dir,
    metadata,
    started,
    identity,
    product_ratio,
    delta_error,
    quantization_error,
):
    metadata.update(
        {
            "status": "wall_time_exhausted",
            "resumable": True,
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + time.monotonic()
            - started,
            "maximum_score_identity_delta": identity,
            "maximum_product_recomposition_low_precision_ratio": product_ratio,
            "maximum_delta_recomposition_abs_error": delta_error,
            "maximum_actual_product_quantization_abs_error": quantization_error,
        }
    )
    _write_json(run_dir / "metadata.json", metadata)
    return metadata


def _finalize(run_dir, metadata, progress, samples):
    partial = run_dir / "rows.partial.jsonl"
    observed = _audit_partial(partial, samples)
    if observed["completed_rows"] != len(samples):
        raise ValueError("cannot finalize incomplete MLP formation observations")
    rows_path = run_dir / "rows.jsonl"
    os.replace(partial, rows_path)
    progress["status"] = "completed"
    metadata.update(
        {
            "status": "completed",
            "resumable": False,
            "observation_rows": len(samples),
            "rows_path": str(rows_path),
            "rows_sha256": sha256_file(rows_path),
            "complete_finite_observation_coverage": True,
        }
    )
    _write_json(run_dir / "progress.json", progress)
    _write_json(run_dir / "metadata.json", metadata)
    return metadata


def _append_sync(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
