"""Resumable qrels-blind Q1 full/null KV-cache all-layer trajectory runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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
    MAX_WALL_SECONDS,
    _canonical_sha256,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
)
from myrec.mechanism.q0_trajectory_evaluator import GEOMETRY_METRICS
from myrec.mechanism.q1_kv_trajectory import (
    Q1AllStateCapture,
    Q1_METHOD_ID,
    capture_q1_request_trajectory,
    compare_q1_full_null,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


Q1_FULL_BASELINE_DIR = Path(
    "runs/20260717_kuaisearch_q1_instructrec_generalqwen_internal_dev_full_score"
)
Q1_NULL_BASELINE_DIR = Path(
    "runs/20260717_kuaisearch_q1_instructrec_generalqwen_internal_dev_null_score"
)
DEFAULT_REQUESTS_PER_SHARD = 8


def write_q1_kv_trajectory_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = "experiments/motivation/transformer_deep_dive_manifest.yaml",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    requests_per_shard: int = DEFAULT_REQUESTS_PER_SHARD,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run exact Q1 cache scoring while retaining scalar all-layer geometry."""

    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("an explicit Q1 trajectory device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("Q1 trajectory max_wall_seconds must be in (0,13500]")
    if int(requests_per_shard) <= 0:
        raise ValueError("Q1 trajectory requests_per_shard must be positive")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("Q1 trajectory max_requests must be positive")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    config = load_v12_ranker_config(config_path)
    if config["method_id"] != Q1_METHOD_ID:
        raise ValueError("Q1 trajectory runtime received another model")
    frozen = manifest["frozen_inputs"]["models"][Q1_METHOD_ID]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("Q1 trajectory config differs from frozen manifest")
    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, manifest["frozen_inputs"]["records_dev_sha256"]),
        (standardized_dir / "manifest.json", manifest["frozen_inputs"]["dataset_manifest_sha256"]),
        (standardized_dir / "request_manifest.json", manifest["frozen_inputs"]["request_manifest_sha256"]),
        (standardized_dir / "candidate_manifest.json", manifest["frozen_inputs"]["candidate_manifest_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"Q1 trajectory frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("Q1 trajectory requires all 8000 dev requests")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, Q1_METHOD_ID
    )
    if checkpoint_id != frozen["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("Q1 trajectory checkpoint differs from frozen binding")
    full_baseline, full_identity = _load_frozen_baseline(
        Q1_FULL_BASELINE_DIR, Q1_METHOD_ID, checkpoint_id, records
    )
    null_baseline, null_identity = _load_frozen_baseline(
        Q1_NULL_BASELINE_DIR, Q1_METHOD_ID, checkpoint_id, records
    )
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = (
        "registered_descriptive_breadth"
        if max_requests is None
        else "mechanical_smoke_non_result"
    )
    implementation = q1_trajectory_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": Q1_METHOD_ID,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "full_scores_sha256": full_identity["scores_sha256"],
        "null_scores_sha256": null_identity["scores_sha256"],
        "target_request_ids_sha256": _canonical_sha256(
            [record.request_id for record in target_records]
        ),
        "target_requests": len(target_records),
        "requests_per_shard": int(requests_per_shard),
        "device": str(device),
        "implementation_digest": implementation["digest"],
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d6_q1_kv_trajectory",
        "run_id": run_id,
        "method_id": Q1_METHOD_ID,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "hidden_state_indices": list(range(29)),
        "request_positions": ["query_end", "history_summary_end"],
        "native_readout_phases": ["shared_prefix_last", "all_cached_response_tokens"],
        "geometry_metrics": list(GEOMETRY_METRICS),
        "scoring_execution": "unchanged shared-prefix KV-cache with complete candidate slate",
        "implementation_identity": implementation,
        "identity_tolerance": 1.0e-5,
        "qrels_read": False,
        "source_test_opened": False,
        "evidence_mode": evidence_mode,
        "result_eligible": max_requests is None,
        "run_contract": contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    metadata, index = _prepare(
        run_dir, metadata, contract_sha256, target_records, resume=resume
    )
    completed = int(index["request_count"])
    if completed >= len(target_records):
        return _finalize(run_dir, metadata, index, target_records)
    maximum_full_delta = float(metadata.get("maximum_full_baseline_delta", 0.0))
    maximum_null_delta = float(metadata.get("maximum_null_baseline_delta", 0.0))
    maximum_ratio = float(metadata.get("maximum_baseline_low_precision_ratio", 0.0))
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
        metadata.update(_runtime_metadata(Q1_METHOD_ID, torch, transformers))
        metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 2))
        with torch.inference_mode(), Q1AllStateCapture(model) as capture:
            while completed < len(target_records):
                if time.monotonic() - started >= max_wall_seconds:
                    _store_progress(
                        metadata,
                        started,
                        completed,
                        maximum_full_delta,
                        maximum_null_delta,
                        maximum_ratio,
                        status="wall_time_exhausted",
                    )
                    _write_json(run_dir / "metadata.json", metadata)
                    return metadata
                stop = min(completed + int(requests_per_shard), len(target_records))
                shard_records = target_records[completed:stop]
                request_ids = []
                request_geometry = []
                prompt_geometry = []
                candidate_ids = []
                candidate_offsets = [0]
                target_lengths = []
                full_scores = []
                null_scores = []
                response_geometry = []
                first_geometry = []
                continuation_geometry = []
                phase_calls = {"prefix": 0, "cached_continuation": 0}
                phase_shape_hasher = hashlib.sha256()
                for record in shard_records:
                    capture.call_shapes.clear()
                    full = capture_q1_request_trajectory(
                        model,
                        capture,
                        tokenizer,
                        record,
                        record.history,
                        config,
                        device=str(device),
                        batch_size=batch_size,
                    )
                    null = capture_q1_request_trajectory(
                        model,
                        capture,
                        tokenizer,
                        record,
                        [],
                        config,
                        device=str(device),
                        batch_size=batch_size,
                    )
                    expected_continuations = 2 * math.ceil(
                        len(record.candidates) / batch_size
                    )
                    observed_prefix = sum(
                        row["phase"] == "prefix" for row in capture.call_shapes
                    )
                    observed_continuation = sum(
                        row["phase"] == "cached_continuation"
                        for row in capture.call_shapes
                    )
                    if observed_prefix != 2 or observed_continuation != expected_continuations:
                        raise RuntimeError("Q1 prefix/continuation call accounting drifted")
                    phase_calls["prefix"] += observed_prefix
                    phase_calls["cached_continuation"] += observed_continuation
                    phase_shape_hasher.update(
                        json.dumps(
                            capture.call_shapes,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    )
                    comparison = compare_q1_full_null(full, null)
                    request_ids.append(record.request_id)
                    request_geometry.append(
                        _stack_geometry(comparison["request_geometry"])
                    )
                    prompt_geometry.append(
                        _stack_geometry(comparison["prompt_readout_geometry"])
                    )
                    for row in comparison["candidates"]:
                        item_id = str(row["candidate_item_id"])
                        key = (record.request_id, item_id)
                        maximum_full_delta = max(
                            maximum_full_delta,
                            abs(float(row["full_score"]) - full_baseline[key]),
                        )
                        maximum_null_delta = max(
                            maximum_null_delta,
                            abs(float(row["null_score"]) - null_baseline[key]),
                        )
                        for value, reference in (
                            (float(row["full_score"]), full_baseline[key]),
                            (float(row["null_score"]), null_baseline[key]),
                        ):
                            bound = 4.0 * (2.0**-7) * max(1.0, abs(reference))
                            maximum_ratio = max(
                                maximum_ratio, abs(value - reference) / bound
                            )
                        candidate_ids.append(item_id)
                        target_lengths.append(int(row["target_length"]))
                        full_scores.append(float(row["full_score"]))
                        null_scores.append(float(row["null_score"]))
                        response_geometry.append(
                            _stack_geometry(row["mean_token_geometry"])
                        )
                        first_geometry.append(
                            _stack_geometry(row["first_token_geometry"])
                        )
                        continuation_geometry.append(
                            _stack_geometry(row["continuation_geometry"])
                        )
                    candidate_offsets.append(len(candidate_ids))
                shard_name = f"shard_{len(index['shards']):05d}.npz"
                shard = _write_shard(
                    run_dir / "shards" / shard_name,
                    request_ids=request_ids,
                    request_geometry=np.stack(request_geometry),
                    prompt_readout_geometry=np.stack(prompt_geometry),
                    candidate_offsets=np.asarray(candidate_offsets, dtype=np.int64),
                    candidate_ids=candidate_ids,
                    target_lengths=np.asarray(target_lengths, dtype=np.int16),
                    full_scores=np.asarray(full_scores, dtype=np.float32),
                    null_scores=np.asarray(null_scores, dtype=np.float32),
                    response_geometry=np.stack(response_geometry),
                    first_token_geometry=np.stack(first_geometry),
                    continuation_geometry=np.stack(continuation_geometry),
                )
                shard.update(
                    {
                        "start_request_ordinal": completed,
                        "phase_calls": phase_calls,
                        "phase_shapes_sha256": phase_shape_hasher.hexdigest(),
                    }
                )
                index["shards"].append(shard)
                index["request_count"] += len(shard_records)
                index["candidate_count"] += len(candidate_ids)
                index["response_tokens"] += int(sum(target_lengths))
                index["prefix_calls"] += phase_calls["prefix"]
                index["continuation_calls"] += phase_calls["cached_continuation"]
                index["rolling_shards_sha256"] = _rolling_digest(
                    index["rolling_shards_sha256"], shard["sha256"]
                )
                completed = stop
                _write_json(run_dir / "index.partial.json", index)
                _write_json(
                    run_dir / "progress.json",
                    {
                        "schema_version": 1,
                        "run_contract_sha256": contract_sha256,
                        "completed_requests": completed,
                        "completed_candidates": index["candidate_count"],
                        "completed_response_tokens": index["response_tokens"],
                        "last_request_id": request_ids[-1],
                        "partial_index_sha256": sha256_file(
                            run_dir / "index.partial.json"
                        ),
                        "status": "running",
                        "updated_at": _utc_now(),
                    },
                )
    except Exception as exc:
        _store_progress(
            metadata,
            started,
            completed,
            maximum_full_delta,
            maximum_null_delta,
            maximum_ratio,
            status="mechanical_failure",
        )
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(run_dir / "metadata.json", metadata)
        raise
    _store_progress(
        metadata,
        started,
        completed,
        maximum_full_delta,
        maximum_null_delta,
        maximum_ratio,
        status="running",
    )
    return _finalize(run_dir, metadata, index, target_records)


def _stack_geometry(value: Mapping[str, Any]) -> np.ndarray:
    result = np.stack(
        [np.asarray(value[metric], dtype=np.float32) for metric in GEOMETRY_METRICS],
        axis=-1,
    )
    if not np.isfinite(result).all():
        raise FloatingPointError("Q1 geometry contains a non-finite scalar")
    return result


def _write_shard(path: Path, **arrays: Any) -> dict[str, Any]:
    request_ids = list(arrays.pop("request_ids"))
    candidate_ids = list(arrays.pop("candidate_ids"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing.npz")
    np.savez(
        temporary,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        candidate_ids=np.asarray(candidate_ids, dtype=np.str_),
        geometry_metrics=np.asarray(GEOMETRY_METRICS, dtype=np.str_),
        hidden_state_indices=np.arange(29, dtype=np.int16),
        **arrays,
    )
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "request_count": len(request_ids),
        "candidate_count": len(candidate_ids),
        "first_request_id": request_ids[0],
        "last_request_id": request_ids[-1],
    }


def _prepare(
    run_dir: Path,
    metadata: dict[str, Any],
    contract_sha256: str,
    records: Sequence[Any],
    *,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Q1 trajectory run is not empty: {run_dir}")
        (run_dir / "shards").mkdir(parents=True, exist_ok=True)
        index = {
            "schema_version": 1,
            "run_contract_sha256": contract_sha256,
            "request_count": 0,
            "candidate_count": 0,
            "response_tokens": 0,
            "prefix_calls": 0,
            "continuation_calls": 0,
            "rolling_shards_sha256": hashlib.sha256(b"").hexdigest(),
            "shards": [],
        }
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "index.partial.json", index)
        _write_json(
            run_dir / "progress.json",
            {
                "schema_version": 1,
                "run_contract_sha256": contract_sha256,
                "completed_requests": 0,
                "completed_candidates": 0,
                "completed_response_tokens": 0,
                "last_request_id": None,
                "partial_index_sha256": sha256_file(run_dir / "index.partial.json"),
                "status": "initializing",
                "updated_at": _utc_now(),
            },
        )
        return metadata, index
    stored = _read_json(run_dir / "metadata.json")
    index = _read_json(run_dir / "index.partial.json")
    progress = _read_json(run_dir / "progress.json")
    for payload in (stored, index, progress):
        if payload.get("run_contract_sha256") != contract_sha256:
            raise ValueError("Q1 trajectory resume contract drift")
    _audit_partial(run_dir, index, records)
    if progress.get("partial_index_sha256") != sha256_file(
        run_dir / "index.partial.json"
    ):
        raise ValueError("Q1 trajectory progress hash drift")
    lineage = list(stored.get("resume_lineage", []))
    lineage.append(
        {
            "resumed_at": _utc_now(),
            "completed_requests": index["request_count"],
            "rolling_shards_sha256": index["rolling_shards_sha256"],
        }
    )
    stored.update({"resume_lineage": lineage, "status": "initializing"})
    _write_json(run_dir / "metadata.json", stored)
    return stored, index


def _audit_partial(run_dir: Path, index: Mapping[str, Any], records: Sequence[Any]) -> None:
    observed_requests = 0
    observed_candidates = 0
    observed_tokens = 0
    rolling = hashlib.sha256(b"").hexdigest()
    for shard in index.get("shards", []):
        path = run_dir / "shards" / str(shard["path"])
        if sha256_file(path) != shard.get("sha256"):
            raise ValueError("Q1 trajectory shard hash mismatch")
        with np.load(path, allow_pickle=False) as payload:
            request_ids = [str(value) for value in payload["request_ids"].tolist()]
            candidate_ids = [str(value) for value in payload["candidate_ids"].tolist()]
            offsets = np.asarray(payload["candidate_offsets"], dtype=np.int64)
            target_lengths = np.asarray(payload["target_lengths"], dtype=np.int64)
            expected = records[observed_requests : observed_requests + len(request_ids)]
            if request_ids != [record.request_id for record in expected]:
                raise ValueError("Q1 trajectory request order drift")
            expected_candidates = [
                str(candidate["item_id"]) for record in expected for candidate in record.candidates
            ]
            if candidate_ids != expected_candidates or int(offsets[-1]) != len(candidate_ids):
                raise ValueError("Q1 trajectory candidate coverage drift")
            if payload["geometry_metrics"].tolist() != list(GEOMETRY_METRICS):
                raise ValueError("Q1 trajectory geometry metric order drift")
            for name in (
                "request_geometry",
                "prompt_readout_geometry",
                "response_geometry",
                "first_token_geometry",
                "continuation_geometry",
                "full_scores",
                "null_scores",
            ):
                if not np.isfinite(payload[name]).all():
                    raise FloatingPointError("Q1 trajectory shard is non-finite")
        observed_requests += len(request_ids)
        observed_candidates += len(candidate_ids)
        observed_tokens += int(target_lengths.sum())
        rolling = _rolling_digest(rolling, str(shard["sha256"]))
    if (
        observed_requests != int(index["request_count"])
        or observed_candidates != int(index["candidate_count"])
        or observed_tokens != int(index["response_tokens"])
        or rolling != index["rolling_shards_sha256"]
    ):
        raise ValueError("Q1 trajectory partial index counters drifted")


def _finalize(
    run_dir: Path,
    metadata: dict[str, Any],
    index: dict[str, Any],
    records: Sequence[Any],
) -> dict[str, Any]:
    _audit_partial(run_dir, index, records)
    expected_candidates = sum(len(record.candidates) for record in records)
    if index["request_count"] != len(records) or index["candidate_count"] != expected_candidates:
        raise ValueError("cannot finalize incomplete Q1 trajectory coverage")
    maximum = max(
        float(metadata.get("maximum_full_baseline_delta", math.inf)),
        float(metadata.get("maximum_null_baseline_delta", math.inf)),
    )
    if maximum > 1.0e-5:
        raise ValueError(f"Q1 frozen native-score identity failed: {maximum}")
    index_path = run_dir / "index.json"
    os.replace(run_dir / "index.partial.json", index_path)
    metadata.update(
        {
            "status": "completed",
            "resumable": False,
            "request_count": index["request_count"],
            "candidate_count": index["candidate_count"],
            "response_tokens": index["response_tokens"],
            "prefix_calls": index["prefix_calls"],
            "continuation_calls": index["continuation_calls"],
            "index_path": str(index_path),
            "index_sha256": sha256_file(index_path),
            "complete_finite_trajectory_coverage": True,
            "native_score_identity_passed": True,
            "maximum_native_score_identity_delta": maximum,
        }
    )
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "progress.json",
        {
            "schema_version": 1,
            "run_contract_sha256": metadata["run_contract_sha256"],
            "completed_requests": index["request_count"],
            "completed_candidates": index["candidate_count"],
            "completed_response_tokens": index["response_tokens"],
            "last_request_id": records[-1].request_id,
            "index_sha256": metadata["index_sha256"],
            "status": "completed",
            "updated_at": _utc_now(),
        },
    )
    return metadata


def _store_progress(
    metadata: dict[str, Any],
    started: float,
    completed: int,
    full_delta: float,
    null_delta: float,
    ratio: float,
    *,
    status: str,
) -> None:
    metadata.update(
        {
            "status": status,
            "resumable": status != "running",
            "completed_requests": completed,
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + time.monotonic()
            - started,
            "maximum_full_baseline_delta": full_delta,
            "maximum_null_baseline_delta": null_delta,
            "maximum_baseline_low_precision_ratio": ratio,
        }
    )


def q1_trajectory_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/q1_trajectory_runtime.py",
        "src/myrec/mechanism/q1_kv_trajectory.py",
        "src/myrec/mechanism/q0_trajectory_evaluator.py",
        "src/myrec/mechanism/representation_runtime.py",
        "src/myrec/baselines/motivation_v12_ranker.py",
        "scripts/score_deep_dive_q1_trajectory.py",
    )
    files = [
        {"path": path, "sha256": sha256_file(root / path), "size_bytes": (root / path).stat().st_size}
        for path in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}


def _rolling_digest(previous: str, current: str) -> str:
    return sha256_text(f"{previous}\0{current}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
