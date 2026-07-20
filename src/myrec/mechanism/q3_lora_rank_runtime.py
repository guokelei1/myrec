"""Resumable qrels-blind N10 Q3 LoRA rank-path scorer."""

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
    _canonical_sha256,
    _load_content_controls,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.postblock_sweep_runtime import NULL_BASELINE_DIRS
from myrec.mechanism.q3_lora_rank_scoring import (
    LORA_PATH_CONDITIONS,
    score_q3_lora_rank_chunk,
)
from myrec.mechanism.representation_probe import normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


N10_RANK_MANIFEST_PATH = Path("experiments/motivation/transformer_n10_q3_lora_rank_manifest_v1.yaml")
N10_RANK_MANIFEST_SHA256 = "10373ff6ad55d0e23af739c479dce0d9748af66de45d519aa61344f4919e23f6"
MAX_WALL_SECONDS = 13_500.0


def write_q3_lora_rank_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = N10_RANK_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("N10 Q3 LoRA rank scorer requires an explicit device")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("N10 Q3 LoRA max_wall_seconds must be in (0,13500]")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("N10 Q3 LoRA max_requests must be positive")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    rank_manifest = _load_rank_manifest(manifest_path)
    deep_manifest = _load_manifest(DEEP_DIVE_MANIFEST_PATH)
    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, deep_manifest["frozen_inputs"]["records_dev_sha256"]),
        (standardized_dir / "manifest.json", deep_manifest["frozen_inputs"]["dataset_manifest_sha256"]),
        (standardized_dir / "request_manifest.json", deep_manifest["frozen_inputs"]["request_manifest_sha256"]),
        (standardized_dir / "candidate_manifest.json", deep_manifest["frozen_inputs"]["candidate_manifest_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"N10 Q3 LoRA frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("N10 Q3 LoRA scorer requires 8000 requests")
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id != "q3_tallrec_generalqwen":
        raise ValueError("N10 Q3 LoRA scorer received another model")
    frozen_model = deep_manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("N10 Q3 LoRA config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(model_dir, method_id)
    if checkpoint_id != frozen_model["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("N10 Q3 LoRA checkpoint differs from frozen binding")
    controls, control_identity = _load_content_controls(deep_manifest, method_id, records)
    baseline_full, full_identity = _load_frozen_baseline(BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records)
    baseline_null, null_identity = _load_frozen_baseline(NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records)
    fold1_records = [record for record in records if normalized_query_fold(record.query) == 1]
    target_records = fold1_records if max_requests is None else fold1_records[: int(max_requests)]
    implementation = lora_rank_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "n10_rank_manifest_sha256": rank_manifest["_sha256"],
        "deep_dive_manifest_sha256": deep_manifest["_sha256"],
        "score_conditions": list(LORA_PATH_CONDITIONS),
        "target_requests": len(target_records),
        "normalized_query_fold": 1,
        "device": str(device),
        "evidence_mode": "registered_n10_q3_lora_rank_path" if max_requests is None else "mechanical_smoke_non_result",
        "implementation_digest": implementation["digest"],
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_n10_q3_lora_rank_path",
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "n10_rank_manifest_path": str(manifest_path),
        "n10_rank_manifest_sha256": rank_manifest["_sha256"],
        "deep_dive_manifest_sha256": deep_manifest["_sha256"],
        "content_neutral_control": control_identity,
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "score_conditions": list(LORA_PATH_CONDITIONS),
        "lora_rank": 8,
        "lora_paths": 56,
        "qrels_read": False,
        "source_test_opened": False,
        "result_eligible": max_requests is None,
        "complete_finite_score_coverage_required": True,
        "run_contract": contract,
        "run_contract_sha256": contract_sha256,
        "implementation_identity": implementation,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    prepared = prepare_scalar_bundle(
        run_dir,
        metadata=metadata,
        contract_sha256=contract_sha256,
        records=target_records,
        conditions=LORA_PATH_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    max_full_delta = float(prepared.metadata.get("maximum_full_baseline_delta", 0.0))
    max_null_delta = float(prepared.metadata.get("maximum_null_baseline_delta", 0.0))
    eligible_completed = int(prepared.metadata.get("eligible_requests_completed", 0))
    if completed >= len(target_records):
        return finalize_scalar_bundle(run_dir, prepared, target_records, LORA_PATH_CONDITIONS, maximum_identity_delta=max(max_full_delta, max_null_delta))

    started = time.monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(config, device=str(device), training=False, checkpoint_model_dir=model_dir)
        model.eval()
        prepared.metadata.update(_runtime_metadata(method_id, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    prepared.metadata.update({"status": "wall_time_exhausted", "resumable": True, "completed_requests": ordinal, "maximum_full_baseline_delta": max_full_delta, "maximum_null_baseline_delta": max_null_delta, "eligible_requests_completed": eligible_completed})
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                control = controls[record.request_id]
                rows: list[dict[str, Any]] = []
                if control["eligible"] is True:
                    eligible_completed += 1
                    full_rows: dict[str, list[float]] = {condition: [] for condition in LORA_PATH_CONDITIONS if condition != "baseline_null"}
                    for start in range(0, len(record.candidates), batch_size):
                        candidates = list(record.candidates[start : start + batch_size])
                        full = score_q3_lora_rank_chunk(model, tokenizer, record, candidates, config, history=record.history, device=str(device))
                        for condition, values in full.items():
                            full_rows[condition].extend(float(value) for value in values)
                        null_context = __import__("myrec.mechanism.deep_dive_native_patch", fromlist=["_q3_context"])._q3_context(tokenizer, record, candidates, [], config, str(device))
                        from myrec.mechanism.q3_native_readout_scoring import capture_q3_native_readout
                        null_values = capture_q3_native_readout(model, null_context)["score"].detach().float().cpu().numpy()
                        if start == 0:
                            null_rows: list[float] = []
                        null_rows.extend(float(value) for value in null_values)
                    values_by_condition = {condition: full_rows[condition] for condition in full_rows}
                    values_by_condition["baseline_null"] = null_rows
                    for candidate_ordinal, candidate in enumerate(record.candidates):
                        values = {condition: float(values_by_condition[condition][candidate_ordinal]) for condition in LORA_PATH_CONDITIONS}
                        key = (record.request_id, str(candidate["item_id"]))
                        max_full_delta = max(max_full_delta, abs(values["baseline_full"] - baseline_full[key]))
                        max_null_delta = max(max_null_delta, abs(values["baseline_null"] - baseline_null[key]))
                        rows.append({"request_id": record.request_id, "candidate_item_id": str(candidate["item_id"]), "candidate_ordinal": candidate_ordinal, "conditions": values})
                else:
                    for candidate_ordinal, candidate in enumerate(record.candidates):
                        key = (record.request_id, str(candidate["item_id"]))
                        full = baseline_full[key]
                        null = baseline_null[key]
                        rows.append({"request_id": record.request_id, "candidate_item_id": str(candidate["item_id"]), "candidate_ordinal": candidate_ordinal, "conditions": {condition: (null if condition == "baseline_null" else full) for condition in LORA_PATH_CONDITIONS}})
                append_scalar_request(run_dir, {"ordinal": ordinal, "request_id": record.request_id, "rows": rows, "rows_sha256": _canonical_sha256(rows)}, prepared)
    except Exception as exc:
        prepared.metadata.update({"status": "mechanical_failure", "resumable": True, "qrels_read": False, "error": {"type": type(exc).__name__, "message": str(exc)}, "maximum_full_baseline_delta": max_full_delta, "maximum_null_baseline_delta": max_null_delta, "eligible_requests_completed": eligible_completed})
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    prepared.metadata.update({"elapsed_seconds": float(prepared.metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started, "maximum_full_baseline_delta": max_full_delta, "maximum_null_baseline_delta": max_null_delta, "eligible_requests_completed": eligible_completed})
    return finalize_scalar_bundle(run_dir, prepared, target_records, LORA_PATH_CONDITIONS, maximum_identity_delta=max(max_full_delta, max_null_delta))


def _load_rank_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    observed = sha256_file(path)
    if observed != N10_RANK_MANIFEST_SHA256:
        raise ValueError("N10 Q3 LoRA rank manifest differs from frozen digest")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("N10 Q3 LoRA rank manifest is not a mapping")
    value["_sha256"] = observed
    return value


def lora_rank_implementation_identity() -> dict[str, Any]:
    from myrec.mechanism.q3_lora_rank_scoring import lora_rank_implementation_identity as identity

    return identity()
