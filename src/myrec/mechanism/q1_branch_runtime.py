"""Resumable full-dev Q1 listwise KV-cache branch causal scorer."""

from __future__ import annotations

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
    DEEP_DIVE_MANIFEST_PATH,
    MAX_WALL_SECONDS,
    _canonical_sha256,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.q1_branch_scoring import (
    Q1_BRANCH_CONDITIONS,
    score_q1_branch_record,
)
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


Q1_METHOD_ID = "q1_instructrec_generalqwen"
Q1_FULL_BASELINE = Path(
    "runs/20260717_kuaisearch_q1_instructrec_generalqwen_internal_dev_full_score"
)
Q1_NULL_BASELINE = Path(
    "runs/20260717_kuaisearch_q1_instructrec_generalqwen_internal_dev_null_score"
)


def write_q1_branch_bundle(
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
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    block = int(block)
    if block not in (13, 20, 27):
        raise ValueError("Q1 branch block must be 13,20,or27")
    if not str(device).strip():
        raise ValueError("an explicit Q1 branch device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("Q1 branch wall time must be in (0,13500]")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("Q1 branch smoke request cap must be positive")
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    records_path = standardized_dir / "records_dev.jsonl"
    if sha256_file(records_path) != manifest["frozen_inputs"]["records_dev_sha256"]:
        raise ValueError("Q1 branch records hash differs")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("Q1 branch requires full 8000-request dev")
    config = load_v12_ranker_config(config_path)
    if config["method_id"] != Q1_METHOD_ID:
        raise ValueError("Q1 branch received another model")
    frozen = manifest["frozen_inputs"]["models"][Q1_METHOD_ID]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("Q1 branch config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(model_dir, Q1_METHOD_ID)
    if checkpoint_id != frozen["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("Q1 branch checkpoint differs from frozen binding")
    baseline_full, full_identity = _load_frozen_baseline(
        Q1_FULL_BASELINE, Q1_METHOD_ID, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        Q1_NULL_BASELINE, Q1_METHOD_ID, checkpoint_id, records
    )
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = "registered_mechanism_diagnostic" if max_requests is None else "mechanical_smoke_non_result"
    implementation = q1_branch_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": Q1_METHOD_ID,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "full_scores_sha256": full_identity["scores_sha256"],
        "null_scores_sha256": null_identity["scores_sha256"],
        "conditions": list(Q1_BRANCH_CONDITIONS),
        "target_requests": len(target_records),
        "device": str(device),
        "implementation_digest": implementation["digest"],
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d6_q1_branch_extension",
        "run_id": run_id,
        "method_id": Q1_METHOD_ID,
        "block_zero_based": block,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "score_conditions": list(Q1_BRANCH_CONDITIONS),
        "native_scoring": "single listwise prefix cache plus complete per-candidate multi-token cached continuation",
        "patch_scope": "prompt readout and every response token",
        "implementation_identity": implementation,
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
    prepared = prepare_scalar_bundle(
        run_dir, metadata=metadata, contract_sha256=contract_sha256,
        records=target_records, conditions=Q1_BRANCH_CONDITIONS, resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    identity = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    baseline_ratio = float(prepared.metadata.get("maximum_baseline_low_precision_ratio", 0.0))
    response_tokens = int(prepared.metadata.get("response_tokens_processed", 0))
    if completed >= len(target_records):
        _assert_baseline_bound(baseline_ratio)
        return finalize_scalar_bundle(
            run_dir, prepared, target_records, Q1_BRANCH_CONDITIONS,
            maximum_identity_delta=identity,
        )
    started = time.monotonic()
    try:
        import torch
        import transformers
        tokenizer, model = _load_model_and_tokenizer(
            config, device=str(device), training=False, checkpoint_model_dir=model_dir
        )
        model.eval()
        prepared.metadata.update(_runtime_metadata(Q1_METHOD_ID, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    _progress(prepared.metadata, started, identity, baseline_ratio, response_tokens, "wall_time_exhausted")
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                result = score_q1_branch_record(
                    model, tokenizer, record, config, block=block,
                    device=str(device), batch_size=batch_size,
                )
                identity = max(identity, float(result["maximum_identity_delta"]))
                response_tokens += int(result["response_tokens"])
                rows = []
                for candidate_ordinal, candidate in enumerate(record.candidates):
                    values = {
                        name: float(result["conditions"][name][candidate_ordinal])
                        for name in Q1_BRANCH_CONDITIONS
                    }
                    key = (record.request_id, str(candidate["item_id"]))
                    for name, reference in (
                        ("baseline_full", baseline_full[key]),
                        ("baseline_null", baseline_null[key]),
                    ):
                        bound = 8.0 * (2.0**-7) * max(1.0, abs(reference))
                        baseline_ratio = max(
                            baseline_ratio, abs(values[name] - reference) / bound
                        )
                    if not all(math.isfinite(value) for value in values.values()):
                        raise FloatingPointError("Q1 branch score is non-finite")
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
                        "response_tokens": int(result["response_tokens"]),
                        "call_audit": result["call_audit"],
                        "rows": rows,
                        "rows_sha256": _canonical_sha256(rows),
                    },
                    prepared,
                )
    except Exception as exc:
        _progress(prepared.metadata, started, identity, baseline_ratio, response_tokens, "mechanical_failure")
        prepared.metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    _assert_baseline_bound(baseline_ratio)
    _progress(prepared.metadata, started, identity, baseline_ratio, response_tokens, "running")
    return finalize_scalar_bundle(
        run_dir, prepared, target_records, Q1_BRANCH_CONDITIONS,
        maximum_identity_delta=identity,
    )


def _assert_baseline_bound(ratio):
    if ratio > 1.0:
        raise ValueError(f"Q1 branch baseline BF16 bound failed: {ratio}")


def _progress(metadata, started, identity, baseline_ratio, response_tokens, status):
    metadata.update({
        "status": status,
        "resumable": status != "running",
        "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started,
        "maximum_identity_delta": identity,
        "maximum_baseline_low_precision_ratio": baseline_ratio,
        "response_tokens_processed": response_tokens,
    })


def q1_branch_implementation_identity():
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/q1_branch_runtime.py",
        "src/myrec/mechanism/q1_branch_scoring.py",
        "src/myrec/mechanism/q1_kv_trajectory.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "scripts/score_deep_dive_q1_branches.py",
    )
    files = [{"path": path, "sha256": sha256_file(root / path), "size_bytes": (root / path).stat().st_size} for path in paths]
    return {"files": files, "digest": _canonical_sha256(files)}
