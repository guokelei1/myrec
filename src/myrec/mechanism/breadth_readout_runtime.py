"""Resumable native final-RMSNorm/readout causal scoring for Q0 and Q1."""

from __future__ import annotations

import hashlib
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
    _single_token_id,
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
from myrec.mechanism.breadth_readout_scoring import (
    BREADTH_READOUT_CONDITIONS,
    BREADTH_READOUT_NODES,
    score_q0_readout_chunk,
    score_q1_readout_record,
)
from myrec.mechanism.q0_branch_runtime import Q0_FULL_BASELINE, Q0_NULL_BASELINE
from myrec.mechanism.q1_branch_runtime import Q1_FULL_BASELINE, Q1_NULL_BASELINE
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


Q0_METHOD_ID = "q0_qwen3_reranker_06b"
Q1_METHOD_ID = "q1_instructrec_generalqwen"
SUPPORTED_METHODS = (Q0_METHOD_ID, Q1_METHOD_ID)
BASELINE_DIRS = {
    Q0_METHOD_ID: (Q0_FULL_BASELINE, Q0_NULL_BASELINE),
    Q1_METHOD_ID: (Q1_FULL_BASELINE, Q1_NULL_BASELINE),
}
SMOKE_NAMESPACE = "transformer-deep-dive-d6-breadth-readout-smoke-v1"


def write_breadth_readout_bundle(
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
    """Score two final-norm nodes without changing either model's native scorer."""

    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("an explicit breadth readout device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("breadth readout wall time must be in (0,13500]")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("breadth readout smoke request cap must be positive")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    records_path = standardized_dir / "records_dev.jsonl"
    if sha256_file(records_path) != manifest["frozen_inputs"]["records_dev_sha256"]:
        raise ValueError("breadth readout records hash differs")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("breadth readout requires full 8000-request dev")

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("breadth readout supports only Q0/Q1")
    frozen = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("breadth readout config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(model_dir, method_id)
    if checkpoint_id != frozen["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("breadth readout checkpoint differs from frozen binding")

    full_dir, null_dir = BASELINE_DIRS[method_id]
    baseline_full, full_identity = _load_frozen_baseline(
        full_dir, method_id, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        null_dir, method_id, checkpoint_id, records
    )
    target_records = (
        records
        if max_requests is None
        else _stable_smoke_records(records, int(max_requests))
    )
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if max_requests is None
        else "mechanical_smoke_non_result"
    )
    implementation = breadth_readout_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "full_scores_sha256": full_identity["scores_sha256"],
        "null_scores_sha256": null_identity["scores_sha256"],
        "nodes": list(BREADTH_READOUT_NODES),
        "conditions": list(BREADTH_READOUT_CONDITIONS),
        "target_requests": len(target_records),
        "smoke_namespace": SMOKE_NAMESPACE if max_requests is not None else None,
        "device": str(device),
        "implementation_digest": implementation["digest"],
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": f"transformer_deep_dive_d6_{method_id[:2]}_final_readout",
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "request_manifest_sha256": sha256_file(
            standardized_dir / "request_manifest.json"
        ),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "readout_nodes": list(BREADTH_READOUT_NODES),
        "score_conditions": list(BREADTH_READOUT_CONDITIONS),
        "native_scoring": (
            "lowercase yes-minus-no native logits"
            if method_id == Q0_METHOD_ID
            else "single listwise prefix cache plus complete per-candidate multi-token cached continuation"
        ),
        "patch_scope": (
            "native final prompt token"
            if method_id == Q0_METHOD_ID
            else "prompt readout and every response token"
        ),
        "implementation_identity": implementation,
        "q0_specialized_pretraining_boundary": method_id == Q0_METHOD_ID,
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
        run_dir,
        metadata=metadata,
        contract_sha256=contract_sha256,
        records=target_records,
        conditions=BREADTH_READOUT_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    identity = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    baseline_ratio = float(
        prepared.metadata.get("maximum_baseline_low_precision_ratio", 0.0)
    )
    response_tokens = int(prepared.metadata.get("response_tokens_processed", 0))
    if completed >= len(target_records):
        _assert_baseline_bound(baseline_ratio)
        return finalize_scalar_bundle(
            run_dir,
            prepared,
            target_records,
            BREADTH_READOUT_CONDITIONS,
            maximum_identity_delta=identity,
        )

    started = time.monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(
            config,
            device=str(device),
            training=False,
            checkpoint_model_dir=model_dir,
        )
        model.eval()
        if method_id == Q0_METHOD_ID and (
            _single_token_id(tokenizer, "yes") != 9693
            or _single_token_id(tokenizer, "no") != 2152
        ):
            raise ValueError("Q0 native yes/no token identity differs")
        prepared.metadata.update(_runtime_metadata(method_id, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    _progress(
                        prepared.metadata,
                        started,
                        identity,
                        baseline_ratio,
                        response_tokens,
                        "wall_time_exhausted",
                    )
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                if method_id == Q0_METHOD_ID:
                    rows = []
                    for start in range(0, len(record.candidates), batch_size):
                        candidates = list(record.candidates[start : start + batch_size])
                        result = score_q0_readout_chunk(
                            model,
                            tokenizer,
                            record,
                            candidates,
                            config,
                            device=str(device),
                        )
                        identity = max(
                            identity, float(result["maximum_identity_delta"])
                        )
                        rows.extend(
                            _condition_rows(
                                record,
                                candidates,
                                result["conditions"],
                                start,
                            )
                        )
                    request_audit: dict[str, Any] = {}
                else:
                    result = score_q1_readout_record(
                        model,
                        tokenizer,
                        record,
                        config,
                        device=str(device),
                        batch_size=batch_size,
                    )
                    identity = max(identity, float(result["maximum_identity_delta"]))
                    response_tokens += int(result["response_tokens"])
                    rows = _condition_rows(
                        record,
                        list(record.candidates),
                        result["conditions"],
                        0,
                    )
                    request_audit = {
                        "response_tokens": int(result["response_tokens"]),
                        "call_audit": result["call_audit"],
                    }
                for row in rows:
                    key = (record.request_id, str(row["candidate_item_id"]))
                    for name, reference in (
                        ("baseline_full", baseline_full[key]),
                        ("baseline_null", baseline_null[key]),
                    ):
                        value = float(row["conditions"][name])
                        bound = 8.0 * (2.0**-7) * max(1.0, abs(reference))
                        baseline_ratio = max(
                            baseline_ratio, abs(value - reference) / bound
                        )
                    if not all(
                        math.isfinite(float(value))
                        for value in row["conditions"].values()
                    ):
                        raise FloatingPointError("breadth readout score is non-finite")
                append_scalar_request(
                    run_dir,
                    {
                        "ordinal": ordinal,
                        "request_id": record.request_id,
                        **request_audit,
                        "rows": rows,
                        "rows_sha256": _canonical_sha256(rows),
                    },
                    prepared,
                )
    except Exception as exc:
        _progress(
            prepared.metadata,
            started,
            identity,
            baseline_ratio,
            response_tokens,
            "mechanical_failure",
        )
        prepared.metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise

    _assert_baseline_bound(baseline_ratio)
    _progress(
        prepared.metadata,
        started,
        identity,
        baseline_ratio,
        response_tokens,
        "running",
    )
    return finalize_scalar_bundle(
        run_dir,
        prepared,
        target_records,
        BREADTH_READOUT_CONDITIONS,
        maximum_identity_delta=identity,
    )


def _condition_rows(record, candidates, conditions, start):
    return [
        {
            "request_id": record.request_id,
            "candidate_item_id": str(candidate["item_id"]),
            "candidate_ordinal": start + local,
            "conditions": {
                name: float(conditions[name][local])
                for name in BREADTH_READOUT_CONDITIONS
            },
        }
        for local, candidate in enumerate(candidates)
    ]


def _stable_smoke_records(records, limit):
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{SMOKE_NAMESPACE}\0{record.request_id}".encode("utf-8")
        ).hexdigest(),
    )[:limit]


def _assert_baseline_bound(ratio):
    if ratio > 1.0:
        raise ValueError(f"breadth readout baseline BF16 bound failed: {ratio}")


def _progress(metadata, started, identity, baseline_ratio, response_tokens, status):
    metadata.update(
        {
            "status": status,
            "resumable": status != "running",
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + time.monotonic()
            - started,
            "maximum_identity_delta": identity,
            "maximum_baseline_low_precision_ratio": baseline_ratio,
            "response_tokens_processed": response_tokens,
        }
    )


def breadth_readout_implementation_identity():
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/breadth_readout_runtime.py",
        "src/myrec/mechanism/breadth_readout_scoring.py",
        "src/myrec/mechanism/q0_representation_prompt.py",
        "src/myrec/mechanism/q1_kv_trajectory.py",
        "src/myrec/mechanism/q1_branch_scoring.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "scripts/score_deep_dive_breadth_readout.py",
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
