"""Resumable qrels-blind N10 candidate-gap geometry scorer."""

from __future__ import annotations

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
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.breadth_readout_runtime import (
    BASELINE_DIRS as BREADTH_BASELINE_DIRS,
    Q0_METHOD_ID,
    Q1_METHOD_ID,
)
from myrec.mechanism.native_readout_runtime import Q2_NULL_BASELINE_DIR
from myrec.mechanism.postblock_sweep_runtime import NULL_BASELINE_DIRS
from myrec.mechanism.q3_native_readout_runtime import Q3_NULL_BASELINE_DIR
from myrec.mechanism.candidate_gap_scoring import (
    CANDIDATE_GAP_CONDITIONS,
    CANDIDATE_GAP_NODES,
    score_q0_candidate_gap_chunk,
    score_q1_candidate_gap_record,
    score_q2_candidate_gap_chunk,
    score_q3_candidate_gap_chunk,
)
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


N10_CANDIDATE_GAP_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_n10_candidate_gap_manifest_v1.yaml"
)
N10_CANDIDATE_GAP_MANIFEST_SHA256 = "c8f7811581961d37d73e4f011ce447a4b9624396903d08eeecc12018910bbb66"
MAX_WALL_SECONDS = 13_500.0
SUPPORTED_METHODS = (Q0_METHOD_ID, Q1_METHOD_ID, "q2_recranker_generalqwen", "q3_tallrec_generalqwen")
BASELINE_DIRS = {
    Q0_METHOD_ID: BREADTH_BASELINE_DIRS[Q0_METHOD_ID],
    Q1_METHOD_ID: BREADTH_BASELINE_DIRS[Q1_METHOD_ID],
    "q2_recranker_generalqwen": (BASELINE_SCORE_DIRS["q2_recranker_generalqwen"], Q2_NULL_BASELINE_DIR),
    "q3_tallrec_generalqwen": (BASELINE_SCORE_DIRS["q3_tallrec_generalqwen"], Q3_NULL_BASELINE_DIR),
}


def write_candidate_gap_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = N10_CANDIDATE_GAP_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("candidate-gap scorer requires an explicit device")
    if not 0 < float(max_wall_seconds) <= MAX_WALL_SECONDS:
        raise ValueError("candidate-gap max_wall_seconds must be in (0,13500]")
    manifest = _load_candidate_manifest(manifest_path)
    deep_manifest = _load_manifest(DEEP_DIVE_MANIFEST_PATH)
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, deep_manifest["frozen_inputs"]["records_dev_sha256"]),
        (standardized_dir / "manifest.json", deep_manifest["frozen_inputs"]["dataset_manifest_sha256"]),
        (standardized_dir / "request_manifest.json", deep_manifest["frozen_inputs"]["request_manifest_sha256"]),
        (standardized_dir / "candidate_manifest.json", deep_manifest["frozen_inputs"]["candidate_manifest_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"candidate-gap frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("candidate-gap scorer requires all 8000 requests")
    target_records = records if max_requests is None else records[: int(max_requests)]

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError(f"candidate-gap scorer does not support {method_id}")
    frozen_model = deep_manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("candidate-gap config differs from frozen binding")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(model_dir, method_id)
    if checkpoint_id != frozen_model["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("candidate-gap checkpoint differs from frozen binding")

    baseline_full, full_identity = _load_frozen_baseline(BASELINE_DIRS[method_id][0], method_id, checkpoint_id, records)
    baseline_null, null_identity = _load_frozen_baseline(BASELINE_DIRS[method_id][1], method_id, checkpoint_id, records)
    implementation = candidate_gap_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "deep_dive_manifest_sha256": deep_manifest["_sha256"],
        "candidate_gap_manifest_sha256": manifest["_sha256"],
        "nodes": list(CANDIDATE_GAP_NODES),
        "score_conditions": list(CANDIDATE_GAP_CONDITIONS),
        "target_requests": len(target_records),
        "device": str(device),
        "evidence_mode": "registered_n10_candidate_gap" if max_requests is None else "mechanical_smoke_non_result",
        "implementation_digest": implementation["digest"],
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_n10_candidate_gap_geometry",
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_sha256": deep_manifest["_sha256"],
        "candidate_gap_manifest_path": str(manifest_path),
        "candidate_gap_manifest_sha256": manifest["_sha256"],
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "nodes": list(CANDIDATE_GAP_NODES),
        "score_conditions": list(CANDIDATE_GAP_CONDITIONS),
        "qrels_read": False,
        "source_test_opened": False,
        "result_eligible": max_requests is None,
        "complete_finite_score_coverage_required": True,
        "implementation_identity": implementation,
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
        conditions=CANDIDATE_GAP_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    max_identity = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    max_full_delta = float(prepared.metadata.get("maximum_full_baseline_delta", 0.0))
    max_null_delta = float(prepared.metadata.get("maximum_null_baseline_delta", 0.0))
    max_shared_prompt_delta = float(prepared.metadata.get("shared_prompt_path_max_abs_delta", 0.0))
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir, prepared, target_records, CANDIDATE_GAP_CONDITIONS,
            maximum_identity_delta=max(max_identity, max_full_delta, max_null_delta),
        )

    started = time.monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(
            config, device=str(device), training=False, checkpoint_model_dir=model_dir
        )
        model.eval()
        prepared.metadata.update(_runtime_metadata(method_id, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= float(max_wall_seconds):
                    prepared.metadata.update({
                        "status": "wall_time_exhausted", "resumable": True,
                        "maximum_identity_delta": max_identity,
                        "maximum_full_baseline_delta": max_full_delta,
                        "maximum_null_baseline_delta": max_null_delta,
                        "shared_prompt_path_max_abs_delta": max_shared_prompt_delta,
                    })
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                if method_id == Q1_METHOD_ID:
                    result = score_q1_candidate_gap_record(
                        model, tokenizer, record, config, device=str(device), batch_size=batch_size
                    )
                    rows = _condition_rows(record, result["conditions"], 0)
                    request_extra = {
                        "response_tokens": result["response_tokens"],
                        "call_audit": result["call_audit"],
                        "direction_norms": result["direction_norms"],
                    }
                else:
                    scorer = {
                        Q0_METHOD_ID: score_q0_candidate_gap_chunk,
                        "q2_recranker_generalqwen": score_q2_candidate_gap_chunk,
                        "q3_tallrec_generalqwen": score_q3_candidate_gap_chunk,
                    }[method_id]
                    rows = []
                    direction_norms: dict[str, float] = {}
                    for start in range(0, len(record.candidates), batch_size):
                        candidates = list(record.candidates[start : start + batch_size])
                        kwargs = {"device": str(device)}
                        if method_id == "q3_tallrec_generalqwen":
                            kwargs["history"] = record.history
                        result = scorer(model, tokenizer, record, candidates, config, **kwargs)
                        rows.extend(_condition_rows(record, result["conditions"], start))
                        direction_norms.update(result["direction_norms"])
                        max_shared_prompt_delta = max(
                            max_shared_prompt_delta,
                            float(result.get("shared_prompt_path_max_abs_delta", 0.0)),
                        )
                    request_extra = {"direction_norms": direction_norms}
                for row in rows:
                    key = (record.request_id, str(row["candidate_item_id"]))
                    values = row["conditions"]
                    max_identity = max(
                        max_identity,
                        *(
                            abs(float(values[f"{node}_full_identity"]) - float(values["baseline_full"]))
                            for node in CANDIDATE_GAP_NODES
                        ),
                    )
                    max_full_delta = max(max_full_delta, abs(float(values["baseline_full"]) - float(baseline_full[key])))
                    max_null_delta = max(max_null_delta, abs(float(values["baseline_null"]) - float(baseline_null[key])))
                    if not all(math.isfinite(float(value)) for value in values.values()):
                        raise FloatingPointError("candidate-gap score is non-finite")
                append_scalar_request(
                    run_dir,
                    {
                        "ordinal": ordinal,
                        "request_id": record.request_id,
                        **request_extra,
                        "rows": rows,
                        "rows_sha256": _canonical_sha256(rows),
                    },
                    prepared,
                )
    except Exception as exc:
        prepared.metadata.update({
            "status": "mechanical_failure", "resumable": True,
            "maximum_identity_delta": max_identity,
            "maximum_full_baseline_delta": max_full_delta,
            "maximum_null_baseline_delta": max_null_delta,
            "shared_prompt_path_max_abs_delta": max_shared_prompt_delta,
            "qrels_read": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    prepared.metadata.update({
        "elapsed_seconds": float(prepared.metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started,
        "maximum_identity_delta": max_identity,
        "maximum_full_baseline_delta": max_full_delta,
        "maximum_null_baseline_delta": max_null_delta,
        "shared_prompt_path_max_abs_delta": max_shared_prompt_delta,
    })
    return finalize_scalar_bundle(
        run_dir, prepared, target_records, CANDIDATE_GAP_CONDITIONS,
        maximum_identity_delta=max(max_identity, max_full_delta, max_null_delta),
    )


def _condition_rows(record: Any, conditions: Mapping[str, Any], start: int) -> list[dict[str, Any]]:
    rows = []
    for local, candidate in enumerate(record.candidates[start : start + len(next(iter(conditions.values()))) ]):
        rows.append({
            "request_id": record.request_id,
            "candidate_item_id": str(candidate["item_id"]),
            "candidate_ordinal": start + local,
            "conditions": {name: float(values[local]) for name, values in conditions.items()},
        })
    return rows


def _load_candidate_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    observed = sha256_file(path)
    if observed != N10_CANDIDATE_GAP_MANIFEST_SHA256:
        raise ValueError("candidate-gap manifest differs from frozen digest")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("candidate-gap manifest is not a mapping")
    value["_sha256"] = observed
    return value


def candidate_gap_implementation_identity() -> dict[str, Any]:
    from myrec.mechanism.attention_edge_runtime import _canonical_sha256

    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/candidate_gap_scoring.py",
        "src/myrec/mechanism/candidate_gap_runtime.py",
        "src/myrec/mechanism/breadth_readout_scoring.py",
        "src/myrec/mechanism/native_readout_scoring.py",
        "src/myrec/mechanism/q3_native_readout_scoring.py",
        "scripts/score_deep_dive_candidate_gap.py",
        "scripts/evaluate_deep_dive_candidate_gap.py",
        "src/myrec/mechanism/candidate_gap_evaluator.py",
    )
    files = [{"path": path, "sha256": sha256_file(root / path), "size_bytes": (root / path).stat().st_size} for path in paths]
    return {"files": files, "digest": _canonical_sha256(files)}
