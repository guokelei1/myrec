"""Resumable qrels-blind runtime for N17/N18 routing-boundary bundles.

This runtime mirrors the scalar bundle contract used by N11--N16.  It is
registered but intentionally does not launch itself; the N17/N18 queue must
wait for the earlier inference closeout before consuming a GPU.
"""

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
    FIXED_BLOCKS,
    MAX_WALL_SECONDS,
    SUPPORTED_METHODS,
    _assert_native_targets,
    _canonical_sha256,
    _load_content_controls,
    _load_frozen_baseline,
    _read_json,
    _write_json,
)
from myrec.mechanism.postblock_sweep_runtime import NULL_BASELINE_DIRS
from myrec.mechanism.routing_boundary_scoring import (
    GQA_CONDITIONS,
    HEAD_NORM_CONDITIONS,
    score_gqa_grouping_chunk,
    score_qk_head_norm_chunk,
)
from myrec.mechanism.q3_lora_branch_scoring import (
    LORA_BRANCH_CONDITIONS,
    score_q3_lora_branch_chunk,
)
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


N17_N20_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_n17_n20_boundary_manifest_v1.yaml"
)
N17_N20_MANIFEST_SHA256 = (
    "e13177bd48c422002359dd3ba1e98f21c9fe8be7c3c460a12860af0074f01760"
)


def write_qk_head_norm_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    component: str,
    device: str,
    runs_dir: str | Path = "runs",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    return _write_routing_bundle(
        standardized_dir,
        config_path,
        checkpoint_root,
        run_id,
        kind="n17_head_norm",
        block=int(block),
        component=str(component),
        device=device,
        runs_dir=runs_dir,
        resume=resume,
        max_wall_seconds=max_wall_seconds,
        max_requests=max_requests,
        command=command,
    )


def write_gqa_grouping_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    device: str,
    runs_dir: str | Path = "runs",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    return _write_routing_bundle(
        standardized_dir,
        config_path,
        checkpoint_root,
        run_id,
        kind="n18_gqa_grouping",
        block=int(block),
        component=None,
        device=device,
        runs_dir=runs_dir,
        resume=resume,
        max_wall_seconds=max_wall_seconds,
        max_requests=max_requests,
        command=command,
    )


def write_q3_lora_branch_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    component: str,
    device: str,
    runs_dir: str | Path = "runs",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    return _write_routing_bundle(
        standardized_dir,
        config_path,
        checkpoint_root,
        run_id,
        kind="n19_lora_branch",
        block=int(block),
        component=str(component),
        device=device,
        runs_dir=runs_dir,
        resume=resume,
        max_wall_seconds=max_wall_seconds,
        max_requests=max_requests,
        command=command,
    )


def _write_routing_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    kind: str,
    block: int,
    component: str | None,
    device: str,
    runs_dir: str | Path,
    resume: bool,
    max_wall_seconds: float,
    max_requests: int | None,
    command: Sequence[str] | None,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    if kind not in {"n17_head_norm", "n18_gqa_grouping", "n19_lora_branch"}:
        raise ValueError(f"unsupported routing kind={kind}")
    if block not in FIXED_BLOCKS:
        raise ValueError("routing boundary requires fixed block 13/20/27")
    if kind == "n17_head_norm" and component not in {"q", "k"}:
        raise ValueError("N17 requires component q or k")
    if kind == "n19_lora_branch" and component not in {"q", "v"}:
        raise ValueError("N19 requires component q or v")
    if kind == "n18_gqa_grouping" and component is not None:
        raise ValueError("N18 does not take a component")
    if not str(device).strip():
        raise ValueError("routing boundary requires an explicit device")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("routing boundary max_wall_seconds is out of range")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("routing boundary max_requests must be positive")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    boundary_manifest = _load_boundary_manifest()
    parent_manifest = _load_parent_manifest()
    records_path = standardized_dir / "records_dev.jsonl"
    expected_hashes = {
        records_path: parent_manifest["frozen_inputs"]["records_sha256"],
        standardized_dir / "manifest.json": parent_manifest["frozen_inputs"]["dataset_manifest_sha256"],
        standardized_dir / "request_manifest.json": parent_manifest["frozen_inputs"]["request_manifest_sha256"],
        standardized_dir / "candidate_manifest.json": parent_manifest["frozen_inputs"]["candidate_manifest_sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"{kind} frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError(f"{kind} requires all 8000 internal-dev requests")

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError(f"{kind} supports only Q2/Q3")
    if kind == "n19_lora_branch" and method_id != "q3_tallrec_generalqwen":
        raise ValueError("N19 LoRA branch bundle is Q3-only")
    binding = parent_manifest["model_bindings"].get(method_id)
    if not isinstance(binding, Mapping) or config["_config_sha256"] != binding["config_sha256"]:
        raise ValueError(f"{kind} config differs from frozen binding")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != binding["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError(f"{kind} checkpoint differs from frozen binding")

    controls, control_identity = _load_content_controls(
        parent_manifest, method_id, records
    )
    baseline_full, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    conditions = {
        "n17_head_norm": HEAD_NORM_CONDITIONS,
        "n18_gqa_grouping": GQA_CONDITIONS,
        "n19_lora_branch": LORA_BRANCH_CONDITIONS,
    }[kind]
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if max_requests is None
        else "mechanical_smoke_non_result"
    )
    implementation = routing_boundary_runtime_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "analysis_stage": kind,
        "method_id": method_id,
        "block_zero_based": block,
        "component": component,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "boundary_manifest_sha256": boundary_manifest["_sha256"],
        "content_control_sha256": control_identity["rows_sha256"],
        "baseline_full_sha256": full_identity["scores_sha256"],
        "baseline_null_sha256": null_identity["scores_sha256"],
        "score_conditions": list(conditions),
        "target_requests": len(target_records),
        "evidence_mode": evidence_mode,
        "device": str(device),
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": kind,
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "component": component,
        "score_conditions": list(conditions),
        "identity_conditions": [name for name in conditions if name.endswith("identity")],
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
        "boundary_manifest_path": str(N17_N20_MANIFEST_PATH),
        "boundary_manifest_sha256": boundary_manifest["_sha256"],
        "content_control": control_identity,
        "frozen_baseline_full": full_identity,
        "frozen_baseline_null": null_identity,
        "identity_tolerance": 1.0e-5,
        "implementation_identity": implementation,
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage_required": True,
        "ineligible_scoring": "copy_frozen_baseline_pair",
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
        conditions=conditions,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    maximum_identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    maximum_baseline_delta = float(prepared.metadata.get("maximum_frozen_baseline_delta", 0.0))
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir, prepared, target_records, conditions,
            maximum_identity_delta=maximum_identity_delta,
        )

    started = time.monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(
            config, device=str(device), training=False, checkpoint_model_dir=checkpoint_model_dir
        )
        model.eval()
        _assert_native_targets(tokenizer, method_id, parent_manifest)
        prepared.metadata.update(_runtime_metadata(method_id, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    prepared.metadata.update({
                        "status": "wall_time_exhausted",
                        "resumable": True,
                        "maximum_identity_delta": maximum_identity_delta,
                        "maximum_frozen_baseline_delta": maximum_baseline_delta,
                    })
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                control = controls[record.request_id]
                rows: list[dict[str, Any]] = []
                if control.get("eligible") is True:
                    for start in range(0, len(record.candidates), batch_size):
                        candidates = list(record.candidates[start : start + batch_size])
                        if kind == "n17_head_norm":
                            result = score_qk_head_norm_chunk(
                                model, tokenizer, record, candidates, control, config,
                                block=block, component=str(component), device=str(device)
                            )
                        elif kind == "n18_gqa_grouping":
                            result = score_gqa_grouping_chunk(
                                model, tokenizer, record, candidates, control, config,
                                block=block, device=str(device)
                            )
                        else:
                            result = score_q3_lora_branch_chunk(
                                model, tokenizer, record, candidates, control, config,
                                block=block, component=str(component), device=str(device)
                            )
                        maximum_identity_delta = max(
                            maximum_identity_delta, float(result["maximum_identity_delta"])
                        )
                        for local, candidate in enumerate(candidates):
                            values = {
                                name: float(result["conditions"][name][local])
                                for name in conditions
                            }
                            if not all(math.isfinite(value) for value in values.values()):
                                raise FloatingPointError(f"{kind} produced a non-finite score")
                            key = (record.request_id, str(candidate["item_id"]))
                            maximum_baseline_delta = max(
                                maximum_baseline_delta,
                                abs(values["baseline_full"] - baseline_full[key]),
                                abs(values["baseline_null"] - baseline_null[key]),
                            )
                            rows.append({
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "candidate_ordinal": start + local,
                                "conditions": values,
                            })
                else:
                    for candidate_ordinal, candidate in enumerate(record.candidates):
                        key = (record.request_id, str(candidate["item_id"]))
                        full, null = baseline_full[key], baseline_null[key]
                        values = {
                            name: (null if name == "baseline_null" or name.startswith("null_") else full)
                            for name in conditions
                        }
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
                        "block_zero_based": block,
                        "component": component,
                        "content_control_eligible": bool(control.get("eligible") is True),
                        "rows": rows,
                        "rows_sha256": _canonical_sha256(rows),
                    },
                    prepared,
                )
    except Exception as exc:
        prepared.metadata.update({
            "status": "mechanical_failure",
            "resumable": True,
            "qrels_read": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "maximum_identity_delta": maximum_identity_delta,
            "maximum_frozen_baseline_delta": maximum_baseline_delta,
        })
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    prepared.metadata.update({
        "elapsed_seconds": float(prepared.metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started,
        "maximum_identity_delta": maximum_identity_delta,
        "maximum_frozen_baseline_delta": maximum_baseline_delta,
    })
    return finalize_scalar_bundle(
        run_dir, prepared, target_records, conditions,
        maximum_identity_delta=maximum_identity_delta,
    )


def routing_boundary_runtime_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "src/myrec/mechanism/qk_head_rmsnorm_interventions.py",
        "src/myrec/mechanism/gqa_grouping_interventions.py",
        "src/myrec/mechanism/routing_boundary_scoring.py",
        "src/myrec/mechanism/routing_boundary_runtime.py",
        "src/myrec/mechanism/q3_lora_branch_interventions.py",
        "src/myrec/mechanism/q3_lora_branch_scoring.py",
        "src/myrec/mechanism/attention_edge_scoring.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "scripts/score_deep_dive_routing_boundary.py",
    )
    files = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"N17--N19 implementation file is missing: {path}")
        files.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"files": files, "digest": _canonical_sha256(files)}


def _load_boundary_manifest() -> dict[str, Any]:
    observed = sha256_file(N17_N20_MANIFEST_PATH)
    if observed != N17_N20_MANIFEST_SHA256:
        raise ValueError("N17/N20 manifest differs from immutable digest")
    value = yaml.safe_load(N17_N20_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("N17/N20 manifest is not a mapping")
    value["_sha256"] = observed
    return value


def _load_parent_manifest() -> dict[str, Any]:
    value = yaml.safe_load(DEEP_DIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("parent deep-dive manifest is not a mapping")
    return value
