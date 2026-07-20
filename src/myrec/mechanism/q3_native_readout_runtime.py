"""Qrels-blind resumable D6 scorer for Q3's three native readout states."""

from __future__ import annotations

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
from myrec.mechanism.deep_dive_native_patch import _q3_context
from myrec.mechanism.patch_scorer import _cross_request_mapping
from myrec.mechanism.q3_native_readout_scoring import (
    Q3_FINAL_NODES,
    LOW_PRECISION_RATIO_TOLERANCE,
    Q3_READOUT_SCOPES,
    Q3_TERM_NAMES,
    capture_q3_native_readout,
    compose_q3_readout_terms,
    q3_score_low_precision_bound,
)
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


Q3_METHOD_ID = "q3_tallrec_generalqwen"
Q3_NULL_BASELINE_DIR = Path(
    "runs/20260717_kuaisearch_q3_tallrec_generalqwen_internal_dev_null_score"
)
Q3_NATIVE_READOUT_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    *(
        f"{scope}_{suffix}"
        for scope in Q3_READOUT_SCOPES
        for suffix in (
            "same_full_to_null",
            "cross_full_to_null",
            "null_identity",
            "full_identity",
        )
    ),
)


def write_q3_native_readout_bundle(
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
    """Materialize all registered Q3 final-readout conditions without qrels."""

    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("an explicit Q3 native-readout device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("Q3 native-readout max_wall_seconds must be in (0, 13500]")
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
            raise ValueError(f"Q3 native-readout frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("Q3 native-readout scorer requires all 8000 dev requests")
    records_by_id = {row.request_id: row for row in records}
    cross_mapping = _cross_request_mapping(records)
    cross_mapping_sha256 = _canonical_sha256(cross_mapping)

    config = load_v12_ranker_config(config_path)
    if config["method_id"] != Q3_METHOD_ID:
        raise ValueError("Q3 native-readout runtime received another method")
    frozen_model = manifest["frozen_inputs"]["models"][Q3_METHOD_ID]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("Q3 native-readout config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, Q3_METHOD_ID
    )
    if checkpoint_id != frozen_model["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("Q3 native-readout checkpoint differs from frozen binding")
    baseline_full, baseline_full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[Q3_METHOD_ID], Q3_METHOD_ID, checkpoint_id, records
    )
    baseline_null, baseline_null_identity = _load_frozen_baseline(
        Q3_NULL_BASELINE_DIR, Q3_METHOD_ID, checkpoint_id, records
    )
    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if max_requests is None
        else "mechanical_smoke_non_result"
    )
    implementation = q3_native_readout_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": Q3_METHOD_ID,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "cross_request_mapping_sha256": cross_mapping_sha256,
        "full_scores_sha256": baseline_full_identity["scores_sha256"],
        "null_scores_sha256": baseline_null_identity["scores_sha256"],
        "score_conditions": list(Q3_NATIVE_READOUT_CONDITIONS),
        "target_requests": len(target_records),
        "device": str(device),
        "evidence_mode": evidence_mode,
        "implementation_digest": implementation["digest"],
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d6_q3_native_readout",
        "run_id": run_id,
        "method_id": Q3_METHOD_ID,
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
        "score_conditions": list(Q3_NATIVE_READOUT_CONDITIONS),
        "readout_scopes": list(Q3_READOUT_SCOPES),
        "term_names": list(Q3_TERM_NAMES),
        "captured_nodes": list(Q3_FINAL_NODES),
        "causal_node": "final_rmsnorm_output",
        "native_token_ids": {"yes": [9454, 151645], "no": [2753, 151645]},
        "readout_intervention": "exact registered native-term substitution at final RMSNorm output",
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
        conditions=Q3_NATIVE_READOUT_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    state = _resume_state(prepared.metadata)
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir,
            prepared,
            target_records,
            Q3_NATIVE_READOUT_CONDITIONS,
            maximum_identity_delta=state["identity"],
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
        _assert_native_targets(tokenizer, Q3_METHOD_ID, manifest)
        prepared.metadata.update(_runtime_metadata(Q3_METHOD_ID, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    _store_progress(prepared.metadata, state, started, status="wall_time_exhausted")
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
                    full_context = _q3_context(
                        tokenizer, record, candidates, record.history, config, str(device)
                    )
                    null_context = _q3_context(
                        tokenizer, record, candidates, [], config, str(device)
                    )
                    cross_context = _q3_context(
                        tokenizer,
                        donor_record,
                        donor_candidates,
                        donor_record.history,
                        config,
                        str(device),
                    )
                    full = capture_q3_native_readout(model, full_context)
                    null = capture_q3_native_readout(model, null_context)
                    cross = capture_q3_native_readout(model, cross_context)
                    captures = (full, null, cross)
                    state["algebra"] = max(
                        state["algebra"],
                        *(float(value["algebra_max_abs_error"]) for value in captures),
                    )
                    state["algebra_ratio"] = max(
                        state["algebra_ratio"],
                        *(float(value["algebra_low_precision_max_ratio"]) for value in captures),
                    )
                    state["shared"] = max(
                        state["shared"],
                        *(
                            float(delta)
                            for value in captures
                            for delta in value["shared_prompt_path_max_abs_delta"].values()
                        ),
                    )
                    if any(
                        float(delta) != 0.0
                        for value in captures
                        for delta in value["shared_prompt_path_max_abs_delta"].values()
                    ):
                        raise RuntimeError(
                            "Q3 shared prompt state differs across equal-length Yes/No paths"
                        )
                    conditions: dict[str, Any] = {
                        "baseline_full": full["score"],
                        "baseline_null": null["score"],
                    }
                    for scope in Q3_READOUT_SCOPES:
                        conditions[f"{scope}_same_full_to_null"] = compose_q3_readout_terms(
                            null["terms"], full["terms"], scope=scope
                        )["score"]
                        conditions[f"{scope}_cross_full_to_null"] = compose_q3_readout_terms(
                            null["terms"], cross["terms"], scope=scope
                        )["score"]
                        null_identity = compose_q3_readout_terms(
                            null["terms"], null["terms"], scope=scope
                        )["score"]
                        full_identity = compose_q3_readout_terms(
                            full["terms"], full["terms"], scope=scope
                        )["score"]
                        conditions[f"{scope}_null_identity"] = null_identity
                        conditions[f"{scope}_full_identity"] = full_identity
                        state["identity"] = max(
                            state["identity"],
                            float((null_identity - null["score"]).abs().max().item()),
                            float((full_identity - full["score"]).abs().max().item()),
                        )
                    for local, candidate in enumerate(candidates):
                        values = {
                            name: float(conditions[name][local].item())
                            for name in Q3_NATIVE_READOUT_CONDITIONS
                        }
                        if not all(math.isfinite(value) for value in values.values()):
                            raise FloatingPointError("Q3 native-readout produced a non-finite score")
                        key = (record.request_id, str(candidate["item_id"]))
                        state["full_delta"] = max(
                            state["full_delta"],
                            abs(values["baseline_full"] - baseline_full[key]),
                        )
                        state["null_delta"] = max(
                            state["null_delta"],
                            abs(values["baseline_null"] - baseline_null[key]),
                        )
                        for condition, reference, captured in (
                            ("baseline_full", baseline_full[key], full),
                            ("baseline_null", baseline_null[key], null),
                        ):
                            bound = float(
                                q3_score_low_precision_bound(
                                    captured["terms"][local : local + 1]
                                )[0].item()
                            )
                            state["baseline_ratio"] = max(
                                state["baseline_ratio"],
                                abs(values[condition] - reference) / bound,
                            )
                        rows.append(
                            {
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "candidate_ordinal": start + local,
                                "conditions": values,
                                "readout_diagnostics": {
                                    "full_terms": [
                                        float(value) for value in full["terms"][local].tolist()
                                    ],
                                    "null_terms": [
                                        float(value) for value in null["terms"][local].tolist()
                                    ],
                                    "full_prompt_contrast": float(
                                        0.5 * (full["terms"][local, 0] - full["terms"][local, 2])
                                    ),
                                    "full_context_contrast": float(
                                        0.5 * (full["terms"][local, 1] - full["terms"][local, 3])
                                    ),
                                    "full_yes_input_norm": [
                                        float(value)
                                        for value in full["geometry"]["yes"]["input_norm"][local].tolist()
                                    ],
                                    "full_yes_output_norm": [
                                        float(value)
                                        for value in full["geometry"]["yes"]["output_norm"][local].tolist()
                                    ],
                                    "full_no_input_norm": [
                                        float(value)
                                        for value in full["geometry"]["no"]["input_norm"][local].tolist()
                                    ],
                                    "full_no_output_norm": [
                                        float(value)
                                        for value in full["geometry"]["no"]["output_norm"][local].tolist()
                                    ],
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
        _store_progress(prepared.metadata, state, started, status="mechanical_failure")
        prepared.metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    _store_progress(prepared.metadata, state, started, status="running")
    return finalize_scalar_bundle(
        run_dir,
        prepared,
        target_records,
        Q3_NATIVE_READOUT_CONDITIONS,
        maximum_identity_delta=state["identity"],
    )


def q3_native_readout_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/q3_native_readout_runtime.py",
        "src/myrec/mechanism/q3_native_readout_scoring.py",
        "src/myrec/mechanism/deep_dive_native_patch.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "scripts/score_deep_dive_q3_native_readout.py",
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


def _resume_state(metadata: Mapping[str, Any]) -> dict[str, float]:
    return {
        "identity": float(metadata.get("maximum_identity_delta", 0.0)),
        "full_delta": float(metadata.get("maximum_full_baseline_delta", 0.0)),
        "null_delta": float(metadata.get("maximum_null_baseline_delta", 0.0)),
        "algebra": float(metadata.get("maximum_algebra_delta", 0.0)),
        "baseline_ratio": float(metadata.get("maximum_baseline_low_precision_ratio", 0.0)),
        "algebra_ratio": float(metadata.get("maximum_algebra_low_precision_ratio", 0.0)),
        "shared": float(metadata.get("maximum_shared_prompt_path_delta", 0.0)),
    }


def _store_progress(
    metadata: dict[str, Any],
    state: Mapping[str, float],
    started: float,
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
            "maximum_identity_delta": state["identity"],
            "maximum_full_baseline_delta": state["full_delta"],
            "maximum_null_baseline_delta": state["null_delta"],
            "maximum_algebra_delta": state["algebra"],
            "maximum_baseline_low_precision_ratio": state["baseline_ratio"],
            "maximum_algebra_low_precision_ratio": state["algebra_ratio"],
            "maximum_shared_prompt_path_delta": state["shared"],
            "low_precision_algebra_passed": (
                state["baseline_ratio"] <= 1.0 + LOW_PRECISION_RATIO_TOLERANCE
                and state["algebra_ratio"] <= 1.0 + LOW_PRECISION_RATIO_TOLERANCE
                and state["shared"] == 0.0
            ),
        }
    )
