"""Resumable qrels-blind D2 Q2/Q3 post-block causal sweep runtime."""

from __future__ import annotations

import json
import hashlib
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
from myrec.mechanism.postblock_sweep_scoring import (
    POSTBLOCK_CONDITIONS,
    score_q2_null_identity_chunk,
    score_postblock_chunk,
)
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.mechanism.representation_probe import normalized_query_fold
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


def write_postblock_sweep_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    fold: int,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    q3_gate_metrics_path: str | Path | None = None,
    fold0_selection_path: str | Path | None = None,
    q2_reuse_identity_dir: str | Path | None = None,
    q2_reuse_same_dir: str | Path | None = None,
    q2_reuse_cross_dir: str | Path | None = None,
    q2_reuse_audit_path: str | Path | None = None,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Materialize one full-population model/block post-block bundle."""

    _validate_run_id(run_id)
    block = int(block)
    if not 13 <= block <= 27:
        raise ValueError("post-block sweep block must be in [13,27]")
    fold = int(fold)
    if fold not in (0, 1):
        raise ValueError("post-block sweep fold must be 0 or 1")
    if not str(device).strip():
        raise ValueError("an explicit post-block sweep device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("post-block sweep max_wall_seconds must be in (0,13500]")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("post-block sweep max_requests must be positive")

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
            raise ValueError(f"post-block frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("post-block sweep requires all 8000 dev requests")
    records_by_id = {record.request_id: record for record in records}
    cross_mapping = _cross_request_mapping(records)
    cross_mapping_sha256 = _canonical_sha256(cross_mapping)

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("post-block sweep supports only Q2/Q3")
    frozen = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("post-block config differs from frozen manifest")
    gate_identity = None
    if method_id == "q3_tallrec_generalqwen" and max_requests is None:
        if q3_gate_metrics_path is None:
            raise ValueError("formal Q3 post-block sweep requires gate metrics")
        gate_path = Path(q3_gate_metrics_path)
        gate = _read_json(gate_path)
        if (
            gate.get("analysis_type")
            != "transformer_deep_dive_d2_q3_native_position_gate"
            or gate.get("status") != "completed"
            or gate.get("q3_sweep_admitted") is not True
            or gate.get("method_id") != method_id
        ):
            raise ValueError("Q3 post-block scientific gate did not admit sweep")
        gate_identity = {"path": str(gate_path), "sha256": sha256_file(gate_path)}

    reuse_arguments = (
        q2_reuse_identity_dir,
        q2_reuse_same_dir,
        q2_reuse_cross_dir,
        q2_reuse_audit_path,
    )
    if any(value is not None for value in reuse_arguments) and not all(
        value is not None for value in reuse_arguments
    ):
        raise ValueError("Q2 post-block reuse requires all three bundles and audit")
    if any(value is not None for value in reuse_arguments) and (
        method_id != "q2_recranker_generalqwen"
        or block not in (13, 27)
        or max_requests is not None
    ):
        raise ValueError("Q2 post-block reuse is formal-only at blocks 13/27")

    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != frozen["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("post-block checkpoint differs from frozen binding")
    selection_identity = None
    if fold == 1 and max_requests is None:
        if fold0_selection_path is None:
            raise ValueError("formal fold-1 sweep requires a frozen fold-0 selection")
        selection_path = Path(fold0_selection_path)
        selection = _read_json(selection_path)
        if (
            selection.get("analysis_type")
            != "transformer_deep_dive_d2_postblock_fold0_selection"
            or selection.get("status") != "completed"
            or selection.get("selection_frozen_before_fold1") is not True
            or selection.get("method_id") != method_id
            or selection.get("checkpoint_id") != checkpoint_id
            or selection.get("fold") != 0
            or selection.get("registered_blocks") != list(range(13, 28))
        ):
            raise ValueError("fold-0 selection record is not an admissible fold-1 gate")
        selection_identity = {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        }
    baseline_full, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    q2_reuse = _load_q2_reuse(
        records,
        method_id=method_id,
        checkpoint_id=checkpoint_id,
        block=block,
        identity_dir=q2_reuse_identity_dir,
        same_dir=q2_reuse_same_dir,
        cross_dir=q2_reuse_cross_dir,
        audit_path=q2_reuse_audit_path,
    ) if all(value is not None for value in reuse_arguments) else None
    folded_records = [record for record in records if normalized_query_fold(record.query) == fold]
    target_records = (
        folded_records
        if max_requests is None
        else _stable_smoke_records(folded_records, int(max_requests))
    )
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if max_requests is None
        else "mechanical_smoke_non_result"
    )
    implementation = postblock_sweep_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "normalized_query_fold": fold,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "cross_request_mapping_sha256": cross_mapping_sha256,
        "full_scores_sha256": full_identity["scores_sha256"],
        "null_scores_sha256": null_identity["scores_sha256"],
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "q3_gate_metrics": gate_identity,
        "fold0_selection": selection_identity,
        "q2_reused_first_round_conditions": (
            q2_reuse["identity"] if q2_reuse is not None else None
        ),
        "score_conditions": list(POSTBLOCK_CONDITIONS),
        "target_requests": len(target_records),
        "device": str(device),
        "implementation_digest": implementation["digest"],
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d2_postblock_sweep",
        "run_id": run_id,
        "method_id": method_id,
        "block_zero_based": block,
        "normalized_query_fold": fold,
        "full_population_request_count": len(records),
        "fold_request_count": len(folded_records),
        "native_position_scope": (
            "all_three_native_positions"
            if method_id == "q3_tallrec_generalqwen"
            else "single_native_readout"
        ),
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "q3_gate_metrics": gate_identity,
        "fold0_selection": selection_identity,
        "q2_reused_first_round_conditions": (
            q2_reuse["identity"] if q2_reuse is not None else None
        ),
        "cross_request_mapping_rule": "frozen M2 sha256 ring successor; candidate ordinal modulo donor slate",
        "cross_request_mapping_sha256": cross_mapping_sha256,
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "score_conditions": list(POSTBLOCK_CONDITIONS),
        "identity_conditions": ["full_to_full_identity", "null_to_null_identity"],
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
    prepared = prepare_scalar_bundle(
        run_dir,
        metadata=metadata,
        contract_sha256=contract_sha256,
        records=target_records,
        conditions=POSTBLOCK_CONDITIONS,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    full_delta = float(prepared.metadata.get("maximum_full_baseline_delta", 0.0))
    null_delta = float(prepared.metadata.get("maximum_null_baseline_delta", 0.0))
    baseline_ratio = float(
        prepared.metadata.get("maximum_baseline_low_precision_ratio", 0.0)
    )
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir,
            prepared,
            target_records,
            POSTBLOCK_CONDITIONS,
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
                        status="wall_time_exhausted",
                    )
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                donor = records_by_id[cross_mapping[record.request_id]]
                rows = []
                for start in range(0, len(record.candidates), batch_size):
                    candidates = list(record.candidates[start : start + batch_size])
                    donors = [
                        donor.candidates[(start + local) % len(donor.candidates)]
                        for local in range(len(candidates))
                    ]
                    if q2_reuse is None:
                        result = score_postblock_chunk(
                            model,
                            tokenizer,
                            record,
                            candidates,
                            donor,
                            donors,
                            config,
                            block=block,
                            device=str(device),
                        )
                        identity_delta = max(
                            identity_delta, float(result["maximum_identity_delta"])
                        )
                        conditions = result["conditions"]
                    else:
                        result = score_q2_null_identity_chunk(
                            model,
                            tokenizer,
                            record,
                            candidates,
                            config,
                            block=block,
                            device=str(device),
                        )
                        identity_delta = max(
                            identity_delta, float(result["maximum_identity_delta"])
                        )
                        conditions = None
                    for local, candidate in enumerate(candidates):
                        key = (record.request_id, str(candidate["item_id"]))
                        if q2_reuse is None:
                            assert conditions is not None
                            values = {
                                name: float(conditions[name][local].item())
                                for name in POSTBLOCK_CONDITIONS
                            }
                        else:
                            live_null = float(result["baseline_null"][local].item())
                            null_identity = float(
                                result["null_to_null_identity"][local].item()
                            )
                            values = {
                                "baseline_full": baseline_full[key],
                                "baseline_null": baseline_null[key],
                                "full_to_full_identity": q2_reuse["scores"][
                                    "full_to_full_identity"
                                ][key],
                                "null_to_null_identity": null_identity,
                                "same_full_to_null": q2_reuse["scores"][
                                    "same_request_full_to_null"
                                ][key],
                                "cross_full_to_null": q2_reuse["scores"][
                                    "cross_request_same_layer"
                                ][key],
                            }
                            identity_delta = max(
                                identity_delta,
                                abs(
                                    values["full_to_full_identity"]
                                    - values["baseline_full"]
                                ),
                                abs(null_identity - live_null),
                            )
                        if not all(math.isfinite(value) for value in values.values()):
                            raise FloatingPointError("post-block sweep score is non-finite")
                        full_delta = max(
                            full_delta,
                            abs(values["baseline_full"] - baseline_full[key]),
                        )
                        null_delta = max(
                            null_delta,
                            abs(
                                (
                                    float(result["baseline_null"][local].item())
                                    if q2_reuse is not None
                                    else values["baseline_null"]
                                )
                                - baseline_null[key]
                            ),
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
                        rows.append(
                            {
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "candidate_ordinal": start + local,
                                "conditions": values,
                            }
                        )
                append_scalar_request(
                    run_dir,
                    {
                        "ordinal": ordinal,
                        "request_id": record.request_id,
                        "cross_donor_request_id": donor.request_id,
                        "rows": rows,
                        "rows_sha256": _canonical_sha256(rows),
                    },
                    prepared,
                )
    except Exception as exc:
        _store_progress(
            prepared.metadata,
            started,
            identity_delta,
            full_delta,
            null_delta,
            baseline_ratio,
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
            "post-block recomputed baseline exceeded the frozen path-local BF16 bound: "
            f"ratio={baseline_ratio}"
        )
        _store_progress(
            prepared.metadata,
            started,
            identity_delta,
            full_delta,
            null_delta,
            baseline_ratio,
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
        status="running",
    )
    return finalize_scalar_bundle(
        run_dir,
        prepared,
        target_records,
        POSTBLOCK_CONDITIONS,
        maximum_identity_delta=identity_delta,
    )


def postblock_sweep_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/postblock_sweep_runtime.py",
        "src/myrec/mechanism/postblock_sweep_scoring.py",
        "src/myrec/mechanism/deep_dive_native_patch.py",
        "src/myrec/mechanism/native_readout_scoring.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "src/myrec/mechanism/patch_evaluator.py",
        "scripts/score_deep_dive_postblock_sweep.py",
    )
    files = [
        {"path": path, "sha256": sha256_file(root / path), "size_bytes": (root / path).stat().st_size}
        for path in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}


def _stable_smoke_records(records: Sequence[Any], limit: int) -> list[Any]:
    """Select a qrels-blind, order-stable smoke population."""

    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"deep-dive-postblock-smoke-v1\0{record.request_id}".encode("utf-8")
        ).digest(),
    )[:limit]


def _load_q2_reuse(
    records: Sequence[Any],
    *,
    method_id: str,
    checkpoint_id: str,
    block: int,
    identity_dir: str | Path | None,
    same_dir: str | Path | None,
    cross_dir: str | Path | None,
    audit_path: str | Path | None,
) -> dict[str, Any]:
    """Audit and index admitted first-round Q2 intervention scores."""

    from myrec.mechanism.patch_evaluator import _audit_score_bundle

    if any(value is None for value in (identity_dir, same_dir, cross_dir, audit_path)):
        raise ValueError("incomplete Q2 reuse inputs")
    audit_path = Path(audit_path)  # type: ignore[arg-type]
    audit = _read_json(audit_path)
    if (
        audit.get("analysis_type")
        != "transformer_deep_dive_d2_q2_postblock_reuse_equivalence"
        or audit.get("status") != "passed"
        or audit.get("method_id") != method_id
        or audit.get("checkpoint_id") != checkpoint_id
        or int(audit.get("block_zero_based", -1)) != block
        or audit.get("maximum_abs_score_delta", math.inf) > 1.0e-5
    ):
        raise ValueError("Q2 first-round reuse equivalence audit did not pass")
    bundle_specs = {
        "full_to_full_identity": (identity_dir, "full_to_full_identity"),
        "same_request_full_to_null": (same_dir, "same_request_full_to_null"),
        "cross_request_same_layer": (cross_dir, "cross_request_same_layer"),
    }
    scores: dict[str, dict[tuple[str, str], float]] = {}
    identities: dict[str, Any] = {}
    for name, (root, patch_kind) in bundle_specs.items():
        bundle = _audit_score_bundle(
            root,
            records,
            expected_condition=None,
            patch=(patch_kind, block),
        )
        if bundle.metadata.get("method_id") != method_id or bundle.metadata.get(
            "checkpoint_id"
        ) != checkpoint_id:
            raise ValueError("Q2 reused bundle model binding drift")
        scores[name] = {
            (request_id, item_id): value
            for request_id, request_scores in bundle.scores.items()
            for item_id, value in request_scores.items()
        }
        identities[name] = {
            "path": str(bundle.root),
            "scores_sha256": bundle.scores_sha256,
            "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
        }
        audited = audit.get("first_round_bundles", {}).get(name, {})
        if (
            audited.get("scores_sha256") != bundle.scores_sha256
            or audited.get("metadata_sha256")
            != identities[name]["metadata_sha256"]
        ):
            raise ValueError("Q2 reused bundle differs from equivalence-audited bytes")
    identity = {
        "equivalence_audit_path": str(audit_path),
        "equivalence_audit_sha256": sha256_file(audit_path),
        "conditions": identities,
        "newly_computed_condition": "null_to_null_identity",
    }
    return {"scores": scores, "identity": identity}


def _store_progress(
    metadata: dict[str, Any],
    started: float,
    identity: float,
    full_delta: float,
    null_delta: float,
    baseline_ratio: float,
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
        }
    )
