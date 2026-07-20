"""Label-isolated, resumable scorer for Motivation history interventions.

The frozen V1.2 ranker remains the owner of model loading, prompt construction,
and candidate scoring.  This module changes exactly one input: the history
selected by an audited mechanism-assignment manifest.  It deliberately has no
evaluator or qrels imports.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import (
    FORBIDDEN_MODEL_INPUT_FIELDS,
    HISTORY_INPUT_FIELDS,
    SERIALIZED_INPUT_FIELDS,
    ModelRecord,
    sanitize_record_for_model,
)
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _assert_scoring_population,
    _checkpoint_identity,
    _git_revision,
    _implementation_identity as _frozen_scorer_implementation_identity,
    _load_model_and_tokenizer,
    _runtime_metadata,
    _score_instructrec_request,
    _score_yes_no_request,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


MAX_WALL_SECONDS = 13_500.0
MECHANISM_PROBE_MANIFEST_PATH = Path(
    "experiments/motivation/probe_manifest.yaml"
)
MECHANISM_PROBE_MANIFEST_SHA256 = (
    "adedf0e662b9d8529162b8abffedcf6b10962913f28580af6119d807cc5d929c"
)
PARTIAL_FILENAME = "scores.partial.jsonl"
PROGRESS_FILENAME = "progress.json"
METADATA_FILENAME = "metadata.json"
SCORES_FILENAME = "scores.jsonl"
_CONDITION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_ASSIGNMENT_ROW_FIELDS = frozenset({"request_id", "condition_id", "history"})
_HISTORY_FIELDS = frozenset(HISTORY_INPUT_FIELDS)
_BLOCK_FIELDS = frozenset(
    {
        "condition_id",
        "ordinal",
        "prompt_at_max_boundary",
        "request_id",
        "rows",
        "rows_sha256",
    }
)
_SCORE_ROW_FIELDS = frozenset(
    {"candidate_item_id", "method_id", "request_id", "score"}
)


def write_mechanism_intervention_scores(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    assignment_path: str | Path,
    assignment_manifest_path: str | Path,
    condition_id: str,
    reference_run_id: str,
    split: str,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    command: Sequence[str] | None = None,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_score_requests: int | None = None,
    probe_manifest_path: str | Path = MECHANISM_PROBE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Score one audited history intervention on frozen internal-dev records.

    An uncapped invocation is result-eligible only after complete finite
    request/candidate coverage.  A capped invocation is permanently marked as
    a smoke non-result even when its cap happens to cover the whole population.
    """

    _validate_run_id(run_id)
    _validate_run_id(reference_run_id)
    if split != "dev":
        raise ValueError("mechanism scoring is restricted to split=dev")
    if not _CONDITION_PATTERN.fullmatch(condition_id):
        raise ValueError(f"invalid condition_id={condition_id!r}")
    if not device or not str(device).strip():
        raise ValueError("an explicit scoring device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not (
        0.0 < max_wall_seconds <= MAX_WALL_SECONDS
    ):
        raise ValueError(
            f"max_wall_seconds must be in (0, {int(MAX_WALL_SECONDS)}]"
        )
    if max_score_requests is not None and int(max_score_requests) <= 0:
        raise ValueError("max_score_requests must be positive")
    if max_score_requests is not None:
        max_score_requests = int(max_score_requests)

    mechanism_probe_manifest = _load_mechanism_probe_manifest(
        probe_manifest_path
    )

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    assignment_path = Path(assignment_path)
    assignment_manifest_path = Path(assignment_manifest_path)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    records_path = standardized_dir / "records_dev.jsonl"
    dataset_manifest_path = standardized_dir / "manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"

    # Every boundary check happens before model loading.  None of these paths
    # is derived from, or scans for, qrels.
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    population = config["_protocol"]["data"]["development_population"]
    dataset_manifest = _read_json(dataset_manifest_path, "dataset manifest")
    if str(dataset_manifest.get("dataset_id")) != "kuaisearch":
        raise ValueError("mechanism scorer only admits the KuaiSearch dataset")
    if str(dataset_manifest.get("dataset_version")) != str(
        population["dataset_version"]
    ):
        raise ValueError("mechanism scorer only admits frozen V1.1 internal-dev")

    checkpoint_dir = checkpoint_root / CHECKPOINT_DIRNAME
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path, "training metadata")
    _validate_scoring_checkpoint_provenance(
        training_metadata,
        config,
        allow_smoke=max_score_requests is not None,
    )
    checkpoint_id, checkpoint_weight_files = _checkpoint_identity(
        checkpoint_dir / "model", method_id
    )
    if checkpoint_id != training_metadata.get("checkpoint_id"):
        raise ValueError("checkpoint weights changed after training metadata")
    _assert_scoring_population(
        standardized_dir,
        config,
        split="dev",
        dataset_manifest=dataset_manifest,
        checkpoint_id=checkpoint_id,
        checkpoint_weight_files=checkpoint_weight_files,
        training_metadata_path=training_metadata_path,
    )

    hashes = {
        "assignment_sha256": sha256_file(assignment_path),
        "assignment_manifest_sha256": sha256_file(assignment_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "records_sha256": sha256_file(records_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
    }
    expected_hashes = {
        "candidate_manifest_sha256": str(population["candidate_manifest_sha256"]),
        "dataset_manifest_sha256": str(population["manifest_sha256"]),
        "records_sha256": str(population["records_dev_sha256"]),
        "request_manifest_sha256": str(population["request_manifest_sha256"]),
    }
    for key, expected in expected_hashes.items():
        if hashes[key] != expected:
            raise ValueError(f"frozen internal-dev hash mismatch: {key}")

    records = _load_frozen_records(records_path)
    expected_request_count = int(population["internal_dev_requests"])
    if len(records) != expected_request_count:
        raise ValueError(
            "internal-dev request count differs from frozen protocol: "
            f"{len(records)} != {expected_request_count}"
        )
    assignment_manifest, condition_manifest = _validate_assignment_manifest(
        assignment_manifest_path,
        assignment_path=assignment_path,
        condition_id=condition_id,
        hashes=hashes,
        dataset_id="kuaisearch",
        dataset_version=str(population["dataset_version"]),
        split="dev",
        request_count=len(records),
    )
    assignments = _load_assignments(
        assignment_path,
        condition_id=condition_id,
        records=records,
    )

    frozen_identity = _frozen_scorer_implementation_identity()
    reference = _load_and_validate_reference_run(
        runs_dir,
        reference_run_id,
        method_id=method_id,
        checkpoint_id=checkpoint_id,
        checkpoint_weight_files=checkpoint_weight_files,
        config=config,
        dataset_id="kuaisearch",
        dataset_version=str(population["dataset_version"]),
        candidate_manifest_sha256=hashes["candidate_manifest_sha256"],
        request_manifest_sha256=hashes["request_manifest_sha256"],
        request_count=len(records),
        score_rows=sum(len(value["record"].candidates) for value in records),
        frozen_identity=frozen_identity,
    )
    mechanism_identity = mechanism_scorer_implementation_identity()
    evidence_mode = (
        "smoke_non_result"
        if max_score_requests is not None
        else "mechanism_diagnostic"
    )
    target_request_count = (
        min(max_score_requests, len(records))
        if max_score_requests is not None
        else len(records)
    )
    run_contract = {
        "schema_version": 1,
        "assignment_manifest_sha256": hashes["assignment_manifest_sha256"],
        "assignment_sha256": hashes["assignment_sha256"],
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "checkpoint_id": checkpoint_id,
        "condition_id": condition_id,
        "config_sha256": config["_config_sha256"],
        "dataset_id": "kuaisearch",
        "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
        "dataset_version": str(population["dataset_version"]),
        "device": str(device),
        "evidence_mode": evidence_mode,
        "frozen_scorer_implementation_digest": frozen_identity["digest"],
        "max_score_requests": max_score_requests,
        "mechanism_scorer_implementation_digest": mechanism_identity["digest"],
        "mechanism_probe_manifest_sha256": mechanism_probe_manifest["sha256"],
        "records_sha256": hashes["records_sha256"],
        "reference_metadata_sha256": reference["metadata_sha256"],
        "reference_run_id": reference_run_id,
        "reference_scores_sha256": reference["scores_sha256"],
        "request_manifest_sha256": hashes["request_manifest_sha256"],
        "run_id": run_id,
        "split": "dev",
        "target_request_count": target_request_count,
        "training_metadata_sha256": sha256_file(training_metadata_path),
    }
    run_contract_sha256 = _canonical_sha256(run_contract)
    base_metadata = {
        "schema_version": 1,
        "assignment_manifest_path": str(assignment_manifest_path),
        "assignment_manifest_sha256": hashes["assignment_manifest_sha256"],
        "assignment_path": str(assignment_path),
        "assignment_sha256": hashes["assignment_sha256"],
        "base_scoring_signature": copy.deepcopy(reference["scoring_signature"]),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": checkpoint_weight_files,
        "code_revision": _git_revision(),
        "command": list(command or sys.argv),
        "condition_id": condition_id,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "dataset_id": "kuaisearch",
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
        "dataset_version": str(population["dataset_version"]),
        "evidence_mode": evidence_mode,
        "frozen_scorer_implementation_identity": frozen_identity,
        "input_fields_used": list(SERIALIZED_INPUT_FIELDS),
        "intervention": {
            "assignment_manifest_path": str(assignment_manifest_path),
            "assignment_manifest_sha256": hashes["assignment_manifest_sha256"],
            "assignment_path": str(assignment_path),
            "assignment_sha256": hashes["assignment_sha256"],
            "candidate_leakage_audit_passed": True,
            "candidate_or_query_modified": False,
            "causality_audit_passed": True,
            "condition_id": condition_id,
            "full_canary_required": condition_id == "full",
            "history_is_only_model_input_intervention": True,
            "manifest_condition": copy.deepcopy(condition_manifest),
            "manifest_implementation": copy.deepcopy(
                assignment_manifest.get("implementation")
            ),
            "manifest_probe_id": assignment_manifest.get("probe_id"),
            "source_query_and_candidates": "frozen_records_dev_jsonl",
        },
        "mechanism_scorer_implementation_identity": mechanism_identity,
        "mechanism_scorer_sha256": mechanism_identity["scorer_sha256"],
        "mechanism_probe_manifest": mechanism_probe_manifest,
        "method_id": method_id,
        "qrels_read": False,
        "records_path": str(records_path),
        "records_sha256": hashes["records_sha256"],
        "reference_full_run_id": reference_run_id,
        "reference_metadata_sha256": reference["metadata_sha256"],
        "reference_scores_sha256": reference["scores_sha256"],
        "request_manifest_path": str(request_manifest_path),
        "request_manifest_sha256": hashes["request_manifest_sha256"],
        "result_eligible": False,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "run_id": run_id,
        # Copying this value preserves the exact frozen model/scoring identity;
        # the intervention has its own independent identity above.
        "scoring_signature": copy.deepcopy(reference["scoring_signature"]),
        "split": "dev",
        "standardized_dir": str(standardized_dir),
        "status": "initializing",
        "training_evidence_mode": training_metadata.get("evidence_mode"),
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": sha256_file(training_metadata_path),
    }

    state = _prepare_run_state(
        run_dir,
        base_metadata=base_metadata,
        run_contract_sha256=run_contract_sha256,
        records=records[:target_request_count],
        method_id=method_id,
        condition_id=condition_id,
        resume=resume,
    )
    metadata = state["metadata"]
    progress = state["progress"]
    partial_hasher = state["partial_hasher"]
    request_ranges = state["request_ranges"]
    prompt_boundary_count = state["prompt_boundary_count"]
    if int(progress["completed_requests"]) >= target_request_count:
        return _finalize_run(
            run_dir,
            metadata=metadata,
            progress=progress,
            records=records[:target_request_count],
            method_id=method_id,
            condition_id=condition_id,
            reference=reference,
            full_population_requests=len(records),
            smoke=max_score_requests is not None,
        )

    segment_started = _monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(
            config,
            device=str(device),
            training=False,
            checkpoint_model_dir=checkpoint_dir / "model",
        )
        model.eval()
        metadata.update(_runtime_metadata(method_id, torch, transformers))
        metadata["status"] = "running"
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        score_batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        start_index = int(progress["completed_requests"])
        with torch.inference_mode():
            for ordinal in range(start_index, target_request_count):
                if _monotonic() - segment_started >= max_wall_seconds:
                    return _record_wall_exit(
                        run_dir,
                        metadata=metadata,
                        progress=progress,
                        segment_elapsed=_monotonic() - segment_started,
                        target_request_count=target_request_count,
                    )
                frozen = records[ordinal]
                record = frozen["record"]
                history = assignments[record.request_id]
                if method_id == "q1_instructrec_generalqwen":
                    request_scores, at_boundary = _score_instructrec_request(
                        model,
                        tokenizer,
                        record,
                        history,
                        config,
                        device=str(device),
                        batch_size=score_batch_size,
                    )
                else:
                    request_scores, at_boundary = _score_yes_no_request(
                        model,
                        tokenizer,
                        record,
                        history,
                        config,
                        device=str(device),
                        batch_size=score_batch_size,
                    )
                rows, score_range = _validated_score_rows(
                    record,
                    request_scores,
                    method_id=method_id,
                )
                block = _score_block(
                    ordinal=ordinal,
                    condition_id=condition_id,
                    request_id=record.request_id,
                    rows=rows,
                    prompt_at_max_boundary=bool(at_boundary),
                )
                line = _canonical_json(block) + "\n"
                _append_and_sync(run_dir / PARTIAL_FILENAME, line.encode("utf-8"))
                partial_hasher.update(line.encode("utf-8"))
                request_ranges.append(score_range)
                prompt_boundary_count += int(bool(at_boundary))
                progress.update(
                    {
                        "completed_requests": ordinal + 1,
                        "completed_score_rows": int(progress["completed_score_rows"])
                        + len(rows),
                        "last_request_id": record.request_id,
                        "partial_sha256": partial_hasher.hexdigest(),
                        "prompt_at_max_boundary_requests": prompt_boundary_count,
                        "rolling_request_blocks_sha256": _rolling_block_digest(
                            str(progress["rolling_request_blocks_sha256"]),
                            str(block["rows_sha256"]),
                        ),
                        "status": "running",
                        "updated_at": _utc_now(),
                    }
                )
                _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
    except Exception as exc:
        metadata.update(
            {
                "completed_requests": int(progress["completed_requests"]),
                "error": {"message": str(exc), "type": type(exc).__name__},
                "qrels_read": False,
                "resumable": True,
                "status": "failed",
            }
        )
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        raise

    metadata["elapsed_seconds"] = float(metadata.get("elapsed_seconds", 0.0)) + (
        _monotonic() - segment_started
    )
    return _finalize_run(
        run_dir,
        metadata=metadata,
        progress=progress,
        records=records[:target_request_count],
        method_id=method_id,
        condition_id=condition_id,
        reference=reference,
        full_population_requests=len(records),
        smoke=max_score_requests is not None,
    )


def mechanism_scorer_implementation_identity() -> dict[str, Any]:
    """Hash the mechanism scorer and its production CLI independently."""

    root = Path(__file__).resolve().parents[3]
    paths = {
        "scripts/score_mechanism_intervention.py": (
            root / "scripts/score_mechanism_intervention.py"
        ),
        "src/myrec/mechanism/scorer.py": Path(__file__).resolve(),
    }
    files = []
    for relative, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing mechanism scorer implementation: {path}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {
        "digest": _canonical_sha256(files),
        "files": files,
        "scorer_sha256": next(
            value["sha256"]
            for value in files
            if value["path"] == "src/myrec/mechanism/scorer.py"
        ),
    }


def _load_mechanism_probe_manifest(path: str | Path) -> dict[str, Any]:
    """Fail closed unless the exact frozen mechanism-stage manifest is read."""

    root = Path(__file__).resolve().parents[3]
    expected_path = (root / MECHANISM_PROBE_MANIFEST_PATH).resolve()
    supplied = Path(path)
    if not supplied.is_absolute():
        supplied = (Path.cwd() / supplied).resolve()
    else:
        supplied = supplied.resolve()
    if supplied != expected_path:
        raise ValueError(
            "mechanism probe manifest path mismatch: expected "
            f"{MECHANISM_PROBE_MANIFEST_PATH}"
        )
    observed_sha256 = sha256_file(supplied)
    if observed_sha256 != MECHANISM_PROBE_MANIFEST_SHA256:
        raise ValueError("frozen mechanism probe manifest hash mismatch")
    payload = _read_structured_manifest(supplied)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("frozen mechanism probe manifest schema mismatch")
    if payload.get("probe_manifest_id") != (
        "motivation_mechanism_first_diagnosis_v1"
    ):
        raise ValueError("frozen mechanism probe manifest identity mismatch")
    if payload.get("status") != "frozen_before_mechanism_outcomes":
        raise ValueError("mechanism probe manifest is not frozen")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("mechanism probe manifest lacks a scope boundary")
    expected_scope = {
        "dataset_id": "kuaisearch",
        "development_population": "full_confirm_preceding40k_v11",
        "evaluation_population": "internal_dev_only",
        "source_test_opened": False,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            raise ValueError(f"mechanism probe manifest scope mismatch: {key}")
    return {
        "expected_sha256": MECHANISM_PROBE_MANIFEST_SHA256,
        "path": MECHANISM_PROBE_MANIFEST_PATH.as_posix(),
        "sha256": observed_sha256,
        "verified": True,
    }


def _load_frozen_records(path: Path) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for raw in iter_jsonl(path):
        _assert_no_forbidden_keys(raw, f"record {raw.get('request_id', '<unknown>')}")
        record = sanitize_record_for_model(raw)
        if record.request_id in seen:
            raise ValueError(f"duplicate record request_id={record.request_id}")
        seen.add(record.request_id)
        try:
            request_ts = int(raw["ts"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"request_id={record.request_id}: invalid ts") from exc
        result.append({"raw": raw, "record": record, "request_ts": request_ts})
    if not result:
        raise ValueError(f"empty internal-dev records: {path}")
    return result


def _validate_assignment_manifest(
    manifest_path: Path,
    *,
    assignment_path: Path,
    condition_id: str,
    hashes: Mapping[str, str],
    dataset_id: str,
    dataset_version: str,
    split: str,
    request_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_structured_manifest(manifest_path)
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("mechanism assignment manifest schema_version must be 1")
    if manifest.get("qrels_read") is not False:
        raise ValueError("assignment manifest must attest qrels_read=false")
    if manifest.get("model_scores_read") is not False:
        raise ValueError("assignment manifest must attest model_scores_read=false")
    source = manifest.get("source")
    if source is None:
        source = {}
    if not isinstance(source, Mapping):
        raise ValueError("assignment manifest source must be an object")

    declared = {
        "dataset_id": _first_present(manifest, source, "dataset_id"),
        "dataset_version": _first_present(manifest, source, "dataset_version"),
        "split": _first_present(manifest, source, "split"),
        "records_sha256": _first_present(
            manifest, source, "source_records_sha256", "records_sha256"
        ),
        "candidate_manifest_sha256": _first_present(
            manifest, source, "candidate_manifest_sha256"
        ),
        "request_manifest_sha256": _first_present(
            manifest, source, "request_manifest_sha256"
        ),
        "dataset_manifest_sha256": _first_present(
            manifest, source, "dataset_manifest_sha256"
        ),
    }
    expected = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "split": split,
        "records_sha256": hashes["records_sha256"],
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "request_manifest_sha256": hashes["request_manifest_sha256"],
        "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
    }
    # The intervention manifest must bind its source records.  Dataset and
    # candidate/request manifest identities are re-derived and protocol-locked
    # above; when the assignment manifest also repeats them, they must agree.
    if str(declared.get("records_sha256")) != str(expected["records_sha256"]):
        raise ValueError("assignment manifest source mismatch: records_sha256")
    for key, value in expected.items():
        if key == "records_sha256" or declared.get(key) is None:
            continue
        if str(declared[key]) != str(value):
            raise ValueError(f"assignment manifest source mismatch: {key}")

    conditions = manifest.get("conditions")
    if not isinstance(conditions, Mapping) or condition_id not in conditions:
        raise ValueError(f"assignment manifest has no condition={condition_id}")
    condition = conditions[condition_id]
    if not isinstance(condition, dict):
        raise ValueError(f"manifest condition={condition_id} must be an object")
    declared_path = condition.get("path")
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise ValueError(f"manifest condition={condition_id} has no path")
    resolved_assignment = assignment_path.resolve()
    raw_declared_path = Path(declared_path)
    possible_paths = {raw_declared_path.resolve()}
    if not raw_declared_path.is_absolute():
        possible_paths.add((manifest_path.parent / raw_declared_path).resolve())
    if resolved_assignment not in possible_paths:
        raise ValueError("assignment path differs from its manifest condition")
    if str(condition.get("sha256")) != hashes["assignment_sha256"]:
        raise ValueError("assignment file differs from its manifest hash")
    try:
        count_value = condition.get("count", condition.get("request_count"))
        declared_count = int(count_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("assignment manifest condition requires integer count") from exc
    if declared_count != request_count:
        raise ValueError("assignment manifest condition count mismatch")
    _require_zero_audit(
        manifest,
        condition,
        names=(
            "candidate_field_leakage_violations",
            "candidate_leakage_violations",
            "candidate_leakage_count",
            "target_candidate_leakage_violations",
        ),
        role="candidate leakage",
    )
    _require_zero_audit(
        manifest,
        condition,
        names=(
            "causality_violations",
            "causality_violation_count",
            "history_not_strictly_before_target_violations",
        ),
        role="causality",
    )
    _require_zero_audit(
        manifest,
        condition,
        names=("forbidden_field_count", "forbidden_fields"),
        role="forbidden-field",
    )
    immutability = manifest.get("query_candidate_immutability")
    if immutability is not None:
        if not isinstance(immutability, Mapping):
            raise ValueError("assignment manifest immutability audit is invalid")
        if int(immutability.get("query_changed_rows", -1)) != 0 or int(
            immutability.get("candidate_changed_rows", -1)
        ) != 0:
            raise ValueError("assignment manifest query/candidate immutability failed")
        if set(immutability.get("assignment_payload_fields", [])) != set(
            _ASSIGNMENT_ROW_FIELDS
        ):
            raise ValueError("assignment manifest payload whitelist drift")
    for boundary_field in ("confirmation_records_read", "source_test_opened"):
        if boundary_field in manifest and manifest[boundary_field] is not False:
            raise ValueError(f"assignment manifest crossed boundary: {boundary_field}")
    return manifest, condition


def _load_assignments(
    path: Path,
    *,
    condition_id: str,
    records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    frozen_by_id = {value["record"].request_id: value for value in records}
    result: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(path):
        if set(row) != _ASSIGNMENT_ROW_FIELDS:
            raise ValueError(
                "assignment rows must contain exactly request_id, condition_id, history"
            )
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError(f"empty or duplicate assignment request_id={request_id!r}")
        if request_id not in frozen_by_id:
            raise ValueError(f"assignment contains unknown request_id={request_id}")
        if str(row.get("condition_id")) != condition_id:
            raise ValueError(f"assignment condition mismatch for {request_id}")
        raw_history = row.get("history")
        if not isinstance(raw_history, list):
            raise ValueError(f"request_id={request_id}: assignment history must be a list")
        for index, event in enumerate(raw_history):
            if not isinstance(event, dict):
                raise ValueError(f"request_id={request_id}: history[{index}] not object")
            unexpected = set(event) - _HISTORY_FIELDS
            if unexpected:
                raise ValueError(
                    f"request_id={request_id}: history[{index}] outside whitelist: "
                    f"{sorted(unexpected)}"
                )
            _assert_no_forbidden_keys(event, f"assignment {request_id} history[{index}]")
        projected = sanitize_record_for_model(
            {
                "request_id": request_id,
                "query": "mechanism-assignment-validation",
                "history": raw_history,
                "candidates": [
                    {"item_id": "__mechanism_placeholder_a"},
                    {"item_id": "__mechanism_placeholder_b"},
                ],
            }
        )
        history = list(projected.history)
        request_ts = int(frozen_by_id[request_id]["request_ts"])
        for index, event in enumerate(history):
            try:
                event_ts = int(event["ts"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"request_id={request_id}: history[{index}] missing valid ts"
                ) from exc
            if event_ts >= request_ts:
                raise ValueError(
                    f"request_id={request_id}: assignment history is not causal"
                )
        if condition_id == "full" and history != list(
            frozen_by_id[request_id]["record"].history
        ):
            raise ValueError(f"full canary history drift for request_id={request_id}")
        result[request_id] = history
    expected = set(frozen_by_id)
    observed = set(result)
    if observed != expected:
        raise ValueError(
            "assignment request coverage mismatch: "
            f"missing={sorted(expected - observed)[:5]} "
            f"extra={sorted(observed - expected)[:5]}"
        )
    return result


def _load_and_validate_reference_run(
    runs_dir: Path,
    run_id: str,
    *,
    method_id: str,
    checkpoint_id: str,
    checkpoint_weight_files: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    dataset_id: str,
    dataset_version: str,
    candidate_manifest_sha256: str,
    request_manifest_sha256: str,
    request_count: int,
    score_rows: int,
    frozen_identity: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    metadata_path = run_dir / METADATA_FILENAME
    scores_path = run_dir / SCORES_FILENAME
    metadata = _read_json(metadata_path, "reference score metadata")
    if metadata.get("run_id") != run_id:
        raise ValueError("reference metadata run_id mismatch")
    if metadata.get("qrels_read") is not False:
        raise ValueError("reference full score run crossed the qrels boundary")
    if metadata.get("evidence_mode") != "first_round_pilot":
        raise ValueError("reference full score run must be a frozen first-round result")
    if metadata.get("history_condition") not in {"true", "full"}:
        raise ValueError("reference run is not a frozen full-history score bundle")
    expected = {
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "method_id": method_id,
        "request_count": request_count,
        "request_manifest_sha256": request_manifest_sha256,
        "score_rows": score_rows,
        "split": "dev",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"reference full score mismatch: {key}")
    if metadata.get("checkpoint_weight_files") != list(checkpoint_weight_files):
        raise ValueError("reference checkpoint artifact identity mismatch")
    if not scores_path.is_file():
        raise FileNotFoundError(scores_path)
    scores_sha256 = sha256_file(scores_path)
    if metadata.get("scores_sha256") != scores_sha256:
        raise ValueError("reference scores differ from their frozen hash")
    signature = metadata.get("scoring_signature")
    if not isinstance(signature, dict):
        raise ValueError("reference full score has no scoring_signature")
    signature_expected = {
        "config_sha256": config["_config_sha256"],
        "implementation_digest": frozen_identity["digest"],
        "method_id": method_id,
        "protocol_sha256": config["protocol"]["sha256"],
    }
    for key, value in signature_expected.items():
        if signature.get(key) != value:
            raise ValueError(f"reference scoring_signature mismatch: {key}")
    if metadata.get("implementation_identity") != dict(frozen_identity):
        raise ValueError("reference frozen scorer implementation identity mismatch")
    return {
        "metadata": metadata,
        "metadata_path": metadata_path,
        "metadata_sha256": sha256_file(metadata_path),
        "scores_path": scores_path,
        "scores_sha256": scores_sha256,
        "scoring_signature": copy.deepcopy(signature),
    }


def _prepare_run_state(
    run_dir: Path,
    *,
    base_metadata: dict[str, Any],
    run_contract_sha256: str,
    records: Sequence[dict[str, Any]],
    method_id: str,
    condition_id: str,
    resume: bool,
) -> dict[str, Any]:
    partial_path = run_dir / PARTIAL_FILENAME
    progress_path = run_dir / PROGRESS_FILENAME
    metadata_path = run_dir / METADATA_FILENAME
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial_path.touch(exist_ok=False)
        progress = {
            "schema_version": 1,
            "completed_requests": 0,
            "completed_score_rows": 0,
            "last_request_id": None,
            "partial_sha256": hashlib.sha256(b"").hexdigest(),
            "prompt_at_max_boundary_requests": 0,
            "resume_count": 0,
            "rolling_request_blocks_sha256": hashlib.sha256(b"").hexdigest(),
            "run_contract_sha256": run_contract_sha256,
            "status": "initializing",
            "updated_at": _utc_now(),
        }
        metadata = dict(base_metadata)
        metadata.update(
            {
                "elapsed_seconds": 0.0,
                "resume_lineage": [],
                "resumable": True,
            }
        )
        _write_json_atomic(progress_path, progress)
        _write_json_atomic(metadata_path, metadata)
        return {
            "metadata": metadata,
            "partial_hasher": hashlib.sha256(),
            "progress": progress,
            "prompt_boundary_count": 0,
            "request_ranges": [],
        }

    if not run_dir.is_dir():
        raise FileNotFoundError(f"resume run directory is missing: {run_dir}")
    metadata = _read_json(metadata_path, "mechanism score metadata")
    progress = _read_json(progress_path, "mechanism score progress")
    if metadata.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("resume metadata run contract drift")
    if progress.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("resume progress run contract drift")
    if metadata.get("status") == "completed":
        raise ValueError("mechanism score run is already completed")
    if metadata.get("status") not in {
        "failed",
        "initializing",
        "running",
        "wall_time_exhausted",
    }:
        raise ValueError(f"run status is not resumable: {metadata.get('status')}")
    observed = _validate_partial(
        partial_path,
        records=records,
        method_id=method_id,
        condition_id=condition_id,
    )
    expected_progress = {
        "completed_requests": observed["completed_requests"],
        "completed_score_rows": observed["completed_score_rows"],
        "last_request_id": observed["last_request_id"],
        "partial_sha256": observed["partial_sha256"],
        "prompt_at_max_boundary_requests": observed[
            "prompt_at_max_boundary_requests"
        ],
        "rolling_request_blocks_sha256": observed[
            "rolling_request_blocks_sha256"
        ],
    }
    for key, value in expected_progress.items():
        if progress.get(key) != value:
            raise ValueError(f"resume partial/progress mismatch: {key}")
    lineage = metadata.get("resume_lineage")
    if not isinstance(lineage, list):
        raise ValueError("resume metadata lineage is invalid")
    lineage.append(
        {
            "completed_requests": observed["completed_requests"],
            "from_status": metadata["status"],
            "partial_sha256": observed["partial_sha256"],
            "resumed_at": _utc_now(),
        }
    )
    metadata.update(
        {"error": None, "resume_lineage": lineage, "status": "initializing"}
    )
    progress.update(
        {
            "resume_count": int(progress.get("resume_count", 0)) + 1,
            "status": "initializing",
            "updated_at": _utc_now(),
        }
    )
    _write_json_atomic(metadata_path, metadata)
    _write_json_atomic(progress_path, progress)
    return {
        "metadata": metadata,
        "partial_hasher": observed["partial_hasher"],
        "progress": progress,
        "prompt_boundary_count": observed["prompt_at_max_boundary_requests"],
        "request_ranges": observed["request_ranges"],
    }


def _validated_score_rows(
    record: ModelRecord,
    request_scores: Mapping[str, Any],
    *,
    method_id: str,
) -> tuple[list[dict[str, Any]], float]:
    expected_ids = [str(candidate["item_id"]) for candidate in record.candidates]
    if set(request_scores) != set(expected_ids) or len(request_scores) != len(
        expected_ids
    ):
        raise ValueError(f"candidate score coverage failed for {record.request_id}")
    rows = []
    values = []
    for item_id in expected_ids:
        value = float(request_scores[item_id])
        if not math.isfinite(value):
            raise FloatingPointError(
                f"non-finite score request_id={record.request_id} item_id={item_id}"
            )
        rows.append(
            {
                "candidate_item_id": item_id,
                "method_id": method_id,
                "request_id": record.request_id,
                "score": value,
            }
        )
        values.append(value)
    return rows, max(values) - min(values)


def _score_block(
    *,
    ordinal: int,
    condition_id: str,
    request_id: str,
    rows: list[dict[str, Any]],
    prompt_at_max_boundary: bool,
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "ordinal": ordinal,
        "prompt_at_max_boundary": prompt_at_max_boundary,
        "request_id": request_id,
        "rows": rows,
        "rows_sha256": _canonical_sha256(rows),
    }


def _validate_partial(
    path: Path,
    *,
    records: Sequence[dict[str, Any]],
    method_id: str,
    condition_id: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError("partial score file ends with an incomplete request block")
    partial_hasher = hashlib.sha256(content)
    request_ranges = []
    prompt_count = 0
    score_rows = 0
    rolling = hashlib.sha256(b"").hexdigest()
    last_request_id = None
    lines = content.splitlines()
    if len(lines) > len(records):
        raise ValueError("partial score file exceeds target request coverage")
    for ordinal, encoded in enumerate(lines):
        try:
            block = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"partial request block {ordinal} is invalid JSON") from exc
        if not isinstance(block, dict) or set(block) != _BLOCK_FIELDS:
            raise ValueError(f"partial request block {ordinal} has invalid fields")
        if (_canonical_json(block) + "\n").encode("utf-8") != encoded + b"\n":
            raise ValueError(f"partial request block {ordinal} is not canonical")
        record = records[ordinal]["record"]
        if int(block["ordinal"]) != ordinal:
            raise ValueError(f"partial request block {ordinal} ordinal mismatch")
        if block["request_id"] != record.request_id:
            raise ValueError(f"partial request block {ordinal} request order mismatch")
        if block["condition_id"] != condition_id:
            raise ValueError(f"partial request block {ordinal} condition mismatch")
        rows = block["rows"]
        if not isinstance(rows, list) or block["rows_sha256"] != _canonical_sha256(
            rows
        ):
            raise ValueError(f"partial request block {ordinal} hash mismatch")
        expected_ids = [str(value["item_id"]) for value in record.candidates]
        if len(rows) != len(expected_ids):
            raise ValueError(f"partial request block {ordinal} coverage mismatch")
        values = []
        for item_id, row in zip(expected_ids, rows):
            if not isinstance(row, dict) or set(row) != _SCORE_ROW_FIELDS:
                raise ValueError(f"partial request block {ordinal} score fields invalid")
            if (
                row["request_id"] != record.request_id
                or row["candidate_item_id"] != item_id
                or row["method_id"] != method_id
            ):
                raise ValueError(f"partial request block {ordinal} score identity drift")
            value = float(row["score"])
            if not math.isfinite(value):
                raise ValueError(f"partial request block {ordinal} non-finite score")
            values.append(value)
        request_ranges.append(max(values) - min(values))
        prompt_count += int(block["prompt_at_max_boundary"] is True)
        score_rows += len(rows)
        rolling = _rolling_block_digest(rolling, str(block["rows_sha256"]))
        last_request_id = record.request_id
    return {
        "completed_requests": len(lines),
        "completed_score_rows": score_rows,
        "last_request_id": last_request_id,
        "partial_hasher": partial_hasher,
        "partial_sha256": partial_hasher.hexdigest(),
        "prompt_at_max_boundary_requests": prompt_count,
        "request_ranges": request_ranges,
        "rolling_request_blocks_sha256": rolling,
    }


def _finalize_run(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    records: Sequence[dict[str, Any]],
    method_id: str,
    condition_id: str,
    reference: Mapping[str, Any],
    full_population_requests: int,
    smoke: bool,
) -> dict[str, Any]:
    observed = _validate_partial(
        run_dir / PARTIAL_FILENAME,
        records=records,
        method_id=method_id,
        condition_id=condition_id,
    )
    if observed["completed_requests"] != len(records):
        raise ValueError("cannot finalize incomplete mechanism score coverage")
    for key in (
        "completed_requests",
        "completed_score_rows",
        "partial_sha256",
        "prompt_at_max_boundary_requests",
        "rolling_request_blocks_sha256",
    ):
        if progress.get(key) != observed[key]:
            raise ValueError(f"final progress mismatch: {key}")

    temporary = run_dir / f".{SCORES_FILENAME}.tmp-{os.getpid()}"
    with temporary.open("x", encoding="utf-8") as handle:
        for block in _iter_partial_blocks(run_dir / PARTIAL_FILENAME):
            for row in block["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, run_dir / SCORES_FILENAME)
    scores_sha256 = sha256_file(run_dir / SCORES_FILENAME)
    full_canary = None
    if condition_id == "full" and len(records) == full_population_requests:
        full_canary = scores_sha256 == reference["scores_sha256"]
        if not full_canary:
            metadata.update(
                {
                    "full_canary_passed": False,
                    "qrels_read": False,
                    "result_eligible": False,
                    "scores_sha256": scores_sha256,
                    "status": "failed_full_canary",
                }
            )
            _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
            raise ValueError("full canary scores differ from frozen reference bundle")
    nonconstant = sum(value > 1.0e-8 for value in observed["request_ranges"])
    if not smoke and nonconstant == 0:
        metadata.update(
            {
                "qrels_read": False,
                "result_eligible": False,
                "status": "failed_globally_degenerate",
            }
        )
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        raise ValueError("uncapped mechanism score run is globally degenerate")
    coverage_complete = len(records) == full_population_requests
    result_eligible = not smoke and coverage_complete and full_canary is not False
    metadata.update(
        {
            "completed_requests": observed["completed_requests"],
            "coverage_complete": coverage_complete,
            "full_canary_passed": full_canary,
            "non_result_reason": (
                "max_score_requests_cap" if smoke else None
            ),
            "prompt_at_max_boundary_requests": observed[
                "prompt_at_max_boundary_requests"
            ],
            "qrels_read": False,
            "request_count": observed["completed_requests"],
            "result_eligible": result_eligible,
            "score_non_degeneracy": {
                "max_request_range": max(observed["request_ranges"], default=0.0),
                "mean_request_range": (
                    sum(observed["request_ranges"])
                    / len(observed["request_ranges"])
                    if observed["request_ranges"]
                    else 0.0
                ),
                "nonconstant_requests_at_1e_8": nonconstant,
                "threshold": 1.0e-8,
            },
            "score_rows": observed["completed_score_rows"],
            "scores_sha256": scores_sha256,
            "status": "completed",
        }
    )
    progress.update({"status": "completed", "updated_at": _utc_now()})
    _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
    return metadata


def _record_wall_exit(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    segment_elapsed: float,
    target_request_count: int,
) -> dict[str, Any]:
    metadata.update(
        {
            "completed_requests": int(progress["completed_requests"]),
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + float(segment_elapsed),
            "qrels_read": False,
            "remaining_requests": target_request_count
            - int(progress["completed_requests"]),
            "result_eligible": False,
            "resumable": True,
            "status": "wall_time_exhausted",
        }
    )
    progress.update({"status": "wall_time_exhausted", "updated_at": _utc_now()})
    _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
    return metadata


def _iter_partial_blocks(path: Path):
    for value in iter_jsonl(path):
        yield value


def _read_structured_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mechanism assignment manifest must be an object")
    return value


def _read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {role}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return value


def _first_present(
    primary: Mapping[str, Any], secondary: Mapping[str, Any], *names: str
) -> Any:
    for source in (primary, secondary):
        for name in names:
            if name in source:
                return source[name]
    return None


def _require_zero_audit(
    manifest: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    names: Sequence[str],
    role: str,
) -> None:
    values = []
    for source in (
        manifest,
        manifest.get("audit", {}),
        condition,
        condition.get("audit", {}),
    ):
        if not isinstance(source, Mapping):
            continue
        values.extend(source[name] for name in names if name in source)
    if not values:
        raise ValueError(f"assignment manifest lacks a {role} audit")
    try:
        numeric = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"assignment manifest {role} audit is not numeric") from exc
    if any(value != 0 for value in numeric):
        raise ValueError(f"assignment manifest {role} audit failed")


def _assert_no_forbidden_keys(value: Any, location: str) -> None:
    if isinstance(value, dict):
        forbidden = set(value) & FORBIDDEN_MODEL_INPUT_FIELDS
        if forbidden:
            raise ValueError(f"{location} contains forbidden fields: {sorted(forbidden)}")
        for key, nested in value.items():
            _assert_no_forbidden_keys(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_forbidden_keys(nested, f"{location}[{index}]")


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_and_sync(path: Path, payload: bytes) -> None:
    with path.open("ab", buffering=0) as handle:
        written = handle.write(payload)
        if written != len(payload):
            raise OSError("short write while committing a request score block")
        os.fsync(handle.fileno())


def _rolling_block_digest(previous: str, block_digest: str) -> str:
    return sha256_text(f"{previous}:{block_digest}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return sha256_text(_canonical_json(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _monotonic() -> float:
    return time.perf_counter()


__all__ = [
    "MAX_WALL_SECONDS",
    "MECHANISM_PROBE_MANIFEST_PATH",
    "MECHANISM_PROBE_MANIFEST_SHA256",
    "mechanism_scorer_implementation_identity",
    "write_mechanism_intervention_scores",
]
