"""Qrels-free full/null scorer for the Q2 M3 matched training control.

This module intentionally has no evaluator import and never constructs a
qrels path.  It admits only completed formal matched-control checkpoints, then
scores the frozen 8,000-request internal-dev population with complete finite
coverage over all 160,753 candidates.  Request-block persistence makes each
score job independently resumable within the registered wall-time boundary.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import (
    FORBIDDEN_MODEL_INPUT_FIELDS,
    SERIALIZED_INPUT_FIELDS,
    ModelRecord,
    sanitize_record_for_model,
)
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _checkpoint_identity,
    _git_revision,
    _implementation_identity as frozen_ranker_implementation_identity,
    _load_model_and_tokenizer,
    _runtime_metadata,
    _score_yes_no_request,
    _validate_run_id,
    load_v12_ranker_config,
)
from myrec.mechanism.matched_training_control import (
    BALANCED_QUOTAS,
    CONDITIONS,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_WALL_SECONDS,
    METHOD_ID,
    OPTIMIZER_STEPS,
    ROLE,
    TRAIN_GROUPS,
    _canonical_sha256,
    _read_json,
    _validate_q2_recipe,
    _validate_registered_probe,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


HISTORY_CONDITIONS = ("full", "null")
EXPECTED_REQUESTS = 8000
EXPECTED_SCORE_ROWS = 160753
PARTIAL_FILENAME = "request_blocks.partial.jsonl"
PROGRESS_FILENAME = "progress.json"
METADATA_FILENAME = "metadata.json"
SCORES_FILENAME = "scores.jsonl"


def write_q2_matched_control_scores(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    sampling_condition: str,
    history_condition: str,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    probe_manifest_path: str | Path = (
        "experiments/motivation/probe_manifest.yaml"
    ),
    command: Sequence[str] | None = None,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
) -> dict[str, Any]:
    """Score one completed formal checkpoint under full or null history."""

    started = time.perf_counter()
    _validate_run_id(run_id)
    if sampling_condition not in CONDITIONS:
        raise ValueError(f"unsupported sampling condition={sampling_condition!r}")
    if history_condition not in HISTORY_CONDITIONS:
        raise ValueError(f"unsupported history condition={history_condition!r}")
    if not device or not str(device).strip():
        raise ValueError("an explicit scoring device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not (
        0.0 < max_wall_seconds <= MAX_WALL_SECONDS
    ):
        raise ValueError(
            f"max_wall_seconds must be in (0, {int(MAX_WALL_SECONDS)}]"
        )

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    probe = _validate_registered_probe(probe_manifest_path)
    config = load_v12_ranker_config(config_path)
    _validate_q2_recipe(config, config_path=config_path, probe=probe["payload"])
    population = config["_protocol"]["data"]["development_population"]

    # No qrels path is constructed in this module.  All source bindings below
    # are label-free manifests or records.
    records_path = standardized_dir / "records_dev.jsonl"
    dataset_manifest_path = standardized_dir / "manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    hashes = {
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "records_dev_sha256": sha256_file(records_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
    }
    expected_hashes = {
        "candidate_manifest_sha256": population["candidate_manifest_sha256"],
        "dataset_manifest_sha256": population["manifest_sha256"],
        "records_dev_sha256": population["records_dev_sha256"],
        "request_manifest_sha256": population["request_manifest_sha256"],
    }
    for key, expected in expected_hashes.items():
        if hashes[key] != str(expected):
            raise ValueError(f"frozen internal-dev hash mismatch: {key}")
    dataset_manifest = _read_json(dataset_manifest_path)
    if dataset_manifest.get("dataset_id") != "kuaisearch" or str(
        dataset_manifest.get("dataset_version")
    ) != str(population["dataset_version"]):
        raise ValueError("dataset manifest differs from frozen internal dev")

    records = _load_label_free_records(records_path)
    candidates = _load_dev_candidate_manifest(candidate_manifest_path)
    _validate_record_candidate_binding(records, candidates)
    expected_requests = int(population["internal_dev_requests"])
    expected_rows = int(
        config["_protocol"]["data"]["development_population"].get(
            "internal_dev_candidate_rows", EXPECTED_SCORE_ROWS
        )
    )
    # The frozen protocol records this count under model_serialization rather
    # than development_population, so keep the registered constant explicit.
    if expected_rows != EXPECTED_SCORE_ROWS:
        expected_rows = EXPECTED_SCORE_ROWS
    observed_rows = sum(len(record.candidates) for record in records)
    if len(records) != EXPECTED_REQUESTS or expected_requests != EXPECTED_REQUESTS:
        raise ValueError("internal-dev request coverage must be exactly 8000")
    if observed_rows != EXPECTED_SCORE_ROWS or expected_rows != EXPECTED_SCORE_ROWS:
        raise ValueError("internal-dev candidate coverage must be exactly 160753")

    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    training_admission = validate_matched_training_checkpoint_metadata(
        training_metadata,
        sampling_condition=sampling_condition,
        config=config,
        training_metadata_path=training_metadata_path,
    )
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, METHOD_ID
    )
    if checkpoint_id != training_metadata.get("checkpoint_id"):
        raise ValueError("matched-control checkpoint artifacts changed")

    frozen_identity = frozen_ranker_implementation_identity()
    scorer_identity = matched_control_scorer_implementation_identity()
    base_scoring_signature = {
        "schema_version": 1,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "frozen_ranker_implementation_digest": frozen_identity["digest"],
        "input_fields_used": list(SERIALIZED_INPUT_FIELDS),
        "matched_recipe_sha256": training_metadata["matched_recipe_sha256"],
        "method_id": METHOD_ID,
        "prompt_contract": "motivation_v12_project_owned_v2",
        "protocol_sha256": config["protocol"]["sha256"],
        "score_batch_size": int(config.get("scoring", {}).get("batch_size", 16)),
    }
    condition_id = f"{sampling_condition}__{history_condition}"
    run_contract = {
        "schema_version": 1,
        "base_scoring_signature": base_scoring_signature,
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "checkpoint_id": checkpoint_id,
        "condition_id": condition_id,
        "history_condition": history_condition,
        "matched_recipe_sha256": training_metadata["matched_recipe_sha256"],
        "probe_manifest_sha256": probe["sha256"],
        "records_dev_sha256": hashes["records_dev_sha256"],
        "request_manifest_sha256": hashes["request_manifest_sha256"],
        "run_id": run_id,
        "sampling_condition": sampling_condition,
        "scorer_implementation_digest": scorer_identity["digest"],
        "training_metadata_sha256": training_admission["metadata_sha256"],
    }
    run_contract_sha256 = _canonical_sha256(run_contract)
    base_metadata = {
        "schema_version": 1,
        "base_scoring_signature": base_scoring_signature,
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "checkpoint_files": checkpoint_files,
        "checkpoint_id": checkpoint_id,
        "code_revision": _git_revision(),
        "command": list(command or sys.argv),
        "condition_id": condition_id,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "dataset_id": "kuaisearch",
        "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
        "dataset_version": str(population["dataset_version"]),
        "evidence_mode": "mechanism_diagnostic",
        "frozen_ranker_implementation_identity": frozen_identity,
        "history_condition": history_condition,
        "input_fields_used": list(SERIALIZED_INPUT_FIELDS),
        "matched_control_scorer_implementation_identity": scorer_identity,
        "matched_training_control": {
            "matched_recipe": copy.deepcopy(training_metadata["matched_recipe"]),
            "matched_recipe_sha256": training_metadata["matched_recipe_sha256"],
            "role": ROLE,
            "sampling": copy.deepcopy(training_metadata["sampling"]),
            "sampling_condition": sampling_condition,
            "training_checkpoint_id": checkpoint_id,
            "training_metadata_path": str(training_metadata_path),
            "training_metadata_sha256": training_admission["metadata_sha256"],
            "training_run_contract_sha256": training_metadata[
                "run_contract_sha256"
            ],
        },
        "mechanism_probe_manifest": probe["identity"],
        "method_id": METHOD_ID,
        "qrels_read": False,
        "records_path": str(records_path),
        "records_sha256": hashes["records_dev_sha256"],
        "request_manifest_path": str(request_manifest_path),
        "request_manifest_sha256": hashes["request_manifest_sha256"],
        "result_eligible": False,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "run_id": run_id,
        "sampling_condition": sampling_condition,
        "scoring_signature": copy.deepcopy(base_scoring_signature),
        "split": "dev",
        "status": "initialized",
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": training_admission["metadata_sha256"],
    }
    metadata, progress, request_ranges = _prepare_score_run(
        run_dir,
        records=records,
        base_metadata=base_metadata,
        run_contract_sha256=run_contract_sha256,
        condition_id=condition_id,
        resume=resume,
    )
    start_ordinal = int(progress["completed_requests"])
    if start_ordinal >= EXPECTED_REQUESTS:
        return _finalize_score_run(
            run_dir,
            records=records,
            metadata=metadata,
            progress=progress,
            request_ranges=request_ranges,
        )

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
        metadata.update(_runtime_metadata(METHOD_ID, torch, transformers))
        metadata["status"] = "running"
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        score_batch_size = int(config.get("scoring", {}).get("batch_size", 16))
        with torch.inference_mode():
            for ordinal in range(start_ordinal, EXPECTED_REQUESTS):
                if time.perf_counter() - started >= max_wall_seconds:
                    metadata.update(
                        {
                            "elapsed_seconds": float(
                                metadata.get("elapsed_seconds", 0.0)
                            )
                            + (time.perf_counter() - started),
                            "progress": progress,
                            "result_eligible": False,
                            "status": "wall_time_exhausted",
                        }
                    )
                    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
                    return metadata
                record = records[ordinal]
                history = list(record.history) if history_condition == "full" else []
                request_scores, at_boundary = _score_yes_no_request(
                    model,
                    tokenizer,
                    record,
                    history,
                    config,
                    device=str(device),
                    batch_size=score_batch_size,
                )
                rows, score_range = _validated_request_rows(record, request_scores)
                block = {
                    "condition_id": condition_id,
                    "ordinal": ordinal,
                    "prompt_at_max_boundary": bool(at_boundary),
                    "request_id": record.request_id,
                    "rows": rows,
                    "rows_sha256": _canonical_sha256(rows),
                }
                _append_and_sync(
                    run_dir / PARTIAL_FILENAME,
                    _canonical_json(block) + "\n",
                )
                request_ranges.append(score_range)
                progress.update(
                    {
                        "completed_requests": ordinal + 1,
                        "completed_score_rows": int(progress["completed_score_rows"])
                        + len(rows),
                        "last_request_id": record.request_id,
                        "prompt_at_max_boundary_requests": int(
                            progress["prompt_at_max_boundary_requests"]
                        )
                        + int(bool(at_boundary)),
                        "status": "running",
                        "updated_at": _utc_now(),
                    }
                )
                _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
    except Exception as exc:
        metadata.update(
            {
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "progress": progress,
                "qrels_read": False,
                "result_eligible": False,
                "status": "failed",
            }
        )
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        raise

    metadata["elapsed_seconds"] = float(metadata.get("elapsed_seconds", 0.0)) + (
        time.perf_counter() - started
    )
    return _finalize_score_run(
        run_dir,
        records=records,
        metadata=metadata,
        progress=progress,
        request_ranges=request_ranges,
    )


def validate_matched_training_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    sampling_condition: str,
    config: Mapping[str, Any],
    training_metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Reject smoke/incomplete checkpoints before model loading."""

    expected = {
        "condition": sampling_condition,
        "config_sha256": config["_config_sha256"],
        "evidence_mode": "mechanism_diagnostic",
        "formal_checkpoint": True,
        "method_id": METHOD_ID,
        "result_eligible": True,
        "role": ROLE,
        "status": "completed",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"matched training checkpoint admission failed: {key}")
    if int(metadata.get("base_initialization_seed", -1)) != 20260714:
        raise ValueError("matched training base initialization seed mismatch")
    progress = metadata.get("progress")
    if not isinstance(progress, Mapping):
        raise ValueError("matched training checkpoint lacks progress")
    if int(progress.get("group_exposures", -1)) != TRAIN_GROUPS or int(
        progress.get("optimizer_steps", -1)
    ) != OPTIMIZER_STEPS:
        raise ValueError("matched training checkpoint is not exactly 4096/256")
    if int(progress.get("micro_steps", -1)) != TRAIN_GROUPS:
        raise ValueError("matched training checkpoint micro-step count mismatch")
    recipe = metadata.get("matched_recipe")
    if not isinstance(recipe, Mapping) or _canonical_sha256(recipe) != metadata.get(
        "matched_recipe_sha256"
    ):
        raise ValueError("matched training recipe hash mismatch")
    _validate_matched_recipe_against_config(recipe, config)
    implementation = metadata.get(
        "matched_training_control_implementation_identity"
    )
    if not isinstance(implementation, Mapping) or implementation.get(
        "digest"
    ) != recipe.get("implementation_digest"):
        raise ValueError("matched training implementation identity mismatch")
    sampling = metadata.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("condition") != sampling_condition:
        raise ValueError("matched training sampling declaration mismatch")
    selection_sha256 = sampling.get("selection_sha256")
    if not selection_sha256 or selection_sha256 != metadata.get("selection_sha256"):
        raise ValueError("matched training selection hash mismatch")
    surface_counts = sampling.get("selected_surface_counts")
    if not isinstance(surface_counts, Mapping) or sum(
        int(surface_counts.get(surface, 0)) for surface in BALANCED_QUOTAS
    ) != TRAIN_GROUPS:
        raise ValueError("matched training sampling counts do not total 4096")
    if sampling_condition == "surface_balanced" and {
        surface: int(surface_counts.get(surface, -1)) for surface in BALANCED_QUOTAS
    } != BALANCED_QUOTAS:
        raise ValueError("surface-balanced checkpoint quotas differ from registration")
    run_contract = metadata.get("run_contract")
    if not isinstance(run_contract, Mapping) or _canonical_sha256(
        run_contract
    ) != metadata.get("run_contract_sha256"):
        raise ValueError("matched training run-contract hash mismatch")
    probe = metadata.get("probe_manifest")
    if (
        not isinstance(probe, Mapping)
        or probe.get("verified") is not True
        or not recipe.get("probe_manifest_sha256")
        or probe.get("sha256") != recipe.get("probe_manifest_sha256")
    ):
        raise ValueError("matched training probe-manifest binding mismatch")
    qrels_access = metadata.get("qrels_access")
    if not isinstance(qrels_access, Mapping) or qrels_access.get(
        "qrels_dev_read"
    ) is not False:
        raise ValueError("matched training checkpoint crossed dev qrels boundary")
    if training_metadata_path is not None:
        selection_path = _resolve_declared_path(str(metadata.get("selection_path", "")))
        if not selection_path.is_file() or sha256_file(selection_path) != metadata.get(
            "selection_file_sha256"
        ):
            raise ValueError("matched training persisted selection file mismatch")
        if _canonical_sha256(_read_json(selection_path)) != metadata.get(
            "selection_sha256"
        ):
            raise ValueError("matched training persisted selection content mismatch")
        contract_path = selection_path.parent / "run_contract.json"
        if not contract_path.is_file() or sha256_file(contract_path) != metadata.get(
            "run_contract_file_sha256"
        ):
            raise ValueError("matched training persisted run-contract file mismatch")
        if _canonical_sha256(_read_json(contract_path)) != metadata.get(
            "run_contract_sha256"
        ):
            raise ValueError("matched training persisted run-contract content mismatch")
    return {
        "matched_recipe_sha256": metadata["matched_recipe_sha256"],
        "metadata_sha256": (
            sha256_file(training_metadata_path)
            if training_metadata_path is not None
            else _canonical_sha256(metadata)
        ),
        "status": "admitted",
    }


def _validate_matched_recipe_against_config(
    recipe: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    training = config["training"]
    model = config["model"]
    method = config["method"]
    expected = {
        "anchor": METHOD_ID,
        "batch_requests": 1,
        "config_sha256": config["_config_sha256"],
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "group_exposures": TRAIN_GROUPS,
        "history_budget": training["history_budget"],
        "input_fields_used": list(SERIALIZED_INPUT_FIELDS),
        "max_grad_norm": training["max_grad_norm"],
        "max_length": training["max_length"],
        "optimizer_steps": OPTIMIZER_STEPS,
        "protocol_sha256": config["protocol"]["sha256"],
        "role": ROLE,
    }
    for key, value in expected.items():
        if recipe.get(key) != value:
            raise ValueError(f"matched training fixed recipe mismatch: {key}")
    backbone = recipe.get("backbone_initialization")
    expected_backbone = {
        "artifact_manifest_sha256": model["base_artifact_manifest_sha256"],
        "base_model_path": model["base_model_path"],
        "seed": 20260714,
        "tokenizer_sha256": model["tokenizer_sha256"],
        "weights_sha256": model["base_weights_sha256"],
    }
    if backbone != expected_backbone:
        raise ValueError("matched training backbone initialization mismatch")
    optimizer = recipe.get("optimizer")
    if not isinstance(optimizer, Mapping) or optimizer.get(
        "learning_rate"
    ) != training["learning_rate"] or optimizer.get("weight_decay") != training[
        "weight_decay"
    ]:
        raise ValueError("matched training optimizer/LR mismatch")
    objective = recipe.get("objective")
    if not isinstance(objective, Mapping) or objective.get(
        "pairwise_loss_weight"
    ) != method["pairwise_loss_weight"] or objective.get(
        "listwise_loss_weight"
    ) != method[
        "listwise_loss_weight"
    ]:
        raise ValueError("matched training objective mismatch")
    scheduler = recipe.get("scheduler")
    if not isinstance(scheduler, Mapping) or int(
        scheduler.get("total_steps", -1)
    ) != OPTIMIZER_STEPS:
        raise ValueError("matched training total-step schedule mismatch")


def matched_control_scorer_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = {
        "scripts/score_q2_matched_control.py": root
        / "scripts/score_q2_matched_control.py",
        "src/myrec/mechanism/matched_control_scorer.py": Path(__file__).resolve(),
    }
    files = []
    for relative, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing matched-control scorer: {path}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {"digest": _canonical_sha256(files), "files": files}


def _load_label_free_records(path: Path) -> list[ModelRecord]:
    records = []
    seen: set[str] = set()
    for raw in iter_jsonl(path):
        _assert_no_forbidden_keys(raw, f"record {raw.get('request_id', '<unknown>')}")
        record = sanitize_record_for_model(raw)
        if record.request_id in seen:
            raise ValueError(f"duplicate records_dev request_id={record.request_id}")
        seen.add(record.request_id)
        records.append(record)
    if not records:
        raise ValueError("records_dev is empty")
    return records


def _load_dev_candidate_manifest(path: Path) -> dict[str, list[str]]:
    payload = _read_json(path)
    result: dict[str, list[str]] = {}
    for entry in payload.get("entries", []):
        if entry.get("split") != "dev":
            continue
        request_id = str(entry.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError(f"empty or duplicate candidate request_id={request_id!r}")
        item_ids = [str(value) for value in entry.get("candidate_item_ids", [])]
        if len(item_ids) < 2 or len(set(item_ids)) != len(item_ids):
            raise ValueError(f"invalid candidate slate request_id={request_id}")
        result[request_id] = item_ids
    if not result:
        raise ValueError("candidate manifest has no internal-dev entries")
    return result


def _validate_record_candidate_binding(
    records: Sequence[ModelRecord], candidates: Mapping[str, Sequence[str]]
) -> None:
    if {record.request_id for record in records} != set(candidates):
        raise ValueError("records_dev and candidate manifest request coverage differ")
    for record in records:
        observed = [str(row["item_id"]) for row in record.candidates]
        if observed != list(candidates[record.request_id]):
            raise ValueError(
                f"candidate order/identity mismatch request_id={record.request_id}"
            )


def _validated_request_rows(
    record: ModelRecord, request_scores: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], float]:
    expected_ids = [str(row["item_id"]) for row in record.candidates]
    if set(request_scores) != set(expected_ids) or len(request_scores) != len(
        expected_ids
    ):
        raise ValueError(f"candidate score coverage failed for {record.request_id}")
    rows = []
    values = []
    for item_id in expected_ids:
        try:
            score = float(request_scores[item_id])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid score request_id={record.request_id} item_id={item_id}"
            ) from exc
        if not math.isfinite(score):
            raise ValueError(
                f"non-finite score request_id={record.request_id} item_id={item_id}"
            )
        values.append(score)
        rows.append(
            {
                "candidate_item_id": item_id,
                "method_id": METHOD_ID,
                "request_id": record.request_id,
                "score": score,
            }
        )
    return rows, max(values) - min(values)


def _prepare_score_run(
    run_dir: Path,
    *,
    records: Sequence[ModelRecord],
    base_metadata: dict[str, Any],
    run_contract_sha256: str,
    condition_id: str,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
    partial_path = run_dir / PARTIAL_FILENAME
    metadata_path = run_dir / METADATA_FILENAME
    progress_path = run_dir / PROGRESS_FILENAME
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"score run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial_path.touch(exist_ok=False)
        metadata = {**base_metadata, "elapsed_seconds": 0.0, "resume_lineage": []}
        progress = {
            "schema_version": 1,
            "completed_requests": 0,
            "completed_score_rows": 0,
            "last_request_id": None,
            "prompt_at_max_boundary_requests": 0,
            "resume_count": 0,
            "run_contract_sha256": run_contract_sha256,
            "status": "initialized",
            "updated_at": _utc_now(),
        }
        _write_json_atomic(metadata_path, metadata)
        _write_json_atomic(progress_path, progress)
        return metadata, progress, []

    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    metadata = _read_json(metadata_path)
    progress = _read_json(progress_path)
    if metadata.get("run_contract_sha256") != run_contract_sha256 or progress.get(
        "run_contract_sha256"
    ) != run_contract_sha256:
        raise ValueError("score resume run contract mismatch")
    if metadata.get("status") not in {
        "failed",
        "initialized",
        "running",
        "wall_time_exhausted",
    }:
        raise ValueError(f"score run is not resumable: {metadata.get('status')}")
    observed = _validate_partial_blocks(
        partial_path, records=records, condition_id=condition_id
    )
    expected_progress = {
        "completed_requests": observed["completed_requests"],
        "completed_score_rows": observed["completed_score_rows"],
        "last_request_id": observed["last_request_id"],
        "prompt_at_max_boundary_requests": observed[
            "prompt_at_max_boundary_requests"
        ],
    }
    persisted_count = int(progress.get("completed_requests", -1))
    observed_count = int(observed["completed_requests"])
    if persisted_count > observed_count:
        raise ValueError("score resume progress is ahead of the durable partial file")
    if persisted_count < observed_count:
        # A complete request block is fsynced before progress is replaced.  If
        # interruption occurs in that narrow window, the validated block is
        # authoritative and progress can be recovered without rescoring it.
        progress.update(expected_progress)
        metadata["partial_progress_recovery"] = {
            "durable_completed_requests": observed_count,
            "persisted_completed_requests": persisted_count,
            "recovered_at": _utc_now(),
        }
    else:
        for key, value in expected_progress.items():
            if progress.get(key) != value:
                raise ValueError(f"score resume partial/progress mismatch: {key}")
    lineage = list(metadata.get("resume_lineage", []))
    lineage.append(
        {
            "completed_requests": observed["completed_requests"],
            "from_status": metadata["status"],
            "resumed_at": _utc_now(),
        }
    )
    metadata.update({"resume_lineage": lineage, "status": "initialized"})
    progress.update(
        {
            "resume_count": int(progress.get("resume_count", 0)) + 1,
            "status": "initialized",
            "updated_at": _utc_now(),
        }
    )
    _write_json_atomic(metadata_path, metadata)
    _write_json_atomic(progress_path, progress)
    return metadata, progress, observed["request_ranges"]


def _validate_partial_blocks(
    path: Path,
    *,
    records: Sequence[ModelRecord],
    condition_id: str,
) -> dict[str, Any]:
    request_ranges = []
    score_rows = 0
    prompt_boundaries = 0
    last_request_id = None
    count = 0
    for ordinal, block in enumerate(iter_jsonl(path)):
        if ordinal >= len(records):
            raise ValueError("partial score file exceeds frozen request coverage")
        record = records[ordinal]
        if block.get("ordinal") != ordinal or block.get("request_id") != record.request_id:
            raise ValueError("partial score request order mismatch")
        if block.get("condition_id") != condition_id:
            raise ValueError("partial score condition mismatch")
        rows, score_range = _validated_request_rows(
            record,
            {
                str(row["candidate_item_id"]): row["score"]
                for row in block.get("rows", [])
            },
        )
        if rows != block.get("rows") or _canonical_sha256(rows) != block.get(
            "rows_sha256"
        ):
            raise ValueError("partial score block row hash mismatch")
        request_ranges.append(score_range)
        score_rows += len(rows)
        prompt_boundaries += int(bool(block.get("prompt_at_max_boundary")))
        last_request_id = record.request_id
        count += 1
    return {
        "completed_requests": count,
        "completed_score_rows": score_rows,
        "last_request_id": last_request_id,
        "prompt_at_max_boundary_requests": prompt_boundaries,
        "request_ranges": request_ranges,
    }


def _finalize_score_run(
    run_dir: Path,
    *,
    records: Sequence[ModelRecord],
    metadata: dict[str, Any],
    progress: dict[str, Any],
    request_ranges: Sequence[float],
) -> dict[str, Any]:
    observed = _validate_partial_blocks(
        run_dir / PARTIAL_FILENAME,
        records=records,
        condition_id=str(metadata["condition_id"]),
    )
    if observed["completed_requests"] != EXPECTED_REQUESTS:
        raise ValueError("cannot finalize without exactly 8000 requests")
    if observed["completed_score_rows"] != EXPECTED_SCORE_ROWS:
        raise ValueError("cannot finalize without exactly 160753 score rows")
    if list(request_ranges) != observed["request_ranges"]:
        raise ValueError("in-memory and persisted score ranges differ")
    scores_path = run_dir / SCORES_FILENAME
    if scores_path.exists():
        raise FileExistsError(scores_path)
    temporary = scores_path.with_name(f".{scores_path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        for block in iter_jsonl(run_dir / PARTIAL_FILENAME):
            for row in block["rows"]:
                handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, scores_path)
    nonconstant = sum(value > 1.0e-8 for value in observed["request_ranges"])
    if nonconstant <= 0:
        raise ValueError("formal matched-control scores are degenerate on every request")
    progress.update({"status": "completed", "updated_at": _utc_now()})
    metadata.update(
        {
            "error": None,
            "progress": progress,
            "prompt_at_max_boundary_requests": observed[
                "prompt_at_max_boundary_requests"
            ],
            "qrels_read": False,
            "request_count": EXPECTED_REQUESTS,
            "result_eligible": True,
            "score_non_degeneracy": {
                "nonconstant_requests_at_1e_8": nonconstant,
                "threshold": 1.0e-8,
            },
            "score_rows": EXPECTED_SCORE_ROWS,
            "scores_sha256": sha256_file(scores_path),
            "status": "completed",
        }
    )
    _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
    return metadata


def _assert_no_forbidden_keys(value: Any, role: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(FORBIDDEN_MODEL_INPUT_FIELDS & set(value))
        if forbidden:
            raise ValueError(f"{role} contains forbidden model fields: {forbidden}")
        for key, nested in value.items():
            _assert_no_forbidden_keys(nested, f"{role}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_forbidden_keys(nested, f"{role}[{index}]")


def _resolve_declared_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_and_sync(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "EXPECTED_REQUESTS",
    "EXPECTED_SCORE_ROWS",
    "HISTORY_CONDITIONS",
    "matched_control_scorer_implementation_identity",
    "validate_matched_training_checkpoint_metadata",
    "write_q2_matched_control_scores",
]
