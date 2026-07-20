"""Train-only M3 gradient diagnostics for the frozen Q2/Q3 objectives.

This module never constructs a dev/confirmation/test qrels path.  It fixes the
three 96-request train surfaces before model loading or any loss/gradient
access, then measures the project-owned frozen objective request by request.
No optimizer step is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import (
    TrainingGroup,
    load_training_groups,
)
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _assert_frozen_training_population,
    _checkpoint_identity,
    _git_revision,
    _implementation_identity as _frozen_ranker_implementation_identity,
    _load_model_and_tokenizer,
    _runtime_metadata,
    _seed_everything,
    _single_token_id,
    _training_batch_loss,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


PROBE_MANIFEST_PATH = Path("experiments/motivation/probe_manifest.yaml")
PROBE_MANIFEST_SHA256 = (
    "adedf0e662b9d8529162b8abffedcf6b10962913f28580af6119d807cc5d929c"
)
PROBE_MANIFEST_ID = "motivation_mechanism_first_diagnosis_v1"
SUPPORTED_METHODS = (
    "q2_recranker_generalqwen",
    "q3_tallrec_generalqwen",
)
SUPPORTED_STATES = ("base_initialization", "frozen_final_checkpoint")
SURFACES = ("recurrence", "strict_transfer", "other_overlap")
CONTROLS = ("observed", "within_request_label_shuffle")
REQUESTS_PER_SURFACE = 96
SELECTION_SEED = 20_260_717
MODEL_INITIALIZATION_SEED = 20_260_714
Q2_BLOCKS = (0, 6, 13, 20, 27)
MAX_WALL_SECONDS = 13_500.0
_Q2_PARAMETER_PATTERN = re.compile(
    r"(?:^|\.)layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.(weight|bias)$"
)


@dataclass(frozen=True)
class GradientScope:
    """One registered parameter scope, optionally restricted to tensor rows."""

    name: str
    parameter_name: str
    parameter: Any
    row_indices: tuple[int, ...] | None = None


class NumericalGradientError(RuntimeError):
    """A finite-value contract failed without changing mechanical coverage."""


def run_gradient_diagnostic(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    state: str,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    probe_manifest_path: str | Path = PROBE_MANIFEST_PATH,
    command: Sequence[str] | None = None,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests_per_surface: int | None = None,
) -> dict[str, Any]:
    """Run one Q2/Q3 and base/final M3 job with resumable surface cells."""

    _validate_run_id(run_id)
    if state not in SUPPORTED_STATES:
        raise ValueError(f"unsupported gradient diagnostic state={state!r}")
    if not device or not str(device).strip():
        raise ValueError("an explicit device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not (
        0.0 < max_wall_seconds <= MAX_WALL_SECONDS
    ):
        raise ValueError(
            f"max_wall_seconds must be in (0, {int(MAX_WALL_SECONDS)}]"
        )
    if max_requests_per_surface is not None:
        max_requests_per_surface = int(max_requests_per_surface)
        if not 0 < max_requests_per_surface < REQUESTS_PER_SURFACE:
            raise ValueError(
                "max_requests_per_surface must be in [1, 95] and is smoke-only"
            )
    target_per_surface = max_requests_per_surface or REQUESTS_PER_SURFACE
    evidence_mode = (
        "smoke_non_result"
        if max_requests_per_surface is not None
        else "mechanism_gradient_diagnostic"
    )

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    dataset_manifest_path = standardized_dir / "manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"

    probe = _load_probe_manifest(probe_manifest_path)
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("M3 gradient diagnostics are registered only for Q2/Q3")
    population = config["_protocol"]["data"]["development_population"]
    _assert_frozen_training_population(standardized_dir, config)
    hashes = {
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "qrels_train_sha256": sha256_file(qrels_path),
        "records_train_sha256": sha256_file(records_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
    }
    expected_hashes = {
        "candidate_manifest_sha256": population["candidate_manifest_sha256"],
        "dataset_manifest_sha256": population["manifest_sha256"],
        "qrels_train_sha256": population["qrels_train_sha256"],
        "records_train_sha256": population["records_train_sha256"],
        "request_manifest_sha256": population["request_manifest_sha256"],
    }
    for key, expected in expected_hashes.items():
        if hashes[key] != str(expected):
            raise ValueError(f"frozen train population hash mismatch: {key}")

    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path, "training metadata")
    _validate_scoring_checkpoint_provenance(
        training_metadata,
        config,
        allow_smoke=False,
    )
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir,
        method_id,
    )
    if checkpoint_id != training_metadata.get("checkpoint_id"):
        raise ValueError("frozen final checkpoint identity drift")
    _validate_probe_model_binding(
        probe["payload"],
        method_id=method_id,
        config_path=config_path,
        config_sha256=config["_config_sha256"],
        checkpoint_root=checkpoint_root,
        checkpoint_id=checkpoint_id,
    )

    # This is the only qrels parser in the module.  Selection is completed and
    # persisted before importing/loading the model or touching a loss function.
    gains = _load_train_gains(qrels_path)
    groups, group_stats = load_training_groups(
        records_path,
        qrels_path,
        seed=int(config["training"]["seed"]),
        negatives_per_positive=int(config["training"]["negatives_per_positive"]),
        max_group_size=int(config["training"].get("list_size", 8)),
    )
    selected, selection = select_surface_training_groups(
        groups,
        gains,
        requests_per_surface=target_per_surface,
        selection_seed=SELECTION_SEED,
    )
    selection.update(
        {
            "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
            "config_sha256": config["_config_sha256"],
            "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
            "finalized_before_model_load_and_loss": True,
            "group_construction": group_stats,
            "method_id": method_id,
            "probe_manifest_sha256": probe["sha256"],
            "qrels_train_sha256": hashes["qrels_train_sha256"],
            "registered_requests_per_surface": REQUESTS_PER_SURFACE,
            "records_train_sha256": hashes["records_train_sha256"],
            "request_manifest_sha256": hashes["request_manifest_sha256"],
            "smoke_request_cap": max_requests_per_surface,
            "state": state,
        }
    )
    selection_sha256 = _canonical_sha256(selection)
    implementation = gradient_diagnostic_implementation_identity()
    frozen_ranker_identity = _frozen_ranker_implementation_identity()
    run_contract = {
        "schema_version": 1,
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
        "device": str(device),
        "evidence_mode": evidence_mode,
        "frozen_ranker_implementation_digest": frozen_ranker_identity["digest"],
        "gradient_diagnostic_implementation_digest": implementation["digest"],
        "method_id": method_id,
        "probe_manifest_sha256": probe["sha256"],
        "qrels_train_sha256": hashes["qrels_train_sha256"],
        "records_train_sha256": hashes["records_train_sha256"],
        "request_manifest_sha256": hashes["request_manifest_sha256"],
        "run_id": run_id,
        "selection_sha256": selection_sha256,
        "state": state,
        "target_requests_per_surface": target_per_surface,
        "training_metadata_sha256": sha256_file(training_metadata_path),
    }
    run_contract_sha256 = _canonical_sha256(run_contract)
    base_metadata = {
        "schema_version": 1,
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "code_revision": _git_revision(),
        "command": list(command or sys.argv),
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "dataset_id": "kuaisearch",
        "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
        "dataset_version": population["dataset_version"],
        "dev_confirmation_test_qrels_read": False,
        "evidence_mode": evidence_mode,
        "frozen_ranker_implementation_identity": frozen_ranker_identity,
        "gradient_diagnostic_implementation_identity": implementation,
        "matched_4096_group_256_step_training_control_executed": False,
        "method_id": method_id,
        "model_initialization_seed": MODEL_INITIALIZATION_SEED,
        "negative_control": {
            "id": "within_request_label_shuffle",
            "preserves_gain_multiset": True,
            "seed": SELECTION_SEED,
            "selection": (
                "stable_sha256_permutation_then_deterministic_rotation_until_"
                "at_least_one_gain_position_changes"
            ),
        },
        "optimizer_steps_performed": 0,
        "parameter_contract": _registered_parameter_contract(method_id),
        "probe_manifest": probe["identity"],
        "qrels_access": {
            "qrels_train_path": str(qrels_path),
            "qrels_train_read": True,
            "qrels_train_sha256": hashes["qrels_train_sha256"],
            "qrels_dev_read": False,
            "qrels_confirmation_read": False,
            "qrels_test_read": False,
        },
        "qrels_train_sha256": hashes["qrels_train_sha256"],
        "records_train_sha256": hashes["records_train_sha256"],
        "request_manifest_sha256": hashes["request_manifest_sha256"],
        "registered_requests_per_surface": REQUESTS_PER_SURFACE,
        "result_eligible": False,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "run_id": run_id,
        "selection_sha256": selection_sha256,
        "state": state,
        "status": "selection_finalized",
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": sha256_file(training_metadata_path),
    }
    metadata, progress = _prepare_run(
        run_dir,
        base_metadata=base_metadata,
        selection=selection,
        selection_sha256=selection_sha256,
        run_contract_sha256=run_contract_sha256,
        resume=resume,
    )
    if metadata.get("status") == "completed":
        raise ValueError("gradient diagnostic run is already completed")

    segment_started = _monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_state_model(
            config,
            state=state,
            device=str(device),
            checkpoint_model_dir=checkpoint_model_dir,
            torch_module=torch,
        )
        model.train(True)
        scopes = discover_gradient_scopes(model, tokenizer, method_id)
        parameter_boundary_audit = trainable_parameter_audit(
            model,
            scopes,
            method_id,
        )
        scope_manifest = [
            {
                "name": scope.name,
                "parameter_name": scope.parameter_name,
                "row_indices": list(scope.row_indices or ()),
                "shape": list(scope.parameter.shape),
            }
            for scope in scopes
        ]
        scope_manifest_sha256 = _canonical_sha256(scope_manifest)
        prior_scope_manifest_sha256 = metadata.get("parameter_scope_manifest_sha256")
        if (
            prior_scope_manifest_sha256 is not None
            and prior_scope_manifest_sha256 != scope_manifest_sha256
        ):
            raise ValueError("resumed parameter scope manifest drift")
        prior_parameter_boundary_audit = metadata.get("parameter_boundary_audit")
        if (
            prior_parameter_boundary_audit is not None
            and prior_parameter_boundary_audit != parameter_boundary_audit
        ):
            raise ValueError("resumed trainable parameter boundary drift")
        initialization_fingerprint = (
            _q3_trainable_initialization_fingerprint(scopes)
            if method_id == "q3_tallrec_generalqwen"
            else None
        )
        prior_fingerprint = metadata.get("model_initialization_fingerprint")
        if prior_fingerprint is not None and prior_fingerprint != initialization_fingerprint:
            raise ValueError("resumed model initialization fingerprint drift")
        metadata.update(
            {
                **_runtime_metadata(method_id, torch, transformers),
                "loss_or_gradient_access_started_after_selection": True,
                "model_initialization_fingerprint": initialization_fingerprint,
                "parameter_boundary_audit": parameter_boundary_audit,
                "parameter_scope_manifest": scope_manifest,
                "parameter_scope_manifest_sha256": scope_manifest_sha256,
                "status": "running",
            }
        )
        _write_json_atomic(run_dir / "metadata.json", metadata)

        completed_cells = list(progress.get("completed_cells", []))
        cell_order = [
            f"{control}__{surface}"
            for control in CONTROLS
            for surface in SURFACES
        ]
        _validate_completed_cells(run_dir, completed_cells)
        for cell_id in cell_order:
            if cell_id in completed_cells:
                continue
            control, surface = cell_id.split("__", maxsplit=1)
            attempt_number = _next_attempt_number(run_dir, cell_id)
            attempt_path = (
                run_dir
                / "attempts"
                / f"{cell_id}.attempt{attempt_number:03d}.jsonl"
            )
            cell = _run_cell(
                model,
                tokenizer,
                config,
                scopes,
                selected[surface],
                surface=surface,
                control=control,
                state=state,
                device=str(device),
                attempt_path=attempt_path,
                segment_started=segment_started,
                max_wall_seconds=max_wall_seconds,
                torch_module=torch,
            )
            if not cell["completed"]:
                return _record_wall_exit(
                    run_dir,
                    metadata=metadata,
                    progress=progress,
                    cell_id=cell_id,
                    attempt_path=attempt_path,
                    segment_elapsed=_monotonic() - segment_started,
                )
            _commit_cell(
                run_dir,
                cell_id=cell_id,
                attempt_path=attempt_path,
                cell=cell,
                torch_module=torch,
            )
            completed_cells.append(cell_id)
            progress.update(
                {
                    "completed_cells": completed_cells,
                    "last_completed_cell": cell_id,
                    "status": "running",
                    "updated_at": _utc_now(),
                }
            )
            _write_json_atomic(run_dir / "progress.json", progress)
    except Exception as exc:
        metadata.update(
            {
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "result_eligible": False,
                "status": "failed",
            }
        )
        _write_json_atomic(run_dir / "metadata.json", metadata)
        raise

    metadata["elapsed_seconds"] = float(metadata.get("elapsed_seconds", 0.0)) + (
        _monotonic() - segment_started
    )
    return _finalize_run(
        run_dir,
        metadata=metadata,
        progress=progress,
        target_per_surface=target_per_surface,
        smoke=max_requests_per_surface is not None,
    )


def select_surface_training_groups(
    groups: Sequence[TrainingGroup],
    gains_by_request: Mapping[str, Mapping[str, float]],
    *,
    requests_per_surface: int = REQUESTS_PER_SURFACE,
    selection_seed: int = SELECTION_SEED,
) -> tuple[dict[str, list[TrainingGroup]], dict[str, Any]]:
    """Stable-hash select each train surface without accessing model loss."""

    if requests_per_surface <= 0:
        raise ValueError("requests_per_surface must be positive")
    buckets: dict[str, list[TrainingGroup]] = {surface: [] for surface in SURFACES}
    seen: set[str] = set()
    for group in groups:
        request_id = group.record.request_id
        if request_id in seen:
            raise ValueError(f"duplicate training group request_id={request_id}")
        seen.add(request_id)
        if request_id not in gains_by_request:
            raise ValueError(f"training gains missing request_id={request_id}")
        surface = classify_train_surface(group, gains_by_request[request_id])
        if surface is not None:
            buckets[surface].append(group)
    selected: dict[str, list[TrainingGroup]] = {}
    surface_manifest: dict[str, Any] = {}
    for surface in SURFACES:
        values = sorted(
            buckets[surface],
            key=lambda group: (
                _stable_digest(
                    selection_seed,
                    "m3-surface-selection",
                    surface,
                    group.record.request_id,
                ),
                group.record.request_id,
            ),
        )
        if len(values) < requests_per_surface:
            raise ValueError(
                f"surface={surface} has {len(values)} eligible groups; "
                f"requires {requests_per_surface}"
            )
        chosen = values[:requests_per_surface]
        selected[surface] = chosen
        identities = [_group_identity(group) for group in chosen]
        request_ids = [group.record.request_id for group in chosen]
        surface_manifest[surface] = {
            "eligible_requests": len(values),
            "request_count": len(chosen),
            "request_ids": request_ids,
            "request_ids_sha256": _canonical_sha256(request_ids),
            "training_groups_sha256": _canonical_sha256(identities),
        }
    return selected, {
        "schema_version": 1,
        "access_order": (
            "records_train_and_qrels_train_to_surface_selection_then_persisted_"
            "selection_then_model_load_then_loss_gradient"
        ),
        "controls": list(CONTROLS),
        "label_shuffle_seed": selection_seed,
        "requests_per_surface": requests_per_surface,
        "selection_algorithm": (
            "ascending_sha256(seed|m3-surface-selection|surface|request_id)"
        ),
        "selection_seed": selection_seed,
        "surface_order": list(SURFACES),
        "surfaces": surface_manifest,
    }


def classify_train_surface(
    group: TrainingGroup,
    gains: Mapping[str, float],
) -> str | None:
    """Map one positive-eligible train request to the registered M3 surface."""

    candidate_ids = {str(row["item_id"]) for row in group.record.candidates}
    positive_ids = {
        str(item_id)
        for item_id, gain in gains.items()
        if float(gain) > 0.0 and str(item_id) in candidate_ids
    }
    history_ids = {str(row["item_id"]) for row in group.record.history}
    if not positive_ids or not history_ids:
        return None
    if positive_ids & history_ids:
        return "recurrence"
    if history_ids & candidate_ids:
        return "other_overlap"
    return "strict_transfer"


def deterministic_label_shuffle(
    group: TrainingGroup,
    *,
    seed: int = SELECTION_SEED,
) -> tuple[TrainingGroup, dict[str, Any]]:
    """Permute gains within one request deterministically, preserving the multiset."""

    length = len(group.gains)
    if length < 2:
        raise ValueError("within-request label shuffle needs at least two candidates")
    base_order = sorted(
        range(length),
        key=lambda index: (
            _stable_digest(
                seed,
                "within-request-label-shuffle",
                group.record.request_id,
                str(index),
            ),
            index,
        ),
    )
    original = tuple(float(value) for value in group.gains)
    order = None
    shuffled = None
    rotation = None
    for candidate_rotation in range(length):
        candidate_order = (
            base_order[candidate_rotation:] + base_order[:candidate_rotation]
        )
        candidate_gains = tuple(original[index] for index in candidate_order)
        if candidate_gains != original:
            order = candidate_order
            shuffled = candidate_gains
            rotation = candidate_rotation
            break
    if order is None or shuffled is None or rotation is None:
        raise ValueError(
            "within-request label shuffle requires at least two distinct gains"
        )
    if sorted(shuffled) != sorted(original):
        raise AssertionError("label shuffle changed the gain multiset")
    changed_positions = sum(
        left != right for left, right in zip(original, shuffled)
    )
    if changed_positions <= 0:
        raise AssertionError("label shuffle did not change any gain position")
    return TrainingGroup(
        record=group.record,
        candidates=group.candidates,
        gains=shuffled,
    ), {
        "base_permutation_sha256": _canonical_sha256(base_order),
        "changed_positions": changed_positions,
        "permutation": order,
        "permutation_sha256": _canonical_sha256(order),
        "preserved_gain_multiset": True,
        "rotation_search_offset": rotation,
        "seed": seed,
    }


def discover_gradient_scopes(
    model: Any,
    tokenizer: Any,
    method_id: str,
) -> list[GradientScope]:
    """Resolve the exact registered Q2 or Q3 gradient parameter boundary."""

    named = list(model.named_parameters())
    if method_id == "q2_recranker_generalqwen":
        scopes = []
        observed: set[tuple[int, str]] = set()
        for name, parameter in named:
            match = _Q2_PARAMETER_PATTERN.search(name)
            if match is None:
                continue
            block = int(match.group(1))
            projection = str(match.group(2))
            if block not in Q2_BLOCKS:
                continue
            scopes.append(
                GradientScope(
                    name=f"block_{block:02d}.{projection}.{match.group(3)}",
                    parameter_name=name,
                    parameter=parameter,
                )
            )
            observed.add((block, projection))
        required = {(block, projection) for block in Q2_BLOCKS for projection in ("q_proj", "v_proj")}
        if observed != required:
            raise ValueError(
                "Q2 registered q/v block coverage mismatch: "
                f"missing={sorted(required - observed)} extra={sorted(observed - required)}"
            )
        output = model.get_output_embeddings()
        if output is None or not hasattr(output, "weight"):
            raise ValueError("Q2 model has no lm_head/output embedding weight")
        yes_id = int(_single_token_id(tokenizer, "yes"))
        no_id = int(_single_token_id(tokenizer, "no"))
        if yes_id == no_id:
            raise ValueError("Q2 yes/no token rows are not distinct")
        output_names = sorted(
            name for name, parameter in named if parameter is output.weight
        )
        if not output_names:
            raise ValueError(
                "Q2 output embedding weight is not a registered model parameter"
            )
        scopes.append(
            GradientScope(
                name="lm_head.yes_no_rows",
                parameter_name=output_names[0],
                parameter=output.weight,
                row_indices=(yes_id, no_id),
            )
        )
        registered_parameter_ids = {id(scope.parameter) for scope in scopes}
        for _, parameter in named:
            parameter.requires_grad_(id(parameter) in registered_parameter_ids)
        observed_trainable_ids = {
            id(parameter) for _, parameter in named if parameter.requires_grad
        }
        if observed_trainable_ids != registered_parameter_ids:
            raise ValueError("Q2 trainable parameter object boundary mismatch")
        return sorted(scopes, key=lambda scope: scope.name)
    if method_id != "q3_tallrec_generalqwen":
        raise ValueError(f"unsupported gradient scope method={method_id}")
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    if not trainable:
        raise ValueError("Q3 diagnostic model has no trainable LoRA parameters")
    invalid = [
        name
        for name, _ in trainable
        if "lora_" not in name
        or not (".q_proj." in name or ".v_proj." in name)
    ]
    if invalid:
        raise ValueError(
            "Q3 trainable boundary includes non-LoRA-q/v parameters: "
            f"{invalid[:5]}"
        )
    return [
        GradientScope(name=name, parameter_name=name, parameter=parameter)
        for name, parameter in sorted(trainable)
    ]


def trainable_parameter_audit(
    model: Any,
    scopes: Sequence[GradientScope],
    method_id: str,
) -> dict[str, Any]:
    """Audit unique trainable objects after the registered scope boundary."""

    named = list(model.named_parameters())
    names_by_object: dict[int, list[str]] = {}
    parameter_by_object: dict[int, Any] = {}
    for name, parameter in named:
        names_by_object.setdefault(id(parameter), []).append(name)
        parameter_by_object[id(parameter)] = parameter
    trainable_ids = {
        parameter_id
        for parameter_id, parameter in parameter_by_object.items()
        if parameter.requires_grad
    }
    registered_ids = {id(scope.parameter) for scope in scopes}
    if trainable_ids != registered_ids:
        raise ValueError(
            f"{method_id} trainable objects differ from registered gradient scopes"
        )
    rows = []
    for parameter_id in sorted(
        trainable_ids,
        key=lambda value: tuple(sorted(names_by_object[value])),
    ):
        parameter = parameter_by_object[parameter_id]
        rows.append(
            {
                "names": sorted(names_by_object[parameter_id]),
                "numel": int(parameter.numel()),
                "shape": list(parameter.shape),
            }
        )
    scope_numel = {
        scope.name: (
            int(scope.parameter[0].numel()) * len(scope.row_indices)
            if scope.row_indices
            else int(scope.parameter.numel())
        )
        for scope in scopes
    }
    return {
        "all_nonregistered_parameters_frozen": True,
        "registered_scope_count": len(scopes),
        "registered_scope_monitored_numel": scope_numel,
        "trainable_parameter_names": sorted(
            name for row in rows for name in row["names"]
        ),
        "trainable_parameter_numel": sum(row["numel"] for row in rows),
        "trainable_parameter_object_count": len(rows),
        "trainable_parameters": rows,
    }


def gradient_vector_metrics(
    gradients: Mapping[str, Any],
) -> dict[str, Any]:
    """Hand-auditable norm and monitored-scope update-share conversion."""

    squared = {}
    for name, gradient in gradients.items():
        value = float(gradient.double().square().sum().item())
        if not math.isfinite(value):
            raise NumericalGradientError(f"non-finite squared gradient norm: {name}")
        squared[name] = value
    total_squared = sum(squared.values())
    if not math.isfinite(total_squared) or total_squared <= 0.0:
        raise NumericalGradientError("monitored gradient norm is zero or non-finite")
    return {
        "gradient_norm": math.sqrt(total_squared),
        "scope_gradient_norms": {
            name: math.sqrt(value) for name, value in sorted(squared.items())
        },
        "scope_normalized_update_share": {
            name: value / total_squared for name, value in sorted(squared.items())
        },
        "squared_gradient_norm": total_squared,
    }


def mean_gradient_cosine(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float | None:
    """Exact cosine between two surface gradient sums over registered scopes."""

    if set(left) != set(right) or not left:
        raise ValueError("gradient sums have different or empty scope coverage")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for name in sorted(left):
        lvalue = left[name].double().reshape(-1)
        rvalue = right[name].double().reshape(-1)
        if lvalue.shape != rvalue.shape:
            raise ValueError(f"gradient sum shape mismatch: {name}")
        dot += float((lvalue * rvalue).sum().item())
        left_norm += float(lvalue.square().sum().item())
        right_norm += float(rvalue.square().sum().item())
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    value = dot / math.sqrt(left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def normalized_surface_update_shares(
    update_masses: Mapping[str, float],
) -> dict[str, float | None]:
    """Normalize equal-count surface squared-gradient mass into update shares."""

    if set(update_masses) != set(SURFACES):
        raise ValueError("surface update masses do not cover registered surfaces")
    total = sum(float(value) for value in update_masses.values())
    if not math.isfinite(total) or total <= 0.0:
        return {surface: None for surface in SURFACES}
    return {surface: float(update_masses[surface]) / total for surface in SURFACES}


def gradient_diagnostic_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = {
        "scripts/run_mechanism_gradient_diagnostic.py": (
            root / "scripts/run_mechanism_gradient_diagnostic.py"
        ),
        "src/myrec/mechanism/gradient_diagnostic.py": Path(__file__).resolve(),
    }
    files = []
    for relative, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing gradient diagnostic implementation: {path}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {"digest": _canonical_sha256(files), "files": files}


def _run_cell(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    scopes: Sequence[GradientScope],
    groups: Sequence[TrainingGroup],
    *,
    surface: str,
    control: str,
    state: str,
    device: str,
    attempt_path: Path,
    segment_started: float,
    max_wall_seconds: float,
    torch_module: Any,
) -> dict[str, Any]:
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    gradient_sums: dict[str, Any] = {}
    valid_requests = 0
    numerical_failures = 0
    mechanical_failures = 0
    loss_sum = 0.0
    gradient_norm_sum = 0.0
    squared_gradient_norm_sum = 0.0
    scope_norm_sums: dict[str, float] = {}
    processed = 0
    with attempt_path.open("x", encoding="utf-8") as handle:
        for ordinal, original_group in enumerate(groups):
            if _monotonic() - segment_started >= max_wall_seconds:
                handle.flush()
                os.fsync(handle.fileno())
                return {
                    "completed": False,
                    "processed_requests": processed,
                }
            model.zero_grad(set_to_none=True)
            rng_seed = _request_gradient_seed(
                original_group.record.request_id,
                state=state,
            )
            _seed_everything(torch_module, rng_seed)
            group = original_group
            shuffle_audit = None
            if control == "within_request_label_shuffle":
                group, shuffle_audit = deterministic_label_shuffle(original_group)
            row = {
                "control": control,
                "gradient_rng_seed": rng_seed,
                "group_identity_sha256": _canonical_sha256(_group_identity(group)),
                "label_shuffle": shuffle_audit,
                "method_id": config["method_id"],
                "ordinal_within_cell": ordinal,
                "request_id": group.record.request_id,
                "state": state,
                "surface": surface,
            }
            try:
                loss, components = _training_batch_loss(
                    model,
                    tokenizer,
                    [group],
                    config,
                    device=device,
                )
                loss_value = float(loss.detach().float().cpu().item())
                if not math.isfinite(loss_value):
                    raise NumericalGradientError("loss is non-finite")
                loss.backward()
                gradients = _extract_scope_gradients(scopes)
                metrics = gradient_vector_metrics(gradients)
                for name, gradient in gradients.items():
                    if name not in gradient_sums:
                        gradient_sums[name] = torch_module.zeros_like(
                            gradient, device="cpu"
                        )
                    gradient_sums[name].add_(gradient)
                valid_requests += 1
                loss_sum += loss_value
                gradient_norm_sum += float(metrics["gradient_norm"])
                squared_gradient_norm_sum += float(
                    metrics["squared_gradient_norm"]
                )
                for name, value in metrics["scope_gradient_norms"].items():
                    scope_norm_sums[name] = scope_norm_sums.get(name, 0.0) + float(
                        value
                    )
                row.update(
                    {
                        **metrics,
                        "loss": loss_value,
                        "loss_components": components,
                        "status": "completed",
                    }
                )
            except NumericalGradientError as exc:
                numerical_failures += 1
                row.update(
                    {
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "status": "numerical_failure",
                    }
                )
            except Exception as exc:
                mechanical_failures += 1
                row.update(
                    {
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "status": "mechanical_failure",
                    }
                )
            finally:
                model.zero_grad(set_to_none=True)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            processed += 1
        os.fsync(handle.fileno())
    return {
        "completed": True,
        "gradient_sums": gradient_sums,
        "summary": {
            "control": control,
            "gradient_norm_mean": (
                gradient_norm_sum / valid_requests if valid_requests else None
            ),
            "loss_mean": loss_sum / valid_requests if valid_requests else None,
            "mechanical_failures": mechanical_failures,
            "numerical_failures": numerical_failures,
            "processed_requests": processed,
            "scope_gradient_norm_means": {
                name: value / valid_requests
                for name, value in sorted(scope_norm_sums.items())
            }
            if valid_requests
            else {},
            "squared_gradient_norm_sum": squared_gradient_norm_sum,
            "surface": surface,
            "valid_requests": valid_requests,
        },
    }


def _extract_scope_gradients(scopes: Sequence[GradientScope]) -> dict[str, Any]:
    result = {}
    for scope in scopes:
        gradient = scope.parameter.grad
        if gradient is None:
            raise RuntimeError(f"registered parameter has no gradient: {scope.name}")
        if scope.row_indices:
            gradient = gradient[list(scope.row_indices)]
        value = gradient.detach().float().cpu().reshape(-1).clone()
        if not bool(value.isfinite().all().item()):
            raise NumericalGradientError(f"non-finite gradient: {scope.name}")
        result[scope.name] = value
    return result


def _commit_cell(
    run_dir: Path,
    *,
    cell_id: str,
    attempt_path: Path,
    cell: Mapping[str, Any],
    torch_module: Any,
) -> None:
    cells_dir = run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    results_path = cells_dir / f"{cell_id}.jsonl"
    gradients_path = cells_dir / f"{cell_id}.gradient_sums.pt"
    metadata_path = cells_dir / f"{cell_id}.json"
    if any(path.exists() for path in (results_path, gradients_path, metadata_path)):
        raise FileExistsError(f"cell output already exists: {cell_id}")
    temporary_gradients = gradients_path.with_name(
        f".{gradients_path.name}.tmp-{os.getpid()}"
    )
    torch_module.save(
        {
            "gradient_sums": dict(cell["gradient_sums"]),
            "valid_requests": int(cell["summary"]["valid_requests"]),
        },
        temporary_gradients,
    )
    os.replace(temporary_gradients, gradients_path)
    os.replace(attempt_path, results_path)
    cell_metadata = {
        **dict(cell["summary"]),
        "cell_id": cell_id,
        "gradient_sums_path": str(gradients_path),
        "gradient_sums_sha256": sha256_file(gradients_path),
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
    }
    _write_json_atomic(metadata_path, cell_metadata)


def _finalize_run(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    target_per_surface: int,
    smoke: bool,
) -> dict[str, Any]:
    cell_order = [
        f"{control}__{surface}"
        for control in CONTROLS
        for surface in SURFACES
    ]
    if progress.get("completed_cells") != cell_order:
        raise ValueError("cannot finalize incomplete gradient diagnostic cells")
    _validate_completed_cells(run_dir, cell_order)
    import torch

    cell_summaries = {}
    gradient_sums = {}
    per_request_path = run_dir / "per_request.jsonl"
    with per_request_path.open("x", encoding="utf-8") as output:
        for cell_id in cell_order:
            cell_metadata = _read_json(
                run_dir / "cells" / f"{cell_id}.json",
                f"cell metadata {cell_id}",
            )
            cell_summaries[cell_id] = cell_metadata
            payload = torch.load(
                run_dir / "cells" / f"{cell_id}.gradient_sums.pt",
                map_location="cpu",
                weights_only=True,
            )
            gradient_sums[cell_id] = payload["gradient_sums"]
            with (run_dir / "cells" / f"{cell_id}.jsonl").open(
                "r", encoding="utf-8"
            ) as source:
                shutil.copyfileobj(source, output)
        output.flush()
        os.fsync(output.fileno())

    cosine_rows = []
    update_shares = {}
    for control in CONTROLS:
        for left_index, left in enumerate(SURFACES):
            for right in SURFACES[left_index + 1 :]:
                cosine_rows.append(
                    {
                        "control": control,
                        "cosine": mean_gradient_cosine(
                            gradient_sums[f"{control}__{left}"],
                            gradient_sums[f"{control}__{right}"],
                        )
                        if gradient_sums[f"{control}__{left}"]
                        and gradient_sums[f"{control}__{right}"]
                        else None,
                        "left_surface": left,
                        "right_surface": right,
                    }
                )
        masses = {
            surface: float(
                cell_summaries[f"{control}__{surface}"][
                    "squared_gradient_norm_sum"
                ]
            )
            for surface in SURFACES
        }
        update_shares[control] = normalized_surface_update_shares(masses)
    failures = sum(
        int(value["mechanical_failures"]) + int(value["numerical_failures"])
        for value in cell_summaries.values()
    )
    valid_counts_exact = all(
        int(value["valid_requests"]) == target_per_surface
        for value in cell_summaries.values()
    )
    report = {
        "schema_version": 1,
        "cell_summaries": cell_summaries,
        "endpoints": {
            "gradient_norm": "per_request_and_cell_mean_registered_scope_l2",
            "loss": "per_request_and_cell_mean_frozen_training_objective",
            "normalized_update_share": (
                "surface_sum_of_per_request_squared_registered_gradient_norm_"
                "normalized_across_equal_count_surfaces"
            ),
            "surface_mean_gradient_cosine": cosine_rows,
        },
        "method_id": metadata["method_id"],
        "normalized_update_share": update_shares,
        "qrels_train_only": True,
        "state": metadata["state"],
    }
    report_path = run_dir / "gradient_diagnostics.json"
    _write_json_atomic(report_path, report)
    result_eligible = not smoke and failures == 0 and valid_counts_exact
    metadata.update(
        {
            "cell_count": len(cell_order),
            "completed_request_diagnostics": sum(
                int(value["processed_requests"])
                for value in cell_summaries.values()
            ),
            "gradient_diagnostics_path": str(report_path),
            "gradient_diagnostics_sha256": sha256_file(report_path),
            "mechanical_or_numerical_failures": failures,
            "non_result_reason": "request_cap" if smoke else None,
            "per_request_path": str(per_request_path),
            "per_request_sha256": sha256_file(per_request_path),
            "result_eligible": result_eligible,
            "status": "completed",
        }
    )
    progress.update({"status": "completed", "updated_at": _utc_now()})
    _write_json_atomic(run_dir / "progress.json", progress)
    _write_json_atomic(run_dir / "metadata.json", metadata)
    return metadata


def _load_state_model(
    config: dict[str, Any],
    *,
    state: str,
    device: str,
    checkpoint_model_dir: Path,
    torch_module: Any,
) -> tuple[Any, Any]:
    _seed_everything(torch_module, MODEL_INITIALIZATION_SEED)
    return _load_model_and_tokenizer(
        config,
        device=device,
        training=True,
        checkpoint_model_dir=(
            None if state == "base_initialization" else checkpoint_model_dir
        ),
    )


def _q3_trainable_initialization_fingerprint(
    scopes: Sequence[GradientScope],
) -> dict[str, Any]:
    rows = []
    digest = hashlib.sha256()
    for scope in sorted(scopes, key=lambda value: value.name):
        tensor = scope.parameter.detach().float().cpu().contiguous()
        tensor_sha256 = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        row = {
            "name": scope.name,
            "shape": list(scope.parameter.shape),
            "sha256_float32": tensor_sha256,
        }
        rows.append(row)
        digest.update(_canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
    return {"digest": digest.hexdigest(), "parameters": rows}


def _load_probe_manifest(path: str | Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    expected_path = (root / PROBE_MANIFEST_PATH).resolve()
    supplied = Path(path)
    supplied = (
        (Path.cwd() / supplied).resolve()
        if not supplied.is_absolute()
        else supplied.resolve()
    )
    if supplied != expected_path:
        raise ValueError(f"probe manifest path must be {PROBE_MANIFEST_PATH}")
    if sha256_file(supplied) != PROBE_MANIFEST_SHA256:
        raise ValueError("frozen probe manifest hash mismatch")
    import yaml

    payload = yaml.safe_load(supplied.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("probe_manifest_id") != PROBE_MANIFEST_ID:
        raise ValueError("frozen probe manifest identity mismatch")
    m3 = payload.get("m3_objective_and_gradients")
    expected_m3 = {
        "anchors": list(SUPPORTED_METHODS),
        "surfaces": list(SURFACES),
        "requests_per_surface": REQUESTS_PER_SURFACE,
        "states": list(SUPPORTED_STATES),
        "q2_parameter_blocks": {
            "attention_qv_blocks_zero_based": list(Q2_BLOCKS),
            "readout_rows": ["yes_token", "no_token"],
        },
        "q3_parameter_blocks": "all_trainable_lora_qv_parameters",
        "endpoints": [
            "loss",
            "gradient_norm",
            "gradient_cosine",
            "normalized_update_share",
        ],
        "negative_control": "within_request_label_shuffle",
    }
    if not isinstance(m3, Mapping):
        raise ValueError("probe manifest lacks M3 registration")
    for key, expected in expected_m3.items():
        if m3.get(key) != expected:
            raise ValueError(f"probe manifest M3 drift: {key}")
    return {
        "identity": {
            "expected_sha256": PROBE_MANIFEST_SHA256,
            "path": PROBE_MANIFEST_PATH.as_posix(),
            "sha256": PROBE_MANIFEST_SHA256,
            "verified": True,
        },
        "payload": payload,
        "sha256": PROBE_MANIFEST_SHA256,
    }


def _validate_probe_model_binding(
    probe: Mapping[str, Any],
    *,
    method_id: str,
    config_path: Path,
    config_sha256: str,
    checkpoint_root: Path,
    checkpoint_id: str,
) -> None:
    root = Path(__file__).resolve().parents[3]
    entry = probe.get("frozen_inputs", {}).get("models", {}).get(method_id)
    if not isinstance(entry, Mapping):
        raise ValueError(f"probe manifest lacks frozen model={method_id}")
    expected = {
        "config": (root / str(entry["config"])).resolve(),
        "checkpoint": (root / str(entry["checkpoint"])).resolve(),
    }
    if config_path.resolve() != expected["config"]:
        raise ValueError("config path differs from frozen probe manifest")
    if checkpoint_root.resolve() != expected["checkpoint"]:
        raise ValueError("checkpoint path differs from frozen probe manifest")
    if config_sha256 != entry.get("config_sha256"):
        raise ValueError("config hash differs from frozen probe manifest")
    if checkpoint_id != entry.get("checkpoint_id"):
        raise ValueError("checkpoint id differs from frozen probe manifest")


def _load_train_gains(path: Path) -> dict[str, dict[str, float]]:
    if path.name != "qrels_train.jsonl":
        raise ValueError("gradient diagnostic may only read qrels_train.jsonl")
    result = {}
    for row in iter_jsonl(path):
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError(f"empty or duplicate train qrels request_id={request_id!r}")
        relevance = row.get("relevance") or {}
        if not isinstance(relevance, dict):
            raise ValueError(f"train qrels relevance is not an object: {request_id}")
        gains = {
            str(item_id): float(gain)
            for item_id, gain in relevance.items()
            if float(gain) > 0.0
        }
        if not gains:
            gains = {
                **{str(item_id): 1.0 for item_id in row.get("clicked", [])},
                **{str(item_id): 2.0 for item_id in row.get("purchased", [])},
            }
        if any(not math.isfinite(value) or value <= 0.0 for value in gains.values()):
            raise ValueError(f"invalid train gain: {request_id}")
        result[request_id] = gains
    if not result:
        raise ValueError("qrels_train is empty")
    return result


def _prepare_run(
    run_dir: Path,
    *,
    base_metadata: dict[str, Any],
    selection: dict[str, Any],
    selection_sha256: str,
    run_contract_sha256: str,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = run_dir / "metadata.json"
    progress_path = run_dir / "progress.json"
    selection_path = run_dir / "selection_manifest.json"
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "cells").mkdir()
        (run_dir / "attempts").mkdir()
        _write_json_atomic(selection_path, selection)
        if sha256_file(selection_path) != _pretty_json_sha256(selection):
            raise AssertionError("selection manifest write hash mismatch")
        metadata = {
            **base_metadata,
            "elapsed_seconds": 0.0,
            "resume_lineage": [],
            "selection_manifest_path": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
        }
        progress = {
            "schema_version": 1,
            "completed_cells": [],
            "last_completed_cell": None,
            "resume_count": 0,
            "run_contract_sha256": run_contract_sha256,
            "selection_sha256": selection_sha256,
            "status": "selection_finalized",
            "updated_at": _utc_now(),
        }
        _write_json_atomic(metadata_path, metadata)
        _write_json_atomic(progress_path, progress)
        return metadata, progress
    if not run_dir.is_dir():
        raise FileNotFoundError(f"resume run directory missing: {run_dir}")
    metadata = _read_json(metadata_path, "gradient metadata")
    progress = _read_json(progress_path, "gradient progress")
    observed_selection = _read_json(selection_path, "selection manifest")
    if observed_selection != selection:
        raise ValueError("resume selection manifest drift")
    if metadata.get("run_contract_sha256") != run_contract_sha256 or progress.get(
        "run_contract_sha256"
    ) != run_contract_sha256:
        raise ValueError("resume run contract drift")
    if progress.get("selection_sha256") != selection_sha256:
        raise ValueError("resume selection identity drift")
    lineage = metadata.get("resume_lineage")
    if not isinstance(lineage, list):
        raise ValueError("resume lineage is invalid")
    lineage.append(
        {
            "completed_cells": list(progress.get("completed_cells", [])),
            "from_status": metadata.get("status"),
            "partial_attempt_path": metadata.get("partial_attempt_path"),
            "partial_attempt_sha256": metadata.get("partial_attempt_sha256"),
            "resumed_at": _utc_now(),
        }
    )
    metadata.pop("active_cell_at_exit", None)
    metadata.pop("partial_attempt_path", None)
    metadata.pop("partial_attempt_sha256", None)
    metadata.update({"error": None, "resume_lineage": lineage, "status": "selection_finalized"})
    progress.update(
        {
            "resume_count": int(progress.get("resume_count", 0)) + 1,
            "status": "selection_finalized",
            "updated_at": _utc_now(),
        }
    )
    _write_json_atomic(metadata_path, metadata)
    _write_json_atomic(progress_path, progress)
    return metadata, progress


def _validate_completed_cells(run_dir: Path, completed_cells: Sequence[str]) -> None:
    valid_ids = {
        f"{control}__{surface}" for control in CONTROLS for surface in SURFACES
    }
    if len(completed_cells) != len(set(completed_cells)) or not set(
        completed_cells
    ).issubset(valid_ids):
        raise ValueError("progress contains invalid completed cells")
    for cell_id in completed_cells:
        metadata = _read_json(
            run_dir / "cells" / f"{cell_id}.json", f"cell metadata {cell_id}"
        )
        for key, suffix in (
            ("results_sha256", ".jsonl"),
            ("gradient_sums_sha256", ".gradient_sums.pt"),
        ):
            path = run_dir / "cells" / f"{cell_id}{suffix}"
            if metadata.get(key) != sha256_file(path):
                raise ValueError(f"completed cell hash mismatch: {cell_id} {key}")


def _record_wall_exit(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    cell_id: str,
    attempt_path: Path,
    segment_elapsed: float,
) -> dict[str, Any]:
    metadata.update(
        {
            "active_cell_at_exit": cell_id,
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + segment_elapsed,
            "partial_attempt_path": str(attempt_path),
            "partial_attempt_sha256": sha256_file(attempt_path),
            "result_eligible": False,
            "status": "wall_time_exhausted",
        }
    )
    progress.update({"status": "wall_time_exhausted", "updated_at": _utc_now()})
    _write_json_atomic(run_dir / "progress.json", progress)
    _write_json_atomic(run_dir / "metadata.json", metadata)
    return metadata


def _next_attempt_number(run_dir: Path, cell_id: str) -> int:
    directory = run_dir / "attempts"
    return 1 + len(list(directory.glob(f"{cell_id}.attempt*.jsonl")))


def _registered_parameter_contract(method_id: str) -> dict[str, Any]:
    if method_id == "q2_recranker_generalqwen":
        return {
            "attention_blocks_zero_based": list(Q2_BLOCKS),
            "attention_projections": ["q_proj", "v_proj"],
            "readout_rows": ["yes_token", "no_token"],
        }
    return {"all_trainable_parameters": "LoRA q_proj/v_proj only"}


def _group_identity(group: TrainingGroup) -> dict[str, Any]:
    return {
        "candidate_item_ids": [str(row["item_id"]) for row in group.candidates],
        "gains": [float(value) for value in group.gains],
        "request_id": group.record.request_id,
    }


def _request_gradient_seed(request_id: str, *, state: str) -> int:
    return int(
        _stable_digest(
            MODEL_INITIALIZATION_SEED,
            "per-request-gradient-rng",
            state,
            request_id,
        )[:16],
        16,
    ) % (2**31)


def _stable_digest(seed: int, *values: str) -> str:
    payload = "|".join([str(seed), *[str(value) for value in values]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return sha256_text(_canonical_json(value))


def _pretty_json_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _monotonic() -> float:
    return time.perf_counter()


__all__ = [
    "CONTROLS",
    "GradientScope",
    "MODEL_INITIALIZATION_SEED",
    "PROBE_MANIFEST_SHA256",
    "Q2_BLOCKS",
    "REQUESTS_PER_SURFACE",
    "SELECTION_SEED",
    "SURFACES",
    "classify_train_surface",
    "deterministic_label_shuffle",
    "discover_gradient_scopes",
    "gradient_diagnostic_implementation_identity",
    "gradient_vector_metrics",
    "mean_gradient_cosine",
    "normalized_surface_update_shares",
    "run_gradient_diagnostic",
    "select_surface_training_groups",
    "trainable_parameter_audit",
]
