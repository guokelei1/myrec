"""Matched Q2 diagnostic training control for the registered M3 probe.

This module is deliberately train-only.  It may read ``qrels_train.jsonl`` to
construct Q2 ranking groups and the three registered M3 surfaces, but it has no
dev evaluator or score-bundle imports.  The surface classifier is imported
from :mod:`myrec.mechanism.gradient_diagnostic` so the gradient diagnostic and
the matched control cannot silently disagree about surface membership.

The production path trains two independent Q2 checkpoints from the same base
Qwen initialization.  The only registered difference is the pre-model-load
sampling condition.  ``mock_cpu=True`` executes the complete deterministic
schedule without importing torch or loading Qwen; every such checkpoint is
permanently marked ``smoke_non_result`` and is rejected by the scorer.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import (
    SERIALIZED_INPUT_FIELDS,
    TrainingGroup,
    load_training_groups,
)
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _assert_frozen_training_population,
    _checkpoint_identity,
    _git_revision,
    _load_model_and_tokenizer,
    _recover_training_checkpoint,
    _restore_rng_state,
    _save_training_checkpoint,
    _seed_everything,
    _training_batch_loss,
    _validate_run_id,
    load_v12_ranker_config,
)
from myrec.mechanism.gradient_diagnostic import (
    MODEL_INITIALIZATION_SEED,
    PROBE_MANIFEST_PATH,
    SELECTION_SEED,
    SURFACES,
    _load_probe_manifest,
    _load_train_gains,
    classify_train_surface,
)
from myrec.utils.hashing import sha256_file, sha256_text


METHOD_ID = "q2_recranker_generalqwen"
CONDITIONS = ("original_mixture", "surface_balanced")
ROLE = "diagnostic_control_not_paper_method"
TRAIN_GROUPS = 4096
OPTIMIZER_STEPS = 256
BATCH_REQUESTS = 1
GRADIENT_ACCUMULATION_STEPS = 16
BALANCED_QUOTAS = {
    "recurrence": 1366,
    "strict_transfer": 1365,
    "other_overlap": 1365,
}
MAX_WALL_SECONDS = 13_500.0
CHECKPOINT_EVERY_OPTIMIZER_STEPS = 16
SELECTION_FILENAME = "selection.json"
RUN_CONTRACT_FILENAME = "run_contract.json"
PROGRESS_FILENAME = "progress.json"
METADATA_FILENAME = "metadata.json"
MOCK_STATE_FILENAME = "mock_trainer_state.json"


def select_matched_training_groups(
    groups: Sequence[TrainingGroup],
    gains_by_request: Mapping[str, Mapping[str, float]],
    condition: str,
    *,
    train_groups: int = TRAIN_GROUPS,
    selection_seed: int = SELECTION_SEED,
    balanced_quotas: Mapping[str, int] | None = None,
) -> tuple[list[TrainingGroup], dict[str, Any]]:
    """Freeze one deterministic M3 sampling condition before model access.

    ``original_mixture`` is an unstratified stable-hash sample from the union
    of all eligible surfaces, retaining the natural union mixture.
    ``surface_balanced`` uses fixed quotas in the registered ``SURFACES``
    order and then applies a separate stable-hash training order.
    """

    _require_condition(condition)
    train_groups = int(train_groups)
    if train_groups <= 0:
        raise ValueError("train_groups must be positive")
    quotas = dict(balanced_quotas or BALANCED_QUOTAS)
    if set(quotas) != set(SURFACES):
        raise ValueError("balanced quotas must cover the registered surfaces")
    if any(int(quotas[surface]) <= 0 for surface in SURFACES):
        raise ValueError("balanced quotas must be positive")
    if condition == "surface_balanced" and sum(
        int(quotas[surface]) for surface in SURFACES
    ) != train_groups:
        raise ValueError("balanced quotas must sum to train_groups")

    buckets: dict[str, list[TrainingGroup]] = {surface: [] for surface in SURFACES}
    surface_by_request: dict[str, str] = {}
    seen: set[str] = set()
    excluded_without_registered_surface = 0
    for group in groups:
        request_id = str(group.record.request_id)
        if request_id in seen:
            raise ValueError(f"duplicate training group request_id={request_id}")
        seen.add(request_id)
        if request_id not in gains_by_request:
            raise ValueError(f"training gains missing request_id={request_id}")
        # This exact function is shared with the M3 gradient diagnostic.
        surface = classify_train_surface(group, gains_by_request[request_id])
        if surface is None:
            excluded_without_registered_surface += 1
            continue
        if surface not in buckets:
            raise ValueError(f"unregistered surface returned for {request_id}: {surface}")
        buckets[surface].append(group)
        surface_by_request[request_id] = surface

    eligible = [group for surface in SURFACES for group in buckets[surface]]
    if len(eligible) < train_groups:
        raise ValueError(
            f"registered surface union has {len(eligible)} groups; requires {train_groups}"
        )
    if condition == "original_mixture":
        selected = sorted(
            eligible,
            key=lambda group: (
                _stable_digest(
                    selection_seed,
                    "m3-q2-matched-selection",
                    condition,
                    group.record.request_id,
                ),
                group.record.request_id,
            ),
        )[:train_groups]
        selection_algorithm = (
            "ascending_sha256(seed|m3-q2-matched-selection|"
            "original_mixture|request_id)_over_registered_surface_union"
        )
    else:
        chosen: list[TrainingGroup] = []
        for surface in SURFACES:
            values = sorted(
                buckets[surface],
                key=lambda group: (
                    _stable_digest(
                        selection_seed,
                        "m3-q2-matched-balanced-selection",
                        surface,
                        group.record.request_id,
                    ),
                    group.record.request_id,
                ),
            )
            quota = int(quotas[surface])
            if len(values) < quota:
                raise ValueError(
                    f"surface={surface} has {len(values)} groups; requires {quota}"
                )
            chosen.extend(values[:quota])
        selected = sorted(
            chosen,
            key=lambda group: (
                _stable_digest(
                    selection_seed,
                    "m3-q2-matched-training-order",
                    condition,
                    group.record.request_id,
                ),
                group.record.request_id,
            ),
        )
        selection_algorithm = (
            "fixed_surface_order_"
            + "_".join(
                f"{surface}_{int(quotas[surface])}" for surface in SURFACES
            )
            + "_then_ascending_sha256_training_order"
        )

    request_ids = [str(group.record.request_id) for group in selected]
    if len(request_ids) != train_groups or len(set(request_ids)) != train_groups:
        raise AssertionError("matched-control selection is incomplete or duplicated")
    selected_counts = Counter(surface_by_request[request_id] for request_id in request_ids)
    if condition == "surface_balanced" and dict(selected_counts) != {
        surface: int(quotas[surface]) for surface in SURFACES
    }:
        raise AssertionError("surface-balanced selection does not match fixed quotas")

    exposures = []
    for ordinal, group in enumerate(selected):
        identity = _group_identity(group)
        exposures.append(
            {
                "group_identity_sha256": _canonical_sha256(identity),
                "ordinal": ordinal,
                "request_id": group.record.request_id,
                "surface": surface_by_request[group.record.request_id],
            }
        )
    eligible_counts = {surface: len(buckets[surface]) for surface in SURFACES}
    eligible_total = sum(eligible_counts.values())
    selected_count_map = {
        surface: int(selected_counts.get(surface, 0)) for surface in SURFACES
    }
    manifest = {
        "schema_version": 1,
        "balanced_quotas": {surface: int(quotas[surface]) for surface in SURFACES},
        "condition": condition,
        "duplicates": 0,
        "eligible_surface_counts": eligible_counts,
        "eligible_surface_proportions": {
            surface: eligible_counts[surface] / eligible_total for surface in SURFACES
        },
        "eligible_union_requests": eligible_total,
        "excluded_without_registered_surface": excluded_without_registered_surface,
        "exposure_schedule": exposures,
        "exposure_schedule_sha256": _canonical_sha256(exposures),
        "finalized_before_model_load_and_loss": True,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "group_exposures": len(selected),
        "optimizer_steps": len(selected) // GRADIENT_ACCUMULATION_STEPS,
        "request_ids": request_ids,
        "request_ids_sha256": _canonical_sha256(request_ids),
        "selection_algorithm": selection_algorithm,
        "selection_seed": int(selection_seed),
        "selected_surface_counts": selected_count_map,
        "selected_surface_proportions": {
            surface: selected_count_map[surface] / len(selected) for surface in SURFACES
        },
        "surface_classifier": (
            "myrec.mechanism.gradient_diagnostic.classify_train_surface"
        ),
        "surface_order": list(SURFACES),
        "training_order_algorithm": (
            "selection_hash_order"
            if condition == "original_mixture"
            else "ascending_sha256(seed|m3-q2-matched-training-order|"
            "surface_balanced|request_id)"
        ),
        "training_groups_sha256": _canonical_sha256(
            [_group_identity(group) for group in selected]
        ),
    }
    if manifest["optimizer_steps"] != OPTIMIZER_STEPS and train_groups == TRAIN_GROUPS:
        raise AssertionError("formal matched-control schedule is not exactly 256 steps")
    return selected, manifest


def run_q2_matched_training_control(
    standardized_dir: str | Path,
    config_path: str | Path,
    condition: str,
    run_id: str,
    output_model_dir: str | Path,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    probe_manifest_path: str | Path = PROBE_MANIFEST_PATH,
    command: Sequence[str] | None = None,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    mock_cpu: bool = False,
    mock_stop_after_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    """Run or resume one registered Q2 matched-control condition.

    The formal path always consumes exactly 4096 request groups in one fixed
    order, one request per micro-batch, and performs exactly 256 optimizer
    updates.  No cap is exposed on the formal path.
    """

    started = time.perf_counter()
    _validate_run_id(run_id)
    _require_condition(condition)
    if not device or not str(device).strip():
        raise ValueError("an explicit device is required")
    if mock_cpu and str(device) != "cpu":
        raise ValueError("mock_cpu requires device=cpu")
    if not mock_cpu and not str(device).startswith("cuda:"):
        raise ValueError("formal matched-control training requires an explicit cuda:N")
    if not mock_cpu and mock_stop_after_optimizer_steps is not None:
        raise ValueError("mock_stop_after_optimizer_steps is mock-only")
    if mock_stop_after_optimizer_steps is not None and int(
        mock_stop_after_optimizer_steps
    ) <= 0:
        raise ValueError("mock_stop_after_optimizer_steps must be positive")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not (
        0.0 < max_wall_seconds <= MAX_WALL_SECONDS
    ):
        raise ValueError(
            f"max_wall_seconds must be in (0, {int(MAX_WALL_SECONDS)}]"
        )

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    output_model_dir = Path(output_model_dir)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    _assert_run_output_separation(run_dir, output_model_dir)
    probe = _validate_registered_probe(probe_manifest_path)
    config = load_v12_ranker_config(config_path)
    _validate_q2_recipe(config, config_path=config_path, probe=probe["payload"])
    _assert_independent_output_path(output_model_dir, probe["payload"])
    _assert_frozen_training_population(standardized_dir, config)

    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    dataset_manifest_path = standardized_dir / "manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    hashes = {
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "qrels_train_sha256": sha256_file(qrels_path),
        "records_train_sha256": sha256_file(records_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
    }
    population = config["_protocol"]["data"]["development_population"]
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

    # Selection and its stable hash are fully materialized before torch/model
    # loading or any loss access.
    gains = _load_train_gains(qrels_path)
    groups, group_stats = load_training_groups(
        records_path,
        qrels_path,
        seed=int(config["training"]["seed"]),
        negatives_per_positive=int(config["training"]["negatives_per_positive"]),
        max_group_size=int(config["training"]["list_size"]),
    )
    selected, selection = select_matched_training_groups(groups, gains, condition)
    selection.update(
        {
            **hashes,
            "config_sha256": config["_config_sha256"],
            "group_construction": group_stats,
            "method_id": METHOD_ID,
            "probe_manifest_sha256": probe["sha256"],
        }
    )
    selection_sha256 = _canonical_sha256(selection)
    implementation = matched_training_control_implementation_identity()
    matched_recipe = _matched_recipe(
        config,
        hashes=hashes,
        probe=probe,
        implementation=implementation,
    )
    matched_recipe_sha256 = _canonical_sha256(matched_recipe)
    evidence_mode = "smoke_non_result" if mock_cpu else "mechanism_diagnostic"
    run_contract = {
        "schema_version": 1,
        "condition": condition,
        "config_sha256": config["_config_sha256"],
        "evidence_mode": evidence_mode,
        "matched_recipe_sha256": matched_recipe_sha256,
        "method_id": METHOD_ID,
        "mock_cpu": bool(mock_cpu),
        "output_model_dir": str(output_model_dir),
        "probe_manifest_sha256": probe["sha256"],
        "role": ROLE,
        "run_id": run_id,
        "selection_sha256": selection_sha256,
    }
    run_contract_sha256 = _canonical_sha256(run_contract)
    base_metadata = {
        "schema_version": 1,
        "base_initialization_seed": MODEL_INITIALIZATION_SEED,
        "code_revision": _git_revision(),
        "command": list(command or sys.argv),
        "condition": condition,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "dataset_id": "kuaisearch",
        "dataset_version": str(population["dataset_version"]),
        "evidence_mode": evidence_mode,
        "formal_checkpoint": not mock_cpu,
        "input_fields_used": list(SERIALIZED_INPUT_FIELDS),
        "matched_recipe": matched_recipe,
        "matched_recipe_sha256": matched_recipe_sha256,
        "matched_training_control_implementation_identity": implementation,
        "method_id": METHOD_ID,
        "output_model_dir": str(output_model_dir),
        "probe_manifest": probe["identity"],
        "qrels_access": {
            "qrels_train_read": True,
            "qrels_train_sha256": hashes["qrels_train_sha256"],
            "qrels_dev_read": False,
            "qrels_confirmation_read": False,
            "qrels_test_read": False,
        },
        "result_eligible": False,
        "role": ROLE,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "run_id": run_id,
        "sampling": {
            "condition": condition,
            "selection_seed": SELECTION_SEED,
            "selection_sha256": selection_sha256,
            "selected_surface_counts": selection["selected_surface_counts"],
        },
        "selection_path": str(run_dir / SELECTION_FILENAME),
        "selection_sha256": selection_sha256,
        "segments": [],
        "status": "selection_finalized",
        **hashes,
    }
    metadata, progress = _prepare_run(
        run_dir,
        output_model_dir=output_model_dir,
        base_metadata=base_metadata,
        selection=selection,
        selection_sha256=selection_sha256,
        run_contract=run_contract,
        run_contract_sha256=run_contract_sha256,
        resume=resume,
    )
    if metadata.get("status") == "completed":
        raise ValueError("matched-control training run is already completed")

    if mock_cpu:
        return _run_mock_schedule(
            selected,
            run_dir=run_dir,
            output_model_dir=output_model_dir,
            metadata=metadata,
            progress=progress,
            started=started,
            max_wall_seconds=max_wall_seconds,
            stop_after_optimizer_steps=mock_stop_after_optimizer_steps,
        )
    return _run_formal_schedule(
        selected,
        config=config,
        condition=condition,
        run_dir=run_dir,
        output_model_dir=output_model_dir,
        metadata=metadata,
        progress=progress,
        run_contract_sha256=run_contract_sha256,
        selection_sha256=selection_sha256,
        matched_recipe_sha256=matched_recipe_sha256,
        device=str(device),
        started=started,
        max_wall_seconds=max_wall_seconds,
        resume=resume,
    )


def _run_formal_schedule(
    selected: Sequence[TrainingGroup],
    *,
    config: dict[str, Any],
    condition: str,
    run_dir: Path,
    output_model_dir: Path,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    run_contract_sha256: str,
    selection_sha256: str,
    matched_recipe_sha256: str,
    device: str,
    started: float,
    max_wall_seconds: float,
    resume: bool,
) -> dict[str, Any]:
    """Production Q2 optimizer path; called only after selection persistence."""

    import torch
    import transformers
    from transformers.optimization import get_linear_schedule_with_warmup

    _seed_everything(torch, MODEL_INITIALIZATION_SEED)
    if resume:
        _recover_training_checkpoint(output_model_dir)
    checkpoint_dir = output_model_dir / CHECKPOINT_DIRNAME
    tokenizer, model = _load_model_and_tokenizer(
        config,
        device=device,
        training=True,
        checkpoint_model_dir=(checkpoint_dir / "model") if resume else None,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Q2 matched control has no trainable parameters")
    training = config["training"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=float(training["weight_decay"]),
        amsgrad=False,
    )
    warmup_steps = int(OPTIMIZER_STEPS * float(training["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=OPTIMIZER_STEPS,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=str(training["dtype"]) == "float16"
    )
    checkpoint_contract = {
        "schema_version": 1,
        "run_contract_sha256": run_contract_sha256,
        "selection_sha256": selection_sha256,
        "matched_recipe_sha256": matched_recipe_sha256,
        "group_exposures": TRAIN_GROUPS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "condition": condition,
        "role": ROLE,
    }
    if resume:
        state = torch.load(
            checkpoint_dir / "trainer_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if state.get("config_sha256") != config["_config_sha256"]:
            raise ValueError("resume checkpoint config hash mismatch")
        if state.get("training_contract") != checkpoint_contract:
            raise ValueError("resume checkpoint matched-control contract mismatch")
        observed_checkpoint_id, _ = _checkpoint_identity(
            checkpoint_dir / "model", METHOD_ID
        )
        if state.get("checkpoint_id") != observed_checkpoint_id:
            raise ValueError("resume checkpoint model artifacts changed")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        _restore_rng_state(torch, state["rng"])
        checkpoint_progress = dict(state["progress"])
        checkpoint_exposures = int(checkpoint_progress.get("group_exposures", -1))
        persisted_exposures = int(progress.get("group_exposures", -1))
        if checkpoint_exposures < 0 or checkpoint_exposures > persisted_exposures:
            raise ValueError("resume checkpoint progress is ahead of persisted progress")
        if checkpoint_exposures % GRADIENT_ACCUMULATION_STEPS:
            raise ValueError("resume checkpoint is not at an optimizer boundary")
        if int(checkpoint_progress.get("optimizer_steps", -1)) != (
            checkpoint_exposures // GRADIENT_ACCUMULATION_STEPS
        ):
            raise ValueError("resume checkpoint exposure/optimizer counts disagree")
        if checkpoint_exposures != persisted_exposures:
            metadata["checkpoint_resume_rollback"] = {
                "checkpoint_group_exposures": checkpoint_exposures,
                "persisted_group_exposures": persisted_exposures,
                "reason": "checkpoint_is_authoritative_after_interrupted_period",
            }
            _truncate_learning_curve(
                run_dir / "learning_curve.jsonl",
                checkpoint_optimizer_steps=int(
                    checkpoint_progress["optimizer_steps"]
                ),
            )
        checkpoint_progress["resume_count"] = int(progress.get("resume_count", 0))
        checkpoint_progress["status"] = "selection_finalized"
        checkpoint_progress["updated_at"] = _utc_now()
        progress = checkpoint_progress
        _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)

    cursor = int(progress["group_exposures"])
    if cursor % GRADIENT_ACCUMULATION_STEPS:
        raise ValueError("resume cursor is not at an optimizer-step boundary")
    optimizer.zero_grad(set_to_none=True)
    metadata.update(
        {
            "package_versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "python_version": platform.python_version(),
            "status": "running",
        }
    )
    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
    block_losses: list[float] = []
    status = "completed"
    try:
        while cursor < len(selected):
            group = selected[cursor]
            autocast_dtype = (
                torch.float16 if training["dtype"] == "float16" else torch.bfloat16
            )
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=device.startswith("cuda") and training["dtype"] != "float32",
            ):
                raw_loss, loss_parts = _training_batch_loss(
                    model, tokenizer, [group], config, device=device
                )
            if not bool(torch.isfinite(raw_loss).item()):
                raise FloatingPointError(
                    f"non-finite Q2 loss at group exposure={cursor}"
                )
            scaler.scale(raw_loss / GRADIENT_ACCUMULATION_STEPS).backward()
            block_losses.append(float(raw_loss.detach().float().cpu()))
            cursor += 1
            if cursor % GRADIENT_ACCUMULATION_STEPS:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable, float(training["max_grad_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step = cursor // GRADIENT_ACCUMULATION_STEPS
            progress = {
                **progress,
                "group_exposures": cursor,
                "micro_steps": cursor,
                "optimizer_steps": optimizer_step,
                "status": "running",
                "updated_at": _utc_now(),
            }
            _append_jsonl(
                run_dir / "learning_curve.jsonl",
                {
                    "condition": condition,
                    "group_exposure_end_exclusive": cursor,
                    "loss_mean": sum(block_losses) / len(block_losses),
                    "last_loss_parts": loss_parts,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "optimizer_step": optimizer_step,
                },
            )
            block_losses.clear()
            _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
            periodic = optimizer_step == 1 or (
                optimizer_step % CHECKPOINT_EVERY_OPTIMIZER_STEPS == 0
            )
            wall_exit = time.perf_counter() - started >= max_wall_seconds
            if periodic or wall_exit or optimizer_step == OPTIMIZER_STEPS:
                _save_training_checkpoint(
                    output_model_dir,
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    scaler,
                    progress,
                    config,
                    checkpoint_contract,
                    torch,
                )
            if wall_exit and optimizer_step < OPTIMIZER_STEPS:
                status = "wall_time_exhausted"
                break
    except Exception as exc:
        metadata.update(
            {
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "result_eligible": False,
                "status": "failed",
            }
        )
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        raise

    if status == "completed":
        if cursor != TRAIN_GROUPS or int(progress["optimizer_steps"]) != OPTIMIZER_STEPS:
            raise AssertionError("completed formal schedule has incorrect exposure counts")
        progress.update({"status": "completed", "updated_at": _utc_now()})
        _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
    else:
        progress.update({"status": status, "updated_at": _utc_now()})
        _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        output_model_dir / CHECKPOINT_DIRNAME / "model", METHOD_ID
    )
    segment_elapsed = time.perf_counter() - started
    segments = list(metadata.get("segments", []))
    segments.append(
        {
            "device": device,
            "elapsed_seconds": segment_elapsed,
            "ended_at": _utc_now(),
            "formal_checkpoint": True,
            "progress": dict(progress),
            "status": status,
        }
    )
    metadata.update(
        {
            "checkpoint_dir": str(output_model_dir / CHECKPOINT_DIRNAME),
            "checkpoint_files": checkpoint_files,
            "checkpoint_id": checkpoint_id,
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + segment_elapsed,
            "error": None,
            "progress": progress,
            "result_eligible": status == "completed",
            "resume_state_complete": True,
            "segments": segments,
            "status": status,
        }
    )
    _persist_training_metadata(run_dir, output_model_dir, metadata)
    return metadata


def _run_mock_schedule(
    selected: Sequence[TrainingGroup],
    *,
    run_dir: Path,
    output_model_dir: Path,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    started: float,
    max_wall_seconds: float,
    stop_after_optimizer_steps: int | None,
) -> dict[str, Any]:
    """Execute the exact schedule using deterministic scalar arithmetic only."""

    cursor = int(progress["group_exposures"])
    if cursor:
        state_path = output_model_dir / CHECKPOINT_DIRNAME / MOCK_STATE_FILENAME
        state = _read_json(state_path)
        if state.get("run_contract_sha256") != metadata.get("run_contract_sha256"):
            raise ValueError("mock resume checkpoint run contract mismatch")
        if state.get("selection_sha256") != metadata.get("selection_sha256"):
            raise ValueError("mock resume checkpoint selection mismatch")
        state_progress = state.get("progress")
        if not isinstance(state_progress, Mapping):
            raise ValueError("mock resume checkpoint lacks progress")
        for key in (
            "group_exposures",
            "micro_steps",
            "mock_scalar",
            "optimizer_steps",
            "run_contract_sha256",
            "selection_sha256",
        ):
            if state_progress.get(key) != progress.get(key):
                raise ValueError(f"mock resume checkpoint/progress mismatch: {key}")
    scalar = float(progress.get("mock_scalar", 0.0))
    start_step = int(progress["optimizer_steps"])
    target_step = OPTIMIZER_STEPS
    if stop_after_optimizer_steps is not None:
        target_step = min(
            OPTIMIZER_STEPS, start_step + int(stop_after_optimizer_steps)
        )
    status = "completed"
    while cursor < TRAIN_GROUPS:
        group = selected[cursor]
        digest = _canonical_sha256(_group_identity(group))
        scalar += int(digest[:12], 16) / float(16**12)
        cursor += 1
        if cursor % GRADIENT_ACCUMULATION_STEPS:
            continue
        optimizer_step = cursor // GRADIENT_ACCUMULATION_STEPS
        scalar = scalar / (1.0 + optimizer_step / 10_000.0)
        progress = {
            **progress,
            "group_exposures": cursor,
            "micro_steps": cursor,
            "mock_scalar": scalar,
            "optimizer_steps": optimizer_step,
            "status": "running",
            "updated_at": _utc_now(),
        }
        _append_jsonl(
            run_dir / "learning_curve.jsonl",
            {
                "mock_cpu": True,
                "optimizer_step": optimizer_step,
                "scalar": scalar,
            },
        )
        _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
        _write_mock_checkpoint(output_model_dir, metadata, progress)
        if optimizer_step >= target_step and optimizer_step < OPTIMIZER_STEPS:
            status = "mock_pause"
            break
        if time.perf_counter() - started >= max_wall_seconds:
            status = "wall_time_exhausted"
            break
    if cursor == TRAIN_GROUPS:
        status = "completed"
    if status == "completed" and (
        cursor != TRAIN_GROUPS or int(progress["optimizer_steps"]) != OPTIMIZER_STEPS
    ):
        raise AssertionError("completed mock schedule has incorrect exposure counts")
    progress.update({"status": status, "updated_at": _utc_now()})
    _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
    _write_mock_checkpoint(output_model_dir, metadata, progress)
    mock_digest = _canonical_sha256(
        {
            "progress": progress,
            "run_contract_sha256": metadata["run_contract_sha256"],
        }
    )
    segment_elapsed = time.perf_counter() - started
    segments = list(metadata.get("segments", []))
    segments.append(
        {
            "device": "cpu",
            "elapsed_seconds": segment_elapsed,
            "ended_at": _utc_now(),
            "formal_checkpoint": False,
            "progress": dict(progress),
            "status": status,
        }
    )
    metadata.update(
        {
            "checkpoint_dir": str(output_model_dir / CHECKPOINT_DIRNAME),
            "checkpoint_files": [
                {
                    "name": MOCK_STATE_FILENAME,
                    "sha256": sha256_file(
                        output_model_dir / CHECKPOINT_DIRNAME / MOCK_STATE_FILENAME
                    ),
                }
            ],
            "checkpoint_id": f"mock-{METHOD_ID}@{mock_digest[:20]}",
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + segment_elapsed,
            "formal_checkpoint": False,
            "mock_cpu": True,
            "progress": progress,
            "result_eligible": False,
            "resume_state_complete": True,
            "segments": segments,
            "status": status,
        }
    )
    _persist_training_metadata(run_dir, output_model_dir, metadata)
    return metadata


def _matched_recipe(
    config: Mapping[str, Any],
    *,
    hashes: Mapping[str, str],
    probe: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    method = config["method"]
    return {
        "schema_version": 1,
        "anchor": METHOD_ID,
        "backbone_initialization": {
            "artifact_manifest_sha256": config["model"][
                "base_artifact_manifest_sha256"
            ],
            "base_model_path": config["model"]["base_model_path"],
            "tokenizer_sha256": config["model"]["tokenizer_sha256"],
            "weights_sha256": config["model"]["base_weights_sha256"],
            "seed": MODEL_INITIALIZATION_SEED,
        },
        "batch_requests": BATCH_REQUESTS,
        "checkpoint_first_optimizer_step": True,
        "checkpoint_every_optimizer_steps": CHECKPOINT_EVERY_OPTIMIZER_STEPS,
        "config_sha256": config["_config_sha256"],
        "dataset": dict(hashes),
        "dtype": training["dtype"],
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "gradient_checkpointing": training["gradient_checkpointing"],
        "group_construction": {
            "list_size": training["list_size"],
            "negatives_per_positive": training["negatives_per_positive"],
            "seed": training["seed"],
        },
        "group_exposures": TRAIN_GROUPS,
        "history_budget": training["history_budget"],
        "implementation_digest": implementation["digest"],
        "input_fields_used": list(SERIALIZED_INPUT_FIELDS),
        "max_grad_norm": training["max_grad_norm"],
        "max_length": training["max_length"],
        "objective": {
            "listwise_loss_weight": method["listwise_loss_weight"],
            "name": "0.5_ranknet_plus_0.5_tie_aware_listnet",
            "pairwise_loss_weight": method["pairwise_loss_weight"],
        },
        "optimizer": {
            "amsgrad": False,
            "betas": [0.9, 0.999],
            "class": "torch.optim.AdamW",
            "eps": 1.0e-8,
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
        },
        "optimizer_steps": OPTIMIZER_STEPS,
        "probe_manifest_sha256": probe["sha256"],
        "protocol_sha256": config["protocol"]["sha256"],
        "role": ROLE,
        "runtime_contract": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "scheduler": {
            "class": "linear_with_warmup",
            "total_steps": OPTIMIZER_STEPS,
            "warmup_ratio": training["warmup_ratio"],
            "warmup_steps": int(OPTIMIZER_STEPS * training["warmup_ratio"]),
        },
        "surface_classifier": {
            "function": (
                "myrec.mechanism.gradient_diagnostic.classify_train_surface"
            ),
            "module_sha256": sha256_file(
                Path(__file__).with_name("gradient_diagnostic.py")
            ),
            "surface_order": list(SURFACES),
        },
    }


def _validate_registered_probe(path: str | Path) -> dict[str, Any]:
    probe = _load_probe_manifest(path)
    matched = probe["payload"].get("m3_objective_and_gradients", {}).get(
        "matched_training_control"
    )
    expected = {
        "anchor": METHOD_ID,
        "train_groups": TRAIN_GROUPS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "conditions": list(CONDITIONS),
        "fixed": [
            "backbone_initialization",
            "optimizer",
            "learning_rate",
            "visible_fields",
            "total_steps",
        ],
        "role": ROLE,
    }
    if matched != expected:
        raise ValueError("frozen probe manifest matched-training-control drift")
    return probe


def matched_training_control_implementation_identity() -> dict[str, Any]:
    """Bind formal checkpoints to the project-owned trainer and CLI."""

    root = Path(__file__).resolve().parents[3]
    paths = {
        "scripts/train_q2_matched_control.py": root
        / "scripts/train_q2_matched_control.py",
        "src/myrec/mechanism/matched_training_control.py": Path(__file__).resolve(),
    }
    files = []
    for relative, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing matched-control trainer: {path}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {"digest": _canonical_sha256(files), "files": files}


def _validate_q2_recipe(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    probe: Mapping[str, Any],
) -> None:
    if config.get("method_id") != METHOD_ID:
        raise ValueError("matched training control is registered only for Q2")
    entry = probe.get("frozen_inputs", {}).get("models", {}).get(METHOD_ID)
    if not isinstance(entry, Mapping):
        raise ValueError("probe manifest lacks the frozen Q2 binding")
    root = Path(__file__).resolve().parents[3]
    if config_path.resolve() != (root / str(entry["config"])).resolve():
        raise ValueError("Q2 config path differs from the frozen probe manifest")
    if config["_config_sha256"] != entry.get("config_sha256"):
        raise ValueError("Q2 config hash differs from the frozen probe manifest")
    training = config["training"]
    expected = {
        "batch_requests": BATCH_REQUESTS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": 1.0e-5,
        "history_dropout_probability": 0.0,
    }
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f"frozen Q2 matched recipe drift: training.{key}")
    if TRAIN_GROUPS != OPTIMIZER_STEPS * GRADIENT_ACCUMULATION_STEPS:
        raise AssertionError("registered group/optimizer schedule is inconsistent")


def _assert_independent_output_path(
    output_model_dir: Path, probe: Mapping[str, Any]
) -> None:
    entry = probe.get("frozen_inputs", {}).get("models", {}).get(METHOD_ID)
    if not isinstance(entry, Mapping):
        raise ValueError("probe manifest lacks frozen Q2 checkpoint path")
    root = Path(__file__).resolve().parents[3]
    frozen = (root / str(entry["checkpoint"])).resolve()
    output = output_model_dir.resolve()
    if output == frozen or frozen in output.parents or output in frozen.parents:
        raise ValueError("diagnostic checkpoint path overlaps the frozen Q2 checkpoint")


def _assert_run_output_separation(run_dir: Path, output_model_dir: Path) -> None:
    run = run_dir.resolve()
    output = output_model_dir.resolve()
    if run == output or run in output.parents or output in run.parents:
        raise ValueError("run metadata and model checkpoint directories must be independent")


def _prepare_run(
    run_dir: Path,
    *,
    output_model_dir: Path,
    base_metadata: dict[str, Any],
    selection: dict[str, Any],
    selection_sha256: str,
    run_contract: dict[str, Any],
    run_contract_sha256: str,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = run_dir / METADATA_FILENAME
    progress_path = run_dir / PROGRESS_FILENAME
    selection_path = run_dir / SELECTION_FILENAME
    contract_path = run_dir / RUN_CONTRACT_FILENAME
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {run_dir}")
        if output_model_dir.exists() and any(output_model_dir.iterdir()):
            raise FileExistsError(
                f"model output directory is not empty: {output_model_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        output_model_dir.mkdir(parents=True, exist_ok=True)
        progress = {
            "schema_version": 1,
            "group_exposures": 0,
            "micro_steps": 0,
            "optimizer_steps": 0,
            "resume_count": 0,
            "run_contract_sha256": run_contract_sha256,
            "selection_sha256": selection_sha256,
            "status": "selection_finalized",
            "updated_at": _utc_now(),
        }
        metadata = {**base_metadata, "elapsed_seconds": 0.0, "resume_lineage": []}
        _write_json_atomic(selection_path, selection)
        if _canonical_sha256(_read_json(selection_path)) != selection_sha256:
            raise AssertionError("persisted selection hash changed")
        _write_json_atomic(contract_path, run_contract)
        metadata["selection_file_sha256"] = sha256_file(selection_path)
        metadata["run_contract_file_sha256"] = sha256_file(contract_path)
        _write_json_atomic(progress_path, progress)
        _write_json_atomic(metadata_path, metadata)
        return metadata, progress

    if not run_dir.is_dir() or not output_model_dir.is_dir():
        raise FileNotFoundError("resume run/checkpoint directory is missing")
    metadata = _read_json(metadata_path)
    progress = _read_json(progress_path)
    persisted_selection = _read_json(selection_path)
    persisted_contract = _read_json(contract_path)
    if _canonical_sha256(persisted_selection) != selection_sha256:
        raise ValueError("resume selection differs from frozen selection")
    if _canonical_sha256(persisted_contract) != run_contract_sha256:
        raise ValueError("resume run contract differs")
    if metadata.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("resume metadata run contract differs")
    if metadata.get("selection_file_sha256") != sha256_file(selection_path):
        raise ValueError("resume selection file hash differs")
    if metadata.get("run_contract_file_sha256") != sha256_file(contract_path):
        raise ValueError("resume run-contract file hash differs")
    if progress.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("resume progress run contract differs")
    if int(progress.get("group_exposures", -1)) % GRADIENT_ACCUMULATION_STEPS:
        raise ValueError("resume progress is not at an optimizer boundary")
    if int(progress.get("optimizer_steps", -1)) != int(
        progress["group_exposures"]
    ) // GRADIENT_ACCUMULATION_STEPS:
        raise ValueError("resume exposure/optimizer counts disagree")
    if metadata.get("status") not in {
        "failed",
        "mock_pause",
        "running",
        "selection_finalized",
        "wall_time_exhausted",
    }:
        raise ValueError(f"training status is not resumable: {metadata.get('status')}")
    lineage = list(metadata.get("resume_lineage", []))
    lineage.append(
        {
            "from_group_exposures": progress["group_exposures"],
            "from_optimizer_steps": progress["optimizer_steps"],
            "from_status": metadata["status"],
            "resumed_at": _utc_now(),
        }
    )
    metadata.update({"resume_lineage": lineage, "status": "selection_finalized"})
    progress.update(
        {
            "resume_count": int(progress.get("resume_count", 0)) + 1,
            "status": "selection_finalized",
            "updated_at": _utc_now(),
        }
    )
    _write_json_atomic(progress_path, progress)
    _write_json_atomic(metadata_path, metadata)
    return metadata, progress


def _write_mock_checkpoint(
    output_model_dir: Path,
    metadata: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> None:
    checkpoint_dir = output_model_dir / CHECKPOINT_DIRNAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        checkpoint_dir / MOCK_STATE_FILENAME,
        {
            "formal_checkpoint": False,
            "progress": dict(progress),
            "run_contract_sha256": metadata["run_contract_sha256"],
            "selection_sha256": metadata["selection_sha256"],
        },
    )


def _persist_training_metadata(
    run_dir: Path, output_model_dir: Path, metadata: Mapping[str, Any]
) -> None:
    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
    _write_json_atomic(output_model_dir / TRAINING_METADATA, metadata)


def _group_identity(group: TrainingGroup) -> dict[str, Any]:
    return {
        "candidate_item_ids": [str(row["item_id"]) for row in group.candidates],
        "gains": [float(value) for value in group.gains],
        "request_id": str(group.record.request_id),
    }


def _stable_digest(seed: int, *values: str) -> str:
    payload = "|".join([str(seed), *[str(value) for value in values]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return sha256_text(_canonical_json(value))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _truncate_learning_curve(
    path: Path, *, checkpoint_optimizer_steps: int
) -> None:
    if not path.is_file():
        if checkpoint_optimizer_steps:
            raise FileNotFoundError(path)
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "optimizer_step" not in row:
            raise ValueError("learning curve contains an invalid row")
        if int(row["optimizer_step"]) <= checkpoint_optimizer_steps:
            retained.append(row)
    observed_steps = [int(row["optimizer_step"]) for row in retained]
    if observed_steps != list(range(1, checkpoint_optimizer_steps + 1)):
        raise ValueError("learning curve checkpoint prefix is incomplete or duplicated")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _require_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported matched-control condition={condition!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BALANCED_QUOTAS",
    "CONDITIONS",
    "GRADIENT_ACCUMULATION_STEPS",
    "MAX_WALL_SECONDS",
    "METHOD_ID",
    "OPTIMIZER_STEPS",
    "ROLE",
    "TRAIN_GROUPS",
    "matched_training_control_implementation_identity",
    "run_q2_matched_training_control",
    "select_matched_training_groups",
]
