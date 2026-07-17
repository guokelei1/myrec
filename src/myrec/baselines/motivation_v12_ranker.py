"""Resumable shared Qwen harness for Motivation V1.2 Q0--Q3.

The project owns the loader, label boundary, training loop, checkpoint state,
counterfactual scorer, and score export.  Method-specific behavior is limited
to the frozen prompt/response boundary, adaptation mode, and ranking loss.
"""

from __future__ import annotations

import copy
import json
import math
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from myrec.baselines.motivation_v12_contracts import (
    METHOD_IDS,
    SERIALIZED_INPUT_FIELDS,
    TrainingGroup,
    batched_indices,
    build_prompt_sections,
    complete_candidate_chunks,
    encode_instructrec_selection_prompt,
    encode_prompt_sections,
    epoch_batch_order,
    instructrec_template_index,
    listwise_target_distribution,
    load_training_groups,
    pairwise_index_pairs,
    sanitize_record_for_model,
)
from myrec.data.kuaisearch_holdout import (
    V12_DATASET_VERSION,
    verify_published_holdout,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json


RUN_ID_PATTERN = re.compile(r"^\d{8}_[a-z0-9][a-z0-9_]*$")
CHECKPOINT_DIRNAME = "checkpoint_latest"
TRAINING_METADATA = "training_metadata.json"
_BASE_INFERENCE_SUFFIXES = {
    ".jinja",
    ".json",
    ".model",
    ".safetensors",
    ".txt",
}


def load_v12_ranker_config(path: str | Path) -> dict[str, Any]:
    """Load and verify a frozen method config plus its protocol/model hashes."""

    import yaml

    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"invalid V1.2 ranker config: {path}")
    method_id = str(config.get("method_id") or "")
    if method_id not in METHOD_IDS:
        raise ValueError(f"unsupported V1.2 method_id={method_id}")
    protocol = Path(config["protocol"]["path"])
    if sha256_file(protocol) != str(config["protocol"]["sha256"]):
        raise ValueError("frozen V1.2 protocol hash mismatch")
    protocol_config = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    base_path = Path(config["model"]["base_model_path"])
    base_artifacts = _base_inference_artifact_identity(base_path)
    expected_artifact_digest = str(
        config["model"].get("base_artifact_manifest_sha256") or ""
    )
    if base_artifacts["sha256"] != expected_artifact_digest:
        raise ValueError("frozen base-model inference artifact manifest mismatch")
    weights = base_path / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(weights)
    if sha256_file(weights) != str(config["model"]["base_weights_sha256"]):
        raise ValueError("frozen base-model weight hash mismatch")
    tokenizer_path = base_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)
    if sha256_file(tokenizer_path) != str(config["model"]["tokenizer_sha256"]):
        raise ValueError("frozen base-model tokenizer hash mismatch")
    training = config.get("training", {})
    for key in (
        "batch_requests",
        "epochs",
        "gradient_accumulation_steps",
        "history_budget",
        "learning_rate",
        "max_length",
        "negatives_per_positive",
        "seed",
    ):
        if key not in training:
            raise ValueError(f"training config missing {key}")
    _validate_frozen_recipe(config, protocol_config)
    config["_config_path"] = str(path)
    config["_config_sha256"] = sha256_file(path)
    config["_protocol"] = protocol_config
    return config


def _validate_frozen_recipe(
    config: dict[str, Any], protocol: dict[str, Any]
) -> None:
    """Reject method configs that drift from the pre-outcome protocol."""

    method_id = str(config["method_id"])
    training = config["training"]
    common = protocol["common_training"]
    protocol_method = protocol["methods"][method_id]
    common_keys = (
        "dtype",
        "epochs",
        "gradient_checkpointing",
        "history_budget",
        "max_grad_norm",
        "max_length",
        "negatives_per_positive",
        "safe_exit_seconds",
        "warmup_ratio",
        "weight_decay",
    )
    for key in common_keys:
        if training.get(key) != common[key]:
            raise ValueError(f"training.{key} drifted from frozen protocol")
    if training.get("max_continuous_job_seconds") != common[
        "max_continuous_job_seconds"
    ]:
        raise ValueError("training.max_continuous_job_seconds drifted")
    if training.get("history_dropout_probability") != common[
        "history_dropout_probability"
    ]:
        raise ValueError("training.history_dropout_probability drifted")
    if int(training["seed"]) != int(protocol["seed_policy"]["pilot_seed"]):
        raise ValueError("training.seed drifted from frozen pilot seed")
    for key in (
        "batch_requests",
        "gradient_accumulation_steps",
        "learning_rate",
    ):
        if training.get(key) != protocol_method[key]:
            raise ValueError(f"training.{key} drifted from frozen method recipe")
    for key in ("context_token_budget", "list_size", "max_target_length"):
        if key in protocol_method and training.get(key) != protocol_method[key]:
            raise ValueError(f"training.{key} drifted from frozen method recipe")

    if method_id == "q0_qwen3_reranker_06b":
        frozen_model = protocol["models"]["q0"]
        expected_path = frozen_model["base_model"]
        expected_adaptation = "full_parameter"
    else:
        frozen_model = protocol["models"]["shared_general_qwen"]
        expected_path = frozen_model["local_path"]
        expected_adaptation = protocol_method["adaptation"]
    if config["model"].get("base_model_path") != expected_path:
        raise ValueError("base_model_path drifted from frozen protocol")
    expected_weight = frozen_model.get("base_weights_sha256") or frozen_model.get(
        "weights_sha256"
    )
    if config["model"].get("base_weights_sha256") != expected_weight:
        raise ValueError("base_weights_sha256 drifted from frozen protocol")
    if config["model"].get("tokenizer_sha256") != frozen_model[
        "tokenizer_sha256"
    ]:
        raise ValueError("tokenizer_sha256 drifted from frozen protocol")
    if config["model"].get("base_artifact_manifest_sha256") != frozen_model[
        "artifact_manifest_sha256"
    ]:
        raise ValueError(
            "base_artifact_manifest_sha256 drifted from frozen protocol"
        )
    if config["model"].get("adaptation") != expected_adaptation:
        raise ValueError("model adaptation drifted from frozen protocol")

    method = config.get("method", {})
    if method.get("mechanism") != protocol_method["mechanism"]:
        raise ValueError("method mechanism drifted from frozen protocol")
    for key in (
        "lora_alpha",
        "lora_dropout",
        "lora_rank",
        "lora_targets",
        "listwise_loss_weight",
        "pairwise_loss_weight",
    ):
        if key in protocol_method and method.get(key) != protocol_method[key]:
            raise ValueError(f"method.{key} drifted from frozen protocol")


def train_motivation_v12_ranker(
    standardized_dir: str | Path,
    config_path: str | Path,
    run_id: str,
    output_model_dir: str | Path,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    resume: bool = False,
    command: Sequence[str] | None = None,
    max_wall_seconds: float | None = None,
    max_train_groups: int | None = None,
) -> dict[str, Any]:
    """Train or resume one frozen Q0--Q3 recipe at an optimizer-step boundary."""

    started = time.perf_counter()
    _validate_run_id(run_id)
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    training = config["training"]
    implementation_identity = _implementation_identity()
    standardized_dir = Path(standardized_dir)
    output_model_dir = Path(output_model_dir)
    run_dir = Path(runs_dir) / run_id
    _prepare_run_directory(run_dir)
    if resume:
        _recover_training_checkpoint(output_model_dir)
        if not (output_model_dir / CHECKPOINT_DIRNAME).is_dir():
            raise FileNotFoundError(
                f"resume checkpoint missing: {output_model_dir / CHECKPOINT_DIRNAME}"
            )
    elif output_model_dir.exists() and any(output_model_dir.iterdir()):
        raise FileExistsError(f"model output directory is not empty: {output_model_dir}")
    output_model_dir.mkdir(parents=True, exist_ok=True)

    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    manifest_path = standardized_dir / "manifest.json"
    _assert_frozen_training_population(standardized_dir, config)
    groups, group_stats = load_training_groups(
        records_path,
        qrels_path,
        seed=int(training["seed"]),
        negatives_per_positive=int(training["negatives_per_positive"]),
        max_group_size=int(training.get("list_size", 8)),
    )
    evidence_mode = "first_round_pilot"
    if max_train_groups is not None:
        if max_train_groups <= 0:
            raise ValueError("max_train_groups must be positive")
        groups = groups[:max_train_groups]
        evidence_mode = "smoke_non_result"
    if not groups:
        raise ValueError("training group cap produced an empty run")

    import torch
    import transformers
    from transformers.optimization import get_linear_schedule_with_warmup

    seed = int(training["seed"])
    _seed_everything(torch, seed)
    checkpoint_dir = output_model_dir / CHECKPOINT_DIRNAME
    tokenizer, model = _load_model_and_tokenizer(
        config,
        device=device,
        training=True,
        checkpoint_model_dir=(checkpoint_dir / "model") if resume else None,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    batches_per_epoch = math.ceil(len(groups) / int(training["batch_requests"]))
    updates_per_epoch = math.ceil(
        batches_per_epoch / int(training["gradient_accumulation_steps"])
    )
    total_optimizer_steps = updates_per_epoch * int(training["epochs"])
    warmup_steps = int(total_optimizer_steps * float(training.get("warmup_ratio", 0.1)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    dtype = str(training.get("dtype", "bfloat16"))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == "float16")
    training_contract = _build_training_contract(
        batches_per_epoch=batches_per_epoch,
        config_sha256=config["_config_sha256"],
        evidence_mode=evidence_mode,
        group_count=len(groups),
        implementation_digest=implementation_identity["digest"],
        manifest_sha256=sha256_file(manifest_path),
        max_train_groups=max_train_groups,
        qrels_train_sha256=sha256_file(qrels_path),
        records_train_sha256=sha256_file(records_path),
        total_optimizer_steps=total_optimizer_steps,
        updates_per_epoch=updates_per_epoch,
    )
    progress = {
        "batch_cursor": 0,
        "epoch": 0,
        "micro_steps": 0,
        "optimizer_steps": 0,
    }
    prior_segments: list[dict[str, Any]] = []
    if resume:
        state = torch.load(
            checkpoint_dir / "trainer_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if state.get("config_sha256") != config["_config_sha256"]:
            raise ValueError("resume checkpoint config hash mismatch")
        if state.get("training_contract") != training_contract:
            raise ValueError("resume checkpoint training/data/cap contract mismatch")
        observed_checkpoint_id, _ = _checkpoint_identity(
            checkpoint_dir / "model", method_id
        )
        if state.get("checkpoint_id") != observed_checkpoint_id:
            raise ValueError("resume checkpoint inference artifacts changed")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        progress = dict(state["progress"])
        _restore_rng_state(torch, state["rng"])
        prior_metadata_path = output_model_dir / TRAINING_METADATA
        if prior_metadata_path.exists():
            prior = json.loads(prior_metadata_path.read_text(encoding="utf-8"))
            prior_segments = list(prior.get("segments", []))

    max_seconds = (
        float(max_wall_seconds)
        if max_wall_seconds is not None
        else float(training.get("safe_exit_seconds", 13500))
    )
    safe_exit_limit = float(training.get("safe_exit_seconds", 13500))
    if not 0 < max_seconds <= safe_exit_limit:
        raise ValueError(
            f"one continuous training segment must be in (0, {safe_exit_limit}] seconds"
        )
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    loss_count = 0
    status = "completed"
    learning_curve_path = run_dir / "learning_curve.jsonl"
    checkpoint_every = int(training.get("checkpoint_every_optimizer_steps", 500))
    current_progress = dict(progress)

    for epoch in range(int(progress["epoch"]), int(training["epochs"])):
        order = epoch_batch_order(len(groups), seed=seed, epoch=epoch)
        batches = batched_indices(order, int(training["batch_requests"]))
        start_cursor = int(progress["batch_cursor"]) if epoch == int(progress["epoch"]) else 0
        accumulated = 0
        for batch_cursor in range(start_cursor, len(batches)):
            batch = [groups[index] for index in batches[batch_cursor]]
            autocast_dtype = (
                torch.float16 if dtype == "float16" else torch.bfloat16
            )
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=str(device).startswith("cuda") and dtype != "float32",
            ):
                raw_loss, loss_parts = _training_batch_loss(
                    model,
                    tokenizer,
                    batch,
                    config,
                    device=device,
                )
            if not bool(torch.isfinite(raw_loss).item()):
                raise FloatingPointError(
                    f"non-finite {method_id} loss at epoch={epoch} batch={batch_cursor}"
                )
            accumulation_steps = int(training["gradient_accumulation_steps"])
            block_start = batch_cursor - accumulated
            accumulation_target = min(
                accumulation_steps,
                len(batches) - block_start,
            )
            scaler.scale(raw_loss / accumulation_target).backward()
            accumulated += 1
            loss_value = float(raw_loss.detach().cpu())
            loss_sum += loss_value
            loss_count += 1
            progress["micro_steps"] = int(progress["micro_steps"]) + 1
            should_step = (
                accumulated >= int(training["gradient_accumulation_steps"])
                or batch_cursor + 1 == len(batches)
            )
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable, float(training.get("max_grad_norm", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            progress["optimizer_steps"] = int(progress["optimizer_steps"]) + 1
            next_epoch = epoch
            next_cursor = batch_cursor + 1
            if next_cursor == len(batches):
                next_epoch = epoch + 1
                next_cursor = 0
            current_progress = {
                **progress,
                "epoch": next_epoch,
                "batch_cursor": next_cursor,
            }
            _append_jsonl(
                learning_curve_path,
                {
                    "batch_cursor": batch_cursor,
                    "epoch": epoch,
                    "loss": loss_value,
                    "loss_parts": loss_parts,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "optimizer_step": progress["optimizer_steps"],
                },
            )
            elapsed = time.perf_counter() - started
            periodic = (
                checkpoint_every > 0
                and int(progress["optimizer_steps"]) % checkpoint_every == 0
            )
            time_exit = elapsed >= max_seconds
            if periodic and not time_exit:
                _save_training_checkpoint(
                    output_model_dir,
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    scaler,
                    current_progress,
                    config,
                    training_contract,
                    torch,
                )
            if time_exit:
                status = "safe_exit_pending"
                break
        progress = current_progress
        if status != "completed":
            break
    if status == "completed":
        progress = {
            **progress,
            "epoch": int(training["epochs"]),
            "batch_cursor": 0,
        }
    _save_training_checkpoint(
        output_model_dir,
        model,
        tokenizer,
        optimizer,
        scheduler,
        scaler,
        progress,
        config,
        training_contract,
        torch,
    )
    elapsed = time.perf_counter() - started
    checkpoint_id, weight_files = _checkpoint_identity(
        output_model_dir / CHECKPOINT_DIRNAME / "model", method_id
    )
    segment = {
        "elapsed_seconds": elapsed,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "gpu_hours": elapsed / 3600.0,
        "progress": progress,
        "run_id": run_id,
        "status": status,
    }
    metadata = {
        "base_model_path": str(config["model"]["base_model_path"]),
        "base_artifact_manifest_sha256": config["model"][
            "base_artifact_manifest_sha256"
        ],
        "base_tokenizer_sha256": config["model"]["tokenizer_sha256"],
        "base_weights_sha256": config["model"]["base_weights_sha256"],
        "checkpoint_dir": str(output_model_dir / CHECKPOINT_DIRNAME),
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": weight_files,
        "code_revision": _git_revision(),
        "command": list(command or sys.argv),
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "evidence_mode": evidence_mode,
        "example_stats": group_stats,
        "input_fields_used": list(SERIALIZED_INPUT_FIELDS),
        "implementation_identity": implementation_identity,
        "method_id": method_id,
        "objective": _objective_name(method_id),
        **_runtime_metadata(method_id, torch, transformers),
        "progress": progress,
        "protocol_path": config["protocol"]["path"],
        "protocol_sha256": config["protocol"]["sha256"],
        "qrels_read": True,
        "qrels_train_sha256": sha256_file(qrels_path),
        "records_train_sha256": sha256_file(records_path),
        "resume_state_complete": True,
        "run_id": run_id,
        "seed": seed,
        "segments": [*prior_segments, segment],
        "status": status,
        "training": dict(training),
        "training_loss_mean_this_segment": loss_sum / loss_count if loss_count else None,
    }
    if method_id == "q3_tallrec_generalqwen":
        metadata["trainable_parameters"] = _parameter_counts(model)
    write_json(output_model_dir / TRAINING_METADATA, metadata)
    write_json(run_dir / "metadata.json", metadata)
    shutil.copyfile(config_path, run_dir / f"config_snapshot{Path(config_path).suffix}")
    return metadata


def write_motivation_v12_scores(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    history_assignments_path: str | Path,
    history_condition: str,
    split: str,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    command: Sequence[str] | None = None,
    max_score_requests: int | None = None,
) -> dict[str, Any]:
    """Score one full/null/wrong condition without opening evaluation qrels."""

    _validate_run_id(run_id)
    if history_condition not in {"true", "null", "wrong"}:
        raise ValueError(f"unsupported history_condition={history_condition}")
    if split not in {"dev", "confirmation"}:
        raise ValueError("V1.2 scoring supports dev or confirmation only")
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    training = config["training"]
    scoring = config.get("scoring", {})
    standardized_dir = Path(standardized_dir)
    checkpoint_root = Path(checkpoint_root)
    checkpoint_dir = checkpoint_root / CHECKPOINT_DIRNAME
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = json.loads(
        training_metadata_path.read_text(encoding="utf-8")
    )
    _validate_scoring_checkpoint_provenance(
        training_metadata,
        config,
        allow_smoke=max_score_requests is not None,
    )
    assignments_path = Path(history_assignments_path)
    records_path = standardized_dir / f"records_{split}.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    dataset_manifest_path = standardized_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    checkpoint_id, weight_files = _checkpoint_identity(
        checkpoint_dir / "model", method_id
    )
    if checkpoint_id != training_metadata["checkpoint_id"]:
        raise ValueError("checkpoint weights changed after training metadata was written")
    population_integrity = _assert_scoring_population(
        standardized_dir,
        config,
        split=split,
        dataset_manifest=dataset_manifest,
        checkpoint_id=checkpoint_id,
        checkpoint_weight_files=weight_files,
        training_metadata_path=training_metadata_path,
    )
    assignments = _load_history_assignments(
        assignments_path, expected_condition=history_condition
    )
    assignment_manifest = _load_history_assignment_manifest(
        assignments_path,
        expected_condition=history_condition,
        records_path=records_path,
    )

    import torch
    import transformers

    tokenizer, model = _load_model_and_tokenizer(
        config,
        device=device,
        training=False,
        checkpoint_model_dir=checkpoint_dir / "model",
    )
    model.eval()
    run_dir = Path(runs_dir) / run_id
    _prepare_run_directory(run_dir)
    score_batch_size = int(scoring.get("batch_size", 8))
    scores_path = run_dir / "scores.jsonl"
    rows = 0
    requests = 0
    seen_requests: set[str] = set()
    prompt_boundary_count = 0
    request_score_ranges: list[float] = []
    started = time.perf_counter()
    with scores_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for raw_record in iter_jsonl(records_path):
            if max_score_requests is not None and requests >= max_score_requests:
                break
            record = sanitize_record_for_model(raw_record)
            if record.request_id not in assignments:
                raise ValueError(f"history assignment missing {record.request_id}")
            history = _validate_assigned_history(
                raw_record,
                record,
                assignments[record.request_id],
                history_condition=history_condition,
            )
            seen_requests.add(record.request_id)
            requests += 1
            if method_id == "q1_instructrec_generalqwen":
                request_scores, at_boundary = _score_instructrec_request(
                    model,
                    tokenizer,
                    record,
                    history,
                    config,
                    device=device,
                    batch_size=score_batch_size,
                )
            else:
                request_scores, at_boundary = _score_yes_no_request(
                    model,
                    tokenizer,
                    record,
                    history,
                    config,
                    device=device,
                    batch_size=score_batch_size,
                )
            prompt_boundary_count += int(at_boundary)
            if set(request_scores) != {str(row["item_id"]) for row in record.candidates}:
                raise AssertionError(
                    f"candidate scoring coverage failed for {record.request_id}"
                )
            values_for_request = list(request_scores.values())
            request_score_ranges.append(
                max(values_for_request) - min(values_for_request)
            )
            for candidate in record.candidates:
                item_id = str(candidate["item_id"])
                value = float(request_scores[item_id])
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"non-finite score request_id={record.request_id} item_id={item_id}"
                    )
                handle.write(
                    json.dumps(
                        {
                            "candidate_item_id": item_id,
                            "method_id": method_id,
                            "request_id": record.request_id,
                            "score": value,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                rows += 1
    if max_score_requests is None:
        extras = set(assignments) - seen_requests
        if extras:
            raise ValueError(f"assignments contain unknown request IDs: {sorted(extras)[:5]}")
    nonconstant_requests = _enforce_score_nondegeneracy(
        run_dir,
        request_score_ranges,
        capped=max_score_requests is not None,
    )
    elapsed = time.perf_counter() - started
    runtime_metadata = _runtime_metadata(method_id, torch, transformers)
    scoring_signature = {
        "batch_size": score_batch_size,
        "base_artifact_manifest_sha256": config["model"][
            "base_artifact_manifest_sha256"
        ],
        "config_sha256": config["_config_sha256"],
        "dtype": training.get("dtype", "bfloat16"),
        "history_budget": int(training["history_budget"]),
        "implementation_digest": _implementation_identity()["digest"],
        "inference_score": (
            "normalized_candidate_response_log_likelihood"
            if method_id == "q1_instructrec_generalqwen"
            else (
                "complete_answer_log_probability_yes_minus_no"
                if method_id == "q3_tallrec_generalqwen"
                else "raw_yes_minus_no_next_token_logit"
            )
        ),
        "max_length": int(training["max_length"]),
        "max_target_length": training.get("max_target_length"),
        "context_token_budget": training.get("context_token_budget"),
        "method_id": method_id,
        "prompt_contract": "motivation_v12_project_owned_v2",
        "protocol_sha256": config["protocol"]["sha256"],
        "request_aligned_batches": True,
        "runtime_versions": {
            "python": runtime_metadata["python_version"],
            **runtime_metadata["package_versions"],
        },
        "holdout_integrity_lock_sha256": (
            population_integrity["integrity_lock_sha256"]
            if population_integrity is not None
            else None
        ),
        "holdout_release_lock_sha256": (
            population_integrity["post_selection_recipe_checkpoint_lock_sha256"]
            if population_integrity is not None
            else None
        ),
        "target_terminator": (
            "qwen_im_end" if method_id in {
                "q1_instructrec_generalqwen",
                "q3_tallrec_generalqwen",
            } else None
        ),
        "tokenizer_sha256": config["model"]["tokenizer_sha256"],
    }
    metadata = {
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": weight_files,
        "code_revision": _git_revision(),
        "command": list(command or sys.argv),
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "dataset_id": str(dataset_manifest["dataset_id"]),
        "dataset_version": str(dataset_manifest["dataset_version"]),
        "elapsed_seconds": elapsed,
        "evidence_mode": (
            "smoke_non_result" if max_score_requests is not None else "first_round_pilot"
        ),
        "training_evidence_mode": training_metadata["evidence_mode"],
        "history_assignment_sha256": sha256_file(assignments_path),
        "history_assignment_manifest_path": assignment_manifest["path"],
        "history_assignment_manifest_sha256": assignment_manifest["sha256"],
        "history_assignment_semantics_verified": True,
        "history_assignments_path": str(assignments_path),
        "history_condition": history_condition,
        "holdout_integrity": population_integrity,
        "input_fields_used": list(SERIALIZED_INPUT_FIELDS),
        "implementation_identity": _implementation_identity(),
        "method_id": method_id,
        **runtime_metadata,
        "prompt_at_max_boundary_requests": prompt_boundary_count,
        "qrels_read": False,
        "request_count": requests,
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "score_non_degeneracy": {
            "max_request_range": max(request_score_ranges, default=0.0),
            "mean_request_range": (
                sum(request_score_ranges) / len(request_score_ranges)
                if request_score_ranges
                else 0.0
            ),
            "nonconstant_requests_at_1e_8": nonconstant_requests,
            "threshold": 1.0e-8,
        },
        "run_id": run_id,
        "score_rows": rows,
        "scores_sha256": sha256_file(scores_path),
        "scoring_signature": scoring_signature,
        "split": split,
        "standardized_dir": str(standardized_dir),
    }
    write_json(run_dir / "metadata.json", metadata)
    shutil.copyfile(config_path, run_dir / f"config_snapshot{Path(config_path).suffix}")
    return metadata


def pairwise_ranknet_loss(scores: Any, gains: Sequence[float]) -> Any:
    """Mean RankNet loss over all grade-different pairs."""

    import torch
    from torch.nn import functional as F

    pairs = pairwise_index_pairs(gains)
    if not pairs:
        raise ValueError("pairwise loss requires at least one grade-different pair")
    margins = torch.stack([scores[high] - scores[low] for high, low in pairs])
    return F.softplus(-margins).mean()


def listwise_softmax_loss(scores: Any, gains: Sequence[float]) -> Any:
    """Tie-aware ListNet cross entropy against normalized graded gains."""

    import torch
    from torch.nn import functional as F

    target = torch.tensor(
        listwise_target_distribution(gains),
        dtype=scores.dtype,
        device=scores.device,
    )
    return -(target * F.log_softmax(scores, dim=0)).sum()


def _training_batch_loss(
    model: Any,
    tokenizer: Any,
    groups: Sequence[TrainingGroup],
    config: dict[str, Any],
    *,
    device: str,
) -> tuple[Any, dict[str, float]]:
    method_id = str(config["method_id"])
    if method_id == "q1_instructrec_generalqwen":
        loss = _instructrec_training_loss(
            model, tokenizer, groups, config, device=device
        )
        return loss, {"candidate_response_nll": float(loss.detach().cpu())}
    if method_id == "q3_tallrec_generalqwen":
        loss = _tallrec_training_loss(model, tokenizer, groups, config, device=device)
        return loss, {"answer_only_nll": float(loss.detach().cpu())}
    scores_by_group = _yes_no_group_scores(
        model, tokenizer, groups, config, device=device
    )
    import torch
    from torch.nn import functional as F

    if method_id == "q0_qwen3_reranker_06b":
        logits = torch.cat(scores_by_group)
        labels = torch.tensor(
            [float(gain > 0) for group in groups for gain in group.gains],
            dtype=logits.dtype,
            device=logits.device,
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, {"pointwise_bce": float(loss.detach().cpu())}
    pair_losses = [
        pairwise_ranknet_loss(scores, group.gains)
        for scores, group in zip(scores_by_group, groups)
    ]
    list_losses = [
        listwise_softmax_loss(scores, group.gains)
        for scores, group in zip(scores_by_group, groups)
    ]
    pair = torch.stack(pair_losses).mean()
    listwise = torch.stack(list_losses).mean()
    weights = config["method"]
    loss = float(weights["pairwise_loss_weight"]) * pair + float(
        weights["listwise_loss_weight"]
    ) * listwise
    return loss, {
        "pairwise_ranknet": float(pair.detach().cpu()),
        "listwise_softmax": float(listwise.detach().cpu()),
    }


def _yes_no_group_scores(
    model: Any,
    tokenizer: Any,
    groups: Sequence[TrainingGroup],
    config: dict[str, Any],
    *,
    device: str,
) -> list[Any]:
    token_lists: list[list[int]] = []
    sizes: list[int] = []
    training = config["training"]
    for group in groups:
        sizes.append(len(group.candidates))
        for candidate in group.candidates:
            sections = build_prompt_sections(
                config["method_id"],
                group.record,
                candidate,
                history_budget=int(training["history_budget"]),
            )
            token_lists.append(
                encode_prompt_sections(
                    tokenizer, sections, max_length=int(training["max_length"])
                )
            )
    input_ids, attention_mask = _left_pad(token_lists, tokenizer.pad_token_id, device)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=1,
    )
    logits = output.logits[:, -1, :]
    yes_id = _single_token_id(tokenizer, "yes")
    no_id = _single_token_id(tokenizer, "no")
    differences = logits[:, yes_id] - logits[:, no_id]
    result = []
    offset = 0
    for size in sizes:
        result.append(differences[offset : offset + size])
        offset += size
    return result


def _instructrec_training_loss(
    model: Any,
    tokenizer: Any,
    groups: Sequence[TrainingGroup],
    config: dict[str, Any],
    *,
    device: str,
) -> Any:
    training = config["training"]
    prompt_tokens: list[list[int]] = []
    targets: list[list[int]] = []
    max_target = int(training.get("max_target_length", 96))
    for group in groups:
        prompt, response_by_item, _audit = encode_instructrec_selection_prompt(
            tokenizer,
            group.record,
            group.record.candidates,
            history_budget=int(training["history_budget"]),
            template_index=instructrec_template_index(
                group.record.request_id, seed=int(training["seed"])
            ),
            max_length=int(training["max_length"]) - max_target,
            context_token_budget=int(training["context_token_budget"]),
            max_target_length=max_target,
        )
        for candidate, gain in zip(group.candidates, group.gains):
            if gain <= 0:
                continue
            prompt_tokens.append(prompt)
            targets.append(response_by_item[str(candidate["item_id"])])
    return _mean_target_sequence_nll(
        model, prompt_tokens, targets, tokenizer.pad_token_id, device
    )


def _tallrec_training_loss(
    model: Any,
    tokenizer: Any,
    groups: Sequence[TrainingGroup],
    config: dict[str, Any],
    *,
    device: str,
) -> Any:
    training = config["training"]
    prompts: list[list[int]] = []
    targets: list[list[int]] = []
    yes_target = _answer_target_tokens(tokenizer, "Yes")
    no_target = _answer_target_tokens(tokenizer, "No")
    target_reserve = max(len(yes_target), len(no_target))
    for group in groups:
        for candidate, gain in zip(group.candidates, group.gains):
            sections = build_prompt_sections(
                config["method_id"],
                group.record,
                candidate,
                history_budget=int(training["history_budget"]),
            )
            prompts.append(
                encode_prompt_sections(
                    tokenizer,
                    sections,
                    max_length=int(training["max_length"]) - target_reserve,
                )
            )
            targets.append(yes_target if gain > 0 else no_target)
    return _mean_target_sequence_nll(
        model, prompts, targets, tokenizer.pad_token_id, device
    )


def _mean_target_sequence_nll(
    model: Any,
    prompts: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
    pad_token_id: int,
    device: str,
) -> Any:
    import torch
    from torch.nn import functional as F

    if len(prompts) != len(targets) or not prompts:
        raise ValueError("prompt/target sequence batch is empty or misaligned")
    sequences = [list(prompt) + list(target) for prompt, target in zip(prompts, targets)]
    input_ids, attention_mask = _left_pad(sequences, pad_token_id, device)
    max_target = max(len(target) for target in targets)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=max_target + 1,
    )
    losses = []
    kept = output.logits
    for row, target in enumerate(targets):
        length = len(target)
        prediction_logits = kept[row, -(length + 1) : -1, :]
        target_tensor = torch.tensor(target, dtype=torch.long, device=device)
        losses.append(F.cross_entropy(prediction_logits.float(), target_tensor))
    return torch.stack(losses).mean()


def _score_yes_no_request(
    model: Any,
    tokenizer: Any,
    record: Any,
    history: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    device: str,
    batch_size: int,
) -> tuple[dict[str, float], bool]:
    method_id = str(config["method_id"])
    training = config["training"]
    result: dict[str, float] = {}
    at_boundary = False
    yes_target = (
        _answer_target_tokens(tokenizer, "Yes")
        if method_id == "q3_tallrec_generalqwen"
        else []
    )
    no_target = (
        _answer_target_tokens(tokenizer, "No")
        if method_id == "q3_tallrec_generalqwen"
        else []
    )
    target_reserve = max(len(yes_target), len(no_target)) if yes_target else 0
    chunks = complete_candidate_chunks(record.candidates, batch_size)
    for chunk in chunks:
        prompts = []
        for candidate in chunk:
            sections = build_prompt_sections(
                method_id,
                record,
                candidate,
                history=history,
                history_budget=int(training["history_budget"]),
            )
            prompt = encode_prompt_sections(
                tokenizer,
                sections,
                max_length=int(training["max_length"]) - target_reserve,
            )
            prompts.append(prompt)
            at_boundary = at_boundary or len(prompt) == (
                int(training["max_length"]) - target_reserve
            )
        if method_id == "q3_tallrec_generalqwen":
            yes_scores = _target_sequence_log_likelihoods(
                model,
                prompts,
                [yes_target] * len(prompts),
                tokenizer.pad_token_id,
                device,
            )
            no_scores = _target_sequence_log_likelihoods(
                model,
                prompts,
                [no_target] * len(prompts),
                tokenizer.pad_token_id,
                device,
            )
            values = [left - right for left, right in zip(yes_scores, no_scores)]
        else:
            input_ids, attention_mask = _left_pad(
                prompts, tokenizer.pad_token_id, device
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
            logits = output.logits[:, -1, :]
            yes_id = _single_token_id(tokenizer, "yes")
            no_id = _single_token_id(tokenizer, "no")
            values = (logits[:, yes_id] - logits[:, no_id]).float().cpu().tolist()
        for candidate, value in zip(chunk, values):
            result[str(candidate["item_id"])] = float(value)
    return result, at_boundary


def _score_instructrec_request(
    model: Any,
    tokenizer: Any,
    record: Any,
    history: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    device: str,
    batch_size: int,
) -> tuple[dict[str, float], bool]:
    training = config["training"]
    max_target = int(training.get("max_target_length", 96))
    prompt, response_by_item, _audit = encode_instructrec_selection_prompt(
        tokenizer,
        record,
        record.candidates,
        history=history,
        history_budget=int(training["history_budget"]),
        template_index=instructrec_template_index(
            record.request_id, seed=int(training["seed"])
        ),
        max_length=int(training["max_length"]) - max_target,
        context_token_budget=int(training["context_token_budget"]),
        max_target_length=max_target,
    )
    at_boundary = len(prompt) == int(training["max_length"]) - max_target
    result: dict[str, float] = {}
    prepared_prefix = _prepare_target_sequence_prefix_cache(
        model,
        prompt,
        device,
    )
    for chunk in complete_candidate_chunks(record.candidates, batch_size):
        targets = [response_by_item[str(row["item_id"])] for row in chunk]
        values = _cached_prefix_target_sequence_log_likelihoods(
            model,
            prompt,
            targets,
            tokenizer.pad_token_id,
            device,
            prepared_prefix=prepared_prefix,
        )
        for candidate, value in zip(chunk, values):
            result[str(candidate["item_id"])] = float(value)
    return result, at_boundary


def _target_sequence_log_likelihoods(
    model: Any,
    prompts: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
    pad_token_id: int,
    device: str,
) -> list[float]:
    import torch
    from torch.nn import functional as F

    if len(prompts) != len(targets) or not prompts:
        raise ValueError("prompt/target likelihood batch is empty or misaligned")
    if any(not target for target in targets):
        raise ValueError("target sequence must contain at least one token")
    sequences = [list(prompt) + list(target) for prompt, target in zip(prompts, targets)]
    input_ids, attention_mask = _left_pad(sequences, pad_token_id, device)
    max_target = max(len(target) for target in targets)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=max_target + 1,
    )
    kept = output.logits
    values = []
    for row, target in enumerate(targets):
        length = len(target)
        logits = kept[row, -(length + 1) : -1, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        target_tensor = torch.tensor(target, dtype=torch.long, device=device)
        token_values = log_probs.gather(1, target_tensor[:, None]).squeeze(1)
        values.append(float(token_values.mean().cpu()))
    return values


def _cached_prefix_target_sequence_log_likelihoods(
    model: Any,
    prompt: Sequence[int],
    targets: Sequence[Sequence[int]],
    pad_token_id: int,
    device: str,
    *,
    prepared_prefix: tuple[Any, Any, int] | None = None,
) -> list[float]:
    """Score Q1 alternatives from a reusable, exact shared-prompt KV cache."""

    import torch
    from torch.nn import functional as F

    if not prompt or not targets or any(not target for target in targets):
        raise ValueError("cached target likelihood received an empty prompt/target")
    if prepared_prefix is None:
        prepared_prefix = _prepare_target_sequence_prefix_cache(
            model,
            prompt,
            device,
        )
    first_log_probs, prefix_cache, prompt_length = prepared_prefix
    if prompt_length != len(prompt):
        raise ValueError("prepared Q1 prefix length differs from the scoring prompt")
    cache = copy.deepcopy(prefix_cache)
    if not hasattr(cache, "batch_repeat_interleave"):
        raise TypeError("Q1 prefix cache lacks batch_repeat_interleave")
    cache.batch_repeat_interleave(len(targets))
    continuation_lengths = [len(target) - 1 for target in targets]
    width = max(continuation_lengths)
    continuation_values: Any | None = None
    if width > 0:
        continuation_ids = torch.full(
            (len(targets), width),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        continuation_mask = torch.zeros_like(continuation_ids)
        for row, target in enumerate(targets):
            values = torch.tensor(target[:-1], dtype=torch.long, device=device)
            continuation_ids[row, : len(values)] = values
            continuation_mask[row, : len(values)] = 1
        attention_mask = torch.cat(
            [
                torch.ones(
                    (len(targets), prompt_length),
                    dtype=torch.long,
                    device=device,
                ),
                continuation_mask,
            ],
            dim=1,
        )
        continuation_values = model(
            input_ids=continuation_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=False,
        ).logits
    result = []
    for row, target in enumerate(targets):
        token_values = [first_log_probs[int(target[0])]]
        length = continuation_lengths[row]
        if length:
            log_probs = F.log_softmax(
                continuation_values[row, :length].float(), dim=-1
            )
            expected = torch.tensor(target[1:], dtype=torch.long, device=device)
            token_values.extend(log_probs.gather(1, expected[:, None]).squeeze(1))
        result.append(float(torch.stack(token_values).mean().cpu()))
    return result


def _prepare_target_sequence_prefix_cache(
    model: Any,
    prompt: Sequence[int],
    device: str,
) -> tuple[Any, Any, int]:
    """Run one Q1 prompt forward and retain immutable state for all chunks."""

    import torch
    from torch.nn import functional as F

    if not prompt:
        raise ValueError("cannot prepare an empty Q1 scoring prompt")
    prompt_ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    prompt_attention = torch.ones_like(prompt_ids)
    prefix = model(
        input_ids=prompt_ids,
        attention_mask=prompt_attention,
        use_cache=True,
        logits_to_keep=1,
    )
    first_log_probs = F.log_softmax(prefix.logits[0, -1].float(), dim=-1)
    prefix_cache = prefix.past_key_values
    if not hasattr(prefix_cache, "batch_repeat_interleave"):
        raise TypeError("Q1 prefix cache lacks batch_repeat_interleave")
    return first_log_probs, prefix_cache, len(prompt)


def _load_model_and_tokenizer(
    config: dict[str, Any],
    *,
    device: str,
    training: bool,
    checkpoint_model_dir: Path | None,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    method_id = str(config["method_id"])
    base_path = str(config["model"]["base_model_path"])
    tokenizer_path = str(checkpoint_model_dir or base_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if method_id == "q3_tallrec_generalqwen":
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        if checkpoint_model_dir is None:
            from peft import LoraConfig, TaskType, get_peft_model

            method = config["method"]
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(method["lora_rank"]),
                lora_alpha=int(method["lora_alpha"]),
                lora_dropout=float(method["lora_dropout"]),
                target_modules=list(method["lora_targets"]),
                bias="none",
            )
            model = get_peft_model(base, peft_config)
        else:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                base, str(checkpoint_model_dir), is_trainable=training
            )
        if training and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        source = str(checkpoint_model_dir or base_path)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.float32 if training else torch.bfloat16,
        )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if training and bool(config["training"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.to(device)
    model.train(training)
    return tokenizer, model


def _save_training_checkpoint(
    output_root: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    progress: dict[str, Any],
    config: dict[str, Any],
    training_contract: dict[str, Any],
    torch: Any,
) -> None:
    temporary = output_root / ".checkpoint_writing"
    latest = output_root / CHECKPOINT_DIRNAME
    backup = output_root / ".checkpoint_previous"
    for path in (temporary, backup):
        if path.exists():
            shutil.rmtree(path)
    temporary.mkdir(parents=True, exist_ok=False)
    model_dir = temporary / "model"
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    checkpoint_id, _artifacts = _checkpoint_identity(
        model_dir, str(config["method_id"])
    )
    state = {
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "optimizer": optimizer.state_dict(),
        "progress": dict(progress),
        "rng": _capture_rng_state(torch),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_contract": dict(training_contract),
    }
    torch.save(state, temporary / "trainer_state.pt")
    write_json(temporary / "progress.json", progress)
    if latest.exists():
        latest.rename(backup)
    temporary.rename(latest)
    if backup.exists():
        shutil.rmtree(backup)


def _recover_training_checkpoint(output_root: Path) -> None:
    latest = output_root / CHECKPOINT_DIRNAME
    backup = output_root / ".checkpoint_previous"
    temporary = output_root / ".checkpoint_writing"
    if latest.exists():
        if temporary.exists():
            shutil.rmtree(temporary)
        return
    if backup.is_dir():
        backup.rename(latest)
        if temporary.exists():
            shutil.rmtree(temporary)
        return
    if temporary.exists():
        shutil.rmtree(temporary)


def _capture_rng_state(torch: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _left_pad(
    token_lists: Sequence[Sequence[int]], pad_token_id: int, device: str
) -> tuple[Any, Any]:
    import torch

    if not token_lists:
        raise ValueError("cannot pad an empty token batch")
    width = max(len(row) for row in token_lists)
    input_ids = torch.full(
        (len(token_lists), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros(
        (len(token_lists), width), dtype=torch.long, device=device
    )
    for index, row in enumerate(token_lists):
        values = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[index, -len(row) :] = values
        attention[index, -len(row) :] = 1
    return input_ids, attention


def _load_history_assignments(
    path: Path, *, expected_condition: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate assignment request_id={request_id}")
        if str(row.get("assignment")) != expected_condition:
            raise ValueError(f"assignment condition mismatch for {request_id}")
        # Apply the same whitelist as unified records; assignment files are not
        # trusted to carry arbitrary donor fields into a model prompt.
        projected = sanitize_record_for_model(
            {
                "request_id": request_id,
                "query": "assignment-validation-placeholder",
                "history": list(row.get("history", [])),
                "candidates": [
                    {"item_id": "__placeholder_a"},
                    {"item_id": "__placeholder_b"},
                ],
            }
        )
        result[request_id] = {
            "donor_request_id": row.get("donor_request_id"),
            "donor_user_id": row.get("donor_user_id"),
            "history": list(projected.history),
            "match_type": row.get("match_type"),
        }
    if not result:
        raise ValueError(f"empty history assignment file: {path}")
    return result


def _load_history_assignment_manifest(
    assignments_path: Path,
    *,
    expected_condition: str,
    records_path: Path,
) -> dict[str, str]:
    manifest_path = assignments_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"history assignment manifest is required: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("qrels_read") is not False or manifest.get(
        "model_scores_read"
    ) is not False:
        raise ValueError("history assignment manifest crossed the label/model boundary")
    file_entry = manifest.get("files", {}).get(expected_condition, {})
    if file_entry.get("sha256") != sha256_file(assignments_path):
        raise ValueError("history assignment file differs from its manifest")
    if manifest.get("source_records_sha256") != sha256_file(records_path):
        raise ValueError("history assignment source records hash mismatch")
    if int(manifest.get("target_candidate_leakage_violations", -1)) != 0:
        raise ValueError("history assignment candidate leakage audit failed")
    if int(manifest.get("history_not_strictly_before_target_violations", -1)) != 0:
        raise ValueError("history assignment causality audit failed")
    return {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}


def _validate_assigned_history(
    raw_record: dict[str, Any],
    record: Any,
    assignment: dict[str, Any],
    *,
    history_condition: str,
) -> list[dict[str, Any]]:
    history = list(assignment["history"])
    if history_condition == "true":
        if history != list(record.history):
            raise ValueError(
                f"true assignment differs from record history: {record.request_id}"
            )
        return history
    if history_condition == "null":
        if history:
            raise ValueError(f"null assignment is non-empty: {record.request_id}")
        return history
    candidate_ids = {str(row["item_id"]) for row in record.candidates}
    if any(str(event["item_id"]) in candidate_ids for event in history):
        raise ValueError(f"wrong history contains a target candidate: {record.request_id}")
    request_ts = int(raw_record["ts"])
    if any(int(event["ts"]) >= request_ts for event in history):
        raise ValueError(f"wrong history is not causal: {record.request_id}")
    if history:
        donor_user_id = str(assignment.get("donor_user_id") or "")
        if not donor_user_id or donor_user_id == str(raw_record["user_id"]):
            raise ValueError(f"wrong history is not cross-user: {record.request_id}")
        if str(assignment.get("donor_request_id") or "") == record.request_id:
            raise ValueError(f"wrong history reused the target request: {record.request_id}")
    return history


def _base_inference_artifact_identity(base_path: str | Path) -> dict[str, Any]:
    """Hash every local file type that Transformers/PEFT may load at inference.

    The individual weight/tokenizer hashes remain human-readable protocol
    fields, while this canonical digest also freezes model/tokenizer configs,
    vocabularies, merges, chat templates, and any shard/index JSON files.
    """

    base_path = Path(base_path)
    if not base_path.is_dir():
        raise FileNotFoundError(f"base-model directory is missing: {base_path}")
    paths = sorted(
        path
        for path in base_path.rglob("*")
        if path.is_file()
        and path.suffix in _BASE_INFERENCE_SUFFIXES
        and not any(
            part.startswith(".") for part in path.relative_to(base_path).parts
        )
    )
    if not paths or not any(path.suffix == ".safetensors" for path in paths):
        raise FileNotFoundError(
            f"base-model inference artifacts are incomplete: {base_path}"
        )
    files = [
        {
            "name": path.relative_to(base_path).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return {
        "files": files,
        "sha256": sha256_text(
            json.dumps(files, sort_keys=True, separators=(",", ":"))
        ),
    }


def _checkpoint_identity(
    model_dir: Path, method_id: str
) -> tuple[str, list[dict[str, Any]]]:
    allowed_suffixes = {".jinja", ".json", ".model", ".safetensors", ".txt"}
    files = sorted(
        path
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix in allowed_suffixes
    )
    if not any(path.suffix == ".safetensors" for path in files):
        raise FileNotFoundError(f"checkpoint contains no safetensors weights: {model_dir}")
    values = [
        {
            "name": str(path.relative_to(model_dir)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    digest = sha256_text(json.dumps(values, sort_keys=True, separators=(",", ":")))
    return f"{method_id}@{digest[:20]}", values


def _single_token_id(tokenizer: Any, text: str) -> int:
    values = tokenizer.encode(text, add_special_tokens=False)
    if len(values) != 1:
        raise ValueError(f"expected one token for {text!r}, observed {values}")
    return int(values[0])


def _answer_target_tokens(tokenizer: Any, answer: str) -> list[int]:
    values = tokenizer.encode(answer, add_special_tokens=False)
    terminator = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if not values or not terminator:
        raise ValueError(f"empty answer or terminator tokenization for {answer!r}")
    return [*values, *terminator]


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_run_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must use YYYYMMDD_<dataset_id>_<method_id>_<short_purpose> "
            f"characters: {run_id}"
        )


def _objective_name(method_id: str) -> str:
    return {
        "q0_qwen3_reranker_06b": "pointwise_binary_cross_entropy_on_yes_no_logits",
        "q1_instructrec_generalqwen": "output_only_normalized_candidate_response_nll",
        "q2_recranker_generalqwen": "0.5_ranknet_plus_0.5_tie_aware_listnet",
        "q3_tallrec_generalqwen": "output_only_yes_no_recommendation_alignment_nll_with_lora",
    }[method_id]


def _build_training_contract(
    *,
    batches_per_epoch: int,
    config_sha256: str,
    evidence_mode: str,
    group_count: int,
    implementation_digest: str,
    manifest_sha256: str,
    max_train_groups: int | None,
    qrels_train_sha256: str,
    records_train_sha256: str,
    total_optimizer_steps: int,
    updates_per_epoch: int,
) -> dict[str, Any]:
    """Bind every resumable optimizer state to its data, recipe, and code."""

    if not implementation_digest:
        raise ValueError("training contract requires an implementation digest")
    return {
        "batches_per_epoch": batches_per_epoch,
        "config_sha256": config_sha256,
        "evidence_mode": evidence_mode,
        "group_count": group_count,
        "implementation_digest": implementation_digest,
        "manifest_sha256": manifest_sha256,
        "max_train_groups": max_train_groups,
        "qrels_train_sha256": qrels_train_sha256,
        "records_train_sha256": records_train_sha256,
        "total_optimizer_steps": total_optimizer_steps,
        "updates_per_epoch": updates_per_epoch,
    }


def _assert_frozen_training_population(
    standardized_dir: Path, config: dict[str, Any]
) -> None:
    population = config["_protocol"]["data"]["development_population"]
    expected = {
        "manifest.json": population["manifest_sha256"],
        "candidate_manifest.json": population["candidate_manifest_sha256"],
        "request_manifest.json": population["request_manifest_sha256"],
        "records_train.jsonl": population["records_train_sha256"],
        "qrels_train.jsonl": population["qrels_train_sha256"],
    }
    _assert_file_hashes(standardized_dir, expected, role="frozen training population")


def _validate_scoring_checkpoint_provenance(
    metadata: dict[str, Any],
    config: dict[str, Any],
    *,
    allow_smoke: bool,
) -> None:
    if metadata.get("status") != "completed":
        raise ValueError("cannot score an under-converged or incomplete checkpoint")
    evidence_mode = metadata.get("evidence_mode")
    if evidence_mode == "smoke_non_result" and not allow_smoke:
        raise ValueError("a capped smoke checkpoint cannot produce an uncapped result")
    if evidence_mode not in {"first_round_pilot", "smoke_non_result"}:
        raise ValueError("checkpoint has an unknown evidence mode")
    expected = {
        "base_artifact_manifest_sha256": config["model"][
            "base_artifact_manifest_sha256"
        ],
        "base_model_path": str(config["model"]["base_model_path"]),
        "base_tokenizer_sha256": config["model"]["tokenizer_sha256"],
        "base_weights_sha256": config["model"]["base_weights_sha256"],
        "config_sha256": config["_config_sha256"],
        "method_id": config["method_id"],
        "protocol_sha256": config["protocol"]["sha256"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint provenance mismatch: {key}")
    population = config["_protocol"]["data"]["development_population"]
    if metadata.get("records_train_sha256") != population["records_train_sha256"]:
        raise ValueError("checkpoint provenance mismatch: records_train_sha256")
    if metadata.get("qrels_train_sha256") != population["qrels_train_sha256"]:
        raise ValueError("checkpoint provenance mismatch: qrels_train_sha256")
    implementation = metadata.get("implementation_identity", {})
    if implementation.get("digest") != _implementation_identity()["digest"]:
        raise ValueError("training/scoring implementation identity mismatch")
    progress = metadata.get("progress", {})
    if int(progress.get("epoch", -1)) != int(config["training"]["epochs"]):
        raise ValueError("completed checkpoint did not reach the frozen final epoch")


def _assert_scoring_population(
    standardized_dir: Path,
    config: dict[str, Any],
    *,
    split: str,
    dataset_manifest: dict[str, Any],
    checkpoint_id: str,
    checkpoint_weight_files: Sequence[dict[str, Any]],
    training_metadata_path: Path,
) -> dict[str, Any] | None:
    population = config["_protocol"]["data"]["development_population"]
    dataset_version = str(dataset_manifest.get("dataset_version"))
    if dataset_version != str(population["dataset_version"]):
        if dataset_version != V12_DATASET_VERSION:
            raise ValueError(
                f"unregistered V1.2 scoring dataset_version={dataset_version!r}"
            )
        if split != "confirmation":
            raise ValueError("the registered V1.2 holdout may only score confirmation")
        audit = verify_published_holdout(
            standardized_dir,
            protocol_path=config["protocol"]["path"],
            open_qrels=False,
        )
        frozen = audit["checkpoint_identities"].get(str(config["method_id"]))
        if not isinstance(frozen, dict):
            raise ValueError("holdout release lock lacks the scoring method identity")
        expected = {
            "checkpoint_id": checkpoint_id,
            "config_sha256": config["_config_sha256"],
            "implementation_digest": _implementation_identity()["digest"],
            "protocol_sha256": config["protocol"]["sha256"],
            "training_metadata_sha256": sha256_file(training_metadata_path),
        }
        for key, observed in expected.items():
            if frozen.get(key) != observed:
                raise ValueError(f"holdout frozen checkpoint mismatch: {key}")
        observed_files = _canonical_checkpoint_file_entries(
            checkpoint_weight_files
        )
        frozen_files = _canonical_checkpoint_file_entries(
            frozen.get("checkpoint_files", [])
        )
        if observed_files != frozen_files:
            raise ValueError("holdout frozen checkpoint artifact list mismatch")
        return {
            "checkpoint_identity_manifest_sha256": frozen[
                "identity_manifest_sha256"
            ],
            "checkpoint_id": checkpoint_id,
            "integrity_lock_sha256": audit["integrity_lock_sha256"],
            "manifest_sha256": audit["manifest_sha256"],
            "post_selection_recipe_checkpoint_lock_sha256": audit[
                "post_selection_recipe_checkpoint_lock_sha256"
            ],
            "protocol_sha256": audit["protocol_sha256"],
            "qrels_opened": False,
            "verified_before_model_load": True,
        }
    records_key = (
        "records_dev_sha256"
        if split == "dev"
        else "records_legacy_compatibility_sha256"
    )
    expected = {
        "manifest.json": population["manifest_sha256"],
        "candidate_manifest.json": population["candidate_manifest_sha256"],
        "request_manifest.json": population["request_manifest_sha256"],
        f"records_{split}.jsonl": population[records_key],
    }
    _assert_file_hashes(standardized_dir, expected, role="frozen scoring population")
    return None


def _canonical_checkpoint_file_entries(
    values: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for value in values:
        result.append(
            {
                "name": str(value["name"]),
                "sha256": str(value["sha256"]),
                "size_bytes": int(value["size_bytes"]),
            }
        )
    names = [value["name"] for value in result]
    if not result or len(names) != len(set(names)):
        raise ValueError("checkpoint artifact identity is empty or duplicated")
    return sorted(result, key=lambda value: value["name"])


def _assert_file_hashes(
    root: Path, expected: dict[str, str], *, role: str
) -> None:
    for name, digest in expected.items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(digest):
            raise ValueError(f"{role} hash mismatch: {name}")


def _parameter_counts(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _runtime_metadata(
    method_id: str,
    torch: Any,
    transformers: Any,
) -> dict[str, Any]:
    """Record the interpreter and method-relevant inference packages."""

    package_versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if method_id == "q3_tallrec_generalqwen":
        import peft

        package_versions["peft"] = peft.__version__
    return {
        "package_versions": package_versions,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }


def _enforce_score_nondegeneracy(
    run_dir: Path,
    request_score_ranges: Sequence[float],
    *,
    capped: bool,
) -> int:
    """Refuse to publish an uncapped run that is constant on every request."""

    nonconstant_requests = sum(value > 1.0e-8 for value in request_score_ranges)
    if not capped and nonconstant_requests == 0:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        raise ValueError(
            "uncapped score run is globally degenerate at the frozen 1e-8 threshold"
        )
    return nonconstant_requests


def _implementation_identity() -> dict[str, Any]:
    files = [
        Path(__file__),
        Path(__file__).with_name("motivation_v12_contracts.py"),
    ]
    values = [
        {"path": str(path), "sha256": sha256_file(path)} for path in sorted(files)
    ]
    return {
        "digest": sha256_text(
            json.dumps(values, sort_keys=True, separators=(",", ":"))
        ),
        "files": values,
    }


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
