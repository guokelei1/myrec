"""Exact train-only step-501 replay for Q3 LoRA A/B/joint coordinates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import load_training_groups
from myrec.baselines.motivation_v12_ranker import (
    _assert_frozen_training_population,
    _git_revision,
    _load_model_and_tokenizer,
    _runtime_metadata,
    _training_batch_loss,
    _validate_run_id,
    load_v12_ranker_config,
)
from myrec.mechanism.attention_edge_runtime import _load_manifest
from myrec.mechanism.gradient_diagnostic import (
    CONTROLS,
    REQUESTS_PER_SURFACE,
    SELECTION_SEED,
    SURFACES,
    _load_train_gains,
    deterministic_label_shuffle,
    select_surface_training_groups,
)
from myrec.mechanism.optimizer_replay_binding import (
    TRAINING_CUDA_RNG_INDEX,
    load_bound_step500_state,
    restore_bound_rng_state,
)
from myrec.mechanism.optimizer_replay_math import (
    adamw_exact_delta,
    clip_gradients,
    lora_function_delta,
    lora_parameter_identity,
    lora_singular_values,
    parameter_order_digest,
    parameter_order_rows,
    vector_cosine,
    vector_relative_error,
    vector_summary,
)
from myrec.utils.hashing import sha256_file, sha256_text


Q3_METHOD_ID = "q3_tallrec_generalqwen"
EXPECTED_PARAMETER_ORDER_DIGEST = (
    "cae9b185ec9486a366247a8222598fc645aefe76aede1374123a72f6d583368f"
)
BLOCK_REQUESTS = 16
BLOCKS_PER_SURFACE = 6
MAX_WALL_SECONDS = 13_500.0
MODES = ("a_only", "b_only", "joint")


def run_q3_step501_replay(
    standardized_dir: str | Path,
    config_path: str | Path,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = "experiments/motivation/transformer_deep_dive_manifest.yaml",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_tasks: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Replay each frozen 16-request block from the identical step-500 state."""

    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("an explicit Q3 optimizer replay device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("Q3 optimizer replay wall time must be in (0,13500]")
    if max_tasks is not None and not 0 < int(max_tasks) < 36:
        raise ValueError("Q3 replay smoke max_tasks must be in [1,35]")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    config = load_v12_ranker_config(config_path)
    if config["method_id"] != Q3_METHOD_ID:
        raise ValueError("step-501 LoRA replay is Q3-only")
    frozen = manifest["frozen_inputs"]["models"][Q3_METHOD_ID]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("Q3 replay config differs from frozen manifest")
    _assert_frozen_training_population(standardized_dir, config)
    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
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
        requests_per_surface=REQUESTS_PER_SURFACE,
        selection_seed=SELECTION_SEED,
    )
    tasks = []
    for control in CONTROLS:
        for surface in SURFACES:
            for block in range(BLOCKS_PER_SURFACE):
                start = block * BLOCK_REQUESTS
                tasks.append((control, surface, block, selected[surface][start : start + BLOCK_REQUESTS]))
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]
    selection.update(
        {
            "method_id": Q3_METHOD_ID,
            "group_construction": group_stats,
            "blocks_per_surface": BLOCKS_PER_SURFACE,
            "requests_per_block": BLOCK_REQUESTS,
            "task_order": [
                {
                    "control": control,
                    "surface": surface,
                    "block": block,
                    "request_ids": [group.record.request_id for group in block_groups],
                }
                for control, surface, block, block_groups in tasks
            ],
            "finalized_before_model_load_and_gradient": True,
        }
    )
    selection_sha256 = _canonical_sha256(selection)
    step_state, step_audit = load_bound_step500_state(
        Q3_METHOD_ID, manifest_path=manifest_path
    )
    if step_audit["parameter_order_digest"] != EXPECTED_PARAMETER_ORDER_DIGEST:
        raise ValueError("Q3 replay binding parameter digest differs")
    implementation = q3_optimizer_replay_implementation_identity()
    evidence_mode = (
        "registered_optimizer_replay"
        if max_tasks is None
        else "mechanical_smoke_non_result"
    )
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": Q3_METHOD_ID,
        "checkpoint_id": step_audit["checkpoint_id"],
        "config_sha256": config["_config_sha256"],
        "selection_sha256": selection_sha256,
        "tasks": len(tasks),
        "parameter_order_digest": EXPECTED_PARAMETER_ORDER_DIGEST,
        "trainer_state_sha256": step_audit["observed_hashes"]["trainer_state_sha256"],
        "training_cuda_rng_index": TRAINING_CUDA_RNG_INDEX[Q3_METHOD_ID],
        "rng_restore_policy": "training_logical_cuda_rng_to_replay_device",
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "device": str(device),
        "implementation_digest": implementation["digest"],
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d7_q3_step501_lora_replay",
        "run_id": run_id,
        "method_id": Q3_METHOD_ID,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "records_train_sha256": sha256_file(records_path),
        "qrels_train_sha256": sha256_file(qrels_path),
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "step500_binding": step_audit,
        "selection_sha256": selection_sha256,
        "controls": list(CONTROLS),
        "surfaces": list(SURFACES),
        "coordinate_modes": list(MODES),
        "native_batch_requests": 2,
        "native_accumulation_steps": 8,
        "requests_per_replay": BLOCK_REQUESTS,
        "optimizer_update": "real AdamW.step at step-501 followed by exact theta restore; algebra separately audited",
        "inactive_factor_contract": "grad=None; no moment update; no weight decay",
        "training_cuda_rng_index": TRAINING_CUDA_RNG_INDEX[Q3_METHOD_ID],
        "rng_restore_policy": "training_logical_cuda_rng_to_replay_device",
        "implementation_identity": implementation,
        "evidence_mode": evidence_mode,
        "result_eligible": max_tasks is None,
        "run_contract": contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "selection_finalized",
    }
    metadata, progress = _prepare(
        run_dir, metadata, selection, contract_sha256, tasks, resume=resume
    )
    completed = int(progress["completed_tasks"])
    if completed >= len(tasks):
        del step_state
        return _finalize(run_dir, metadata, progress, tasks)

    started = time.monotonic()
    try:
        import torch
        import transformers
        from transformers.optimization import get_linear_schedule_with_warmup

        tokenizer, model = _load_model_and_tokenizer(
            config,
            device=str(device),
            training=True,
            checkpoint_model_dir=Path(step_audit["model_dir"]),
        )
        model.train(True)
        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if len(named) != 112 or parameter_order_digest(named) != EXPECTED_PARAMETER_ORDER_DIGEST:
            raise ValueError("Q3 replay model parameter order differs from step-500 optimizer")
        names = [name for name, _ in named]
        parameters = [parameter for _, parameter in named]
        original_parameters = {
            name: parameter.detach().clone() for name, parameter in named
        }
        identities = {name: lora_parameter_identity(name) for name in names}
        family_by_name = {
            name: (
                f"block_{identity['block_zero_based']:02d}."
                f"{identity['projection']}.{identity['factor']}"
            )
            for name, identity in identities.items()
        }
        pairs = _lora_pairs(identities)
        optimizer_state = step_state["optimizer"]
        group = optimizer_state["param_groups"][0]
        state_by_name = {
            name: optimizer_state["state"][index] for index, name in enumerate(names)
        }
        next_lr = _next_linear_lr(step_state)
        rng_restore_binding = restore_bound_rng_state(
            torch,
            step_state["rng"],
            method_id=Q3_METHOD_ID,
            device=str(device),
        )
        metadata.update(
            {
                **_runtime_metadata(Q3_METHOD_ID, torch, transformers),
                "parameter_order_digest": parameter_order_digest(named),
                "parameter_order_rows": parameter_order_rows(named),
                "next_scheduler_lr_after_step501": next_lr,
                "rng_restore_binding": rng_restore_binding,
                "status": "running",
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        dtype = str(config["training"].get("dtype", "bfloat16"))
        autocast_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        for task_index in range(completed, len(tasks)):
            if time.monotonic() - started >= max_wall_seconds:
                metadata.update(
                    {
                        "status": "wall_time_exhausted",
                        "resumable": True,
                        "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
                        + time.monotonic()
                        - started,
                    }
                )
                _write_json(run_dir / "metadata.json", metadata)
                return metadata
            control, surface, block, original_groups = tasks[task_index]
            block_groups = list(original_groups)
            shuffle_audits = None
            if control == "within_request_label_shuffle":
                converted = [deterministic_label_shuffle(group) for group in block_groups]
                block_groups = [value[0] for value in converted]
                shuffle_audits = [value[1] for value in converted]
            observed_rng_restore = restore_bound_rng_state(
                torch,
                step_state["rng"],
                method_id=Q3_METHOD_ID,
                device=str(device),
            )
            if observed_rng_restore != rng_restore_binding:
                raise ValueError("Q3 replay RNG restore binding drifted")
            model.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for microbatch in range(8):
                micro_groups = block_groups[microbatch * 2 : (microbatch + 1) * 2]
                with torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                    enabled=str(device).startswith("cuda") and dtype != "float32",
                ):
                    loss, _parts = _training_batch_loss(
                        model, tokenizer, micro_groups, config, device=str(device)
                    )
                (loss / 8.0).backward()
                loss_sum += float(loss.detach().float().cpu())
            gradients = {
                name: parameter.grad.detach().clone() if parameter.grad is not None else None
                for name, parameter in named
            }
            if any(value is None for value in gradients.values()):
                raise ValueError("Q3 joint replay produced an unused LoRA parameter")
            mode_results = {}
            mode_deltas: dict[str, dict[str, Any]] = {}
            for mode in MODES:
                active = {
                    name: (
                        gradient
                        if mode == "joint" or identities[name]["factor"].lower() == mode[0]
                        else None
                    )
                    for name, gradient in gradients.items()
                }
                expected_clipped, clip_coefficient, raw_norm = clip_gradients(
                    active, float(config["training"].get("max_grad_norm", 1.0))
                )
                optimizer = torch.optim.AdamW(
                    parameters, lr=float(config["training"]["learning_rate"])
                )
                total_steps = int(step_state["training_contract"]["total_optimizer_steps"])
                warmup_steps = int(
                    total_steps * float(config["training"].get("warmup_ratio", 0.1))
                )
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )
                optimizer.load_state_dict(optimizer_state)
                scheduler.load_state_dict(step_state["scheduler"])
                for name, parameter in named:
                    parameter.grad = (
                        None
                        if active[name] is None
                        else active[name].detach().clone()
                    )
                observed_raw_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        parameters,
                        float(config["training"].get("max_grad_norm", 1.0)),
                    ).item()
                )
                clipped = {
                    name: (
                        None
                        if parameter.grad is None
                        else parameter.grad.detach().clone()
                    )
                    for name, parameter in named
                }
                active_names = [name for name in names if active[name] is not None]
                clip_identity = vector_relative_error(
                    {name: clipped[name] for name in active_names},
                    {name: expected_clipped[name] for name in active_names},
                )
                if clip_identity["relative_l2_error"] > 8.0 * (2.0**-7):
                    raise ValueError(f"Q3 actual/manual clip mismatch: {clip_identity}")
                replay = {
                    name: adamw_exact_delta(
                        parameter,
                        clipped[name],
                        state_by_name[name],
                        group,
                    )
                    for name, parameter in named
                }
                directions = {
                    name: value["preconditioned_direction"] for name, value in replay.items()
                }
                moment = {name: value["moment_delta"] for name, value in replay.items()}
                decay = {name: value["weight_decay_delta"] for name, value in replay.items()}
                algebra_total = {
                    name: value["total_delta"] for name, value in replay.items()
                }
                optimizer.step()
                scheduler.step()
                observed_next_lr = float(scheduler.get_last_lr()[0])
                if not math.isclose(observed_next_lr, next_lr, rel_tol=1.0e-12, abs_tol=0.0):
                    raise ValueError(
                        f"Q3 actual scheduler step differs: {observed_next_lr} != {next_lr}"
                    )
                total = {}
                maximum_inactive_delta = 0.0
                for name, parameter in named:
                    delta = parameter.detach() - original_parameters[name]
                    if active[name] is None:
                        maximum_inactive_delta = max(
                            maximum_inactive_delta,
                            float(delta.float().abs().max().item()),
                        )
                        total[name] = None
                    else:
                        total[name] = delta.detach().clone()
                if maximum_inactive_delta != 0.0:
                    raise ValueError(
                        f"Q3 inactive LoRA factor changed: {maximum_inactive_delta}"
                    )
                actual_algebra_identity = vector_relative_error(
                    {name: total[name] for name in active_names},
                    {name: algebra_total[name] for name in active_names},
                )
                with torch.no_grad():
                    for name, parameter in named:
                        parameter.copy_(original_parameters[name])
                        parameter.grad = None
                del optimizer, scheduler
                mode_deltas[mode] = total
                mode_results[mode] = {
                    "clip_coefficient": clip_coefficient,
                    "raw_global_norm": raw_norm,
                    "torch_clip_grad_norm_return": observed_raw_norm,
                    "manual_vs_torch_clip_identity": clip_identity,
                    "actual_step_vs_algebra_identity": actual_algebra_identity,
                    "maximum_inactive_parameter_delta": maximum_inactive_delta,
                    "actual_scheduler_lr_after_step": observed_next_lr,
                    "raw_gradient": vector_summary(active, family_by_name=family_by_name),
                    "clipped_gradient": vector_summary(clipped, family_by_name=family_by_name),
                    "adam_preconditioned_direction": vector_summary(
                        directions, family_by_name=family_by_name
                    ),
                    "moment_delta": vector_summary(moment, family_by_name=family_by_name),
                    "weight_decay_delta": vector_summary(decay, family_by_name=family_by_name),
                    "total_delta": vector_summary(total, family_by_name=family_by_name),
                    "algebra_total_delta": vector_summary(
                        algebra_total, family_by_name=family_by_name
                    ),
                    "raw_gradient_vs_total_delta_cosine": vector_cosine(active, total),
                    "preconditioned_vs_total_delta_cosine": vector_cosine(directions, total),
                }
            paths = _path_function_summaries(named, pairs, mode_deltas)
            row = {
                "task_index": task_index,
                "control": control,
                "surface": surface,
                "block_within_surface": block,
                "request_ids": [group.record.request_id for group in block_groups],
                "request_ids_sha256": _canonical_sha256(
                    [group.record.request_id for group in block_groups]
                ),
                "label_shuffle": shuffle_audits,
                "mean_microbatch_loss": loss_sum / 8.0,
                "modes": mode_results,
                "lora_paths": paths,
                "inactive_factor_none_verified": True,
                "scheduler_lr_used": float(group["lr"]),
                "scheduler_lr_after_step": next_lr,
            }
            _append_sync(run_dir / "replays.partial.jsonl", row)
            progress.update(
                {
                    "completed_tasks": task_index + 1,
                    "last_task": [control, surface, block],
                    "partial_sha256": sha256_file(run_dir / "replays.partial.jsonl"),
                    "status": "running",
                    "updated_at": _utc_now(),
                }
            )
            _write_json(run_dir / "progress.json", progress)
            model.zero_grad(set_to_none=True)
            del gradients, mode_deltas, mode_results, paths
    except Exception as exc:
        metadata.update(
            {
                "status": "mechanical_failure",
                "resumable": True,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        raise
    metadata["elapsed_seconds"] = float(metadata.get("elapsed_seconds", 0.0)) + (
        time.monotonic() - started
    )
    return _finalize(run_dir, metadata, progress, tasks)


def _lora_pairs(identities: Mapping[str, Mapping[str, Any]]) -> dict[tuple[int, str], dict[str, str]]:
    pairs: dict[tuple[int, str], dict[str, str]] = {}
    for name, identity in identities.items():
        key = (int(identity["block_zero_based"]), str(identity["projection"]))
        pairs.setdefault(key, {})[str(identity["factor"])] = name
    expected = {(block, projection) for block in range(28) for projection in ("q", "v")}
    if set(pairs) != expected or any(set(value) != {"A", "B"} for value in pairs.values()):
        raise ValueError("Q3 replay LoRA A/B path coverage drifted")
    return pairs


def _path_function_summaries(
    named: Sequence[tuple[str, Any]],
    pairs: Mapping[tuple[int, str], Mapping[str, str]],
    mode_deltas: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    parameters = dict(named)
    rows = []
    for (block, projection), names in sorted(pairs.items()):
        a = parameters[names["A"]].detach()
        b = parameters[names["B"]].detach()
        joint = lora_function_delta(
            a,
            b,
            mode_deltas["joint"][names["A"]],
            mode_deltas["joint"][names["B"]],
        )
        a_replay = 2.0 * (b @ mode_deltas["a_only"][names["A"]])
        b_replay = 2.0 * (mode_deltas["b_only"][names["B"]] @ a)
        pre_effective_weight = 2.0 * (b @ a)
        post_effective_weight = 2.0 * (
            (b + mode_deltas["joint"][names["B"]])
            @ (a + mode_deltas["joint"][names["A"]])
        )
        rows.append(
            {
                "block_zero_based": block,
                "projection": projection,
                "a_only_replay_function_norm": float(a_replay.double().norm().item()),
                "b_only_replay_function_norm": float(b_replay.double().norm().item()),
                "joint_function_norm": float(joint["joint"].double().norm().item()),
                "joint_a_component_norm": float(joint["a_only"].double().norm().item()),
                "joint_b_component_norm": float(joint["b_only"].double().norm().item()),
                "joint_second_order_interaction_norm": float(
                    joint["interaction"].double().norm().item()
                ),
                "function_recomposition_max_abs_error": joint[
                    "recomposition_max_abs_error"
                ],
                "step500_effective_weight_norm": float(
                    pre_effective_weight.double().norm().item()
                ),
                "step500_spectrum": lora_singular_values(a, b),
                "post_step501_spectrum": lora_singular_values(
                    a + mode_deltas["joint"][names["A"]],
                    b + mode_deltas["joint"][names["B"]],
                ),
                "post_step501_effective_weight_norm": float(
                    post_effective_weight.double().norm().item()
                ),
                "step501_effective_delta_norm": float(
                    joint["joint"].double().norm().item()
                ),
            }
        )
    return rows


def _next_linear_lr(state: Mapping[str, Any]) -> float:
    contract = state["training_contract"]
    total = int(contract["total_optimizer_steps"])
    base_lr = float(state["scheduler"]["base_lrs"][0])
    warmup = int(total * 0.1)
    next_epoch = int(state["scheduler"]["last_epoch"]) + 1
    if next_epoch < warmup:
        return base_lr * next_epoch / max(1, warmup)
    return base_lr * max(0.0, (total - next_epoch) / max(1, total - warmup))


def q3_optimizer_replay_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/q3_optimizer_replay_runtime.py",
        "src/myrec/mechanism/optimizer_replay_binding.py",
        "src/myrec/mechanism/optimizer_replay_math.py",
        "src/myrec/mechanism/gradient_diagnostic.py",
        "src/myrec/baselines/motivation_v12_ranker.py",
        "scripts/run_deep_dive_q3_optimizer_replay.py",
    )
    files = [
        {"path": path, "sha256": sha256_file(root / path), "size_bytes": (root / path).stat().st_size}
        for path in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}


def _prepare(
    run_dir: Path,
    metadata: dict[str, Any],
    selection: dict[str, Any],
    contract_sha256: str,
    tasks: Sequence[Any],
    *,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    partial = run_dir / "replays.partial.jsonl"
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Q3 replay run is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial.touch(exist_ok=False)
        _write_json(run_dir / "selection_manifest.json", selection)
        metadata.update({"elapsed_seconds": 0.0, "resumable": True, "resume_lineage": []})
        progress = {
            "schema_version": 1,
            "run_contract_sha256": contract_sha256,
            "completed_tasks": 0,
            "last_task": None,
            "partial_sha256": sha256_file(partial),
            "status": "selection_finalized",
            "updated_at": _utc_now(),
        }
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "progress.json", progress)
        return metadata, progress
    stored = _read_json(run_dir / "metadata.json")
    progress = _read_json(run_dir / "progress.json")
    if stored.get("run_contract_sha256") != contract_sha256 or progress.get(
        "run_contract_sha256"
    ) != contract_sha256:
        raise ValueError("Q3 replay resume contract drift")
    if _read_json(run_dir / "selection_manifest.json") != selection:
        raise ValueError("Q3 replay selection drift")
    observed = _audit_partial(partial, tasks)
    if progress.get("completed_tasks") != observed["completed_tasks"] or progress.get(
        "partial_sha256"
    ) != observed["partial_sha256"]:
        raise ValueError("Q3 replay progress differs from partial")
    lineage = list(stored.get("resume_lineage", []))
    lineage.append({"completed_tasks": observed["completed_tasks"], "partial_sha256": observed["partial_sha256"]})
    stored.update({"resume_lineage": lineage, "status": "selection_finalized"})
    _write_json(run_dir / "metadata.json", stored)
    return stored, progress


def _audit_partial(path: Path, tasks: Sequence[Any]) -> dict[str, Any]:
    count = 0
    for row in _iter_jsonl(path):
        if count >= len(tasks):
            raise ValueError("Q3 replay partial has excess rows")
        control, surface, block, groups = tasks[count]
        if (
            row.get("task_index") != count
            or row.get("control") != control
            or row.get("surface") != surface
            or row.get("block_within_surface") != block
            or row.get("request_ids") != [group.record.request_id for group in groups]
            or len(row.get("lora_paths", [])) != 56
            or set(row.get("modes", {})) != set(MODES)
        ):
            raise ValueError("Q3 replay partial task identity/coverage drift")
        count += 1
    return {"completed_tasks": count, "partial_sha256": sha256_file(path)}


def _finalize(run_dir: Path, metadata: dict[str, Any], progress: dict[str, Any], tasks: Sequence[Any]) -> dict[str, Any]:
    observed = _audit_partial(run_dir / "replays.partial.jsonl", tasks)
    if observed["completed_tasks"] != len(tasks):
        raise ValueError("cannot finalize incomplete Q3 optimizer replay")
    path = run_dir / "replays.jsonl"
    os.replace(run_dir / "replays.partial.jsonl", path)
    progress.update({"status": "completed", "updated_at": _utc_now()})
    metadata.update(
        {
            "status": "completed",
            "resumable": False,
            "completed_replay_blocks": len(tasks),
            "replays_path": str(path),
            "replays_sha256": sha256_file(path),
            "optimizer_steps_performed_then_exactly_restored": len(tasks) * len(MODES),
            "exact_effective_updates_computed": len(tasks) * len(MODES),
        }
    )
    _write_json(run_dir / "progress.json", progress)
    _write_json(run_dir / "metadata.json", metadata)
    return metadata


def _append_sync(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _canonical_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
