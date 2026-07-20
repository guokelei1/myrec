"""Exact train-only step-501 replay for Q2 loss and optimizer coordinates."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import load_training_groups
from myrec.baselines.motivation_v12_ranker import (
    _assert_frozen_training_population,
    _git_revision,
    _load_model_and_tokenizer,
    _runtime_metadata,
    _validate_run_id,
    _yes_no_group_scores,
    listwise_softmax_loss,
    load_v12_ranker_config,
    pairwise_ranknet_loss,
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
    parameter_order_digest,
    parameter_order_rows,
    q2_parameter_family,
    vector_cosine,
    vector_relative_error as _vector_relative_error,
    vector_summary,
)
from myrec.utils.hashing import sha256_file, sha256_text


Q2_METHOD_ID = "q2_recranker_generalqwen"
EXPECTED_PARAMETER_ORDER_DIGEST = (
    "f6f51ede754a5b360faba0b3acde525a3a03ba4e512d8e618ef321646dc9cf70"
)
OBJECTIVES = ("pairwise_ranknet", "listwise_softmax", "combined_half_half")
OBJECTIVE_PAIRS = (
    ("pairwise_ranknet", "listwise_softmax"),
    ("pairwise_ranknet", "combined_half_half"),
    ("listwise_softmax", "combined_half_half"),
)
BLOCK_REQUESTS = 16
BLOCKS_PER_SURFACE = 6
MAX_WALL_SECONDS = 13_500.0


def run_q2_step501_replay(
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
    """Replay three Q2 objectives from identical step-500 state per block."""

    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("an explicit Q2 optimizer replay device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("Q2 optimizer replay wall time must be in (0,13500]")
    if max_tasks is not None and not 0 < int(max_tasks) < 36:
        raise ValueError("Q2 replay smoke max_tasks must be in [1,35]")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    config = load_v12_ranker_config(config_path)
    if config["method_id"] != Q2_METHOD_ID:
        raise ValueError("step-501 full-model replay is Q2-only")
    frozen = manifest["frozen_inputs"]["models"][Q2_METHOD_ID]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("Q2 replay config differs from frozen manifest")
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
            "method_id": Q2_METHOD_ID,
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
        Q2_METHOD_ID, manifest_path=manifest_path
    )
    if step_audit["parameter_order_digest"] != EXPECTED_PARAMETER_ORDER_DIGEST:
        raise ValueError("Q2 replay binding parameter digest differs")
    implementation = q2_optimizer_replay_implementation_identity()
    evidence_mode = "registered_optimizer_replay" if max_tasks is None else "mechanical_smoke_non_result"
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": Q2_METHOD_ID,
        "checkpoint_id": step_audit["checkpoint_id"],
        "config_sha256": config["_config_sha256"],
        "selection_sha256": selection_sha256,
        "tasks": len(tasks),
        "objectives": list(OBJECTIVES),
        "parameter_order_digest": EXPECTED_PARAMETER_ORDER_DIGEST,
        "trainer_state_sha256": step_audit["observed_hashes"]["trainer_state_sha256"],
        "training_cuda_rng_index": TRAINING_CUDA_RNG_INDEX[Q2_METHOD_ID],
        "rng_restore_policy": "training_logical_cuda_rng_to_replay_device",
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "device": str(device),
        "implementation_digest": implementation["digest"],
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d7_q2_step501_optimizer_replay",
        "run_id": run_id,
        "method_id": Q2_METHOD_ID,
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
        "objectives": list(OBJECTIVES),
        "native_batch_requests": 1,
        "native_accumulation_steps": 16,
        "requests_per_replay": BLOCK_REQUESTS,
        "optimizer_update": "real AdamW.step at step-501 followed by exact theta restore; algebra separately audited",
        "combined_identity": "separate backward of 0.5*RankNet+0.5*ListNet",
        "training_cuda_rng_index": TRAINING_CUDA_RNG_INDEX[Q2_METHOD_ID],
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
        named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name != "lm_head.weight"
        ]
        if len(named) != 310 or parameter_order_digest(named) != EXPECTED_PARAMETER_ORDER_DIGEST:
            raise ValueError("Q2 replay model parameter order differs from step-500 optimizer")
        names = [name for name, _ in named]
        parameters = [parameter for _, parameter in named]
        original_parameters = {
            name: parameter.detach().clone() for name, parameter in named
        }
        families = {name: q2_parameter_family(name) for name in names}
        optimizer_state = step_state["optimizer"]
        group = optimizer_state["param_groups"][0]
        state_by_name = {
            name: optimizer_state["state"][index] for index, name in enumerate(names)
        }
        next_lr = _next_linear_lr(step_state)
        rng_restore_binding = restore_bound_rng_state(
            torch,
            step_state["rng"],
            method_id=Q2_METHOD_ID,
            device=str(device),
        )
        metadata.update(
            {
                **_runtime_metadata(Q2_METHOD_ID, torch, transformers),
                "parameter_order_digest": parameter_order_digest(named),
                "parameter_order_rows": parameter_order_rows(named),
                "parameter_families": sorted(set(families.values())),
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
                        + time.monotonic() - started,
                    }
                )
                _write_json(run_dir / "metadata.json", metadata)
                return metadata
            control, surface, block, original_groups = tasks[task_index]
            block_groups = list(original_groups)
            shuffle_audits = None
            if control == "within_request_label_shuffle":
                converted = [deterministic_label_shuffle(value) for value in block_groups]
                block_groups = [value[0] for value in converted]
                shuffle_audits = [value[1] for value in converted]
            raw: dict[str, dict[str, Any]] = {}
            losses: dict[str, float] = {}
            for objective in OBJECTIVES:
                observed_rng_restore = restore_bound_rng_state(
                    torch,
                    step_state["rng"],
                    method_id=Q2_METHOD_ID,
                    device=str(device),
                )
                if observed_rng_restore != rng_restore_binding:
                    raise ValueError("Q2 replay RNG restore binding drifted")
                model.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for micro_group in block_groups:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                        enabled=str(device).startswith("cuda") and dtype != "float32",
                    ):
                        scores = _yes_no_group_scores(
                            model, tokenizer, [micro_group], config, device=str(device)
                        )[0]
                        pair = pairwise_ranknet_loss(scores, micro_group.gains)
                        listwise = listwise_softmax_loss(scores, micro_group.gains)
                        loss = (
                            pair
                            if objective == "pairwise_ranknet"
                            else listwise
                            if objective == "listwise_softmax"
                            else 0.5 * pair + 0.5 * listwise
                        )
                    (loss / 16.0).backward()
                    loss_sum += float(loss.detach().float().cpu())
                gradients = {
                    name: parameter.grad.detach().cpu().clone()
                    if parameter.grad is not None else None
                    for name, parameter in named
                }
                if any(value is None for value in gradients.values()):
                    raise ValueError("Q2 replay produced an unused full-model parameter")
                raw[objective] = gradients
                losses[objective] = loss_sum / 16.0
                model.zero_grad(set_to_none=True)
            raw_identity = _combined_gradient_identity(raw)
            if raw_identity["passed"] is not True:
                raise ValueError(
                    "separate combined-loss backward failed the registered BF16 "
                    f"gradient identity: {raw_identity}"
                )
            stages: dict[str, dict[str, dict[str, Any]]] = {
                name: {} for name in (
                    "raw_gradient", "clipped_gradient", "adam_preconditioned_direction",
                    "moment_delta", "weight_decay_delta", "total_delta",
                )
            }
            objective_results = {}
            for objective in OBJECTIVES:
                expected_clipped, clip_coefficient, raw_norm = clip_gradients(
                    raw[objective], float(config["training"].get("max_grad_norm", 1.0))
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
                    parameter.grad = raw[objective][name].to(parameter.device).clone()
                observed_raw_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        parameters,
                        float(config["training"].get("max_grad_norm", 1.0)),
                    ).item()
                )
                clipped = {
                    name: parameter.grad.detach().cpu().clone()
                    for name, parameter in named
                }
                clip_identity = _vector_relative_error(clipped, expected_clipped)
                if clip_identity["relative_l2_error"] > 8.0 * (2.0**-7):
                    raise ValueError(f"Q2 actual/manual clip mismatch: {clip_identity}")
                components = {
                    "adam_preconditioned_direction": {},
                    "moment_delta": {},
                    "weight_decay_delta": {},
                    "algebra_total_delta": {},
                }
                for (name, parameter) in named:
                    replay = adamw_exact_delta(
                        parameter,
                        clipped[name].to(parameter.device),
                        state_by_name[name],
                        group,
                    )
                    components["adam_preconditioned_direction"][name] = replay[
                        "preconditioned_direction"
                    ].detach().cpu()
                    components["moment_delta"][name] = replay["moment_delta"].detach().cpu()
                    components["weight_decay_delta"][name] = replay[
                        "weight_decay_delta"
                    ].detach().cpu()
                    components["algebra_total_delta"][name] = replay["total_delta"].detach().cpu()
                    del replay
                optimizer.step()
                scheduler.step()
                observed_next_lr = float(scheduler.get_last_lr()[0])
                if not math.isclose(observed_next_lr, next_lr, rel_tol=1.0e-12, abs_tol=0.0):
                    raise ValueError(
                        f"Q2 actual scheduler step differs: {observed_next_lr} != {next_lr}"
                    )
                actual_total = {
                    name: (parameter.detach() - original_parameters[name]).cpu()
                    for name, parameter in named
                }
                actual_algebra_identity = _vector_relative_error(
                    actual_total, components["algebra_total_delta"]
                )
                with torch.no_grad():
                    for name, parameter in named:
                        parameter.copy_(original_parameters[name])
                        parameter.grad = None
                del optimizer, scheduler
                stages["raw_gradient"][objective] = raw[objective]
                stages["clipped_gradient"][objective] = clipped
                for stage, vectors in components.items():
                    if stage == "algebra_total_delta":
                        continue
                    stages[stage][objective] = vectors
                stages["total_delta"][objective] = actual_total
                objective_results[objective] = {
                    "mean_microbatch_loss": losses[objective],
                    "clip_coefficient": clip_coefficient,
                    "raw_global_norm": raw_norm,
                    "torch_clip_grad_norm_return": observed_raw_norm,
                    "manual_vs_torch_clip_identity": clip_identity,
                    "actual_step_vs_algebra_identity": actual_algebra_identity,
                    "actual_scheduler_lr_after_step": observed_next_lr,
                    **{
                        stage: vector_summary(vectors[objective], family_by_name=families)
                        for stage, vectors in stages.items()
                    },
                    "algebra_total_delta": vector_summary(
                        components["algebra_total_delta"], family_by_name=families
                    ),
                    "raw_gradient_vs_total_delta_cosine": vector_cosine(
                        raw[objective], actual_total
                    ),
                    "preconditioned_vs_total_delta_cosine": vector_cosine(
                        components["adam_preconditioned_direction"],
                        actual_total,
                    ),
                }
            stage_cosines = {
                stage: {
                    f"{left}_vs_{right}": vector_cosine(values[left], values[right])
                    for left, right in OBJECTIVE_PAIRS
                }
                for stage, values in stages.items()
            }
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
                "objectives": objective_results,
                "stage_objective_cosines": stage_cosines,
                "combined_raw_gradient_identity": raw_identity,
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
            del raw, stages, objective_results, stage_cosines
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


def _combined_gradient_identity(
    gradients: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | bool]:
    pair = gradients["pairwise_ranknet"]
    listwise = gradients["listwise_softmax"]
    combined = gradients["combined_half_half"]
    maximum = squared_error = reference_mass = 0.0
    for name in pair:
        reference = 0.5 * pair[name].double() + 0.5 * listwise[name].double()
        error = combined[name].double() - reference
        maximum = max(maximum, float(error.abs().max().item()))
        squared_error += float(error.square().sum().item())
        reference_mass += float(reference.square().sum().item())
    relative = math.sqrt(squared_error / reference_mass) if reference_mass > 0 else 0.0
    return {
        "maximum_absolute_error": maximum,
        "relative_l2_error": relative,
        "path_local_bf16_bound": 8.0 * (2.0**-7),
        "passed": relative <= 8.0 * (2.0**-7),
    }


def _next_linear_lr(state: Mapping[str, Any]) -> float:
    contract = state["training_contract"]
    total = int(contract["total_optimizer_steps"])
    base_lr = float(state["scheduler"]["base_lrs"][0])
    warmup = int(total * 0.1)
    next_epoch = int(state["scheduler"]["last_epoch"]) + 1
    if next_epoch < warmup:
        return base_lr * next_epoch / max(1, warmup)
    return base_lr * max(0.0, (total - next_epoch) / max(1, total - warmup))


def q2_optimizer_replay_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/q2_optimizer_replay_runtime.py",
        "src/myrec/mechanism/optimizer_replay_binding.py",
        "src/myrec/mechanism/optimizer_replay_math.py",
        "src/myrec/mechanism/gradient_diagnostic.py",
        "src/myrec/baselines/motivation_v12_ranker.py",
        "scripts/run_deep_dive_q2_optimizer_replay.py",
    )
    files = [
        {"path": path, "sha256": sha256_file(root / path), "size_bytes": (root / path).stat().st_size}
        for path in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}


def _prepare(run_dir, metadata, selection, contract_sha256, tasks, *, resume):
    partial = run_dir / "replays.partial.jsonl"
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Q2 replay run is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial.touch(exist_ok=False)
        _write_json(run_dir / "selection_manifest.json", selection)
        metadata.update({"elapsed_seconds": 0.0, "resumable": True, "resume_lineage": []})
        progress = {
            "schema_version": 1, "run_contract_sha256": contract_sha256,
            "completed_tasks": 0, "last_task": None,
            "partial_sha256": sha256_file(partial), "status": "selection_finalized",
            "updated_at": _utc_now(),
        }
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "progress.json", progress)
        return metadata, progress
    stored = _read_json(run_dir / "metadata.json")
    progress = _read_json(run_dir / "progress.json")
    if stored.get("run_contract_sha256") != contract_sha256 or progress.get("run_contract_sha256") != contract_sha256:
        raise ValueError("Q2 replay resume contract drift")
    if _read_json(run_dir / "selection_manifest.json") != selection:
        raise ValueError("Q2 replay selection drift")
    observed = _audit_partial(partial, tasks)
    if progress.get("completed_tasks") != observed["completed_tasks"] or progress.get("partial_sha256") != observed["partial_sha256"]:
        raise ValueError("Q2 replay progress differs from partial")
    lineage = list(stored.get("resume_lineage", []))
    lineage.append({"completed_tasks": observed["completed_tasks"], "partial_sha256": observed["partial_sha256"]})
    stored.update({"resume_lineage": lineage, "status": "selection_finalized"})
    _write_json(run_dir / "metadata.json", stored)
    return stored, progress


def _audit_partial(path, tasks):
    count = 0
    for row in _iter_jsonl(path):
        if count >= len(tasks):
            raise ValueError("Q2 replay partial has excess rows")
        control, surface, block, groups = tasks[count]
        if (
            row.get("task_index") != count
            or row.get("control") != control
            or row.get("surface") != surface
            or row.get("block_within_surface") != block
            or row.get("request_ids") != [group.record.request_id for group in groups]
            or set(row.get("objectives", {})) != set(OBJECTIVES)
        ):
            raise ValueError("Q2 replay partial task identity/coverage drift")
        count += 1
    return {"completed_tasks": count, "partial_sha256": sha256_file(path)}


def _finalize(run_dir, metadata, progress, tasks):
    observed = _audit_partial(run_dir / "replays.partial.jsonl", tasks)
    if observed["completed_tasks"] != len(tasks):
        raise ValueError("cannot finalize incomplete Q2 optimizer replay")
    path = run_dir / "replays.jsonl"
    os.replace(run_dir / "replays.partial.jsonl", path)
    progress.update({"status": "completed", "updated_at": _utc_now()})
    metadata.update(
        {
            "status": "completed", "resumable": False,
            "completed_replay_blocks": len(tasks), "replays_path": str(path),
            "replays_sha256": sha256_file(path),
            "optimizer_steps_performed_then_exactly_restored": len(tasks) * len(OBJECTIVES),
            "exact_effective_updates_computed": len(tasks) * len(OBJECTIVES),
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
    return datetime.now(timezone.utc).isoformat()
