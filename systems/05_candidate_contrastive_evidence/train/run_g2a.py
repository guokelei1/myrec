"""Run the preregistered C05 G1 smoke and G2a train-internal falsifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from model.signal_probe import TargetAttentionSignalProbe  # noqa: E402
from myrec.analysis.supervised_diagnostics import PackedRequestData  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, request_metrics  # noqa: E402
from train.data import (  # noqa: E402
    FrozenG2AFeatures,
    assert_candidate_manifest,
    assert_proposal_lock,
    collate_g2a,
    iter_request_batches,
    load_config,
    read_json,
    seed_everything,
    selected_candidate_key_sha256,
    sha256_file,
    validate_gpu,
    write_json,
)
from train.losses import masked_listwise_loss  # noqa: E402


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines()
        return {"commit": commit, "dirty": bool(status), "status_short": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def _begin_attempt(
    config: dict[str, Any],
    config_path: Path,
    proposal_lock_hash: str,
) -> dict[str, Any]:
    artifact_root = Path(config["paths"]["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_path = artifact_root / "g2a_report.json"
    checkpoint_path = Path(config["paths"]["checkpoint"])
    if report_path.exists():
        raise FileExistsError("immutable G2a outcome report already exists")
    if checkpoint_path.exists():
        raise FileExistsError(
            "formal checkpoint already exists; refusing an outcome-ambiguous overwrite"
        )

    ledger_path = artifact_root / "attempt_ledger.json"
    if ledger_path.exists():
        ledger = read_json(ledger_path)
    else:
        ledger = {
            "candidate_id": "c05",
            "base_run_id": config["run_id"],
            "implementation_attempts_max": int(
                config["training"]["implementation_attempts_max"]
            ),
            "attempts": [],
        }
    attempts = ledger["attempts"]
    if any(
        row.get("stage") in {"internal_evaluation_started", "completed"}
        for row in attempts
    ):
        raise RuntimeError("an internal outcome was already exposed; rerun is forbidden")
    maximum = int(config["training"]["implementation_attempts_max"])
    if len(attempts) >= maximum:
        raise RuntimeError("C05 G2a implementation-attempt budget is exhausted")

    attempt_number = len(attempts) + 1
    actual_run_id = f"{config['run_id']}_attempt{attempt_number}"
    run_dir = Path("runs") / actual_run_id
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"run metadata already exists: {metadata_path}")
    now = datetime.now(timezone.utc).isoformat()
    attempt_row = {
        "attempt": attempt_number,
        "run_id": actual_run_id,
        "stage": "started",
        "started_at": now,
        "internal_outcome_observed": False,
        "config_sha256": sha256_file(config_path),
        "proposal_lock_sha256": proposal_lock_hash,
    }
    attempts.append(attempt_row)
    write_json(ledger_path, ledger)

    metadata = {
        "candidate_id": "c05",
        "method_id": config["method_id"],
        "run_id": actual_run_id,
        "base_run_id": config["run_id"],
        "attempt": attempt_number,
        "current_stage": "started",
        "created_at": now,
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "proposal_lock_path": config["paths"]["proposal_lock"],
        "proposal_lock_sha256": proposal_lock_hash,
        "environment": {
            "name": config["environment"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "visible_gpu_count": torch.cuda.device_count(),
            "visible_gpu_name": torch.cuda.get_device_name(0),
            "program_device": config["program_device"],
            "registered_physical_gpu": int(config["physical_gpu"]),
        },
        "git": _git_metadata(),
        "qrels_read": False,
        "dev_records_read": False,
        "test_read": False,
    }
    write_json(metadata_path, metadata)
    return {
        "attempt": attempt_number,
        "run_id": actual_run_id,
        "ledger_path": str(ledger_path),
        "metadata_path": str(metadata_path),
    }


def _update_attempt(
    context: dict[str, Any], stage: str, **fields: Any
) -> None:
    ledger_path = Path(context["ledger_path"])
    ledger = read_json(ledger_path)
    matching = [
        row
        for row in ledger["attempts"]
        if int(row["attempt"]) == int(context["attempt"])
    ]
    if len(matching) != 1:
        raise ValueError("attempt ledger row is missing or duplicated")
    now = datetime.now(timezone.utc).isoformat()
    matching[0].update(fields)
    matching[0]["stage"] = stage
    matching[0]["updated_at"] = now
    write_json(ledger_path, ledger)

    metadata_path = Path(context["metadata_path"])
    metadata = read_json(metadata_path)
    metadata["current_stage"] = stage
    metadata["updated_at"] = now
    metadata.update(fields)
    write_json(metadata_path, metadata)


def _new_model(config: dict[str, Any], device: str) -> TargetAttentionSignalProbe:
    model_config = config["model"]
    return TargetAttentionSignalProbe(
        input_dim=int(model_config["input_dim"]),
        evidence_dim=int(model_config["evidence_dim"]),
        score_delta_max=float(model_config["score_delta_max"]),
        dropout=float(model_config["dropout"]),
    ).to(device)


def _validate_g0(
    config: dict[str, Any], proposal_lock_hash: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(config["paths"]["artifact_root"])
    report_path = root / "g0_report.json"
    report = read_json(report_path)
    if report.get("status") != "passed" or report.get("gate") != "G0":
        raise ValueError("C05 G0 did not pass")
    if report.get("proposal_lock_sha256") != proposal_lock_hash:
        raise ValueError("G0 used a different proposal lock")
    if report.get("qrels_read") or report.get("dev_records_read") or report.get(
        "test_read"
    ):
        raise ValueError("G0 evidence-hygiene violation")
    if not report["alignment"].get("bitwise_array_equal"):
        raise ValueError("G0 base alignment is not bitwise exact")
    if int(report["alignment"].get("rank_mismatches", -1)) != 0:
        raise ValueError("G0 base alignment changed ranks")
    for filename, metadata in report["outputs"].items():
        if sha256_file(metadata["path"]) != metadata["sha256"]:
            raise ValueError(f"G0 output hash changed: {filename}")
    for filename, metadata in report["packed_candidate_files"].items():
        if sha256_file(metadata["path"]) != metadata["sha256"]:
            raise ValueError(f"packed candidate file changed after G0: {filename}")
    for name, metadata in report["registered_input_files"].items():
        if sha256_file(metadata["path"]) != metadata["sha256"]:
            raise ValueError(f"registered G0/G2a input changed: {name}")
    if sha256_file(config["paths"]["raw_item_embeddings"]) != report["base"][
        "item_embeddings_sha256"
    ]:
        raise ValueError("raw item embeddings changed after G0")

    selection_path = root / "selection.json"
    if sha256_file(selection_path) != report["selection_sha256_frozen_before_labels"]:
        raise ValueError("selection lock changed after G0")
    selection = read_json(selection_path)
    return report, selection


def _batches(
    data: PackedRequestData,
    indices: list[int],
    config: dict[str, Any],
    *,
    seed: int,
    shuffle: bool,
):
    training = config["training"]
    yield from iter_request_batches(
        data,
        indices,
        history_limit=int(config["model"]["max_history"]),
        max_requests=int(training["max_requests_per_batch"]),
        max_padded_candidates=int(training["max_padded_candidate_rows"]),
        max_padded_history=int(training["max_padded_history_rows"]),
        seed=seed,
        shuffle=shuffle,
    )


def _forward(
    model: TargetAttentionSignalProbe,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    return model(
        tensors["query"],
        tensors["candidates"],
        tensors["history"],
        tensors["candidate_mask"],
        tensors["history_mask"],
        tensors["history_event_weights"],
        tensors["base_scores"],
    ).scores


def _smoke(
    config: dict[str, Any],
    data: PackedRequestData,
    features: FrozenG2AFeatures,
    fit_indices: list[int],
    device: str,
) -> dict[str, Any]:
    started = time.time()
    seed_everything(int(config["seed"]))
    batch_indices = None
    batch_shape = None
    best_work = -1
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        for candidate_batch in _batches(
            data,
            fit_indices,
            config,
            seed=int(config["seed"]) + epoch,
            shuffle=True,
        ):
            candidate_counts = [
                int(data.candidate_offsets[index + 1] - data.candidate_offsets[index])
                for index in candidate_batch
            ]
            history_counts = [
                min(
                    int(data.history_offsets[index + 1] - data.history_offsets[index]),
                    int(config["model"]["max_history"]),
                )
                for index in candidate_batch
            ]
            padded_candidates = len(candidate_batch) * max(candidate_counts)
            padded_history = len(candidate_batch) * max(1, max(history_counts))
            attention_pairs = (
                len(candidate_batch)
                * max(candidate_counts)
                * max(1, max(history_counts))
            )
            work = (
                (padded_candidates + padded_history)
                * int(config["model"]["input_dim"]) ** 2
                + attention_pairs * int(config["model"]["evidence_dim"])
            )
            if work > best_work:
                best_work = work
                batch_indices = candidate_batch.copy()
                batch_shape = {
                    "requests": len(candidate_batch),
                    "padded_candidate_rows": padded_candidates,
                    "padded_history_rows": padded_history,
                    "padded_attention_pairs": attention_pairs,
                    "estimated_linear_attention_work": work,
                }
    if batch_indices is None or batch_shape is None:
        raise ValueError("no formal fit batch exists for G1 smoke")
    batch = collate_g2a(
        data, batch_indices, history_limit=int(config["model"]["max_history"])
    )
    tensors = features.tensors(data, batch, device)
    model = _new_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    losses = []
    nonzero_gradients = []
    optimizer_step_seconds = []
    for step in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_started = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scores = _forward(model, tensors)
            loss = masked_listwise_loss(
                scores, tensors["labels"], tensors["candidate_mask"]
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"G1 smoke loss is non-finite at step {step}")
        loss.backward()
        step_nonzero = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"G1 invalid gradient: {name}")
            if bool((parameter.grad != 0).any().item()):
                step_nonzero.append(name)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip_norm"])
        )
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_step_seconds.append(time.time() - step_started)
        losses.append(float(loss.detach().cpu()))
        nonzero_gradients.append(step_nonzero)
    required_prefixes = {
        "query_projection",
        "history_value_projection",
        "output_projection",
        "score_head",
    }
    first_prefixes = {name.split(".")[0] for name in nonzero_gradients[0]}
    if "output_projection" not in first_prefixes:
        raise AssertionError("G1 zero-residual first step did not open the output path")
    second_prefixes = {name.split(".")[0] for name in nonzero_gradients[1]}
    if not required_prefixes.issubset(second_prefixes):
        raise AssertionError(
            f"G1 lacks real second-step gradient paths: {second_prefixes}"
        )

    smoke_path = Path(config["paths"]["artifact_root"]) / "g1_smoke_checkpoint.pt"
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, smoke_path)
    model.eval()
    with torch.inference_mode():
        before = _forward(model, tensors).float().cpu().numpy()
    restored = _new_model(config, device)
    restored.load_state_dict(
        torch.load(smoke_path, map_location=device, weights_only=True)["model_state"],
        strict=True,
    )
    restored.eval()
    with torch.inference_mode():
        after = _forward(restored, tensors).float().cpu().numpy()
    if not np.array_equal(before, after):
        raise AssertionError("G1 checkpoint reload did not rescore bit-identically")
    return {
        "status": "passed",
        "batch_requests": len(batch_indices),
        "batch_selection": "maximum estimated work among both formal epoch batchings",
        "batch_shape": batch_shape,
        "amp_dtype": "bfloat16",
        "optimizer_steps": 2,
        "optimizer_step_seconds": optimizer_step_seconds,
        "losses": losses,
        "nonzero_gradient_parameter_tensors_by_step": nonzero_gradients,
        "checkpoint_path": str(smoke_path),
        "checkpoint_sha256": sha256_file(smoke_path),
        "reload_rescore_bitwise_equal": True,
        "elapsed_seconds": time.time() - started,
    }


def _pretrain_projection(
    config: dict[str, Any],
    data: PackedRequestData,
    fit_indices: list[int],
    internal_indices: list[int],
    smoke: dict[str, Any],
    g0_report: dict[str, Any],
    elapsed_so_far: float,
) -> dict[str, Any]:
    fit_batches = sum(
        sum(
            1
            for _ in _batches(
                data,
                fit_indices,
                config,
                seed=int(config["seed"]) + epoch,
                shuffle=True,
            )
        )
        for epoch in range(1, int(config["training"]["epochs"]) + 1)
    )
    internal_batches = sum(
        1
        for _ in _batches(
            data,
            internal_indices,
            config,
            seed=int(config["seed"]),
            shuffle=False,
        )
    )
    seconds_per_train_batch = max(
        float(smoke["elapsed_seconds"]) / 2.0,
        max(float(value) for value in smoke["optimizer_step_seconds"]),
    )
    # Three internal passes are binding: two deterministic score passes and
    # one shared-metric pass.  Treat each as a train batch for a conservative
    # upper projection.
    projected_remaining_seconds = seconds_per_train_batch * (
        fit_batches + 3 * internal_batches
    )
    projected_hours = (
        float(g0_report["elapsed_seconds"])
        + elapsed_so_far
        + projected_remaining_seconds
    ) / 3600.0
    budget = float(config["training"]["cumulative_a40_gpu_hours_max"])
    result = {
        "fit_batches": fit_batches,
        "internal_batches_per_pass": internal_batches,
        "conservative_seconds_per_batch": seconds_per_train_batch,
        "projected_remaining_seconds": projected_remaining_seconds,
        "projected_cumulative_a40_hours": projected_hours,
        "budget_a40_hours": budget,
        "status": "passed" if projected_hours <= budget else "failed",
    }
    if result["status"] != "passed":
        raise RuntimeError(f"G2a pre-train wall-time projection exceeds budget: {result}")
    return result


def _train(
    model: TargetAttentionSignalProbe,
    config: dict[str, Any],
    data: PackedRequestData,
    features: FrozenG2AFeatures,
    fit_indices: list[int],
    device: str,
) -> tuple[TargetAttentionSignalProbe, list[dict[str, Any]]]:
    seed = int(config["seed"])
    seed_everything(seed)
    initial = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epoch_rows: list[dict[str, Any]] = []
    amp_dtype = torch.bfloat16
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        requests = 0
        batches = 0
        for batch_indices in _batches(
            data,
            fit_indices,
            config,
            seed=seed + epoch,
            shuffle=True,
        ):
            batch = collate_g2a(
                data,
                batch_indices,
                history_limit=int(config["model"]["max_history"]),
            )
            tensors = features.tensors(data, batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                scores = _forward(model, tensors)
                loss = masked_listwise_loss(
                    scores, tensors["labels"], tensors["candidate_mask"]
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite G2a loss at epoch {epoch}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite G2a gradient at epoch {epoch}")
            optimizer.step()
            count = len(batch_indices)
            loss_sum += float(loss.detach().cpu()) * count
            requests += count
            batches += 1
        movement = math.sqrt(
            sum(
                float(((parameter.detach() - initial[name]) ** 2).sum().cpu())
                for name, parameter in model.named_parameters()
            )
        )
        if movement == 0.0:
            raise AssertionError("G2a parameters did not move")
        epoch_rows.append(
            {
                "epoch": epoch,
                "mean_listwise_loss": loss_sum / requests,
                "requests": requests,
                "batches": batches,
                "parameter_l2_movement_from_initial": movement,
            }
        )
    return model, epoch_rows


def _score_internal(
    model: TargetAttentionSignalProbe,
    config: dict[str, Any],
    data: PackedRequestData,
    features: FrozenG2AFeatures,
    internal_indices: list[int],
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    base_metrics: list[float] = []
    personalized_metrics: list[float] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_indices in _batches(
            data,
            internal_indices,
            config,
            seed=int(config["seed"]),
            shuffle=False,
        ):
            batch = collate_g2a(
                data,
                batch_indices,
                history_limit=int(config["model"]["max_history"]),
            )
            tensors = features.tensors(data, batch, device)
            scores = _forward(model, tensors).float().cpu().numpy()
            base = tensors["base_scores"].float().cpu().numpy()
            for row_index, raw_index in enumerate(batch_indices):
                request_index = int(raw_index)
                count = int(batch["candidate_mask"][row_index].sum())
                item_ids = batch["candidate_item_ids"][row_index, :count]
                labels = batch["candidate_labels"][row_index, :count]
                clicked = {
                    str(item_id)
                    for item_id, label in zip(item_ids, labels)
                    if float(label) > 0
                }
                candidates = [str(item_id) for item_id in item_ids]
                request_id = data.request_ids[request_index]
                base_row = request_metrics(
                    request_id,
                    [
                        ScoredCandidate(item_id, float(score))
                        for item_id, score in zip(candidates, base[row_index, :count])
                    ],
                    clicked,
                    set(),
                )
                personalized_row = request_metrics(
                    request_id,
                    [
                        ScoredCandidate(item_id, float(score))
                        for item_id, score in zip(candidates, scores[row_index, :count])
                    ],
                    clicked,
                    set(),
                )
                base_ndcg = float(base_row["ndcg@10"])
                personalized_ndcg = float(personalized_row["ndcg@10"])
                base_metrics.append(base_ndcg)
                personalized_metrics.append(personalized_ndcg)
                rows.append(
                    {
                        "request_index": request_index,
                        "request_id": request_id,
                        "base_ndcg@10": base_ndcg,
                        "personalized_ndcg@10": personalized_ndcg,
                        "delta_ndcg@10": personalized_ndcg - base_ndcg,
                    }
                )
    return (
        np.asarray(base_metrics, dtype=np.float64),
        np.asarray(personalized_metrics, dtype=np.float64),
        rows,
    )


def _flat_scores(
    model: TargetAttentionSignalProbe,
    config: dict[str, Any],
    data: PackedRequestData,
    features: FrozenG2AFeatures,
    internal_indices: list[int],
    device: str,
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch_indices in _batches(
            data,
            internal_indices,
            config,
            seed=int(config["seed"]),
            shuffle=False,
        ):
            batch = collate_g2a(
                data,
                batch_indices,
                history_limit=int(config["model"]["max_history"]),
            )
            tensors = features.tensors(data, batch, device)
            scores = _forward(model, tensors).float().cpu().numpy()
            for row in range(len(batch_indices)):
                count = int(batch["candidate_mask"][row].sum())
                values.append(scores[row, :count].copy())
    return np.concatenate(values).astype(np.float32, copy=False)


def _flat_base_scores(
    config: dict[str, Any],
    data: PackedRequestData,
    features: FrozenG2AFeatures,
    internal_indices: list[int],
    device: str,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for batch_indices in _batches(
        data,
        internal_indices,
        config,
        seed=int(config["seed"]),
        shuffle=False,
    ):
        batch = collate_g2a(
            data,
            batch_indices,
            history_limit=int(config["model"]["max_history"]),
        )
        tensors = features.tensors(data, batch, device)
        base = tensors["base_scores"].float().cpu().numpy()
        for row in range(len(batch_indices)):
            count = int(batch["candidate_mask"][row].sum())
            values.append(base[row, :count].copy())
    return np.concatenate(values).astype(np.float32, copy=False)


def _fold(request_id: str, seed: int, folds: int) -> int:
    payload = f"c05_g2a_fold:{seed}:{request_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16) % folds


def _gate_statistics(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gate = config["g2a_gate"]
    deltas = np.asarray([row["delta_ndcg@10"] for row in rows], dtype=np.float64)
    bootstrap_seed = int(gate["bootstrap_seed"])
    samples = int(gate["bootstrap_samples"])
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(deltas), size=len(deltas))
        bootstrap_means[index] = deltas[draw].mean()
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])

    fold_count = int(gate["hash_folds"])
    fold_rows = []
    for fold in range(fold_count):
        selected = np.asarray(
            [
                _fold(str(row["request_id"]), int(config["seed"]), fold_count)
                == fold
                for row in rows
            ],
            dtype=bool,
        )
        fold_rows.append(
            {
                "fold": fold,
                "requests": int(selected.sum()),
                "mean_delta_ndcg@10": float(deltas[selected].mean()),
            }
        )
    mean_delta = float(deltas.mean())
    checks = {
        "delta_at_least_threshold": mean_delta
        >= float(gate["nonrepeat_ndcg10_delta_min"]),
        "bootstrap_ci_low_strictly_positive": float(low) > 0.0,
        "all_hash_folds_positive": all(
            row["mean_delta_ndcg@10"] > 0.0 for row in fold_rows
        ),
    }
    return {
        "mean_delta_ndcg@10": mean_delta,
        "bootstrap": {
            "samples": samples,
            "seed": bootstrap_seed,
            "percentile_95_ci": [float(low), float(high)],
        },
        "hash_fold_rule": (
            "int(sha256('c05_g2a_fold:<seed>:<request_id>'), 16) mod 3"
        ),
        "hash_folds": fold_rows,
        "checks": checks,
    }


def run(config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    config_path = Path(config_path)
    config = load_config(config_path)
    validate_gpu(device)
    if not config["authorization"].get("g2a"):
        raise PermissionError("G2a is not authorized")
    if config["authorization"].get("g2b") or config["authorization"].get(
        "cceb_training"
    ):
        raise PermissionError("this runner is scoped to G2a only")
    proposal_lock_hash = assert_proposal_lock(config)
    candidate_manifest_hash = assert_candidate_manifest(config)
    g0_report, selection = _validate_g0(config, proposal_lock_hash)

    fit_indices = [int(value) for value in selection["fit"]["indices"]]
    internal_indices = [int(value) for value in selection["internal"]["indices"]]
    if set(fit_indices) & set(internal_indices):
        raise AssertionError("fit/internal selection overlap")
    data = PackedRequestData.load(config["paths"]["packed_train_root"], "train")
    selected_indices = fit_indices + internal_indices
    if selected_candidate_key_sha256(data, selected_indices) != g0_report[
        "requests"
    ]["selected_candidate_key_sha256"]:
        raise ValueError("selected candidate-key hash changed after G0")
    features = FrozenG2AFeatures(config, selection)
    attempt = _begin_attempt(config, config_path, proposal_lock_hash)

    smoke = _smoke(config, data, features, fit_indices, device)
    _update_attempt(attempt, "g1_smoke_passed", g1_smoke_status="passed")
    seed_everything(int(config["seed"]))
    model = _new_model(config, device)
    initial_flat = _flat_scores(
        model, config, data, features, internal_indices, device
    )
    base_flat = _flat_base_scores(
        config, data, features, internal_indices, device
    )
    initial_parity = {
        "candidate_rows": len(initial_flat),
        "bitwise_equal_to_d2p": bool(np.array_equal(initial_flat, base_flat)),
        "max_abs_difference_from_d2p": float(
            np.max(
                np.abs(
                    initial_flat.astype(np.float64) - base_flat.astype(np.float64)
                )
            )
        ),
    }
    if not initial_parity["bitwise_equal_to_d2p"]:
        raise AssertionError(f"same-seed untrained probe is not exact D2p: {initial_parity}")
    projection = _pretrain_projection(
        config,
        data,
        fit_indices,
        internal_indices,
        smoke,
        g0_report,
        time.time() - started,
    )
    _update_attempt(
        attempt,
        "pretrain_projection_passed",
        projected_cumulative_a40_hours=projection[
            "projected_cumulative_a40_hours"
        ],
    )
    _update_attempt(attempt, "formal_training_started")
    model, epochs = _train(model, config, data, features, fit_indices, device)
    checkpoint_path = Path(config["paths"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate_id": "c05",
            "variant": "g2a_target_attention",
            "seed": int(config["seed"]),
            "config_sha256": sha256_file(config_path),
            "proposal_lock_sha256": proposal_lock_hash,
            "g0_report_sha256": sha256_file(
                Path(config["paths"]["artifact_root"]) / "g0_report.json"
            ),
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        checkpoint_path,
    )

    restored_checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    reloaded_model = _new_model(config, device)
    reloaded_model.load_state_dict(restored_checkpoint["model_state"], strict=True)
    reloaded_model.eval()
    model = reloaded_model
    _update_attempt(
        attempt,
        "trained_checkpoint_reloaded",
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
    )

    _update_attempt(
        attempt,
        "internal_evaluation_started",
        internal_outcome_observed=True,
    )
    first_flat = _flat_scores(
        model, config, data, features, internal_indices, device
    )
    second_flat = _flat_scores(
        model, config, data, features, internal_indices, device
    )
    deterministic_max_abs = float(
        np.max(np.abs(first_flat.astype(np.float64) - second_flat.astype(np.float64)))
    )
    deterministic_exact = bool(np.array_equal(first_flat, second_flat))
    base, personalized, per_request = _score_internal(
        model, config, data, features, internal_indices, device
    )
    statistics = _gate_statistics(config, per_request)
    deterministic_required = float(
        config["g2a_gate"]["deterministic_rescore_max_abs_difference"]
    )
    statistics["checks"]["deterministic_rescore"] = (
        deterministic_exact and deterministic_max_abs <= deterministic_required
    )

    elapsed = time.time() - started
    cumulative_gpu_hours = float(g0_report["a40_gpu_hours_reserved"]) + elapsed / 3600.0
    statistics["checks"]["within_gpu_budget"] = cumulative_gpu_hours <= float(
        config["training"]["cumulative_a40_gpu_hours_max"]
    )
    passed = all(bool(value) for value in statistics["checks"].values())

    artifact_root = Path(config["paths"]["artifact_root"])
    np.savez_compressed(
        artifact_root / "g2a_internal_per_request.npz",
        request_indices=np.asarray(internal_indices, dtype=np.int64),
        base_ndcg10=base,
        personalized_ndcg10=personalized,
    )
    report = {
        "candidate_id": "c05",
        "gate": "G2a",
        "run_id": attempt["run_id"],
        "implementation_attempt": attempt["attempt"],
        "attempt_ledger_path": attempt["ledger_path"],
        "run_metadata_path": attempt["metadata_path"],
        "status": "passed" if passed else "failed",
        "decision": (
            "eligible for a separately reviewed and locked G2b; still unauthorized"
            if passed
            else "stop; close shallow non-repeat transfer and do not train CCEB"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "proposal_lock_sha256": proposal_lock_hash,
        "candidate_manifest_sha256": candidate_manifest_hash,
        "g0_report_sha256": sha256_file(artifact_root / "g0_report.json"),
        "selection_sha256": sha256_file(artifact_root / "selection.json"),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "selection": "fixed final epoch only",
            "strictly_reloaded_before_internal_evaluation": True,
        },
        "model": {
            "variant": "ordinary positive target attention",
            "parameter_count": model.parameter_count(),
            "exact_relation_enabled": False,
            "candidate_centering_enabled": False,
            "corruption_training_enabled": False,
            "evidence_budget_enabled": False,
        },
        "g1_smoke": smoke,
        "same_seed_untrained_control": initial_parity,
        "pretrain_wall_time_projection": projection,
        "training": {
            "fit_requests": len(fit_indices),
            "epochs": epochs,
            "amp_dtype": str(config["training"]["amp_dtype"]),
            "optimizer": "AdamW",
        },
        "internal_evaluation": {
            "requests": len(internal_indices),
            "base_mean_ndcg@10": float(base.mean()),
            "personalized_mean_ndcg@10": float(personalized.mean()),
            "deterministic_rescore_bitwise_equal": deterministic_exact,
            "deterministic_rescore_max_abs_difference": deterministic_max_abs,
            **statistics,
        },
        "elapsed_seconds_excluding_g0": elapsed,
        "cumulative_a40_gpu_hours_reserved": cumulative_gpu_hours,
        "primary_dev_evaluator_calls": 0,
        "qrels_read": False,
        "dev_records_read": False,
        "test_read": False,
        "g2b_eligible_for_new_lock": passed,
        "g2b_authorized": False,
        "cceb_training_authorized": False,
    }
    report_path = artifact_root / "g2a_report.json"
    write_json(report_path, report)
    _update_attempt(
        attempt,
        "completed",
        internal_outcome_observed=True,
        gate_status=report["status"],
        report_path=str(report_path),
        report_sha256=sha256_file(report_path),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    run(arguments.config, arguments.device)


if __name__ == "__main__":
    main()
