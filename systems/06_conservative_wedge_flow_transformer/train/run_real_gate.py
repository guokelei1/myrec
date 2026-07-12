"""Run C06 fit-only GPU smoke or the one-shot train-internal real gate.

The formal path trains four frozen variants for exactly two epochs.  It scores
internal-A twice and writes a label-free A0 audit before opening any A label.
Only an A0 pass permits the shared NDCG@10 metric and paired A1 comparisons.
Internal-B and escrow are never loaded by this runner.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import transformers


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from model.complexity import dominant_probe_flops  # noqa: E402
from model.controls import CenteredCrossAttentionProbeRanker  # noqa: E402
from model.wedge_flow import (  # noqa: E402
    ConservativeWedgeFlowProbeRanker,
    low_rank_hodge_calibration,
)
from myrec.eval.metrics import ScoredCandidate, request_metrics, sort_candidates  # noqa: E402
from train.losses import masked_listwise_loss  # noqa: E402
from train.real_data import (  # noqa: E402
    FrozenRealFeatures,
    SelectedLabels,
    StructuralTrainData,
    assert_internal_a_opening_barrier,
    assert_candidate_manifest,
    assert_real_gate_lock,
    collate_structural,
    iter_request_batches,
    load_config,
    open_selected_labels,
    read_json,
    seed_everything,
    selected_candidate_key_sha256,
    sha256_file,
    validate_execution_authority,
    write_json,
)
from train.real_gate_metrics import (  # noqa: E402
    clicked_minus_unclicked,
    compare_primary,
    order_change_summary,
    paired_bootstrap,
)


TRAINED_VARIANTS = (
    "local_hodge",
    "untrusted",
    "direct_learned",
    "centered_cross_attention",
)
COUNTERFACTUAL_VARIANTS = (
    "local_checkpoint_t_one",
    "local_checkpoint_global_hodge",
)


def _git_metadata() -> dict[str, Any]:
    try:
        return {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "status_short": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ).splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def _new_model(
    config: Mapping[str, Any], variant: str, device: str
) -> torch.nn.Module:
    model_config = config["model"]
    kwargs = {
        "input_dim": int(model_config["input_dim"]),
        "score_delta_max": float(model_config["score_delta_max"]),
    }
    if variant == "centered_cross_attention":
        return CenteredCrossAttentionProbeRanker(
            evidence_dim=int(model_config["flow_dim"]),
            compute_rounds=int(model_config["centered_compute_rounds"]),
            **kwargs,
        ).to(device)
    trust_mode = {
        "local_hodge": "local_hodge",
        "untrusted": "untrusted",
        "direct_learned": "direct_learned",
        "local_checkpoint_t_one": "untrusted",
        "local_checkpoint_global_hodge": "global_hodge",
    }.get(variant)
    if trust_mode is None:
        raise ValueError(f"unknown C06 real-gate variant: {variant}")
    return ConservativeWedgeFlowProbeRanker(
        flow_dim=int(model_config["flow_dim"]),
        trust_mode=trust_mode,
        **kwargs,
    ).to(device)


def _forward(
    model: torch.nn.Module, tensors: Mapping[str, torch.Tensor]
) -> Any:
    return model(
        tensors["query"],
        tensors["candidates"],
        tensors["history"],
        tensors["candidate_mask"],
        tensors["history_mask"],
        tensors["history_prior"],
        tensors["base_scores"],
    )


def _batches(
    data: StructuralTrainData,
    indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
    shuffle: bool,
):
    yield from iter_request_batches(
        data,
        indices,
        history_limit=int(config["model"]["max_history"]),
        max_requests=int(config["training"]["max_requests_per_batch"]),
        max_padded_candidates=int(
            config["training"]["max_padded_candidate_rows"]
        ),
        max_padded_history=int(config["training"]["max_padded_history_rows"]),
        seed=seed,
        shuffle=shuffle,
    )


def _validate_capacity(config: Mapping[str, Any], device: str) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    models = {variant: _new_model(config, variant, device) for variant in TRAINED_VARIANTS}
    counts = {variant: model.parameter_count() for variant, model in models.items()}
    primary = counts["local_hodge"]
    maximum_fraction = float(config["controls"]["parameter_difference_max_fraction"])
    checks = {
        variant: abs(count - primary) / primary <= maximum_fraction
        for variant, count in counts.items()
    }
    if not all(checks.values()):
        raise AssertionError(f"C06 capacity match failed: {counts}")
    reference = config["controls"]["flop_reference"]
    dimensions = {
        "input_dim": int(config["model"]["input_dim"]),
        "evidence_dim": int(config["model"]["flow_dim"]),
        "candidates": int(reference["candidates"]),
        "history": int(reference["history"]),
    }
    primary_flops = dominant_probe_flops(variant="local_hodge", **dimensions)
    centered_flops = dominant_probe_flops(
        variant="centered_cross_attention",
        centered_compute_rounds=int(config["model"]["centered_compute_rounds"]),
        **dimensions,
    )
    difference = abs(centered_flops - primary_flops) / primary_flops
    flop_ok = difference <= float(config["controls"]["flop_difference_max_fraction"])
    if not flop_ok:
        raise AssertionError("C06 centered control FLOP match failed")
    del models
    torch.cuda.empty_cache()
    return {
        "parameter_counts": counts,
        "parameter_match_checks": checks,
        "parameter_difference_max_fraction": maximum_fraction,
        "dominant_flops": {
            "reference_shape": reference,
            "local_hodge": primary_flops,
            "centered_cross_attention": centered_flops,
            "fractional_difference": difference,
        },
        "dominant_flop_match": flop_ok,
    }


def _validate_g0(
    config: Mapping[str, Any], lock_hash: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(config["paths"]["artifact_root"])
    report_path = root / "g0_report.json"
    report = read_json(report_path)
    if report.get("status") != "passed" or report.get("gate") != "G0":
        raise ValueError("C06 G0 has not passed")
    repair = config["numeric_repair"]
    if report.get("real_gate_lock_sha256") != repair["parent_lock_sha256"]:
        raise ValueError("C06 G0 did not use the preserved parent v1 lock")
    if sha256_file(report_path) != repair["g0_report_sha256"]:
        raise ValueError("C06 G0 differs from the review1 repair registration")
    if report.get("materialized_label_roles") != ["fit"]:
        raise ValueError("C06 G0 label isolation changed")
    if report.get("internal_A_labels_opened"):
        raise ValueError("C06 G0 improperly opened internal-A labels")
    if not report.get("internal_B_features_or_labels_opened") is False:
        raise ValueError("C06 G0 improperly touched delayed B")
    if not report.get("escrow_features_or_labels_opened") is False:
        raise ValueError("C06 G0 improperly touched escrow")
    if report.get("qrels_read") or report.get("dev_records_read") or report.get("test_read"):
        raise ValueError("C06 G0 evidence-hygiene violation")
    if not report["alignment"].get("bitwise_array_equal"):
        raise ValueError("C06 G0 key alignment is not exact")
    label_source = report.get("train_label_source_verified_after_selection")
    if not isinstance(label_source, dict) or label_source.get(
        "verified_after_selection_sha256"
    ) != report.get("selection_sha256_frozen_before_labels"):
        raise ValueError("C06 G0 lacks post-selection label-source registration")
    for metadata in report["outputs"].values():
        if sha256_file(metadata["path"]) != metadata["sha256"]:
            raise ValueError(f"C06 G0 output changed: {metadata['path']}")
    for metadata in report["registered_structural_input_files"].values():
        if sha256_file(metadata["path"]) != metadata["sha256"]:
            raise ValueError(f"C06 registered structural input changed: {metadata['path']}")
    selection_path = root / "selection.json"
    if sha256_file(selection_path) != report["selection_sha256_frozen_before_labels"]:
        raise ValueError("C06 frozen selection changed")
    selection = read_json(selection_path)
    expected_counts = {
        "fit": 12_000,
        "internal_A": 1_200,
        "internal_B": 600,
        "escrow": 515,
        "nohistory": 512,
    }
    actual_counts = {
        role: len(selection["roles"][role]["indices"]) for role in expected_counts
    }
    if actual_counts != expected_counts:
        raise ValueError(f"C06 frozen cohort count changed: {actual_counts}")
    return report, selection


def _validate_role_candidate_hashes(
    data: StructuralTrainData,
    selection: Mapping[str, Any],
    g0_report: Mapping[str, Any],
) -> dict[str, str]:
    expected = g0_report["requests"].get("role_candidate_key_sha256")
    if not isinstance(expected, dict):
        raise ValueError("C06 G0 lacks per-role candidate-key hashes")
    actual = {
        role: selected_candidate_key_sha256(data, row["indices"])
        for role, row in selection["roles"].items()
    }
    required = {"fit", "internal_A", "nohistory"}
    if not required.issubset(actual) or any(actual[role] != expected.get(role) for role in actual):
        raise ValueError("C06 selected candidate identities changed after G0")
    return actual


def _largest_fit_batch(
    data: StructuralTrainData,
    fit_indices: Sequence[int],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    best: np.ndarray | None = None
    best_shape: dict[str, int] | None = None
    best_work = -1
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        for indices in _batches(
            data,
            fit_indices,
            config,
            seed=int(config["seed"]) + epoch,
            shuffle=True,
        ):
            candidates = max(
                int(data.candidate_offsets[index + 1] - data.candidate_offsets[index])
                for index in indices
            )
            history = max(
                1,
                max(
                    min(
                        int(data.history_offsets[index + 1] - data.history_offsets[index]),
                        int(config["model"]["max_history"]),
                    )
                    for index in indices
                ),
            )
            work = dominant_probe_flops(
                variant="local_hodge",
                input_dim=int(config["model"]["input_dim"]),
                evidence_dim=int(config["model"]["flow_dim"]),
                candidates=candidates,
                history=history,
            ) * len(indices)
            if work > best_work:
                best = indices.copy()
                best_work = work
                best_shape = {
                    "requests": len(indices),
                    "candidate_columns": candidates,
                    "history_columns": history,
                    "dominant_flops": work,
                }
    if best is None or best_shape is None:
        raise ValueError("no C06 fit batch exists")
    return best, best_shape


def _smoke_variant(
    *,
    variant: str,
    batch: Mapping[str, np.ndarray],
    tensors: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    model = _new_model(config, variant, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    losses = []
    gradient_names = []
    fallback_counts = []
    step_seconds = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = _forward(model, tensors)
            fallback_counts.append(
                int(getattr(output, "cycle_energy_fallback_count", 0))
            )
            loss = masked_listwise_loss(
                output.scores, tensors["labels"], tensors["candidate_mask"]
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{variant} smoke loss is non-finite")
        loss.backward()
        nonzero = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"{variant} invalid gradient: {name}")
            if bool((parameter.grad != 0).any().item()):
                nonzero.append(name)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip_norm"])
        )
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds.append(time.time() - started)
        losses.append(float(loss.detach().cpu()))
        gradient_names.append(nonzero)
    if "raw_residual_scale" not in gradient_names[0]:
        raise AssertionError(f"{variant} smoke did not open residual scale")
    second_prefixes = {name.split(".")[0] for name in gradient_names[1]}
    required = {"query_projection", "candidate_projection", "history_projection"}
    required |= (
        {"attention_projection", "value_projection"}
        if variant == "centered_cross_attention"
        else {"factor_a_projection", "factor_b_projection"}
    )
    if variant == "direct_learned":
        required.add("direct_gate_projection")
    if not required.issubset(second_prefixes):
        raise AssertionError(f"{variant} lacks second-step paths: {second_prefixes}")

    root = Path(config["paths"]["artifact_root"]) / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"{variant}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"immutable C06 smoke checkpoint exists: {variant}")
    torch.save({"model_state": model.state_dict()}, checkpoint_path)
    model.eval()
    with torch.inference_mode():
        before = _forward(model, tensors).scores.float().cpu().numpy()
    restored = _new_model(config, variant, device)
    restored.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state"],
        strict=True,
    )
    restored.eval()
    with torch.inference_mode():
        after = _forward(restored, tensors).scores.float().cpu().numpy()
    if not np.array_equal(before, after):
        raise AssertionError(f"{variant} smoke reload changed scores")
    result = {
        "variant": variant,
        "optimizer_steps": 2,
        "losses": losses,
        "nonzero_gradient_parameters_by_step": gradient_names,
        "optimizer_step_seconds": step_seconds,
        "candidate_cycle_energy_fallback_rows_by_step": fallback_counts,
        "candidate_cycle_energy_fallback_rows_total": sum(fallback_counts),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "reload_rescore_bitwise_equal": True,
        "candidate_rows": int(batch["candidate_mask"].sum()),
    }
    del model, restored, optimizer
    torch.cuda.empty_cache()
    return result


def run_smoke(
    config_path: str | Path, device: str, *, variant: str
) -> dict[str, Any]:
    started = time.time()
    config_path = Path(config_path)
    config = load_config(config_path)
    if variant not in TRAINED_VARIANTS:
        raise ValueError(f"invalid C06 smoke variant: {variant}")
    validate_execution_authority(
        config, stage="gpu_smoke", device=device, variant=variant
    )
    lock_hash = assert_real_gate_lock(config)
    candidate_manifest_hash = assert_candidate_manifest(config)
    g0_report, selection = _validate_g0(config, lock_hash)
    report_path = (
        Path(config["paths"]["artifact_root"])
        / "smoke"
        / f"gpu_smoke_{variant}.json"
    )
    if report_path.exists():
        raise FileExistsError("immutable C06 GPU smoke report already exists")
    data = StructuralTrainData.load(config["paths"]["packed_train_root"])
    role_hashes = _validate_role_candidate_hashes(data, selection, g0_report)
    features = FrozenRealFeatures(config, selection)
    fit_indices = [int(value) for value in selection["roles"]["fit"]["indices"]]
    batch_indices, batch_shape = _largest_fit_batch(data, fit_indices, config)
    batch = collate_structural(
        data, batch_indices, history_limit=int(config["model"]["max_history"])
    )
    tensors = features.tensors(batch, device, labels=features.fit_labels)
    capacity = _validate_capacity(config, device)
    variant_report = _smoke_variant(
        variant=variant,
        batch=batch,
        tensors=tensors,
        config=config,
        device=device,
    )
    report = {
        "candidate_id": "c06",
        "gate": "G1_GPU_smoke",
        "variant": variant,
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "real_gate_lock_sha256": lock_hash,
        "candidate_manifest_sha256": candidate_manifest_hash,
        "g0_report_sha256": sha256_file(
            Path(config["paths"]["artifact_root"]) / "g0_report.json"
        ),
        "selection_sha256": sha256_file(
            Path(config["paths"]["artifact_root"]) / "selection.json"
        ),
        "fit_only": True,
        "internal_A_features_scored": False,
        "internal_A_labels_opened": False,
        "internal_B_or_escrow_opened": False,
        "qrels_read": False,
        "dev_records_read": False,
        "test_read": False,
        "batch_selection": "maximum registered dominant work across both formal epoch orders",
        "batch_shape": batch_shape,
        "role_candidate_key_sha256": role_hashes,
        "capacity": capacity,
        "variant_report": variant_report,
        "elapsed_seconds": time.time() - started,
        "environment": {
            "name": config["environment"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "visible_gpu_name": torch.cuda.get_device_name(0),
            "physical_gpu": config["resources"]["variant_physical_gpus"][variant],
        },
        "git": _git_metadata(),
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def _validate_smoke(
    config: Mapping[str, Any], lock_hash: str, variant: str
) -> dict[str, Any]:
    path = (
        Path(config["paths"]["artifact_root"])
        / "smoke"
        / f"gpu_smoke_{variant}.json"
    )
    report = read_json(path)
    if report.get("status") != "passed" or report.get("gate") != "G1_GPU_smoke":
        raise ValueError("C06 GPU smoke has not passed")
    if report.get("real_gate_lock_sha256") != config["numeric_repair"][
        "parent_lock_sha256"
    ]:
        raise ValueError("C06 GPU smoke did not use the parent v1 lock")
    if report.get("variant") != variant:
        raise ValueError("C06 GPU smoke variant mismatch")
    if report.get("internal_A_labels_opened") or report.get("internal_B_or_escrow_opened"):
        raise ValueError("C06 GPU smoke violated role isolation")
    row = report["variant_report"]
    if sha256_file(row["checkpoint_path"]) != row["checkpoint_sha256"]:
        raise ValueError("C06 GPU smoke checkpoint changed")
    return report


def _begin_attempt(
    config: Mapping[str, Any],
    config_path: Path,
    lock_hash: str,
    *,
    kind: str,
    variant: str | None = None,
) -> dict[str, Any]:
    root = Path(config["paths"]["artifact_root"])
    if kind not in {"variant_numeric_repair", "audit"}:
        raise ValueError(f"unknown C06 attempt kind: {kind}")
    if kind == "variant_numeric_repair":
        eligible = tuple(config["numeric_repair"]["eligible_variants"])
        if variant not in eligible:
            raise PermissionError("variant is not eligible for numeric repair")
        suffix = f"repair1_{variant}"
        run_id = str(config["repair_variant_run_ids"][variant])
        physical_gpu = int(config["resources"]["variant_physical_gpus"][variant])
        parent_ledger_path = root / f"formal_attempt_{variant}.json"
        review_lock = read_json(config["paths"]["real_gate_lock"])
        expected_parent_ledger_hash = review_lock["failed_variant_evidence"][
            variant
        ]["ledger_sha256"]
        if sha256_file(parent_ledger_path) != expected_parent_ledger_hash:
            raise PermissionError("numeric-repair parent ledger changed after lock")
        parent_ledger = read_json(parent_ledger_path)
        parent_attempts = parent_ledger.get("attempts", [])
        if len(parent_attempts) != 1 or parent_attempts[0].get("stage") != "started":
            raise PermissionError("numeric repair requires one failed v1 attempt")
        if parent_attempts[0].get("internal_A_features_scored") is not False or parent_attempts[
            0
        ].get("internal_A_labels_opened") is not False:
            raise PermissionError("numeric repair parent attempt touched internal A")
        parent_ledger_sha256 = sha256_file(parent_ledger_path)
    else:
        suffix = "audit"
        run_id = str(config["audit_run_id"])
        physical_gpu = int(config["resources"]["audit_physical_gpu"])
    ledger_path = root / f"formal_attempt_{suffix}.json"
    if ledger_path.exists():
        raise RuntimeError(f"C06 one-shot attempt has already started: {suffix}")
    now = datetime.now(timezone.utc).isoformat()
    ledger = {
        "candidate_id": "c06",
        "gate_id": config["gate_id"],
        "attempt_kind": kind,
        "variant": variant,
        "attempts_max": 1,
        "formal_attempt_number": 2 if kind == "variant_numeric_repair" else 1,
        "parent_attempt_ledger_path": (
            str(parent_ledger_path) if kind == "variant_numeric_repair" else None
        ),
        "parent_attempt_ledger_sha256": (
            parent_ledger_sha256 if kind == "variant_numeric_repair" else None
        ),
        "attempts": [
            {
                "attempt": 1,
                "formal_attempt": 2 if kind == "variant_numeric_repair" else 1,
                "repair_attempt": 1 if kind == "variant_numeric_repair" else None,
                "run_id": run_id,
                "stage": "started",
                "started_at": now,
                "internal_A_features_scored": False,
                "internal_A_labels_opened": False,
                "internal_B_or_escrow_opened": False,
                "config_sha256": sha256_file(config_path),
                "real_gate_lock_sha256": lock_hash,
            }
        ],
    }
    write_json(ledger_path, ledger)
    metadata_path = Path("runs") / run_id / "metadata.json"
    write_json(
        metadata_path,
        {
            "candidate_id": "c06",
            "gate_id": config["gate_id"],
            "attempt_kind": kind,
            "variant": variant,
            "run_id": run_id,
            "attempt": 1,
            "formal_attempt": 2 if kind == "variant_numeric_repair" else 1,
            "repair_attempt": 1 if kind == "variant_numeric_repair" else None,
            "current_stage": "started",
            "created_at": now,
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "real_gate_lock_sha256": lock_hash,
            "environment": config["environment"],
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "qrels_read": False,
            "dev_records_read": False,
            "test_read": False,
            "internal_B_or_escrow_opened": False,
            "git": _git_metadata(),
        },
    )
    return {
        "run_id": run_id,
        "ledger_path": str(ledger_path),
        "metadata_path": str(metadata_path),
    }


def _update_attempt(context: Mapping[str, Any], stage: str, **fields: Any) -> None:
    now = datetime.now(timezone.utc).isoformat()
    ledger = read_json(context["ledger_path"])
    row = ledger["attempts"][0]
    row.update(fields)
    row["stage"] = stage
    row["updated_at"] = now
    write_json(context["ledger_path"], ledger)
    metadata = read_json(context["metadata_path"])
    metadata.update(fields)
    metadata["current_stage"] = stage
    metadata["updated_at"] = now
    write_json(context["metadata_path"], metadata)


def _record_pre_a_attempt_failure(
    context: Mapping[str, Any], error: BaseException
) -> None:
    """Persist a data-free failure state before propagating the exception."""

    _update_attempt(
        context,
        "failed",
        failure_scope="fit_only_before_any_internal_A_score_or_label",
        exception_type=type(error).__name__,
        exception_message=str(error)[:500],
        internal_A_features_scored=False,
        internal_A_labels_opened=False,
        internal_B_or_escrow_opened=False,
    )


def _initial_parity(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    data: StructuralTrainData,
    features: FrozenRealFeatures,
    indices: Sequence[int],
    device: str,
) -> dict[str, Any]:
    scores: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    fallback_rows = 0
    model.eval()
    with torch.inference_mode():
        for batch_indices in _batches(
            data, indices, config, seed=int(config["seed"]), shuffle=False
        ):
            batch = collate_structural(
                data, batch_indices, history_limit=int(config["model"]["max_history"])
            )
            tensors = features.tensors(batch, device)
            output = _forward(model, tensors)
            fallback_rows += int(
                getattr(output, "cycle_energy_fallback_count", 0)
            )
            for row in range(len(batch_indices)):
                count = int(batch["candidate_mask"][row].sum())
                scores.append(output.scores[row, :count].float().cpu().numpy().copy())
                bases.append(tensors["base_scores"][row, :count].cpu().numpy().copy())
    exact = all(np.array_equal(score, base) for score, base in zip(scores, bases))
    maximum = max(
        float(np.max(np.abs(score.astype(np.float64) - base.astype(np.float64))))
        for score, base in zip(scores, bases)
    )
    if not exact:
        raise AssertionError("untrained C06 variant is not exact D2p")
    return {
        "requests": len(scores),
        "bitwise_equal_to_d2p": True,
        "max_abs": maximum,
        "candidate_cycle_energy_fallback_rows": fallback_rows,
    }


def _train_variant(
    *,
    variant: str,
    config: Mapping[str, Any],
    data: StructuralTrainData,
    features: FrozenRealFeatures,
    fit_indices: Sequence[int],
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    seed_everything(int(config["seed"]))
    model = _new_model(config, variant, device)
    parity = _initial_parity(model, config, data, features, fit_indices[:256], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epoch_rows = []
    total_fallback_rows = 0
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses = []
        candidate_rows = 0
        optimizer_steps = 0
        epoch_fallback_rows = 0
        started = time.time()
        for batch_indices in _batches(
            data,
            fit_indices,
            config,
            seed=int(config["seed"]) + epoch,
            shuffle=True,
        ):
            batch = collate_structural(
                data, batch_indices, history_limit=int(config["model"]["max_history"])
            )
            tensors = features.tensors(batch, device, labels=features.fit_labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = _forward(model, tensors)
                epoch_fallback_rows += int(
                    getattr(output, "cycle_energy_fallback_count", 0)
                )
                loss = masked_listwise_loss(
                    output.scores, tensors["labels"], tensors["candidate_mask"]
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant} training loss is non-finite")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"{variant} invalid training gradient: {name}")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            candidate_rows += int(batch["candidate_mask"].sum())
            optimizer_steps += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "candidate_rows": candidate_rows,
                "mean_dynamic_batch_loss": float(np.mean(losses)),
                "elapsed_seconds": time.time() - started,
                "candidate_cycle_energy_fallback_rows": epoch_fallback_rows,
            }
        )
        total_fallback_rows += epoch_fallback_rows
    checkpoint_root = Path(config["paths"]["checkpoint_root"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / f"{variant}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"formal C06 checkpoint already exists: {checkpoint_path}")
    torch.save(
        {
            "candidate_id": "c06",
            "gate_id": config["gate_id"],
            "variant": variant,
            "numeric_repair_id": config["numeric_repair"]["repair_id"],
            "seed": int(config["seed"]),
            "fixed_epoch": 2,
            "candidate_cycle_energy_fallback_rows_total": total_fallback_rows,
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        checkpoint_path,
    )
    restored = _new_model(config, variant, device)
    restored.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state"],
        strict=True,
    )
    restored.eval()
    del model, optimizer
    return restored, {
        "variant": variant,
        "initial_parity": parity,
        "epochs": epoch_rows,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_selection": "fixed final epoch only",
        "parameter_count": restored.parameter_count(),
        "candidate_cycle_energy_fallback_rows_total": total_fallback_rows,
    }


def run_train_variant(
    config_path: str | Path, device: str, *, variant: str
) -> dict[str, Any]:
    """Run one independently bound, immutable two-epoch variant fit."""

    started = time.time()
    config_path = Path(config_path)
    config = load_config(config_path)
    eligible = tuple(config["numeric_repair"]["eligible_variants"])
    if variant not in eligible:
        raise PermissionError(
            "review1 permits repair only for local_hodge/untrusted/direct_learned; "
            "centered v1 is immutable"
        )
    validate_execution_authority(
        config, stage="train_variants", device=device, variant=variant
    )
    lock_hash = assert_real_gate_lock(config)
    candidate_manifest_hash = assert_candidate_manifest(config)
    g0_report, selection = _validate_g0(config, lock_hash)
    smoke_report = _validate_smoke(config, lock_hash, variant)
    artifact_root = Path(config["paths"]["artifact_root"])
    report_path = artifact_root / "training" / f"{variant}.json"
    if report_path.exists():
        raise FileExistsError(f"immutable C06 variant report exists: {variant}")
    data = StructuralTrainData.load(config["paths"]["packed_train_root"])
    role_hashes = _validate_role_candidate_hashes(data, selection, g0_report)
    features = FrozenRealFeatures(config, selection)
    fit_indices = [int(value) for value in selection["roles"]["fit"]["indices"]]
    context = _begin_attempt(
        config,
        config_path,
        lock_hash,
        kind="variant_numeric_repair",
        variant=variant,
    )
    try:
        model, training = _train_variant(
            variant=variant,
            config=config,
            data=data,
            features=features,
            fit_indices=fit_indices,
            device=device,
        )
    except BaseException as error:
        _record_pre_a_attempt_failure(context, error)
        raise
    del model
    report = {
        "candidate_id": "c06",
        "gate": "G2_variant_training",
        "gate_id": config["gate_id"],
        "variant": variant,
        "numeric_repair_id": config["numeric_repair"]["repair_id"],
        "formal_attempt": 2,
        "repair_attempt": 1,
        "parent_lock_sha256": config["numeric_repair"]["parent_lock_sha256"],
        "run_id": context["run_id"],
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "real_gate_lock_sha256": lock_hash,
        "candidate_manifest_sha256": candidate_manifest_hash,
        "g0_report_sha256": sha256_file(artifact_root / "g0_report.json"),
        "gpu_smoke_report_sha256": sha256_file(
            artifact_root / "smoke" / f"gpu_smoke_{variant}.json"
        ),
        "gpu_smoke_elapsed_seconds": smoke_report["elapsed_seconds"],
        "role_candidate_key_sha256": role_hashes,
        "training": training,
        "fit_requests": len(fit_indices),
        "fixed_epochs": 2,
        "candidate_cycle_energy_fallback_rows_total": training[
            "candidate_cycle_energy_fallback_rows_total"
        ],
        "internal_A_features_scored": False,
        "internal_A_labels_opened": False,
        "internal_B_or_escrow_opened": False,
        "qrels_read": False,
        "dev_records_read": False,
        "test_read": False,
        "elapsed_seconds": time.time() - started,
        "environment": {
            "name": config["environment"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_name": torch.cuda.get_device_name(0),
            "physical_gpu": config["resources"]["variant_physical_gpus"][variant],
        },
        "git": _git_metadata(),
    }
    try:
        write_json(report_path, report)
        _update_attempt(
            context,
            "completed",
            gate_status="passed",
            candidate_cycle_energy_fallback_rows_total=training[
                "candidate_cycle_energy_fallback_rows_total"
            ],
            report_path=str(report_path),
            report_sha256=sha256_file(report_path),
        )
    except BaseException as error:
        _record_pre_a_attempt_failure(context, error)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def _load_trained_variants(
    config: Mapping[str, Any],
    *,
    lock_hash: str,
    g0_report_hash: str,
    role_hashes: Mapping[str, str],
    device: str,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    artifact_root = Path(config["paths"]["artifact_root"])
    models: dict[str, torch.nn.Module] = {}
    reports: dict[str, Any] = {}
    repair = config["numeric_repair"]
    eligible = set(repair["eligible_variants"])
    for variant in TRAINED_VARIANTS:
        report_path = artifact_root / "training" / f"{variant}.json"
        report = read_json(report_path)
        if report.get("status") != "passed" or report.get("variant") != variant:
            raise ValueError(f"C06 variant training did not pass: {variant}")
        expected_lock_hash = (
            lock_hash if variant in eligible else repair["parent_lock_sha256"]
        )
        if report.get("real_gate_lock_sha256") != expected_lock_hash:
            raise ValueError(f"C06 variant used a different lock: {variant}")
        if variant == "centered_cross_attention":
            if sha256_file(report_path) != repair["centered_report_sha256"]:
                raise ValueError("preserved centered v1 report changed")
            if report.get("config_sha256") != repair["parent_config_sha256"]:
                raise ValueError("preserved centered v1 used a different config")
        else:
            if report.get("numeric_repair_id") != repair["repair_id"]:
                raise ValueError(f"C06 repaired report lacks repair ID: {variant}")
            if report.get("config_sha256") != sha256_file(
                CANDIDATE_ROOT / "configs" / "c06_real_mechanism_gate.yaml"
            ):
                raise ValueError(f"C06 repaired report used a different config: {variant}")
            if int(report.get("formal_attempt", -1)) != 2 or int(
                report.get("repair_attempt", -1)
            ) != 1:
                raise ValueError(f"C06 repaired attempt number changed: {variant}")
            if "candidate_cycle_energy_fallback_rows_total" not in report:
                raise ValueError(f"C06 repaired report lacks fallback count: {variant}")
            retry_ledger_path = (
                artifact_root / f"formal_attempt_repair1_{variant}.json"
            )
            retry_ledger = read_json(retry_ledger_path)
            retry_attempts = retry_ledger.get("attempts", [])
            if len(retry_attempts) != 1 or retry_attempts[0].get(
                "stage"
            ) != "completed":
                raise ValueError(f"C06 numeric repair is incomplete: {variant}")
            if retry_attempts[0].get("internal_A_features_scored") is not False or retry_attempts[
                0
            ].get("internal_A_labels_opened") is not False:
                raise ValueError(f"C06 numeric repair touched A: {variant}")
            if retry_attempts[0].get("real_gate_lock_sha256") != lock_hash:
                raise ValueError(f"C06 numeric repair used a different lock: {variant}")
        if report.get("g0_report_sha256") != g0_report_hash:
            raise ValueError(f"C06 variant used a different G0: {variant}")
        if report.get("role_candidate_key_sha256") != dict(role_hashes):
            raise ValueError(f"C06 variant used different candidate identities: {variant}")
        smoke_path = artifact_root / "smoke" / f"gpu_smoke_{variant}.json"
        if sha256_file(smoke_path) != report.get("gpu_smoke_report_sha256"):
            raise ValueError(f"C06 variant smoke provenance changed: {variant}")
        if report.get("internal_A_features_scored") or report.get(
            "internal_A_labels_opened"
        ):
            raise ValueError(f"C06 variant training touched A: {variant}")
        training = report["training"]
        checkpoint_path = Path(training["checkpoint_path"])
        if sha256_file(checkpoint_path) != training["checkpoint_sha256"]:
            raise ValueError(f"C06 formal checkpoint changed: {variant}")
        if variant == "centered_cross_attention" and training[
            "checkpoint_sha256"
        ] != repair["centered_checkpoint_sha256"]:
            raise ValueError("preserved centered v1 checkpoint changed")
        if len(training["epochs"]) != 2 or [
            int(row["epoch"]) for row in training["epochs"]
        ] != [1, 2]:
            raise ValueError(f"C06 variant epoch contract changed: {variant}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        if variant in eligible and checkpoint.get("numeric_repair_id") != repair[
            "repair_id"
        ]:
            raise ValueError(f"C06 repaired checkpoint lacks repair ID: {variant}")
        model = _new_model(config, variant, device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        models[variant] = model
        reports[variant] = {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            **training,
            "candidate_cycle_energy_fallback_rows_total": int(
                training.get("candidate_cycle_energy_fallback_rows_total", 0)
            ),
        }
    return models, reports


def _counterfactual_model(
    config: Mapping[str, Any],
    *,
    local_model: torch.nn.Module,
    variant: str,
    device: str,
) -> torch.nn.Module:
    model = _new_model(config, variant, device)
    model.load_state_dict(local_model.state_dict(), strict=True)
    model.eval()
    return model


def _score_role(
    *,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    data: StructuralTrainData,
    features: FrozenRealFeatures,
    indices: Sequence[int],
    device: str,
    collect_trust: bool,
) -> dict[str, Any]:
    scores: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    conservative: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    item_ids: list[np.ndarray] = []
    trust_min = np.inf
    trust_max = -np.inf
    trust_finite = True
    fallback_rows = 0
    model.eval()
    with torch.inference_mode():
        for batch_indices in _batches(
            data, indices, config, seed=int(config["seed"]), shuffle=False
        ):
            batch = collate_structural(
                data, batch_indices, history_limit=int(config["model"]["max_history"])
            )
            tensors = features.tensors(batch, device)
            output = _forward(model, tensors)
            fallback_rows += int(
                getattr(output, "cycle_energy_fallback_count", 0)
            )
            if collect_trust:
                active = tensors["candidate_mask"][:, :, None] & tensors["history_mask"][:, None, :]
                trust = output.candidate_event_trust[active]
                if trust.numel():
                    trust_finite &= bool(torch.isfinite(trust).all().item())
                    trust_min = min(trust_min, float(trust.min().cpu()))
                    trust_max = max(trust_max, float(trust.max().cpu()))
            for row in range(len(batch_indices)):
                count = int(batch["candidate_mask"][row].sum())
                scores.append(output.scores[row, :count].float().cpu().numpy().copy())
                deltas.append(output.applied_score_delta[row, :count].float().cpu().numpy().copy())
                conservative.append(
                    output.conservative_score_delta[row, :count].float().cpu().numpy().copy()
                )
                bases.append(tensors["base_scores"][row, :count].cpu().numpy().copy())
                item_ids.append(batch["candidate_item_ids"][row, :count].copy())
    result: dict[str, Any] = {
        "scores": scores,
        "deltas": deltas,
        "conservative": conservative,
        "base_scores": bases,
        "candidate_item_ids": item_ids,
        "candidate_cycle_energy_fallback_rows": fallback_rows,
    }
    if collect_trust:
        result["trust"] = {
            "finite": trust_finite,
            "minimum": float(trust_min),
            "maximum": float(trust_max),
        }
    return result


def _rankings(
    request_ids: Sequence[str], item_ids: Sequence[np.ndarray], scores: Sequence[np.ndarray]
) -> list[list[str]]:
    return [
        [
            row.item_id
            for row in sort_candidates(
                str(request_id),
                [
                    ScoredCandidate(str(item_id), float(score))
                    for item_id, score in zip(ids, values)
                ],
            )
        ]
        for request_id, ids, values in zip(request_ids, item_ids, scores)
    ]


def _exact_score_collection(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return all(
        np.array_equal(left, right)
        for left, right in zip(first["scores"], second["scores"])
    )


def _difference_fraction(
    first: Sequence[np.ndarray], second: Sequence[np.ndarray]
) -> float:
    return sum(not np.array_equal(left, right) for left, right in zip(first, second)) / len(first)


def _common_factor_contract(device: str) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(20260708)
    base_a = torch.randn(4, 1, 7, 32, generator=generator, device=device)
    base_b = torch.randn(4, 1, 7, 32, generator=generator, device=device)
    factor_a = base_a.expand(-1, 11, -1, -1).clone()
    factor_b = base_b.expand(-1, 11, -1, -1).clone()
    candidate_mask = torch.ones(4, 11, dtype=torch.bool, device=device)
    history_mask = torch.ones(4, 7, dtype=torch.bool, device=device)
    outputs = low_rank_hodge_calibration(
        factor_a, factor_b, candidate_mask, history_mask
    )
    potential, _, _, _, divergence, *_ = outputs
    maximum = max(float(potential.abs().max().cpu()), float(divergence.abs().max().cpu()))
    return {"exact_zero": maximum == 0.0, "max_abs": maximum}


def _pool_intervention_diagnostics(
    *,
    config: Mapping[str, Any],
    data: StructuralTrainData,
    features: FrozenRealFeatures,
    indices: Sequence[int],
    models: Mapping[str, torch.nn.Module],
    device: str,
) -> dict[str, Any]:
    """Run frozen label-free nested/duplicate/distractor diagnostics.

    These summaries have no outcome threshold for invariance.  Only finiteness,
    conservation, and the registered score bound remain binding contracts.
    """

    requested = int(config["a0_gate"]["pool_diagnostic_requests"])
    selected = list(indices[: min(requested, len(indices))])
    if len(selected) < 2:
        raise ValueError("pool diagnostics require at least two A requests")
    bound = float(config["model"]["score_delta_max"])
    tolerance = float(config["a0_gate"]["zero_sum_abs_tolerance"])
    accumulator = {
        name: {
            intervention: {"aligned_max_abs_changes": [], "max_abs_delta": 0.0, "max_abs_sum": 0.0, "finite": True}
            for intervention in ("nested_prefix_half", "duplicate_first", "cross_request_distractor")
        }
        for name in models
    }
    for position, index in enumerate(selected):
        partner = selected[(position + 1) % len(selected)]
        batch = collate_structural(
            data,
            np.asarray([index, partner], dtype=np.int64),
            history_limit=int(config["model"]["max_history"]),
        )
        full = features.tensors(batch, device)
        candidate_count = int(full["candidate_mask"][0].sum())
        partner_count = int(full["candidate_mask"][1].sum())
        history_columns = full["history"].shape[1]
        if candidate_count < 1 or partner_count < 1:
            raise ValueError("pool diagnostic request has no candidate")

        def make(candidate: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
            count = candidate.shape[1]
            return {
                "query": full["query"][0:1],
                "candidates": candidate,
                "history": full["history"][0:1, :history_columns],
                "candidate_mask": torch.ones(1, count, dtype=torch.bool, device=device),
                "history_mask": full["history_mask"][0:1, :history_columns],
                "history_prior": full["history_prior"][0:1, :history_columns],
                "base_scores": base.float(),
            }

        original_candidate = full["candidates"][0:1, :candidate_count]
        original_base = full["base_scores"][0:1, :candidate_count]
        nested_count = max(1, (candidate_count + 1) // 2)
        interventions = {
            "nested_prefix_half": make(
                original_candidate[:, :nested_count], original_base[:, :nested_count]
            ),
            "duplicate_first": make(
                torch.cat([original_candidate, original_candidate[:, :1]], dim=1),
                torch.cat([original_base, original_base[:, :1]], dim=1),
            ),
            # Base scores never enter a probe's evidence generator.  The
            # cross-request item's base is a declared label-free sentinel so
            # this diagnostic concerns only the conservative correction.
            "cross_request_distractor": make(
                torch.cat(
                    [original_candidate, full["candidates"][1:2, :1]], dim=1
                ),
                torch.cat(
                    [original_base, original_base.min(dim=1, keepdim=True).values - 1.0],
                    dim=1,
                ),
            ),
        }
        with torch.inference_mode():
            for name, model in models.items():
                original = _forward(model, make(original_candidate, original_base))
                original_delta = original.conservative_score_delta[0].float()
                for intervention, tensors in interventions.items():
                    output = _forward(model, tensors)
                    delta = output.conservative_score_delta[0].float()
                    aligned = min(candidate_count, len(delta))
                    row = accumulator[name][intervention]
                    row["aligned_max_abs_changes"].append(
                        float((delta[:aligned] - original_delta[:aligned]).abs().max().cpu())
                    )
                    row["max_abs_delta"] = max(
                        row["max_abs_delta"], float(delta.abs().max().cpu())
                    )
                    row["max_abs_sum"] = max(
                        row["max_abs_sum"], abs(float(delta.sum().cpu()))
                    )
                    row["finite"] &= bool(
                        torch.isfinite(output.scores).all().item()
                        and torch.isfinite(delta).all().item()
                    )
    summaries: dict[str, Any] = {}
    valid = True
    for name, rows in accumulator.items():
        summaries[name] = {}
        for intervention, row in rows.items():
            summary = {
                "requests": len(selected),
                "mean_aligned_max_abs_delta_change": float(
                    np.mean(row["aligned_max_abs_changes"])
                ),
                "maximum_aligned_max_abs_delta_change": float(
                    np.max(row["aligned_max_abs_changes"])
                ),
                "maximum_abs_delta": row["max_abs_delta"],
                "maximum_abs_candidate_sum_delta": row["max_abs_sum"],
                "finite": row["finite"],
                "within_bound": row["max_abs_delta"] <= bound + 1e-7,
                "zero_sum": row["max_abs_sum"] <= tolerance,
            }
            valid &= bool(
                summary["finite"] and summary["within_bound"] and summary["zero_sum"]
            )
            summaries[name][intervention] = summary
    return {
        "requests": len(selected),
        "selection": "first packed-order internal-A rows; frozen before labels",
        "binding_invariance_threshold": None,
        "distractor_base_policy": "per-request minimum D2p score minus one; base is outside evidence generator",
        "variants": summaries,
        "numeric_contracts_passed": valid,
    }


def _a0_audit(
    *,
    config: Mapping[str, Any],
    data: StructuralTrainData,
    features: FrozenRealFeatures,
    selection: Mapping[str, Any],
    models: Mapping[str, torch.nn.Module],
    device: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    a_indices = [int(value) for value in selection["roles"]["internal_A"]["indices"]]
    nohistory_indices = [int(value) for value in selection["roles"]["nohistory"]["indices"]]
    request_ids = [data.request_ids[index] for index in a_indices]
    first: dict[str, dict[str, Any]] = {}
    deterministic: dict[str, bool] = {}
    for name, model in models.items():
        collect_trust = name in {
            "local_hodge",
            "local_checkpoint_t_one",
            "local_checkpoint_global_hodge",
        }
        first[name] = _score_role(
            model=model,
            config=config,
            data=data,
            features=features,
            indices=a_indices,
            device=device,
            collect_trust=collect_trust,
        )
        second = _score_role(
            model=model,
            config=config,
            data=data,
            features=features,
            indices=a_indices,
            device=device,
            collect_trust=False,
        )
        deterministic[name] = _exact_score_collection(first[name], second)

    primary = first["local_hodge"]
    base_rankings = _rankings(request_ids, primary["candidate_item_ids"], primary["base_scores"])
    primary_rankings = _rankings(request_ids, primary["candidate_item_ids"], primary["scores"])
    order = order_change_summary(
        base_rankings=base_rankings, personalized_rankings=primary_rankings
    )
    bound = float(config["model"]["score_delta_max"])
    ranges = np.asarray(
        [float(values.max() - values.min()) for values in primary["conservative"]],
        dtype=np.float64,
    )
    range_fraction = float(
        (ranges > float(config["a0_gate"]["delta_range_fraction_of_bound_min"]) * bound).mean()
    )
    common_mode = [
        abs(float(values.mean())) / max(float(np.max(np.abs(values))), 1e-12)
        for values in primary["conservative"]
    ]
    candidate_sums = [abs(float(values.sum())) for values in primary["conservative"]]
    maximum_abs_delta = max(
        float(np.max(np.abs(values))) for values in primary["conservative"]
    )
    local_to_t_one_fraction = _difference_fraction(
        primary["conservative"], first["local_checkpoint_t_one"]["conservative"]
    )
    local_to_global_fraction = _difference_fraction(
        primary["conservative"], first["local_checkpoint_global_hodge"]["conservative"]
    )
    t_one_rankings = _rankings(
        request_ids,
        primary["candidate_item_ids"],
        first["local_checkpoint_t_one"]["scores"],
    )
    global_rankings = _rankings(
        request_ids,
        primary["candidate_item_ids"],
        first["local_checkpoint_global_hodge"]["scores"],
    )
    t_one_order = order_change_summary(
        base_rankings=primary_rankings, personalized_rankings=t_one_rankings
    )
    global_order = order_change_summary(
        base_rankings=primary_rankings, personalized_rankings=global_rankings
    )
    nohistory_checks = {}
    for name, model in models.items():
        rows = _score_role(
            model=model,
            config=config,
            data=data,
            features=features,
            indices=nohistory_indices,
            device=device,
            collect_trust=False,
        )
        nohistory_checks[name] = all(
            np.array_equal(score, base)
            for score, base in zip(rows["scores"], rows["base_scores"])
        )
    trust = primary["trust"]
    common_contract = _common_factor_contract(device)
    pool_diagnostics = _pool_intervention_diagnostics(
        config=config,
        data=data,
        features=features,
        indices=a_indices,
        models=models,
        device=device,
    )
    gate = config["a0_gate"]
    checks = {
        "all_deterministic_rescores_bitwise_equal": all(deterministic.values()),
        "common_mode_ratio_within_limit": max(common_mode) <= float(gate["common_mode_ratio_max"]),
        "conservative_delta_zero_sum": max(candidate_sums) <= float(gate["zero_sum_abs_tolerance"]),
        "conservative_delta_within_bound": maximum_abs_delta <= bound + 1e-7,
        "enough_nontrivial_delta_ranges": range_fraction >= float(gate["requests_with_delta_range_fraction_min"]),
        "enough_order_changes": order["requests_with_any_order_change_fraction"] >= float(gate["requests_with_any_order_change_fraction_min"]),
        "enough_top10_changes": order["requests_with_top10_membership_change_fraction"] >= float(gate["requests_with_top10_membership_change_fraction_min"]),
        "t_one_changes_deltas": local_to_t_one_fraction >= float(gate["counterfactual_delta_change_fraction_min"]),
        "t_one_changes_orders": t_one_order["requests_with_any_order_change_fraction"] >= float(gate["counterfactual_order_change_fraction_min"]),
        "global_changes_deltas": local_to_global_fraction >= float(gate["counterfactual_delta_change_fraction_min"]),
        "all_local_trust_finite_in_unit_interval": trust["finite"] and trust["minimum"] >= 0.0 and trust["maximum"] <= 1.0,
        "candidate_common_factors_exact_zero": common_contract["exact_zero"],
        "all_nohistory_scores_bitwise_base": all(nohistory_checks.values()),
        "all_scores_finite": all(
            np.isfinite(values).all()
            for rows in first.values()
            for values in rows["scores"]
        ),
        "pool_intervention_numeric_contracts": pool_diagnostics[
            "numeric_contracts_passed"
        ],
    }
    report = {
        "candidate_id": "c06",
        "gate": "G2_A0_label_free",
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requests": len(a_indices),
        "candidate_key_sha256": selected_candidate_key_sha256(data, a_indices),
        "full_candidate_sets": True,
        "candidate_sampling": False,
        "deterministic_rescore_bitwise_equal": deterministic,
        "candidate_cycle_energy_fallback_rows_first_rescore": {
            name: int(rows["candidate_cycle_energy_fallback_rows"])
            for name, rows in first.items()
        },
        "maximum_common_mode_ratio": max(common_mode),
        "maximum_abs_candidate_sum_delta": max(candidate_sums),
        "zero_sum_abs_tolerance": float(gate["zero_sum_abs_tolerance"]),
        "maximum_abs_conservative_delta": maximum_abs_delta,
        "score_delta_bound_plus_tolerance": bound + 1e-7,
        "delta_range_fraction_of_bound": range_fraction,
        "base_to_local_order_changes": order,
        "local_to_t_one": {
            "requests_with_different_delta_fraction": local_to_t_one_fraction,
            "order_changes": t_one_order,
        },
        "local_to_global_hodge": {
            "requests_with_different_delta_fraction": local_to_global_fraction,
            "order_changes": global_order,
        },
        "local_trust": trust,
        "candidate_common_factor_contract": common_contract,
        "pool_intervention_diagnostics": pool_diagnostics,
        "nohistory_bitwise_base": nohistory_checks,
        "checks": checks,
        "internal_A_labels_opened": False,
        "internal_B_or_escrow_opened": False,
        "qrels_read": False,
        "dev_records_read": False,
        "test_read": False,
    }
    return report, first


def _shared_ndcg_rows(
    *,
    request_ids: Sequence[str],
    item_ids: Sequence[np.ndarray],
    labels: SelectedLabels,
    batch_template: Sequence[int],
    scored: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    label_positions = labels.positions
    ndcg = {name: [] for name in ["d2p", *scored.keys()]}
    label_rows: list[np.ndarray] = []
    for row, (request_id, ids, request_index) in enumerate(
        zip(request_ids, item_ids, batch_template)
    ):
        position = label_positions[int(request_index)]
        start = int(labels.offsets[position])
        stop = int(labels.offsets[position + 1])
        label = np.asarray(labels.values[start:stop], dtype=np.float32)
        if len(label) != len(ids):
            raise ValueError("A label/candidate count mismatch")
        label_rows.append(label.copy())
        positives = {
            str(item_id) for item_id, value in zip(ids, label) if float(value) > 0.0
        }
        rows = {"d2p": scored["local_hodge"]["base_scores"][row]}
        rows.update({name: values["scores"][row] for name, values in scored.items()})
        for name, values in rows.items():
            metric = request_metrics(
                str(request_id),
                [
                    ScoredCandidate(str(item_id), float(score))
                    for item_id, score in zip(ids, values)
                ],
                positives,
                set(),
            )
            ndcg[name].append(float(metric["ndcg@10"]))
    return {
        name: np.asarray(values, dtype=np.float64) for name, values in ndcg.items()
    }, label_rows


def _a1_gate(
    *,
    config: Mapping[str, Any],
    data: StructuralTrainData,
    selection: Mapping[str, Any],
    scored: Mapping[str, Mapping[str, Any]],
    labels: SelectedLabels,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    indices = [int(value) for value in selection["roles"]["internal_A"]["indices"]]
    request_ids = [data.request_ids[index] for index in indices]
    primary = scored["local_hodge"]
    ndcg, label_rows = _shared_ndcg_rows(
        request_ids=request_ids,
        item_ids=primary["candidate_item_ids"],
        labels=labels,
        batch_template=indices,
        scored=scored,
    )
    references = {
        "d2p": ndcg["d2p"],
        "untrusted": ndcg["untrusted"],
        "direct_learned": ndcg["direct_learned"],
        "centered_cross_attention": ndcg["centered_cross_attention"],
        "local_checkpoint_global_hodge": ndcg["local_checkpoint_global_hodge"],
    }
    gate = config["a1_gate"]
    comparisons = compare_primary(
        request_ids=request_ids,
        primary=ndcg["local_hodge"],
        references=references,
        bootstrap_samples=int(gate["paired_bootstrap_samples"]),
        bootstrap_seed=int(gate["bootstrap_seed"]),
        folds=int(gate["hash_folds"]),
    )
    clicked_advantage = clicked_minus_unclicked(
        deltas=primary["deltas"], labels=label_rows
    )
    clicked_statistics = paired_bootstrap(
        clicked_advantage,
        samples=int(gate["paired_bootstrap_samples"]),
        seed=int(gate["bootstrap_seed"]) + 100,
    )
    base_row = comparisons["d2p"]
    controls = [name for name in references if name != "d2p"]
    checks = {
        "delta_over_d2p_at_least_threshold": base_row["mean"] >= float(gate["ndcg10_delta_over_d2p_min"]),
        "d2p_ci_low_strictly_positive": base_row["percentile_95_ci"][0] > 0.0,
        "d2p_all_hash_folds_positive": all(row["mean_difference"] > 0.0 for row in base_row["hash_folds"]),
        "delta_over_every_control_at_least_threshold": all(
            comparisons[name]["mean"] >= float(gate["ndcg10_delta_over_each_control_min"])
            for name in controls
        ),
        "every_control_ci_low_strictly_positive": all(
            comparisons[name]["percentile_95_ci"][0] > 0.0 for name in controls
        ),
        "clicked_minus_unclicked_ci_low_strictly_positive": clicked_statistics["percentile_95_ci"][0] > 0.0,
    }
    report = {
        "candidate_id": "c06",
        "gate": "G2_A1_shared_ranking",
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requests": len(indices),
        "metric": {
            "implementation": "src/myrec/eval/metrics.py::request_metrics",
            "source_sha256": sha256_file(REPO_ROOT / "src/myrec/eval/metrics.py"),
            "name": "request-equal NDCG@10",
            "tie_break": "shared evaluator tie_break_key",
        },
        "mean_ndcg@10": {name: float(values.mean()) for name, values in ndcg.items()},
        "paired_comparisons_primary_minus_reference": comparisons,
        "clicked_minus_unclicked_score_delta": clicked_statistics,
        "checks": checks,
        "internal_A_labels_opened_after_A0_pass": True,
        "internal_B_or_escrow_opened": False,
        "qrels_read": False,
        "dev_records_read": False,
        "test_read": False,
    }
    return report, ndcg


def run_audit(config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    config_path = Path(config_path)
    config = load_config(config_path)
    validate_execution_authority(config, stage="a0_a1_audit", device=device)
    lock_hash = assert_real_gate_lock(config)
    candidate_manifest_hash = assert_candidate_manifest(config)
    g0_report, selection = _validate_g0(config, lock_hash)
    artifact_root = Path(config["paths"]["artifact_root"])
    if (artifact_root / "real_gate_report.json").exists():
        raise FileExistsError("immutable C06 real-gate report already exists")
    data = StructuralTrainData.load(config["paths"]["packed_train_root"])
    role_hashes = _validate_role_candidate_hashes(data, selection, g0_report)
    features = FrozenRealFeatures(config, selection)
    a_indices = [int(value) for value in selection["roles"]["internal_A"]["indices"]]
    g0_report_hash = sha256_file(artifact_root / "g0_report.json")
    trained, training_rows = _load_trained_variants(
        config,
        lock_hash=lock_hash,
        g0_report_hash=g0_report_hash,
        role_hashes=role_hashes,
        device=device,
    )
    trained["local_checkpoint_t_one"] = _counterfactual_model(
        config,
        local_model=trained["local_hodge"],
        variant="local_checkpoint_t_one",
        device=device,
    )
    trained["local_checkpoint_global_hodge"] = _counterfactual_model(
        config,
        local_model=trained["local_hodge"],
        variant="local_checkpoint_global_hodge",
        device=device,
    )
    context = _begin_attempt(
        config, config_path, lock_hash, kind="audit"
    )
    _update_attempt(context, "fixed_epoch_checkpoints_reloaded")

    # A0 uses features and candidate identities only.  No label-opening helper
    # has been called for internal A at this point.
    a0, scored = _a0_audit(
        config=config,
        data=data,
        features=features,
        selection=selection,
        models=trained,
        device=device,
    )
    a0_path = Path(config["paths"]["artifact_root"]) / "a0_label_free_audit.json"
    write_json(a0_path, a0)
    _update_attempt(
        context,
        "a0_completed",
        internal_A_features_scored=True,
        internal_A_labels_opened=False,
        a0_status=a0["status"],
        a0_report_path=str(a0_path),
        a0_report_sha256=sha256_file(a0_path),
    )
    if a0["status"] != "passed":
        report = {
            "candidate_id": "c06",
            "gate": "G2_real_mechanism_gate",
            "run_id": context["run_id"],
            "status": "failed_A0",
            "decision": "stop before internal-A labels; close the C06 primitive",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(config_path),
            "real_gate_lock_sha256": lock_hash,
            "candidate_manifest_sha256": candidate_manifest_hash,
            "g0_report_sha256": g0_report_hash,
            "gpu_smoke_report_sha256": {
                variant: sha256_file(
                    artifact_root / "smoke" / f"gpu_smoke_{variant}.json"
                )
                for variant in TRAINED_VARIANTS
            },
            "role_candidate_key_sha256": role_hashes,
            "training": training_rows,
            "a0": a0,
            "a1": None,
            "internal_A_labels_opened": False,
            "internal_B_or_escrow_opened": False,
            "qrels_read": False,
            "dev_records_read": False,
            "test_read": False,
            "elapsed_seconds": time.time() - started,
        }
        report_path = artifact_root / "real_gate_report.json"
        write_json(report_path, report)
        _update_attempt(
            context,
            "completed",
            gate_status=report["status"],
            report_path=str(report_path),
            report_sha256=sha256_file(report_path),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return report

    # The A0 report is durable and hash-registered before this first A-label read.
    durable_a0_hash = assert_internal_a_opening_barrier(
        a0_path,
        expected_candidate_key_sha256=role_hashes["internal_A"],
    )
    label_source = g0_report["train_label_source_verified_after_selection"]
    label_source_hash_at_a1 = sha256_file(config["paths"]["train_candidate_labels"])
    if label_source_hash_at_a1 != label_source["sha256"]:
        raise ValueError("train-label source changed between G0 and A1 opening")
    a_labels = open_selected_labels(
        data,
        a_indices,
        label_path=config["paths"]["train_candidate_labels"],
        allowed_indices=set(a_indices),
    )
    _update_attempt(
        context,
        "a1_labels_opened",
        internal_A_features_scored=True,
        internal_A_labels_opened=True,
        a0_report_sha256=durable_a0_hash,
    )
    a1, ndcg = _a1_gate(
        config=config,
        data=data,
        selection=selection,
        scored=scored,
        labels=a_labels,
    )
    per_request_path = artifact_root / "a1_shared_metric_per_request.npz"
    np.savez_compressed(
        per_request_path,
        request_indices=np.asarray(a_indices, dtype=np.int64),
        **{f"ndcg10_{name}": values for name, values in ndcg.items()},
    )
    report = {
        "candidate_id": "c06",
        "gate": "G2_real_mechanism_gate",
        "run_id": context["run_id"],
        "status": "passed" if a1["status"] == "passed" else "failed_A1",
        "decision": (
            "eligible only for a separately reviewed delayed-B lock"
            if a1["status"] == "passed"
            else "stop; close C06 and do not open delayed B or escrow"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "real_gate_lock_sha256": lock_hash,
        "candidate_manifest_sha256": candidate_manifest_hash,
        "g0_report_sha256": g0_report_hash,
        "gpu_smoke_report_sha256": {
            variant: sha256_file(
                artifact_root / "smoke" / f"gpu_smoke_{variant}.json"
            )
            for variant in TRAINED_VARIANTS
        },
        "role_candidate_key_sha256": role_hashes,
        "train_label_source_sha256_reverified_after_A0": label_source_hash_at_a1,
        "training": training_rows,
        "same_checkpoint_counterfactuals": list(COUNTERFACTUAL_VARIANTS),
        "a0": a0,
        "a0_report_path": str(a0_path),
        "a0_report_sha256_before_labels": durable_a0_hash,
        "a1": a1,
        "a1_per_request_path": str(per_request_path),
        "a1_per_request_sha256": sha256_file(per_request_path),
        "internal_A_labels_opened": True,
        "internal_B_or_escrow_opened": False,
        "delayed_B_authorized": False,
        "escrow_authorized": False,
        "primary_dev_evaluator_calls": 0,
        "qrels_read": False,
        "dev_records_read": False,
        "test_read": False,
        "elapsed_seconds": time.time() - started,
        "environment": {
            "name": config["environment"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_name": torch.cuda.get_device_name(0),
            "physical_gpu": config["resources"]["audit_physical_gpu"],
        },
        "git": _git_metadata(),
    }
    report_path = artifact_root / "real_gate_report.json"
    write_json(report_path, report)
    _update_attempt(
        context,
        "completed",
        gate_status=report["status"],
        internal_A_labels_opened=True,
        report_path=str(report_path),
        report_sha256=sha256_file(report_path),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode", choices=("smoke", "train-variant", "audit"), required=True
    )
    parser.add_argument("--variant", choices=TRAINED_VARIANTS)
    arguments = parser.parse_args()
    if arguments.mode == "smoke":
        if arguments.variant is None:
            parser.error("--mode smoke requires --variant")
        run_smoke(arguments.config, arguments.device, variant=arguments.variant)
    elif arguments.mode == "train-variant":
        if arguments.variant is None:
            parser.error("--mode train-variant requires --variant")
        run_train_variant(
            arguments.config, arguments.device, variant=arguments.variant
        )
    else:
        if arguments.variant is not None:
            parser.error("--mode audit does not accept --variant")
        run_audit(arguments.config, arguments.device)


if __name__ == "__main__":
    main()
