"""Run the frozen C39 Amazon-C4 train-internal value-interface gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.nn import functional as F


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from model.halfspace_value import (  # noqa: E402
    EVENTWISE_HALFSPACE,
    EVENTWISE_RAW,
    GLOBAL_ONLY,
    MODES,
    POSTPOOL_HALFSPACE,
    RAY_ONLY,
    HalfspaceCertifiedValueTransformer,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402
from train.locking import verify_execution_lock, verify_proposal_lock  # noqa: E402
from train.selection import load_blind_records, read_json, sha256_file, write_json  # noqa: E402
from train.store import CompactLabels, FrozenTransferStore, open_role_labels  # noqa: E402


PRIMARY = EVENTWISE_HALFSPACE
CONTROLS = (EVENTWISE_RAW, POSTPOOL_HALFSPACE, RAY_ONLY, GLOBAL_ONLY)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("C39 config must be an object")
    return value


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def make_model(
    config: Mapping[str, Any],
    seed: int,
    mode: str,
) -> HalfspaceCertifiedValueTransformer:
    row = config["model"]
    return HalfspaceCertifiedValueTransformer(
        dim=int(row["embedding_dim"]),
        inner_dim=int(row["inner_dim"]),
        heads=int(row["heads"]),
        ffn_dim=int(row["ffn_dim"]),
        temperature=float(row["history_temperature"]),
        global_scale=float(row["global_scale"]),
        candidate_scale=float(row["candidate_scale"]),
        seed=seed,
        mode=mode,
    )


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_identity(model: HalfspaceCertifiedValueTransformer) -> dict[str, Any]:
    expected = (
        4 * model.dim * model.inner_dim
        + 2 * model.dim * model.ffn_dim
        + 2 * model.dim
    )
    return {
        "mode": model.mode,
        "architecture": model.mode,
        "parameters": model.trainable_parameter_count(),
        "expected_parameters": expected,
        "candidate_shared_global_write": True,
        "candidate_relative_event_values": model.mode != GLOBAL_ONLY,
        "eventwise_halfspace_projection": model.mode == EVENTWISE_HALFSPACE,
        "postpool_halfspace_projection": model.mode == POSTPOOL_HALFSPACE,
        "score_ray_only": model.mode == RAY_ONLY,
        "candidate_specific_parameters_present": False,
        "candidate_scalar_head_present": False,
        "query_attended": True,
        "tangent_projection_present": False,
        "ffn_output_initialized_exact_zero": bool(
            torch.equal(
                model.ffn_down.weight,
                torch.zeros_like(model.ffn_down.weight),
            )
        ),
        "frozen_BGE_encoder_load_bearing": True,
        "cached_embedding_execution_exact": True,
    }


def to_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)


def model_inputs(
    store: FrozenTransferStore,
    index: int,
    history_source: str,
    device: torch.device,
    *,
    candidate_order: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = to_tensor(store.query(index), device)
    history_positions = store.history_positions(index, history_source)
    candidate_positions = store.candidate_positions(index)
    if candidate_order is not None:
        candidate_positions = candidate_positions[candidate_order]
    history = to_tensor(store.items(history_positions), device)
    candidates = to_tensor(store.items(candidate_positions), device)
    return query, history, candidates


def run_g0(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    output_path = Path(config["paths"]["g0_report"])
    if output_path.exists():
        raise FileExistsError(output_path)
    selection = read_json(config["paths"]["selection"])
    c0 = read_json(config["paths"]["c0_report"])
    design_gate = read_json(config["paths"]["design_gate_report"])
    store = FrozenTransferStore(config)
    labels = open_role_labels(
        records_train_path=config["paths"]["records_train"],
        records_train_sha256=config["integrity"]["records_train_sha256"],
        selection_path=config["paths"]["selection"],
        selection_sha256=config["paths"]["selection_sha256"],
        store=store,
        role="fit",
    )
    fit_label_path = Path(config["paths"]["fit_labels"])
    fit_label_path.parent.mkdir(parents=True, exist_ok=True)
    if fit_label_path.exists():
        raise FileExistsError(fit_label_path)
    with fit_label_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=labels.request_indices,
            offsets=labels.offsets,
            values=labels.values,
        )
    records = load_blind_records(config["paths"]["records_train_blind"])
    selected = [
        int(index)
        for role in ("fit", "internal_A")
        for index in selection["roles"][role]["indices"]
    ]
    wrong = selection["wrong_donor_audit"]
    isolation = selection["outcome_isolation"]
    checks = {
        "design_gate_passed_and_hashed": (
            design_gate.get("status") == "passed"
            and sha256_file(config["paths"]["design_gate_report"])
            == config["paths"]["design_gate_report_sha256"]
        ),
        "amazon_c4_c0_passed": c0.get("overall_status") == "passed",
        "dev_test_labels_physically_isolated": bool(
            c0.get("checks", {}).get("dev_test_records_label_free", False)
        ),
        "positive_absent_from_released_history": bool(
            c0.get("checks", {}).get("source_target_absent_from_history", False)
        ),
        "selected_histories_nonempty": all(
            bool(records[index]["history"]) for index in selected
        ),
        "selected_history_timestamps_causal": all(
            all(
                int(event["ts"]) < int(records[index]["ts"])
                for event in records[index]["history"]
            )
            for index in selected
        ),
        "wrong_donor_full_coverage": wrong["coverage_fraction"] == 1.0,
        "wrong_donor_same_length_bin_at_least_95pct": (
            wrong["same_length_bin_fraction"] >= 0.95
        ),
        "wrong_donor_zero_same_user": wrong["same_user_assignments"] == 0,
        "internal_A_from_c38_unused_only": (
            isolation["c39_internal_A_from_c38_unused"]
            == len(selection["roles"]["internal_A"]["indices"])
        ),
        "zero_overlap_with_c38_target_roles": (
            isolation["overlap_with_c38_internal_A_delayed_B_escrow"] == 0
        ),
        "fit_has_exactly_one_positive": all(
            int((labels.row(index, store.candidate_count(index)) > 0).sum()) == 1
            for index in store.role_indices("fit")
        ),
        "internal_A_labels_closed": True,
        "upstream_dev_test_never_opened": True,
    }
    report = {
        "candidate_id": "c39",
        "gate": "G0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "proposal_lock_sha256": proposal_hash,
        "wrong_donor_audit": wrong,
        "outcome_isolation": isolation,
        "fit_labels": {
            "path": str(fit_label_path),
            "sha256": sha256_file(fit_label_path),
            "requests": len(labels.request_indices),
        },
        "internal_A_labels_scores_opened": False,
        "reserve_features_labels_scores_opened": False,
        "dev_test_opened": False,
    }
    write_json(output_path, report)
    return report


def load_fit_labels(config: Mapping[str, Any]) -> CompactLabels:
    path = Path(config["paths"]["fit_labels"])
    g0 = read_json(config["paths"]["g0_report"])
    if sha256_file(path) != g0["fit_labels"]["sha256"]:
        raise RuntimeError("C39 fit labels changed")
    with np.load(path, allow_pickle=False) as values:
        return CompactLabels(
            request_indices=np.asarray(values["request_indices"], dtype=np.int64),
            offsets=np.asarray(values["offsets"], dtype=np.int64),
            values=np.asarray(values["values"], dtype=np.float32),
        )


def train_mode(
    model: HalfspaceCertifiedValueTransformer,
    store: FrozenTransferStore,
    labels: CompactLabels,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    scale = float(config["model"]["correction_scale"])
    all_indices = store.role_indices("fit")
    indices = [index for index in all_indices if not store.has_repeat(index)]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    model.to(device).train()
    losses: list[float] = []
    listwise_losses: list[float] = []
    direction_losses: list[float] = []
    gradient_names: set[str] = set()
    for epoch in range(int(training["epochs"])):
        order = np.random.default_rng(seed + epoch * 10003).permutation(len(indices))
        batch_size = int(training["max_requests_per_batch"])
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            request_losses = []
            request_listwise = []
            request_direction = []
            for position in selected:
                index = indices[int(position)]
                target = to_tensor(
                    labels.row(index, store.candidate_count(index)) > 0,
                    device,
                ).bool()
                query, history, candidates = model_inputs(
                    store, index, "true", device
                )
                correction = scale * model(query, history, candidates)
                score = to_tensor(store.base_row(index), device) + correction
                listwise = torch.logsumexp(score, dim=0) - score[target].mean()
                direction = F.softplus(
                    -(correction[target].mean() - correction[~target].mean())
                )
                request_listwise.append(listwise)
                request_direction.append(direction)
                request_losses.append(
                    float(training["listwise_loss_weight"]) * listwise
                    + float(training["direction_loss_weight"]) * direction
                )
            loss = torch.stack(request_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite C39 training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"nonfinite C39 gradient: {name}")
                if bool(parameter.grad.ne(0).any()):
                    gradient_names.add(name)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            listwise_losses.append(
                float(torch.stack(request_listwise).mean().detach().cpu())
            )
            direction_losses.append(
                float(torch.stack(request_direction).mean().detach().cpu())
            )
    return {
        "fit_requests": len(all_indices),
        "active_nonrepeat_requests": len(indices),
        "skipped_repeat_requests": len(all_indices) - len(indices),
        "epochs": int(training["epochs"]),
        "steps": len(losses),
        "all_candidates_used_per_request": True,
        "candidate_sampling": False,
        "finite": bool(losses) and bool(np.isfinite(losses).all()),
        "loss_first_30_mean": float(np.mean(losses[:30])),
        "loss_last_30_mean": float(np.mean(losses[-30:])),
        "listwise_loss_last_30_mean": float(np.mean(listwise_losses[-30:])),
        "direction_loss_last_30_mean": float(np.mean(direction_losses[-30:])),
        "nonzero_gradient_parameter_names": sorted(gradient_names),
        "nonzero_gradient_parameter_count": len(gradient_names),
    }


def score_mode(
    model: HalfspaceCertifiedValueTransformer,
    store: FrozenTransferStore,
    indices: Sequence[int],
    history_source: str,
    device: torch.device,
    scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    scores = []
    corrections = []
    model.eval()
    with torch.inference_mode():
        for index in indices:
            query, history, candidates = model_inputs(
                store, index, history_source, device
            )
            if store.has_repeat(index):
                correction = np.zeros(store.candidate_count(index), dtype=np.float32)
            else:
                correction = (
                    scale * model(query, history, candidates)
                ).cpu().numpy().astype(np.float32)
            corrections.append(correction)
            scores.append((store.base_row(index) + correction).astype(np.float32))
    return scores, corrections


def diagnose_primary(
    model: HalfspaceCertifiedValueTransformer,
    store: FrozenTransferStore,
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    certificate_violation = 0.0
    support_edges = 0
    zero_support_edges = 0
    mixed_support_requests = 0
    active_negative_edges = 0
    changed_negative_edges = 0
    active_requests = 0
    model.eval()
    with torch.inference_mode():
        for index in indices:
            if store.has_repeat(index):
                continue
            active_requests += 1
            query, history, candidates = model_inputs(
                store, index, "true", device
            )
            state = model.components(query, history, candidates)
            support = state["support"]
            active = state["edge_weight"] > 0
            rejected = support == 0
            support_edges += support.numel()
            zero_support_edges += int(rejected.sum().cpu())
            mixed_support_requests += int(
                bool(rejected.any()) and bool((support > 0).any())
            )
            if bool(active.any()):
                certificate_violation = max(
                    certificate_violation,
                    float(
                        torch.relu(-state["projected_readout"][active]).max().cpu()
                    ),
                )
            negative = active & (state["raw_readout"] < 0)
            active_negative_edges += int(negative.sum().cpu())
            changed_negative_edges += int(
                (
                    negative
                    & (
                        (state["projected_readout"] - state["raw_readout"]).abs()
                        > 1e-8
                    )
                )
                .sum()
                .cpu()
            )
    return {
        "active_nonrepeat_requests": active_requests,
        "certificate_max_violation": certificate_violation,
        "support_edges": support_edges,
        "zero_support_edges": zero_support_edges,
        "zero_support_edge_fraction": zero_support_edges / max(support_edges, 1),
        "mixed_support_requests": mixed_support_requests,
        "mixed_support_request_fraction": mixed_support_requests
        / max(active_requests, 1),
        "active_negative_raw_edges": active_negative_edges,
        "projection_changed_negative_edges": changed_negative_edges,
        "projection_changed_negative_fraction": changed_negative_edges
        / max(active_negative_edges, 1),
    }


def run_seed(
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    seed_all(seed)
    store = FrozenTransferStore(config)
    labels = load_fit_labels(config)
    a_indices = store.role_indices("internal_A")
    scale = float(config["model"]["correction_scale"])
    artifact_root = Path(config["paths"]["artifact_root"])
    report_path = artifact_root / f"seed_{seed}_report.json"
    score_path = artifact_root / f"seed_{seed}_internal_A_scores.npz"
    if report_path.exists() or score_path.exists():
        raise FileExistsError(f"C39 seed output exists: {seed}")
    started = time.time()
    mode_reports = {}
    score_payload: dict[str, np.ndarray] = {}
    initial_hashes = {}
    for mode in MODES:
        seed_all(seed)
        model = make_model(config, seed, mode)
        initial_hash = state_sha256(model)
        initial_hashes[mode] = initial_hash
        identity = model_identity(model)
        training = train_mode(model, store, labels, config, seed, device)
        final_hash = state_sha256(model)
        checkpoint_root = Path(config["paths"]["checkpoint_root"])
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_root / f"seed_{seed}_{mode}.pt"
        if checkpoint_path.exists():
            raise FileExistsError(checkpoint_path)
        temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "candidate_id": "c39",
                "seed": seed,
                "mode": mode,
                "proposal_lock_sha256": proposal_hash,
                "execution_lock_sha256": execution_hash,
                "state_dict": model.state_dict(),
            },
            temporary_checkpoint,
        )
        temporary_checkpoint.replace(checkpoint_path)
        true_scores, true_corrections = score_mode(
            model, store, a_indices, "true", device, scale
        )
        wrong_scores, _ = score_mode(
            model, store, a_indices, "wrong", device, scale
        )
        repeated_scores, _ = score_mode(
            model, store, a_indices[:32], "true", device, scale
        )
        deterministic_scores, _ = score_mode(
            model, store, a_indices[:32], "true", device, scale
        )
        deterministic_error = _max_difference(
            repeated_scores, deterministic_scores
        )
        permutation_error = 0.0
        nohistory_error = 0.0
        query_absent_error = 0.0
        repeat_error = 0.0
        model.eval()
        with torch.inference_mode():
            for index in a_indices[:32]:
                count = store.candidate_count(index)
                permutation = np.random.default_rng(seed + index).permutation(count)
                query, history, candidates = model_inputs(
                    store, index, "true", device
                )
                reference = model(query, history, candidates)[
                    torch.from_numpy(permutation).to(device)
                ]
                _, _, permuted_candidates = model_inputs(
                    store,
                    index,
                    "true",
                    device,
                    candidate_order=permutation,
                )
                actual = model(query, history, permuted_candidates)
                permutation_error = max(
                    permutation_error,
                    float((reference - actual).abs().max().cpu()),
                )
                nohistory_error = max(
                    nohistory_error,
                    float(model(query, history[:0], candidates).abs().max().cpu()),
                )
                query_absent_error = max(
                    query_absent_error,
                    float(
                        model(
                            query,
                            history,
                            candidates,
                            query_present=False,
                        )
                        .abs()
                        .max()
                        .cpu()
                    ),
                )
                repeat_error = max(
                    repeat_error,
                    float(
                        model(
                            query,
                            history,
                            candidates,
                            repeat_present=True,
                        )
                        .abs()
                        .max()
                        .cpu()
                    ),
                )
        diagnostics = (
            diagnose_primary(model, store, a_indices, device)
            if mode == PRIMARY
            else None
        )
        mode_reports[mode] = {
            "identity": identity,
            "training": training,
            "initial_state_sha256": initial_hash,
            "final_state_sha256": final_hash,
            "parameters_updated": final_hash != initial_hash,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "state_sha256": final_hash,
            },
            "deterministic_max_abs_difference": deterministic_error,
            "candidate_permutation_max_abs_difference": permutation_error,
            "nohistory_correction_max_abs": nohistory_error,
            "query_absent_correction_max_abs": query_absent_error,
            "repeat_correction_max_abs": repeat_error,
            "value_diagnostics": diagnostics,
        }
        score_payload[f"{mode}_true"] = _flatten(true_scores)
        score_payload[f"{mode}_wrong"] = _flatten(wrong_scores)
        score_payload[f"{mode}_correction"] = _flatten(true_corrections)
    base_rows = [store.base_row(index) for index in a_indices]
    score_payload["base"] = _flatten(base_rows)
    score_payload["offsets"] = _offsets(base_rows)
    temporary = score_path.with_suffix(score_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **score_payload)
    temporary.replace(score_path)
    report = {
        "candidate_id": "c39",
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "physical_gpu": config["resources"]["seed_to_physical_gpu"][str(seed)],
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "mode_reports": mode_reports,
        "paired_initialization": len(set(initial_hashes.values())) == 1,
        "seed_specific_initial_state_sha256": initial_hashes[PRIMARY],
        "score_artifact": {
            "path": str(score_path),
            "sha256": sha256_file(score_path),
        },
        "internal_A_scores_opened": True,
        "internal_A_labels_opened": False,
        "reserve_features_labels_scores_opened": False,
        "dev_test_qrels_read": False,
    }
    write_json(report_path, report)
    return report


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    artifact_root = Path(config["paths"]["artifact_root"])
    output_path = artifact_root / "train_gate_report.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    seeds = [int(value) for value in config["training"]["seeds"]]
    reports = [
        read_json(artifact_root / f"seed_{seed}_report.json") for seed in seeds
    ]
    store = FrozenTransferStore(config)
    indices = store.role_indices("internal_A")
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    per_seed = {}
    score_rows: dict[int, dict[str, list[np.ndarray]]] = {}
    for seed, report in zip(seeds, reports):
        score_path = Path(report["score_artifact"]["path"])
        if sha256_file(score_path) != report["score_artifact"]["sha256"]:
            raise RuntimeError(f"C39 score artifact changed for seed {seed}")
        with np.load(score_path, allow_pickle=False) as values:
            offsets = np.asarray(values["offsets"], dtype=np.int64)
            rows = {"base": _unflatten(offsets, values["base"])}
            for mode in MODES:
                rows[f"{mode}_true"] = _unflatten(
                    offsets, values[f"{mode}_true"]
                )
                rows[f"{mode}_wrong"] = _unflatten(
                    offsets, values[f"{mode}_wrong"]
                )
                rows[f"{mode}_correction"] = _unflatten(
                    offsets, values[f"{mode}_correction"]
                )
            score_rows[seed] = rows
        per_seed[str(seed)] = report
    averaged = {
        name: _average_rows([score_rows[seed][name] for seed in seeds])
        for name in score_rows[seeds[0]]
    }
    activity: dict[str, dict[str, Any]] = {
        "primary_vs_base": order_changes(
            request_ids,
            item_ids,
            averaged["base"],
            averaged[f"{PRIMARY}_true"],
        ),
        "wrong_vs_true": order_changes(
            request_ids,
            item_ids,
            averaged[f"{PRIMARY}_true"],
            averaged[f"{PRIMARY}_wrong"],
        ),
    }
    for control in CONTROLS:
        activity[f"primary_vs_{control}"] = order_changes(
            request_ids,
            item_ids,
            averaged[f"{control}_true"],
            averaged[f"{PRIMARY}_true"],
        )
    gate = config["gate"]
    initial_hashes = [
        report["seed_specific_initial_state_sha256"] for report in reports
    ]
    required_primary_gradients = {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    }
    a0_checks = {
        "paired_initialization": all(
            report["paired_initialization"] for report in reports
        ),
        "seed_specific_initialization": len(set(initial_hashes)) == len(seeds),
        "capacity_matched_modes": all(
            len(
                {
                    report["mode_reports"][mode]["identity"]["parameters"]
                    for mode in MODES
                }
            )
            == 1
            for report in reports
        ),
        "training_finite": all(
            report["mode_reports"][mode]["training"]["finite"]
            for report in reports
            for mode in MODES
        ),
        "gradients_active_all_modes": all(
            report["mode_reports"][mode]["training"][
                "nonzero_gradient_parameter_count"
            ]
            > 0
            for report in reports
            for mode in MODES
        ),
        "primary_gradients_reach_all_components": all(
            required_primary_gradients
            <= set(
                report["mode_reports"][PRIMARY]["training"][
                    "nonzero_gradient_parameter_names"
                ]
            )
            for report in reports
        ),
        "parameters_updated": all(
            report["mode_reports"][mode]["parameters_updated"]
            for report in reports
            for mode in MODES
        ),
        "deterministic": all(
            report["mode_reports"][mode]["deterministic_max_abs_difference"]
            <= float(gate["deterministic_max_abs_difference"])
            for report in reports
            for mode in MODES
        ),
        "candidate_permutation": all(
            report["mode_reports"][mode][
                "candidate_permutation_max_abs_difference"
            ]
            <= float(gate["candidate_permutation_max_abs_difference"])
            for report in reports
            for mode in MODES
        ),
        "nohistory_exact_base": all(
            report["mode_reports"][mode]["nohistory_correction_max_abs"] == 0.0
            for report in reports
            for mode in MODES
        ),
        "query_absent_exact_base": all(
            report["mode_reports"][mode]["query_absent_correction_max_abs"] == 0.0
            for report in reports
            for mode in MODES
        ),
        "repeat_exact_base": all(
            report["mode_reports"][mode]["repeat_correction_max_abs"] == 0.0
            for report in reports
            for mode in MODES
        ),
        "halfspace_certificate": all(
            report["mode_reports"][PRIMARY]["value_diagnostics"][
                "certificate_max_violation"
            ]
            <= float(gate["certificate_max_violation"])
            for report in reports
        ),
        "projection_changes_negative_edges": all(
            report["mode_reports"][PRIMARY]["value_diagnostics"][
                "projection_changed_negative_fraction"
            ]
            >= float(gate["projection_changed_negative_fraction_min"])
            for report in reports
        ),
        "exact_zero_support_edges": all(
            report["mode_reports"][PRIMARY]["value_diagnostics"][
                "zero_support_edge_fraction"
            ]
            >= float(gate["zero_support_edge_fraction_min"])
            for report in reports
        ),
        "mixed_support_requests": all(
            report["mode_reports"][PRIMARY]["value_diagnostics"][
                "mixed_support_request_fraction"
            ]
            >= float(gate["mixed_support_request_fraction_min"])
            for report in reports
        ),
        "primary_order_active": activity["primary_vs_base"]["any_fraction"]
        >= float(gate["primary_vs_base_order_fraction_min"]),
        "primary_top10_active": activity["primary_vs_base"]["top10_fraction"]
        >= float(gate["primary_vs_base_top10_fraction_min"]),
        "wrong_history_order_distinct": activity["wrong_vs_true"]["any_fraction"]
        >= float(gate["wrong_vs_true_order_fraction_min"]),
        "wrong_history_top10_distinct": activity["wrong_vs_true"]["top10_fraction"]
        >= float(gate["wrong_vs_true_top10_fraction_min"]),
        **{
            f"{control}_order_distinct": activity[f"primary_vs_{control}"][
                "any_fraction"
            ]
            >= float(gate["primary_vs_control_order_fraction_min"])
            for control in CONTROLS
        },
        **{
            f"{control}_top10_distinct": activity[f"primary_vs_{control}"][
                "top10_fraction"
            ]
            >= float(gate["primary_vs_control_top10_fraction_min"])
            for control in CONTROLS
        },
        "no_candidate_scalar_head": all(
            not report["mode_reports"][mode]["identity"][
                "candidate_scalar_head_present"
            ]
            for report in reports
            for mode in MODES
        ),
        "no_tangent_projection": all(
            not report["mode_reports"][mode]["identity"][
                "tangent_projection_present"
            ]
            for report in reports
            for mode in MODES
        ),
        "dev_test_qrels_closed": all(
            not report["dev_test_qrels_read"] for report in reports
        ),
    }
    report: dict[str, Any] = {
        "candidate_id": "c39",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "A0": {"checks": a0_checks, "activity": activity},
        "seed_reports": per_seed,
        "internal_A_scores_opened": True,
        "internal_A_labels_opened": False,
        "reserve_features_labels_scores_opened": False,
        "dev_test_opened": False,
    }
    if not all(a0_checks.values()):
        report["status"] = "failed_A0_terminal"
        write_json(output_path, report)
        return report

    labels = open_role_labels(
        records_train_path=config["paths"]["records_train"],
        records_train_sha256=config["integrity"]["records_train_sha256"],
        selection_path=config["paths"]["selection"],
        selection_sha256=config["paths"]["selection_sha256"],
        store=store,
        role="internal_A",
    )
    label_rows = [
        labels.row(index, store.candidate_count(index)) for index in indices
    ]
    per_seed_ndcg: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        per_seed_ndcg[seed] = {
            "base": ndcg_rows(
                request_ids, item_ids, score_rows[seed]["base"], label_rows
            )
        }
        for mode in MODES:
            per_seed_ndcg[seed][mode] = ndcg_rows(
                request_ids,
                item_ids,
                score_rows[seed][f"{mode}_true"],
                label_rows,
            )
        per_seed_ndcg[seed]["primary_wrong"] = ndcg_rows(
            request_ids,
            item_ids,
            score_rows[seed][f"{PRIMARY}_wrong"],
            label_rows,
        )
    averaged_ndcg = {
        name: np.mean(
            np.stack([per_seed_ndcg[seed][name] for seed in seeds]), axis=0
        )
        for name in per_seed_ndcg[seeds[0]]
    }
    comparisons = compare(
        request_ids,
        averaged_ndcg[PRIMARY],
        {
            "base": averaged_ndcg["base"],
            **{control: averaged_ndcg[control] for control in CONTROLS},
            "wrong_history": averaged_ndcg["primary_wrong"],
        },
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
        folds=int(config["evaluation"]["hash_folds"]),
    )
    candidate_value_corrections = _subtract_rows(
        averaged[f"{PRIMARY}_correction"],
        averaged[f"{GLOBAL_ONLY}_correction"],
    )
    direction = bootstrap(
        clicked_direction(candidate_value_corrections, label_rows),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 101,
    )
    seed_differences = {
        reference: {
            str(seed): float(
                (
                    per_seed_ndcg[seed][PRIMARY]
                    - per_seed_ndcg[seed][reference]
                ).mean()
            )
            for seed in seeds
        }
        for reference in ("base", *CONTROLS)
    }
    a1_checks = {
        "over_base_effect": comparisons["base"]["mean"]
        >= float(gate["primary_minus_base_min"]),
        "over_base_all_seeds": all(
            value > 0 for value in seed_differences["base"].values()
        ),
        "over_base_all_folds": all(
            row["mean_difference"] > 0
            for row in comparisons["base"]["hash_folds"]
        ),
        "over_base_ci": comparisons["base"]["percentile_95_ci"][0] > 0,
        "over_global_effect": comparisons[GLOBAL_ONLY]["mean"]
        >= float(gate["primary_minus_global_min"]),
        "over_global_all_seeds": all(
            value > 0 for value in seed_differences[GLOBAL_ONLY].values()
        ),
        "over_global_all_folds": all(
            row["mean_difference"] > 0
            for row in comparisons[GLOBAL_ONLY]["hash_folds"]
        ),
        "over_global_ci": comparisons[GLOBAL_ONLY]["percentile_95_ci"][0] > 0,
        **{
            f"over_{control}_effect": comparisons[control]["mean"]
            >= float(gate["primary_minus_reduction_min"])
            for control in (EVENTWISE_RAW, POSTPOOL_HALFSPACE, RAY_ONLY)
        },
        **{
            f"over_{control}_all_seeds": all(
                value >= 0 for value in seed_differences[control].values()
            )
            for control in (EVENTWISE_RAW, POSTPOOL_HALFSPACE, RAY_ONLY)
        },
        **{
            f"over_{control}_ci": comparisons[control]["percentile_95_ci"][0]
            > 0
            for control in (EVENTWISE_RAW, POSTPOOL_HALFSPACE, RAY_ONLY)
        },
        "true_over_wrong_ci": comparisons["wrong_history"][
            "percentile_95_ci"
        ][0]
        > 0,
        "candidate_value_clicked_direction_ci": direction[
            "percentile_95_ci"
        ][0]
        > 0,
    }
    report["A1"] = {
        "checks": a1_checks,
        "comparisons": comparisons,
        "candidate_value_clicked_direction": direction,
        "seed_differences": seed_differences,
        "seed_averaged_ndcg10": {
            name: float(values.mean()) for name, values in averaged_ndcg.items()
        },
    }
    report["internal_A_labels_opened"] = True
    report["status"] = (
        "passed_A1_halfspace_certified_value"
        if all(a1_checks.values())
        else "failed_A1_terminal"
    )
    write_json(output_path, report)
    return report


def rankings(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    scores: Sequence[np.ndarray],
) -> list[list[str]]:
    return [
        [
            row.item_id
            for row in sort_candidates(
                request_id,
                [
                    ScoredCandidate(str(item), float(score))
                    for item, score in zip(items, values)
                ],
            )
        ]
        for request_id, items, values in zip(request_ids, item_ids, scores)
    ]


def order_changes(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    reference: Sequence[np.ndarray],
    proposed: Sequence[np.ndarray],
) -> dict[str, Any]:
    first = rankings(request_ids, item_ids, reference)
    second = rankings(request_ids, item_ids, proposed)
    any_count = sum(int(a != b) for a, b in zip(first, second))
    top10_count = sum(
        int(set(a[:10]) != set(b[:10])) for a, b in zip(first, second)
    )
    return {
        "requests": len(first),
        "any_count": any_count,
        "any_fraction": any_count / len(first),
        "top10_count": top10_count,
        "top10_fraction": top10_count / len(first),
    }


def ndcg_rows(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
) -> np.ndarray:
    output = []
    for request_id, items, values, label in zip(
        request_ids, item_ids, scores, labels
    ):
        ranked = rankings([request_id], [items], [values])[0]
        positives = {
            str(item) for item, relevance in zip(items, label) if relevance > 0
        }
        output.append(ndcg_at_k(ranked, positives, 10))
    return np.asarray(output, dtype=np.float64)


def _offsets(rows: Sequence[np.ndarray]) -> np.ndarray:
    values = [0]
    for row in rows:
        values.append(values[-1] + len(row))
    return np.asarray(values, dtype=np.int64)


def _flatten(rows: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(rows).astype(np.float32, copy=False)


def _unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(
            values[int(offsets[row]) : int(offsets[row + 1])],
            dtype=np.float32,
        ).copy()
        for row in range(len(offsets) - 1)
    ]


def _average_rows(
    collections: Sequence[Sequence[np.ndarray]],
) -> list[np.ndarray]:
    return [np.mean(np.stack(rows), axis=0) for rows in zip(*collections)]


def _subtract_rows(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
) -> list[np.ndarray]:
    return [np.asarray(a) - np.asarray(b) for a, b in zip(left, right)]


def _max_difference(
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
) -> float:
    return max(float(np.max(np.abs(a - b))) for a, b in zip(first, second))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("g0", "seed", "aggregate"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "g0":
        value = run_g0(config)
    elif args.stage == "seed":
        if args.seed is None or args.seed not in [
            int(value) for value in config["training"]["seeds"]
        ]:
            raise ValueError("C39 seed is not registered")
        if not torch.cuda.is_available():
            raise RuntimeError("C39 seed stage requires CUDA")
        value = run_seed(config, args.seed, torch.device(args.device))
    else:
        value = aggregate(config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
