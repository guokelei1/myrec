"""Hash-locked one-shot GPU execution of the C23-A train-only gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from model.rrst import MODES, RecurrenceResetSurvivalTransformer  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, request_metrics, sort_candidates  # noqa: E402
from train.gate_metrics import (  # noqa: E402
    clicked_minus_unclicked,
    compare_primary,
    paired_bootstrap,
    retention_bootstrap,
)
from train.locking import verify_execution_lock, verify_proposal_lock  # noqa: E402
from train.losses import masked_listwise_loss  # noqa: E402
from train.real_data import (  # noqa: E402
    CompactLabels,
    FrozenFeatureStore,
    iter_batches,
    open_original_selected_labels,
    state_sha256,
    to_device,
)
from train.structure import atomic_json, load_config, read_json, sha256_file  # noqa: E402


PRIMARY = "reset_suffix"
CONTROLS = tuple(mode for mode in MODES if mode != PRIMARY)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def assert_cuda(config: Mapping[str, Any], device: torch.device) -> None:
    if str(device) != "cuda:0":
        raise ValueError("C23 formal process must use cuda:0")
    physical = int(config["resources"]["physical_gpu"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from C23 registration")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C23 requires deterministic CUBLAS_WORKSPACE_CONFIG")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C23 requires exactly one visible CUDA GPU")


def make_model(config: Mapping[str, Any], mode: str) -> RecurrenceResetSurvivalTransformer:
    model = config["model"]
    return RecurrenceResetSurvivalTransformer(
        input_dim=int(model["input_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        heads=int(model["heads"]),
        layers=int(model["layers"]),
        ffn_dim=int(model["ffn_dim"]),
        max_history=int(model["max_history"]),
        dropout=float(model["dropout"]),
        score_delta_max=float(model["score_delta_max"]),
        mode=mode,
    )


def compact_fit_labels(config: Mapping[str, Any]) -> CompactLabels:
    root = Path(config["paths"]["artifact_root"])
    return CompactLabels(
        request_indices=np.load(root / "fit_request_indices.npy", allow_pickle=False),
        offsets=np.load(root / "fit_label_offsets.npy", allow_pickle=False),
        values=np.load(root / "fit_labels.npy", allow_pickle=False),
    )


def batch_options(config: Mapping[str, Any]) -> dict[str, int]:
    training = config["training"]
    return {
        "max_requests": int(training["max_requests_per_batch"]),
        "max_candidate_sequences": int(training["max_candidate_sequences_per_batch"]),
        "max_sequence_tokens": int(training["max_sequence_tokens_per_batch"]),
    }


def build_schedules(
    store: FrozenFeatureStore,
    indices: Sequence[int],
    config: Mapping[str, Any],
    seed: int,
) -> list[list[np.ndarray]]:
    return [
        list(
            iter_batches(
                store.data,
                indices,
                seed=seed + epoch * 10_003,
                shuffle=True,
                **batch_options(config),
            )
        )
        for epoch in range(int(config["training"]["epochs"]))
    ]


def train_model(
    model: RecurrenceResetSurvivalTransformer,
    *,
    store: FrozenFeatureStore,
    labels: CompactLabels,
    schedules: Sequence[Sequence[np.ndarray]],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    losses: list[float] = []
    gradient_names: set[str] = set()
    beta = float(config["base"]["item_only_beta"])
    for epoch_batches in schedules:
        for request_indices in epoch_batches:
            batch = store.collate(request_indices, labels=labels)
            tensors = to_device(batch, device, beta=beta)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                query=tensors["query"],
                candidates=tensors["candidates"],
                history=tensors["history"],
                candidate_mask=tensors["candidate_mask"],
                history_mask=tensors["history_mask"],
                repeat_mask=tensors["repeat_mask"],
                event_weights=tensors["event_weights"],
                base_scores=tensors["base_scores"],
                item_only_scores=tensors["item_only_scores"],
            )
            loss = masked_listwise_loss(
                output.scores, tensors["labels"], tensors["candidate_mask"]
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite C23 loss for {model.mode}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C23 gradient: {model.mode}/{name}")
                    if bool(parameter.grad.ne(0).any()):
                        gradient_names.add(name)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {
        "epochs": len(schedules),
        "steps": len(losses),
        "finite": bool(losses) and bool(np.isfinite(losses).all()),
        "loss_first_50_mean": float(np.mean(losses[:50])),
        "loss_last_50_mean": float(np.mean(losses[-50:])),
        "nonzero_gradient_parameters": sorted(gradient_names),
    }


def suffix_permutation(
    batch: Mapping[str, Any], payload: str
) -> torch.Tensor:
    repeat = np.asarray(batch["repeat_mask_numpy"], dtype=bool)
    history_mask = np.asarray(batch["history_mask_numpy"], dtype=bool)
    batch_size, candidate_count, history_count = repeat.shape
    result = np.broadcast_to(
        np.arange(history_count, dtype=np.int64), (batch_size, candidate_count, history_count)
    ).copy()
    for row in range(batch_size):
        valid_history = int(history_mask[row].sum())
        for candidate in range(candidate_count):
            exact = np.flatnonzero(repeat[row, candidate, :valid_history])
            if len(exact) == 0:
                continue
            anchor = int(exact[-1])
            suffix = np.arange(anchor + 1, valid_history, dtype=np.int64)
            if len(suffix) < 2:
                continue
            key = (
                payload.replace("<request_id>", str(batch["request_ids"][row]))
                .replace("<candidate_item_id>", str(batch["candidate_item_ids"][row, candidate]))
                .encode("utf-8")
            )
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
            result[row, candidate, suffix] = suffix[
                np.random.default_rng(seed).permutation(len(suffix))
            ]
    return torch.from_numpy(result)


def score_dataset(
    model: RecurrenceResetSurvivalTransformer,
    *,
    store: FrozenFeatureStore,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    suffix_shuffle: bool = False,
    query_present: bool = True,
    corrupt_preanchor: bool = False,
) -> dict[str, Any]:
    model.to(device).eval()
    request_ids: list[str] = []
    item_ids: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    base_scores: list[np.ndarray] = []
    item_only_scores: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    centre_sums: list[float] = []
    beta = float(config["base"]["item_only_beta"])
    with torch.no_grad():
        for request_indices in iter_batches(
            store.data,
            indices,
            seed=0,
            shuffle=False,
            **batch_options(config),
        ):
            batch = store.collate(request_indices, labels=None)
            tensors = to_device(batch, device, beta=beta)
            permutation = None
            if suffix_shuffle:
                permutation = suffix_permutation(
                    batch, str(config["evaluation"]["suffix_shuffle_payload"])
                ).to(device)
            present = torch.full(
                (len(request_indices),), query_present, dtype=torch.bool, device=device
            )
            output = model(
                query=tensors["query"],
                candidates=tensors["candidates"],
                history=tensors["history"],
                candidate_mask=tensors["candidate_mask"],
                history_mask=tensors["history_mask"],
                repeat_mask=tensors["repeat_mask"],
                event_weights=tensors["event_weights"],
                base_scores=tensors["base_scores"],
                item_only_scores=tensors["item_only_scores"],
                query_present=present,
                suffix_permutation=permutation,
                corrupt_preanchor=corrupt_preanchor,
            )
            mask = batch["candidate_mask_numpy"]
            cpu_scores = output.scores.cpu().numpy()
            cpu_base = tensors["base_scores"].cpu().numpy()
            cpu_item = output.item_only_scores.cpu().numpy()
            cpu_delta = output.correction.cpu().numpy()
            for row in range(len(request_indices)):
                count = int(mask[row].sum())
                request_ids.append(str(batch["request_ids"][row]))
                item_ids.append(batch["candidate_item_ids"][row, :count].copy())
                scores.append(cpu_scores[row, :count].copy())
                base_scores.append(cpu_base[row, :count].copy())
                item_only_scores.append(cpu_item[row, :count].copy())
                corrections.append(cpu_delta[row, :count].copy())
                centre_sums.append(float(abs(cpu_delta[row, :count].sum())))
    expected_ids = [store.data.request_ids[int(index)] for index in indices]
    if request_ids != expected_ids:
        raise ValueError("C23 scoring order changed")
    return {
        "request_ids": request_ids,
        "item_ids": item_ids,
        "scores": scores,
        "base_scores": base_scores,
        "item_only_scores": item_only_scores,
        "corrections": corrections,
        "maximum_abs_candidate_correction_sum": max(centre_sums, default=0.0),
    }


def maximum_row_difference(
    first: Sequence[np.ndarray], second: Sequence[np.ndarray]
) -> float:
    if len(first) != len(second):
        raise ValueError("C23 score row counts differ")
    return max(
        (float(np.max(np.abs(a - b))) if len(a) else 0.0)
        for a, b in zip(first, second)
    )


def changed_request_fraction(
    first: Sequence[np.ndarray], second: Sequence[np.ndarray], *, tolerance: float
) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("C23 intervention row counts differ")
    changed = sum(
        int(bool(np.max(np.abs(a - b)) > tolerance))
        for a, b in zip(first, second)
    )
    return changed / len(first)


def average_rows(collections: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    if not collections:
        raise ValueError("cannot average empty C23 row collection")
    return [np.mean(np.stack(rows, axis=0), axis=0) for rows in zip(*collections)]


def order_summary(
    request_ids: Sequence[str],
    item_ids: Sequence[np.ndarray],
    reference: Sequence[np.ndarray],
    primary: Sequence[np.ndarray],
) -> dict[str, Any]:
    any_change = 0
    top10_change = 0
    for request_id, ids, base, proposed in zip(request_ids, item_ids, reference, primary):
        base_rank = [
            row.item_id
            for row in sort_candidates(
                str(request_id),
                [ScoredCandidate(str(item), float(score)) for item, score in zip(ids, base)],
            )
        ]
        proposed_rank = [
            row.item_id
            for row in sort_candidates(
                str(request_id),
                [
                    ScoredCandidate(str(item), float(score))
                    for item, score in zip(ids, proposed)
                ],
            )
        ]
        any_change += int(base_rank != proposed_rank)
        top10_change += int(set(base_rank[:10]) != set(proposed_rank[:10]))
    count = len(request_ids)
    return {
        "requests": count,
        "requests_with_any_order_change": any_change,
        "requests_with_any_order_change_fraction": any_change / count,
        "requests_with_top10_membership_change": top10_change,
        "requests_with_top10_membership_change_fraction": top10_change / count,
    }


def metric_rows(
    scored: Mapping[str, Any],
    score_key: str,
    labels: CompactLabels,
    request_indices: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    counts = [len(row) for row in scored["item_ids"]]
    label_rows = labels.rows(request_indices, counts)
    values: list[float] = []
    for request_id, ids, scores, target in zip(
        scored["request_ids"], scored["item_ids"], scored[score_key], label_rows
    ):
        positives = {
            str(item) for item, label in zip(ids, target) if float(label) > 0.0
        }
        row = request_metrics(
            str(request_id),
            [ScoredCandidate(str(item), float(score)) for item, score in zip(ids, scores)],
            positives,
            set(),
        )
        values.append(float(row["ndcg@10"]))
    return np.asarray(values, dtype=np.float64), label_rows


def candidate_permutation_audit(
    model: RecurrenceResetSurvivalTransformer,
    *,
    store: FrozenFeatureStore,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
) -> float:
    selected = list(indices[: min(32, len(indices))])
    batch = store.collate(selected, labels=None)
    tensors = to_device(batch, device, beta=float(config["base"]["item_only_beta"]))
    model.eval()
    with torch.no_grad():
        clean = model(
            query=tensors["query"], candidates=tensors["candidates"],
            history=tensors["history"], candidate_mask=tensors["candidate_mask"],
            history_mask=tensors["history_mask"], repeat_mask=tensors["repeat_mask"],
            event_weights=tensors["event_weights"], base_scores=tensors["base_scores"],
            item_only_scores=tensors["item_only_scores"],
        ).scores
        candidate_count = clean.shape[1]
        permutation = torch.arange(candidate_count - 1, -1, -1, device=device)
        inverse = torch.argsort(permutation)
        permuted = model(
            query=tensors["query"], candidates=tensors["candidates"][:, permutation],
            history=tensors["history"], candidate_mask=tensors["candidate_mask"][:, permutation],
            history_mask=tensors["history_mask"], repeat_mask=tensors["repeat_mask"][:, permutation],
            event_weights=tensors["event_weights"], base_scores=tensors["base_scores"][:, permutation],
            item_only_scores=tensors["item_only_scores"][:, permutation],
        ).scores[:, inverse]
    return float((clean - permuted).abs().max().cpu())


def save_checkpoint(
    model: nn.Module,
    *,
    config: Mapping[str, Any],
    seed: int,
    mode: str,
    proposal_lock_sha256: str,
    execution_lock_sha256: str,
) -> dict[str, Any]:
    root = Path(config["paths"]["checkpoint_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"seed_{seed}_{mode}.pt"
    if path.exists():
        raise FileExistsError(f"C23 checkpoint already exists: {path}")
    torch.save(
        {
            "candidate_id": "c23",
            "gate_id": config["gate_id"],
            "seed": seed,
            "mode": mode,
            "proposal_lock_sha256": proposal_lock_sha256,
            "execution_lock_sha256": execution_lock_sha256,
            "state_dict": model.state_dict(),
        },
        path,
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "state_sha256": state_sha256(model),
    }


def assert_candidate_hashes(store: FrozenFeatureStore) -> dict[str, str]:
    output: dict[str, str] = {}
    for role, row in store.selection["roles"].items():
        indices = [int(value) for value in row["indices"]]
        actual = store.candidate_key_sha256(indices)
        if actual != row["candidate_key_sha256"]:
            raise RuntimeError(f"C23 candidate-set hash changed: {role}")
        output[role] = actual
    return output


def formal(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    proposal_lock, proposal_hash = verify_proposal_lock(config)
    execution_lock, execution_hash = verify_execution_lock(config, proposal_hash)
    root = Path(config["paths"]["artifact_root"])
    report_path = root / "train_gate_report.json"
    attempt_path = root / "formal_attempt.json"
    if report_path.exists() or attempt_path.exists():
        raise FileExistsError("C23 one-shot formal attempt already exists")
    atomic_json(
        attempt_path,
        {
            "candidate_id": "c23",
            "status": "started_before_training_and_internal_A_access",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "proposal_lock_sha256": proposal_hash,
            "execution_lock_sha256": execution_hash,
            "fit_compact_labels_opened": False,
            "internal_A_labels_opened": False,
            "delayed_B_escrow_dev_test_opened": False,
        },
    )
    store = FrozenFeatureStore(config)
    candidate_hashes = assert_candidate_hashes(store)
    fit_labels = compact_fit_labels(config)
    fit_indices = store.role_indices("fit")
    probe_indices = store.role_indices("internal_A")
    request_ids = [store.data.request_ids[index] for index in probe_indices]
    seeds = [int(value) for value in config["training"]["seeds"]]
    seed_outputs: dict[int, dict[str, Any]] = {}
    seed_summaries: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    parameter_counts: dict[str, int] = {}
    deterministic_max = 0.0
    preanchor_max = 0.0
    permutation_max = 0.0
    centre_max = 0.0
    suffix_change_fractions: dict[str, float] = {}
    query_absent_exact: dict[str, bool] = {}
    nohistory_exact: dict[str, bool] = {}
    nonrepeat_exact: dict[str, bool] = {}
    initial_hashes_by_seed: dict[str, dict[str, str]] = {}
    started = time.monotonic()

    for seed in seeds:
        seed_everything(seed)
        template = make_model(config, PRIMARY)
        initial_state = {name: value.detach().clone() for name, value in template.state_dict().items()}
        schedules = build_schedules(store, fit_indices, config, seed)
        mode_outputs: dict[str, Any] = {}
        training_rows: dict[str, Any] = {}
        initial_hashes: dict[str, str] = {}
        primary_diagnostics: dict[str, Any] = {}
        for mode in MODES:
            model = make_model(config, mode)
            model.load_state_dict(initial_state, strict=True)
            initial_hashes[mode] = state_sha256(model)
            parameter_counts[mode] = model.parameter_count()
            training_rows[mode] = train_model(
                model,
                store=store,
                labels=fit_labels,
                schedules=schedules,
                config=config,
                device=device,
            )
            checkpoints[f"{seed}/{mode}"] = save_checkpoint(
                model,
                config=config,
                seed=seed,
                mode=mode,
                proposal_lock_sha256=proposal_hash,
                execution_lock_sha256=execution_hash,
            )
            clean = score_dataset(
                model, store=store, indices=probe_indices, config=config, device=device
            )
            repeated = score_dataset(
                model, store=store, indices=probe_indices, config=config, device=device
            )
            deterministic_max = max(
                deterministic_max,
                maximum_row_difference(clean["scores"], repeated["scores"]),
            )
            mode_outputs[mode] = clean
            centre_max = max(
                centre_max, float(clean["maximum_abs_candidate_correction_sum"])
            )
            if mode == PRIMARY:
                shuffled = score_dataset(
                    model,
                    store=store,
                    indices=probe_indices,
                    config=config,
                    device=device,
                    suffix_shuffle=True,
                )
                preanchor = score_dataset(
                    model,
                    store=store,
                    indices=probe_indices,
                    config=config,
                    device=device,
                    corrupt_preanchor=True,
                )
                query_absent = score_dataset(
                    model,
                    store=store,
                    indices=probe_indices,
                    config=config,
                    device=device,
                    query_present=False,
                )
                nohistory = score_dataset(
                    model,
                    store=store,
                    indices=store.role_indices("structural_nohistory"),
                    config=config,
                    device=device,
                )
                nonrepeat = score_dataset(
                    model,
                    store=store,
                    indices=store.role_indices("structural_nonrepeat"),
                    config=config,
                    device=device,
                )
                suffix_change_fractions[str(seed)] = changed_request_fraction(
                    clean["corrections"], shuffled["corrections"], tolerance=1e-7
                )
                preanchor_max = max(
                    preanchor_max,
                    maximum_row_difference(clean["scores"], preanchor["scores"]),
                )
                query_absent_exact[str(seed)] = all(
                    np.array_equal(score, item)
                    for score, item in zip(
                        query_absent["scores"], query_absent["item_only_scores"]
                    )
                )
                nohistory_exact[str(seed)] = all(
                    np.array_equal(score, base)
                    for score, base in zip(nohistory["scores"], nohistory["base_scores"])
                )
                nonrepeat_exact[str(seed)] = all(
                    np.array_equal(score, base)
                    for score, base in zip(nonrepeat["scores"], nonrepeat["base_scores"])
                )
                permutation_max = max(
                    permutation_max,
                    candidate_permutation_audit(
                        model,
                        store=store,
                        indices=probe_indices,
                        config=config,
                        device=device,
                    ),
                )
                primary_diagnostics = {"shuffled": shuffled}
            del model
            torch.cuda.empty_cache()
        initial_hashes_by_seed[str(seed)] = initial_hashes
        seed_outputs[seed] = {
            "modes": mode_outputs,
            "shuffled": primary_diagnostics["shuffled"],
        }
        seed_summaries[str(seed)] = {
            "training": training_rows,
            "matched_initialization": len(set(initial_hashes.values())) == 1,
            "matched_parameters": len(set(parameter_counts.values())) == 1,
            "initial_state_hashes": initial_hashes,
            "parameter_counts": dict(parameter_counts),
            "suffix_intervention_change_fraction": suffix_change_fractions[str(seed)],
            "query_absent_bitwise_item_only": query_absent_exact[str(seed)],
            "nohistory_bitwise_d2p": nohistory_exact[str(seed)],
            "nonrepeat_bitwise_d2p": nonrepeat_exact[str(seed)],
        }

    averaged_primary = average_rows(
        [seed_outputs[seed]["modes"][PRIMARY]["scores"] for seed in seeds]
    )
    item_only_rows = seed_outputs[seeds[0]]["modes"][PRIMARY]["item_only_scores"]
    order = order_summary(
        request_ids,
        seed_outputs[seeds[0]]["modes"][PRIMARY]["item_ids"],
        item_only_rows,
        averaged_primary,
    )
    gate = config["gate"]
    a0_checks = {
        "all_training_finite": all(
            row["finite"]
            for seed_row in seed_summaries.values()
            for row in seed_row["training"].values()
        ),
        "matched_parameters": len(set(parameter_counts.values())) == 1,
        "matched_initialization_each_seed": all(
            len(set(values.values())) == 1 for values in initial_hashes_by_seed.values()
        ),
        "candidate_centred": centre_max <= float(gate["candidate_center_sum_abs_max"]),
        "deterministic_rescore": deterministic_max
        <= float(gate["deterministic_rescore_max_abs_difference"]),
        "enough_order_changes": order["requests_with_any_order_change_fraction"]
        >= float(gate["requests_with_any_order_change_fraction_min"]),
        "enough_top10_changes": order["requests_with_top10_membership_change_fraction"]
        >= float(gate["requests_with_top10_membership_change_fraction_min"]),
        "suffix_intervention_active": all(
            value >= float(gate["suffix_intervention_change_fraction_min"])
            for value in suffix_change_fractions.values()
        ),
        "preanchor_exact_invariance": preanchor_max
        <= float(gate["preanchor_invariance_max_abs_difference"]),
        "query_absent_item_only": all(query_absent_exact.values()),
        "nohistory_d2p": all(nohistory_exact.values()),
        "nonrepeat_d2p": all(nonrepeat_exact.values()),
        "candidate_permutation_equivariance": permutation_max
        <= float(gate["candidate_permutation_max_abs_difference"]),
        "selection_integrity": (
            store.selection["checks"].get("labels_opened_before_selection") is False
            and store.selection["checks"].get("dev_test_qrels_metrics_read") is False
            and store.selection["checks"].get("registered_prior_selection_overlap") == 0
            and store.selection["checks"].get("repeat_roles_pairwise_disjoint") is True
            and store.selection["checks"].get("post_repeat_partition_exhaustive") is True
        ),
    }
    a0 = {
        "status": "passed" if all(a0_checks.values()) else "failed",
        "checks": a0_checks,
        "order_changes_on_seed_averaged_scores": order,
        "suffix_intervention_change_fraction": suffix_change_fractions,
        "maximum_abs_candidate_correction_sum": centre_max,
        "deterministic_rescore_max_abs_difference": deterministic_max,
        "preanchor_corruption_max_abs_difference": preanchor_max,
        "candidate_permutation_max_abs_difference": permutation_max,
        "internal_A_labels_opened": False,
    }
    if not all(a0_checks.values()):
        report = {
            "candidate_id": "c23",
            "gate_id": config["gate_id"],
            "status": "failed_A0_terminal",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "proposal_lock_sha256": proposal_hash,
            "execution_lock_sha256": execution_hash,
            "candidate_key_sha256": candidate_hashes,
            "A0": a0,
            "seed_summaries": seed_summaries,
            "checkpoints": checkpoints,
            "fit_compact_labels_opened": True,
            "internal_A_labels_opened": False,
            "delayed_B_escrow_dev_test_opened": False,
            "primary_dev_evaluator_calls": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(report_path, report)
        atomic_json(attempt_path, {**read_json(attempt_path), "status": "failed_A0_terminal"})
        return report

    # A0 has passed. This is the first internal-A label-value access.
    internal_labels = open_original_selected_labels(
        data=store.data,
        indices=probe_indices,
        label_path=config["paths"]["train_candidate_labels"],
        selection_sha256=config["paths"]["selection_sha256"],
        selection_path=config["paths"]["selection"],
    )
    candidate_hashes_after_A0 = assert_candidate_hashes(store)
    if candidate_hashes_after_A0 != candidate_hashes:
        raise RuntimeError("C23 candidate hashes changed before A1")
    seed_metric_arrays: dict[int, dict[str, np.ndarray]] = {}
    label_rows: list[np.ndarray] | None = None
    for seed in seeds:
        rows: dict[str, np.ndarray] = {}
        primary_scored = seed_outputs[seed]["modes"][PRIMARY]
        rows["d2p"], label_rows = metric_rows(
            primary_scored, "base_scores", internal_labels, probe_indices
        )
        rows["item_only"], _ = metric_rows(
            primary_scored, "item_only_scores", internal_labels, probe_indices
        )
        for mode in MODES:
            rows[mode], _ = metric_rows(
                seed_outputs[seed]["modes"][mode], "scores", internal_labels, probe_indices
            )
        rows["suffix_shuffle"], _ = metric_rows(
            seed_outputs[seed]["shuffled"], "scores", internal_labels, probe_indices
        )
        seed_metric_arrays[seed] = rows
        seed_summaries[str(seed)]["ndcg10"] = {
            name: float(values.mean()) for name, values in rows.items()
        }
        np.savez_compressed(
            root / f"seed_{seed}_request_metrics.npz",
            request_indices=np.asarray(probe_indices, dtype=np.int64),
            **rows,
        )

    averaged_metrics = {
        name: np.mean(np.stack([seed_metric_arrays[seed][name] for seed in seeds]), axis=0)
        for name in ("d2p", "item_only", *MODES, "suffix_shuffle")
    }
    comparisons = compare_primary(
        request_ids=request_ids,
        primary=averaged_metrics[PRIMARY],
        references={
            "item_only": averaged_metrics["item_only"],
            **{mode: averaged_metrics[mode] for mode in CONTROLS},
            "d2p_descriptive": averaged_metrics["d2p"],
        },
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
        folds=int(config["evaluation"]["hash_folds"]),
    )
    seed_differences = {
        reference: {
            str(seed): float(
                (
                    seed_metric_arrays[seed][PRIMARY]
                    - seed_metric_arrays[seed][reference]
                ).mean()
            )
            for seed in seeds
        }
        for reference in ("item_only", *CONTROLS)
    }
    clean_gain = averaged_metrics[PRIMARY] - averaged_metrics["item_only"]
    retention = retention_bootstrap(
        clean_gain,
        averaged_metrics["suffix_shuffle"] - averaged_metrics["item_only"],
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 101,
    )
    averaged_corrections = average_rows(
        [seed_outputs[seed]["modes"][PRIMARY]["corrections"] for seed in seeds]
    )
    assert label_rows is not None
    clicked = paired_bootstrap(
        clicked_minus_unclicked(averaged_corrections, label_rows),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 201,
    )
    a1_checks = {
        "primary_over_item_only_effect": comparisons["item_only"]["mean"]
        >= float(gate["ndcg10_delta_over_item_only_min"]),
        "primary_over_item_only_ci": comparisons["item_only"]["percentile_95_ci"][0]
        > 0.0,
        "primary_over_item_only_all_seeds": all(
            value > 0.0 for value in seed_differences["item_only"].values()
        ),
        "primary_over_item_only_all_hash_folds": all(
            row["mean_difference"] > 0.0
            for row in comparisons["item_only"]["hash_folds"]
        ),
        "primary_over_each_control_effect": all(
            comparisons[mode]["mean"]
            >= float(gate["ndcg10_delta_over_each_control_min"])
            for mode in CONTROLS
        ),
        "primary_over_each_control_ci": all(
            comparisons[mode]["percentile_95_ci"][0] > 0.0 for mode in CONTROLS
        ),
        "primary_over_each_control_all_seeds": all(
            all(value > 0.0 for value in seed_differences[mode].values())
            for mode in CONTROLS
        ),
        "clicked_correction_direction": clicked["percentile_95_ci"][0] > 0.0,
        "suffix_corruption_retention": bool(retention.get("applicable"))
        and float(retention["retention"]) <= float(gate["corruption_retention_max"]),
        "suffix_corruption_retention_ci": bool(retention.get("applicable"))
        and float(retention["percentile_95_ci"][1])
        <= float(gate["corruption_retention_ci_high_max"]),
    }
    report = {
        "candidate_id": "c23",
        "gate_id": config["gate_id"],
        "status": "passed" if all(a1_checks.values()) else "failed_A1_terminal",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "candidate_key_sha256_asserted_before_A1": candidate_hashes_after_A0,
        "cohort": {"fit": len(fit_indices), "internal_A": len(probe_indices)},
        "A0": {**a0, "internal_A_labels_opened_after_pass": True},
        "A1": {
            "checks": a1_checks,
            "seed_averaged_ndcg10": {
                name: float(values.mean()) for name, values in averaged_metrics.items()
            },
            "comparisons": comparisons,
            "seed_differences": seed_differences,
            "suffix_shuffle_retention": retention,
            "clicked_minus_unclicked_correction": clicked,
        },
        "seed_summaries": seed_summaries,
        "checkpoints": checkpoints,
        "fit_compact_labels_opened": True,
        "internal_A_labels_opened": True,
        "delayed_B_escrow_dev_test_opened": False,
        "primary_dev_evaluator_calls": 0,
        "elapsed_seconds": time.monotonic() - started,
        "execution": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_name": torch.cuda.get_device_name(0),
        },
    }
    atomic_json(report_path, report)
    atomic_json(
        attempt_path,
        {
            **read_json(attempt_path),
            "status": report["status"],
            "internal_A_labels_opened": True,
            "report_sha256": sha256_file(report_path),
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config, require_frozen_selection=True)
    if config["authorization"].get("fit_label_training_after_lock") is not True:
        raise PermissionError("C23 fit training is not authorized")
    device = torch.device(args.device)
    assert_cuda(config, device)
    report = formal(config, device)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
