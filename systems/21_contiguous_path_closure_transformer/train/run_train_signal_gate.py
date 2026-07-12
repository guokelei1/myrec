"""Hash-locked one-shot GPU runner for the C21 train-only signal gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from model.path_closure import MODES, PathClosureProbe
from myrec.eval.metrics import ScoredCandidate, request_metrics, sort_candidates
from train.gate_metrics import (
    clicked_minus_unclicked,
    compare_primary,
    paired_bootstrap,
    retention_bootstrap,
)
from train.losses import masked_listwise_loss
from train.materialize_selection import load_config, sha256_file
from train.real_data import (
    CompactLabels,
    FrozenFitData,
    iter_batches,
    read_json,
    to_device,
    wrong_history_sources,
)


PRIMARY = "contiguous_path"
CONTROLS = tuple(mode for mode in MODES if mode != PRIMARY)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    lock_path = Path(config["paths"]["proposal_lock"])
    if not lock_path.is_file():
        raise PermissionError("C21 proposal lock is missing")
    lock = read_json(lock_path)
    if lock.get("candidate_id") != "c21" or lock.get("status") != "locked_before_any_c21_label_outcome":
        raise ValueError("unexpected C21 proposal lock")
    failures: list[str] = []
    aggregate_lines: list[str] = []
    for relative, expected in sorted(lock["files_sha256"].items()):
        path = SYSTEM_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            failures.append(f"local:{relative}")
        aggregate_lines.append(f"{expected}  {relative}\n")
    aggregate = hashlib.sha256("".join(aggregate_lines).encode("utf-8")).hexdigest()
    if aggregate != lock.get("aggregate_sha256"):
        failures.append("aggregate_sha256")
    for relative, expected in sorted(lock["external_inputs_sha256"].items()):
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            failures.append(f"external:{relative}")
    if failures:
        raise RuntimeError(f"C21 lock mismatch: {failures}")
    selection_path = Path(config["paths"]["selection"])
    if sha256_file(selection_path) != str(config["paths"]["selection_sha256"]):
        raise RuntimeError("C21 selection changed after freeze")
    return lock, sha256_file(lock_path)


def make_model(config: Mapping[str, Any], mode: str) -> PathClosureProbe:
    settings = config["model"]
    return PathClosureProbe(
        input_dim=int(settings["input_dim"]),
        projection_dim=int(settings["projection_dim"]),
        max_history=int(settings["max_history"]),
        max_horizon=int(settings["max_horizon"]),
        evidence_temperature=float(settings["evidence_temperature"]),
        score_delta_max=float(settings["score_delta_max"]),
        mode=mode,
    )


def batch_limits(config: Mapping[str, Any]) -> dict[str, int]:
    training = config["training"]
    return {
        "history_limit": int(config["model"]["max_history"]),
        "max_requests": int(training["max_requests_per_batch"]),
        "max_padded_candidates": int(training["max_padded_candidate_rows"]),
        "max_padded_history": int(training["max_padded_history_rows"]),
    }


def assert_candidate_hashes(
    data: FrozenFitData, selection: Mapping[str, Any]
) -> dict[str, str]:
    output: dict[str, str] = {}
    for role in ("train_fit", "internal_probe"):
        indices = [int(value) for value in selection["roles"][role]["indices"]]
        actual = data.candidate_key_sha256(indices)
        expected = str(selection["roles"][role]["candidate_key_sha256"])
        if actual != expected:
            raise RuntimeError(f"C21 candidate-set hash changed before evaluation: {role}")
        output[role] = actual
    return output


def build_schedules(
    data: FrozenFitData,
    indices: Sequence[int],
    config: Mapping[str, Any],
    seed: int,
) -> list[list[np.ndarray]]:
    limits = batch_limits(config)
    return [
        list(
            iter_batches(
                data,
                indices,
                seed=seed + epoch * 10_003,
                shuffle=True,
                **limits,
            )
        )
        for epoch in range(int(config["training"]["epochs"]))
    ]


def train_model(
    model: PathClosureProbe,
    *,
    data: FrozenFitData,
    labels: CompactLabels,
    schedules: Sequence[Sequence[np.ndarray]],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    settings = config["training"]
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    losses: list[float] = []
    gradient_names: set[str] = set()
    for epoch_batches in schedules:
        for request_indices in epoch_batches:
            batch = data.collate(
                request_indices,
                history_limit=int(config["model"]["max_history"]),
                labels=labels,
            )
            tensors = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                query=tensors["query"],
                candidates=tensors["candidates"],
                history=tensors["history"],
                candidate_mask=tensors["candidate_mask"],
                history_mask=tensors["history_mask"],
                base_scores=tensors["base_scores"],
            )
            loss = masked_listwise_loss(output.scores, tensors["labels"], tensors["candidate_mask"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite C21 loss for {model.mode}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C21 gradient: {model.mode}/{name}")
                    if bool(parameter.grad.ne(0).any()):
                        gradient_names.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
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


def score_dataset(
    model: PathClosureProbe,
    *,
    data: FrozenFitData,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    history_sources: Mapping[int, int] | None = None,
    shuffle_history: bool = False,
    query_present: bool = True,
) -> dict[str, Any]:
    limits = batch_limits(config)
    scores: list[np.ndarray] = []
    base_scores: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    item_ids: list[np.ndarray] = []
    request_ids: list[str] = []
    centre_sums: list[float] = []
    evidence_rows = 0
    model.to(device).eval()
    with torch.no_grad():
        for request_indices in iter_batches(
            data,
            indices,
            seed=0,
            shuffle=False,
            **limits,
        ):
            batch = data.collate(
                request_indices,
                history_limit=int(config["model"]["max_history"]),
                labels=None,
                history_sources=history_sources,
                shuffle_history=shuffle_history,
                shuffle_payload=str(config["evaluation"]["shuffle_payload"]),
            )
            tensors = to_device(batch, device)
            present = torch.full(
                (len(request_indices),), query_present, dtype=torch.bool, device=device
            )
            output = model(
                query=tensors["query"],
                candidates=tensors["candidates"],
                history=tensors["history"],
                candidate_mask=tensors["candidate_mask"],
                history_mask=tensors["history_mask"],
                base_scores=tensors["base_scores"],
                query_present=present,
            )
            mask = batch["candidate_mask_numpy"]
            cpu_scores = output.scores.detach().cpu().numpy()
            cpu_base = output.base_scores.detach().cpu().numpy()
            cpu_delta = output.deltas.detach().cpu().numpy()
            evidence_rows += int(output.has_evidence.sum().cpu())
            for row in range(len(request_indices)):
                count = int(mask[row].sum())
                scores.append(cpu_scores[row, :count].copy())
                base_scores.append(cpu_base[row, :count].copy())
                deltas.append(cpu_delta[row, :count].copy())
                item_ids.append(batch["candidate_item_ids"][row, :count].copy())
                request_ids.append(batch["request_ids"][row])
                centre_sums.append(float(abs(cpu_delta[row, :count].sum())))
    if request_ids != [data.request_ids[int(index)] for index in indices]:
        raise ValueError("C21 scoring order changed")
    return {
        "request_ids": request_ids,
        "item_ids": item_ids,
        "scores": scores,
        "base_scores": base_scores,
        "deltas": deltas,
        "maximum_abs_candidate_delta_sum": max(centre_sums, default=0.0),
        "requests_with_evidence": evidence_rows,
    }


def metric_rows(
    scored: Mapping[str, Any], labels: CompactLabels, request_indices: Sequence[int]
) -> tuple[np.ndarray, list[np.ndarray], list[list[str]]]:
    counts = [len(row) for row in scored["item_ids"]]
    label_rows = labels.rows(request_indices, counts)
    ndcg: list[float] = []
    rankings: list[list[str]] = []
    for request_id, item_ids, scores, label in zip(
        scored["request_ids"], scored["item_ids"], scored["scores"], label_rows
    ):
        positives = {
            str(item_id) for item_id, value in zip(item_ids, label) if float(value) > 0.0
        }
        candidates = [
            ScoredCandidate(str(item_id), float(score)) for item_id, score in zip(item_ids, scores)
        ]
        row = request_metrics(str(request_id), candidates, positives, set())
        ndcg.append(float(row["ndcg@10"]))
        rankings.append([candidate.item_id for candidate in sort_candidates(str(request_id), candidates)])
    return np.asarray(ndcg, dtype=np.float64), label_rows, rankings


def average_rows(collections: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    if not collections:
        raise ValueError("cannot average empty C21 row collection")
    output: list[np.ndarray] = []
    for rows in zip(*collections):
        shapes = {tuple(np.asarray(row).shape) for row in rows}
        if len(shapes) != 1:
            raise ValueError("C21 candidate rows changed across seeds")
        output.append(np.mean(np.stack(rows, axis=0), axis=0))
    return output


def order_summary(
    request_ids: Sequence[str],
    item_ids: Sequence[np.ndarray],
    base_rows: Sequence[np.ndarray],
    primary_rows: Sequence[np.ndarray],
) -> dict[str, Any]:
    any_change = 0
    top10_change = 0
    for request_id, ids, base, primary in zip(request_ids, item_ids, base_rows, primary_rows):
        base_rank = [
            row.item_id
            for row in sort_candidates(
                request_id,
                [ScoredCandidate(str(item_id), float(score)) for item_id, score in zip(ids, base)],
            )
        ]
        primary_rank = [
            row.item_id
            for row in sort_candidates(
                request_id,
                [ScoredCandidate(str(item_id), float(score)) for item_id, score in zip(ids, primary)],
            )
        ]
        any_change += int(base_rank != primary_rank)
        top10_change += int(set(base_rank[:10]) != set(primary_rank[:10]))
    count = len(request_ids)
    return {
        "requests": count,
        "requests_with_any_order_change": any_change,
        "requests_with_any_order_change_fraction": any_change / count,
        "requests_with_top10_membership_change": top10_change,
        "requests_with_top10_membership_change_fraction": top10_change / count,
    }


def exact_base(scored: Mapping[str, Any]) -> bool:
    return all(
        np.array_equal(score, base)
        for score, base in zip(scored["scores"], scored["base_scores"])
    )


def max_row_difference(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    if len(first) != len(second):
        raise ValueError("C21 deterministic row count mismatch")
    return max(
        (float(np.max(np.abs(a - b))) if len(a) else 0.0)
        for a, b in zip(first, second)
    )


def save_checkpoint(
    model: PathClosureProbe,
    *,
    config: Mapping[str, Any],
    seed: int,
    mode: str,
    lock_sha256: str,
) -> dict[str, Any]:
    root = Path(config["paths"]["checkpoint_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"seed_{seed}_{mode}.pt"
    if path.exists():
        raise FileExistsError(f"C21 checkpoint already exists: {path}")
    torch.save(
        {
            "candidate_id": "c21",
            "gate_id": config["gate_id"],
            "seed": seed,
            "mode": mode,
            "proposal_lock_sha256": lock_sha256,
            "state_dict": model.state_dict(),
        },
        path,
    )
    return {"path": str(path), "sha256": sha256_file(path), "state_sha256": state_sha256(model)}


def preflight(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    lock, lock_hash = verify_lock(config)
    selection = read_json(config["paths"]["selection"])
    data = FrozenFitData(config, selection)
    candidate_hashes = assert_candidate_hashes(data, selection)
    indices = selection["roles"]["train_fit"]["indices"][:4]
    batch = data.collate(indices, history_limit=int(config["model"]["max_history"]), labels=None)
    tensors = to_device(batch, device)
    initial_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    torch.manual_seed(20260727 * 10_000 + 2101)
    template = make_model(config, PRIMARY)
    initial = template.state_dict()
    checks: dict[str, bool] = {}
    for mode in MODES:
        model = make_model(config, mode).to(device)
        model.load_state_dict(initial)
        initial_hashes[mode] = state_sha256(model)
        counts[mode] = sum(value.numel() for value in model.parameters())
        output = model(
            query=tensors["query"],
            candidates=tensors["candidates"],
            history=tensors["history"],
            candidate_mask=tensors["candidate_mask"],
            history_mask=tensors["history_mask"],
            base_scores=tensors["base_scores"],
        )
        checks[f"{mode}_finite"] = bool(torch.isfinite(output.scores).all())
    checks["matched_parameters"] = len(set(counts.values())) == 1
    checks["matched_initialization"] = len(set(initial_hashes.values())) == 1
    checks["selection_checks"] = all(bool(value) for value in selection["checks"].values())
    if not all(checks.values()):
        raise RuntimeError(f"C21 preflight failed: {checks}")
    return {
        "status": "passed_label_free",
        "proposal_lock_sha256": lock_hash,
        "lock_status": lock["status"],
        "device": str(device),
        "checks": checks,
        "parameter_counts": counts,
        "initial_state_hashes": initial_hashes,
        "candidate_key_sha256": candidate_hashes,
        "labels_opened": False,
        "outcomes_observed": False,
    }


def formal(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    lock, lock_hash = verify_lock(config)
    root = Path(config["paths"]["artifact_root"])
    report_path = root / "gate_report.json"
    attempt_path = root / "formal_attempt.json"
    if report_path.exists() or attempt_path.exists():
        raise FileExistsError("C21 one-shot formal attempt already exists")
    attempt = {
        "candidate_id": "c21",
        "gate_id": config["gate_id"],
        "status": "started_before_compact_fit_labels_open",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "proposal_lock_sha256": lock_hash,
        "selection_sha256": sha256_file(config["paths"]["selection"]),
        "compact_fit_labels_opened_at_write": False,
        "c06_nonfit_dev_test_qrels_opened": False,
    }
    atomic_json(attempt_path, attempt)

    selection = read_json(config["paths"]["selection"])
    data = FrozenFitData(config, selection)
    candidate_hashes = assert_candidate_hashes(data, selection)
    labels = data.open_labels(config)
    train_indices = [int(value) for value in selection["roles"]["train_fit"]["indices"]]
    probe_indices = [int(value) for value in selection["roles"]["internal_probe"]["indices"]]
    request_ids = [data.request_ids[index] for index in probe_indices]
    donors = wrong_history_sources(
        data,
        probe_indices,
        payload_template=str(config["evaluation"]["wrong_history_payload"]),
        history_limit=int(config["model"]["max_history"]),
    )

    seeds = [int(value) for value in config["training"]["seeds"]]
    seed_arrays: dict[int, dict[str, Any]] = {}
    seed_summaries: dict[str, Any] = {}
    parameter_counts: dict[str, int] = {}
    initial_hashes_by_seed: dict[str, dict[str, str]] = {}
    maximum_centre_sum = 0.0
    deterministic_max = 0.0
    nohistory_exact: dict[str, bool] = {}
    query_absent_exact: dict[str, bool] = {}
    checkpoints: dict[str, Any] = {}
    started = time.monotonic()

    for seed in seeds:
        torch.manual_seed(seed * 10_000 + 2101)
        torch.cuda.manual_seed_all(seed * 10_000 + 2101)
        template = make_model(config, PRIMARY)
        initial_state = template.state_dict()
        schedules = build_schedules(data, train_indices, config, seed)
        mode_outputs: dict[str, Any] = {}
        mode_ndcg: dict[str, np.ndarray] = {}
        training_rows: dict[str, Any] = {}
        initial_hashes: dict[str, str] = {}
        for mode in MODES:
            model = make_model(config, mode)
            model.load_state_dict(initial_state)
            initial_hashes[mode] = state_sha256(model)
            parameter_counts[mode] = sum(value.numel() for value in model.parameters())
            training_rows[mode] = train_model(
                model,
                data=data,
                labels=labels,
                schedules=schedules,
                config=config,
                device=device,
            )
            checkpoints[f"{seed}/{mode}"] = save_checkpoint(
                model,
                config=config,
                seed=seed,
                mode=mode,
                lock_sha256=lock_hash,
            )
            clean = score_dataset(
                model,
                data=data,
                indices=probe_indices,
                config=config,
                device=device,
            )
            ndcg, label_rows, _ = metric_rows(clean, labels, probe_indices)
            mode_outputs[mode] = clean
            mode_ndcg[mode] = ndcg
            maximum_centre_sum = max(
                maximum_centre_sum, float(clean["maximum_abs_candidate_delta_sum"])
            )
            if mode == PRIMARY:
                repeated = score_dataset(
                    model,
                    data=data,
                    indices=probe_indices,
                    config=config,
                    device=device,
                )
                deterministic_max = max(
                    deterministic_max,
                    max_row_difference(clean["scores"], repeated["scores"]),
                )
                wrong = score_dataset(
                    model,
                    data=data,
                    indices=probe_indices,
                    config=config,
                    device=device,
                    history_sources=donors,
                )
                shuffled = score_dataset(
                    model,
                    data=data,
                    indices=probe_indices,
                    config=config,
                    device=device,
                    shuffle_history=True,
                )
                query_absent = score_dataset(
                    model,
                    data=data,
                    indices=probe_indices,
                    config=config,
                    device=device,
                    query_present=False,
                )
                nohistory = score_dataset(
                    model,
                    data=data,
                    indices=data.nohistory_indices.tolist(),
                    config=config,
                    device=device,
                )
                wrong_ndcg, _, _ = metric_rows(wrong, labels, probe_indices)
                shuffled_ndcg, _, _ = metric_rows(shuffled, labels, probe_indices)
                query_absent_exact[str(seed)] = exact_base(query_absent)
                nohistory_exact[str(seed)] = exact_base(nohistory)
                primary_extra = {
                    "wrong_history": wrong,
                    "shuffled_event": shuffled,
                    "wrong_history_ndcg": wrong_ndcg,
                    "shuffled_event_ndcg": shuffled_ndcg,
                    "label_rows": label_rows,
                }
            del model
            torch.cuda.empty_cache()
        initial_hashes_by_seed[str(seed)] = initial_hashes
        base_ndcg, _, _ = metric_rows(
            {
                **mode_outputs[PRIMARY],
                "scores": mode_outputs[PRIMARY]["base_scores"],
            },
            labels,
            probe_indices,
        )
        seed_arrays[seed] = {
            "ndcg": {"d2p": base_ndcg, **mode_ndcg},
            "outputs": mode_outputs,
            "wrong_history_ndcg": primary_extra["wrong_history_ndcg"],
            "shuffled_event_ndcg": primary_extra["shuffled_event_ndcg"],
            "wrong_history_scores": primary_extra["wrong_history"]["scores"],
            "shuffled_event_scores": primary_extra["shuffled_event"]["scores"],
            "label_rows": primary_extra["label_rows"],
        }
        seed_summaries[str(seed)] = {
            "training": training_rows,
            "ndcg10": {name: float(values.mean()) for name, values in seed_arrays[seed]["ndcg"].items()},
            "wrong_history_ndcg10": float(primary_extra["wrong_history_ndcg"].mean()),
            "shuffled_event_ndcg10": float(primary_extra["shuffled_event_ndcg"].mean()),
            "parameter_counts": dict(parameter_counts),
            "initial_state_hashes": initial_hashes,
            "matched_initialization": len(set(initial_hashes.values())) == 1,
            "matched_parameters": len(set(parameter_counts.values())) == 1,
            "nohistory_bitwise_base": nohistory_exact[str(seed)],
            "query_absent_bitwise_base": query_absent_exact[str(seed)],
        }
        np.savez_compressed(
            root / f"seed_{seed}_request_metrics.npz",
            request_indices=np.asarray(probe_indices, dtype=np.int64),
            d2p=base_ndcg,
            **{name: values for name, values in mode_ndcg.items()},
            wrong_history=primary_extra["wrong_history_ndcg"],
            shuffled_event=primary_extra["shuffled_event_ndcg"],
        )

    base = np.mean(np.stack([seed_arrays[seed]["ndcg"]["d2p"] for seed in seeds]), axis=0)
    averaged_ndcg = {
        mode: np.mean(np.stack([seed_arrays[seed]["ndcg"][mode] for seed in seeds]), axis=0)
        for mode in MODES
    }
    references = {"d2p": base, **{mode: averaged_ndcg[mode] for mode in CONTROLS}}
    evaluation = config["evaluation"]
    comparisons = compare_primary(
        request_ids=request_ids,
        primary=averaged_ndcg[PRIMARY],
        references=references,
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]),
        folds=int(evaluation["hash_folds"]),
    )
    seed_differences = {
        reference: {
            str(seed): float(
                (
                    seed_arrays[seed]["ndcg"][PRIMARY]
                    - seed_arrays[seed]["ndcg"][reference]
                ).mean()
            )
            for seed in seeds
        }
        for reference in ("d2p", *CONTROLS)
    }
    wrong_average = np.mean(
        np.stack([seed_arrays[seed]["wrong_history_ndcg"] for seed in seeds]), axis=0
    )
    shuffle_average = np.mean(
        np.stack([seed_arrays[seed]["shuffled_event_ndcg"] for seed in seeds]), axis=0
    )
    clean_gain = averaged_ndcg[PRIMARY] - base
    corruptions = {
        "wrong_history": retention_bootstrap(
            clean_gain,
            wrong_average - base,
            samples=int(evaluation["bootstrap_samples"]),
            seed=int(evaluation["bootstrap_seed"]) + 101,
        ),
        "shuffled_event": retention_bootstrap(
            clean_gain,
            shuffle_average - base,
            samples=int(evaluation["bootstrap_samples"]),
            seed=int(evaluation["bootstrap_seed"]) + 102,
        ),
    }

    primary_score_rows = average_rows(
        [seed_arrays[seed]["outputs"][PRIMARY]["scores"] for seed in seeds]
    )
    base_score_rows = seed_arrays[seeds[0]]["outputs"][PRIMARY]["base_scores"]
    delta_rows = average_rows(
        [seed_arrays[seed]["outputs"][PRIMARY]["deltas"] for seed in seeds]
    )
    item_ids = seed_arrays[seeds[0]]["outputs"][PRIMARY]["item_ids"]
    label_rows = seed_arrays[seeds[0]]["label_rows"]
    order = order_summary(request_ids, item_ids, base_score_rows, primary_score_rows)
    clicked_values = clicked_minus_unclicked(delta_rows, label_rows)
    clicked = paired_bootstrap(
        clicked_values,
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]) + 201,
    )

    gate = config["gate"]
    checks: dict[str, bool] = {
        "primary_over_d2p_effect": comparisons["d2p"]["mean"] >= float(gate["ndcg10_delta_over_d2p_min"]),
        "primary_over_d2p_ci": comparisons["d2p"]["percentile_95_ci"][0] > 0.0,
        "primary_over_d2p_all_seeds": all(value > 0.0 for value in seed_differences["d2p"].values()),
        "all_comparison_hash_folds_positive": all(
            all(fold["mean_difference"] > 0.0 for fold in row["hash_folds"])
            for row in comparisons.values()
        ),
        "all_controls_effect": all(
            comparisons[mode]["mean"] >= float(gate["ndcg10_delta_over_each_control_min"])
            for mode in CONTROLS
        ),
        "all_controls_ci": all(comparisons[mode]["percentile_95_ci"][0] > 0.0 for mode in CONTROLS),
        "all_controls_all_seeds": all(
            all(value > 0.0 for value in seed_differences[mode].values()) for mode in CONTROLS
        ),
        "wrong_history_retention": corruptions["wrong_history"]["retention"] <= float(gate["corruption_retention_max"]),
        "wrong_history_retention_ci": corruptions["wrong_history"]["percentile_95_ci"][1] <= float(gate["corruption_retention_ci_high_max"]),
        "shuffled_event_retention": corruptions["shuffled_event"]["retention"] <= float(gate["corruption_retention_max"]),
        "shuffled_event_retention_ci": corruptions["shuffled_event"]["percentile_95_ci"][1] <= float(gate["corruption_retention_ci_high_max"]),
        "enough_order_changes": order["requests_with_any_order_change_fraction"] >= float(gate["requests_with_any_order_change_fraction_min"]),
        "enough_top10_changes": order["requests_with_top10_membership_change_fraction"] >= float(gate["requests_with_top10_membership_change_fraction_min"]),
        "clicked_delta_direction": clicked["percentile_95_ci"][0] > 0.0,
        "candidate_centred": maximum_centre_sum <= float(gate["candidate_center_sum_abs_max"]),
        "deterministic_rescore": deterministic_max <= float(gate["deterministic_rescore_max_abs_difference"]),
        "all_nohistory_bitwise_base": all(nohistory_exact.values()),
        "all_query_absent_bitwise_base": all(query_absent_exact.values()),
        "matched_parameters": len(set(parameter_counts.values())) == 1,
        "matched_initialization_each_seed": all(
            len(set(values.values())) == 1 for values in initial_hashes_by_seed.values()
        ),
        "all_training_finite": all(
            row["finite"]
            for seed_row in seed_summaries.values()
            for row in seed_row["training"].values()
        ),
        "selection_integrity": all(bool(value) for value in selection["checks"].values()),
    }
    elapsed = time.monotonic() - started
    report = {
        "candidate_id": "c21",
        "gate_id": config["gate_id"],
        "gate": "G1_train_only_path_signal_observability",
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "proposal_lock_sha256": lock_hash,
        "proposal_lock_status": lock["status"],
        "selection_sha256": sha256_file(config["paths"]["selection"]),
        "candidate_key_sha256_asserted_before_label_metrics": candidate_hashes,
        "cohort": {"train_fit": len(train_indices), "internal_probe": len(probe_indices)},
        "seed_summaries": seed_summaries,
        "seed_differences": seed_differences,
        "seed_averaged_ndcg10": {"d2p": float(base.mean()), **{mode: float(values.mean()) for mode, values in averaged_ndcg.items()}},
        "comparisons": comparisons,
        "corruptions": corruptions,
        "order_changes_on_seed_averaged_scores": order,
        "clicked_minus_unclicked_seed_averaged_delta": clicked,
        "maximum_abs_candidate_delta_sum": maximum_centre_sum,
        "deterministic_rescore_max_abs_difference": deterministic_max,
        "nohistory_bitwise_base": nohistory_exact,
        "query_absent_bitwise_base": query_absent_exact,
        "parameter_counts": parameter_counts,
        "initial_state_hashes": initial_hashes_by_seed,
        "checkpoints": checkpoints,
        "checks": checks,
        "boundaries": {
            "compact_c06_fit_labels_opened": True,
            "original_train_label_array_opened": False,
            "c06_internal_A_opened": False,
            "c06_internal_B_opened": False,
            "c06_escrow_opened": False,
            "dev_records_or_qrels_opened": False,
            "test_opened": False,
            "shared_dev_evaluator_calls": 0,
            "paper_claim_authorized": False,
            "full_transformer_authorized": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    atomic_json(report_path, report)
    attempt["status"] = "completed"
    attempt["completed_at"] = datetime.now(timezone.utc).isoformat()
    attempt["compact_fit_labels_opened_after_lock"] = True
    attempt["report_path"] = str(report_path)
    attempt["report_sha256"] = sha256_file(report_path)
    atomic_json(attempt_path, attempt)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("C21 requires CUDA")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("C21 is locked to physical GPU 1")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    result = preflight(config, device) if args.mode == "preflight" else formal(config, device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
