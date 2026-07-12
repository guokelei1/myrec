#!/usr/bin/env python
"""Prepare, train, score, and adjudicate the frozen C03 screening probe."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

TRAIN_DIR = Path(__file__).resolve().parent
CANDIDATE_ROOT = TRAIN_DIR.parent
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from features import (
    format_coarse,
    format_item,
    format_query,
    load_dev_text_store,
    prepare_dev_text_store,
    prepare_train_features,
)
from io_utils import (
    assert_manifest,
    assert_run_id,
    assert_safe_input,
    candidate_source_hash,
    iter_jsonl,
    load_config,
    repo_path,
    set_determinism,
    sha256_file,
    stable_i63,
    stable_u64,
    write_json,
)
from model.triadic_transport import TriadicTransportRanker, center_request_residual
from probe_data import ProbeCandidateDataset, collate_candidates, to_device


CORRUPTIONS = ("wrong_user", "shuffle", "query_mask", "coarse_only")
DIAGNOSTIC_OPERATORS = ("cycle_null", "softmax", "no_null", "no_cycle", "mean_pool")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "command",
        choices=("prepare", "train", "diagnose", "determinism", "score", "adjudicate"),
    )
    return parser.parse_args()


def require_assigned_gpu(config: dict[str, Any]) -> torch.device:
    expected = str(config["candidate"]["physical_gpu"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != expected:
        raise RuntimeError(
            f"C03 requires CUDA_VISIBLE_DEVICES={expected}; observed {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C03 requires exactly one program-visible CUDA device")
    if str(config["candidate"]["visible_device"]) != "cuda:0":
        raise RuntimeError("program device contract must remain cuda:0")
    return torch.device("cuda:0")


def build_model(config: dict[str, Any], device: torch.device) -> TriadicTransportRanker:
    values = config["model"]
    model = TriadicTransportRanker(
        input_dim=int(values["input_dim"]),
        hidden_dim=int(values["hidden_dim"]),
        num_heads=int(values["num_heads"]),
        num_layers=int(values["num_layers"]),
        ff_dim=int(values["ff_dim"]),
        max_history=int(config["data"]["max_history"]),
        transport_dim=int(values["transport_dim"]),
        sinkhorn_iterations=int(values["sinkhorn_iterations"]),
        sinkhorn_temperature=float(values["sinkhorn_temperature"]),
        cycle_lambda=float(values["cycle_lambda"]),
        identity_bonus_floor=float(values["identity_bonus_floor"]),
        dropout=float(values["dropout"]),
    )
    return model.to(device)


def feature_path(config: dict[str, Any]) -> Path:
    return repo_path(config["paths"]["output_dir"]) / "train_features.pt"


def checkpoint_path(config: dict[str, Any]) -> Path:
    return repo_path(config["paths"]["checkpoint"])


def _record_gpu_time(config: dict[str, Any], stage: str, seconds: float) -> None:
    output_dir = repo_path(config["paths"]["output_dir"])
    path = output_dir / "gpu_ledger.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {"physical_gpu": 2, "stages": []}
    payload["stages"].append({"stage": stage, "wall_seconds_on_assigned_gpu": seconds})
    payload["total_seconds"] = sum(
        float(row["wall_seconds_on_assigned_gpu"]) for row in payload["stages"]
    )
    payload["total_gpu_hours_upper_bound"] = payload["total_seconds"] / 3600.0
    limit = float(config["budget"]["max_a40_gpu_hours"])
    if payload["total_gpu_hours_upper_bound"] > limit:
        raise RuntimeError(f"C03 GPU budget exceeded: {payload['total_gpu_hours_upper_bound']} > {limit}")
    write_json(path, payload)


def command_prepare(config: dict[str, Any]) -> None:
    device = require_assigned_gpu(config)
    start = time.monotonic()
    train_metadata = prepare_train_features(config, device)
    dev_metadata = prepare_dev_text_store(config, device)
    seconds = time.monotonic() - start
    _record_gpu_time(config, "prepare_frozen_bge_features", seconds)
    write_json(
        repo_path(config["paths"]["output_dir"]) / "prepare_report.json",
        {
            "train": train_metadata,
            "dev": dev_metadata,
            "seconds": seconds,
            "gpu": torch.cuda.get_device_name(0),
            "qrels_read": False,
            "test_read": False,
        },
    )


def _wrong_permutation(user_hashes: Tensor) -> Tensor:
    count = int(user_hashes.numel())
    if count < 2:
        return torch.arange(count, device=user_hashes.device)
    result = torch.empty(count, dtype=torch.long, device=user_hashes.device)
    for index in range(count):
        donor = (index + 1) % count
        while donor != index and user_hashes[donor].item() == user_hashes[index].item():
            donor = (donor + 1) % count
        result[index] = donor
    return result


def _reverse_valid(tensor: Tensor, mask: Tensor) -> Tensor:
    result = tensor.clone()
    for index in range(tensor.shape[0]):
        count = int(mask[index].sum().item())
        if count > 1:
            result[index, :count] = tensor[index, :count].flip(0)
    return result


def _corrupted_inputs(batch: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
    wrong = _wrong_permutation(batch["user_hash"])
    wrong_history = batch["history"][wrong]
    wrong_mask = batch["history_mask"][wrong]
    wrong_event = batch["event_types"][wrong]
    wrong_hashes = batch["history_item_hashes"][wrong]
    wrong_exact = wrong_hashes.eq(batch["candidate_item_hash"].unsqueeze(1)) & wrong_mask

    shuffled_history = _reverse_valid(batch["history"], batch["history_mask"])
    shuffled_event = _reverse_valid(batch["event_types"], batch["history_mask"])
    shuffled_exact = _reverse_valid(batch["exact_match"], batch["history_mask"])
    return {
        "wrong_user": {
            "query": batch["query"],
            "history": wrong_history,
            "history_mask": wrong_mask,
            "event_types": wrong_event,
            "exact_match": wrong_exact,
        },
        "shuffle": {
            "query": batch["query"],
            "history": shuffled_history,
            "history_mask": batch["history_mask"],
            "event_types": shuffled_event,
            "exact_match": shuffled_exact,
        },
        "query_mask": {
            "query": torch.zeros_like(batch["query"]),
            "history": batch["history"],
            "history_mask": batch["history_mask"],
            "event_types": batch["event_types"],
            "exact_match": batch["exact_match"],
        },
        "coarse_only": {
            "query": batch["query"],
            "history": batch["coarse_history"],
            "history_mask": batch["history_mask"],
            "event_types": batch["event_types"],
            "exact_match": torch.zeros_like(batch["exact_match"]),
        },
    }


def _forward(
    model: TriadicTransportRanker,
    batch: dict[str, Any],
    *,
    operator: str,
    override: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    values = override or batch
    return model(
        values["query"],
        batch["candidate"],
        values["history"],
        values["history_mask"],
        values["exact_match"],
        event_types=values["event_types"],
        operator=operator,
    )


def _training_loss(
    model: TriadicTransportRanker,
    batch: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    train = config["training"]
    scale = float(config["model"]["final_residual_scale"])
    true = _forward(model, batch, operator="cycle_null")
    logits = true["base_logit"] + scale * true["raw_residual"]
    ranking_loss = F.binary_cross_entropy_with_logits(logits, batch["label"])

    eligible = (
        batch["label"].gt(0.5)
        & batch["history_mask"].any(dim=1)
        & ~batch["exact_match"].any(dim=1)
    )
    score_loss = logits.new_zeros(())
    null_loss = logits.new_zeros(())
    if eligible.any():
        corruptions = _corrupted_inputs(batch)
        score_terms = []
        null_terms = []
        for name in CORRUPTIONS:
            corrupt = _forward(
                model,
                batch,
                operator="cycle_null",
                override=corruptions[name],
            )
            score_terms.append(
                F.relu(
                    float(train["corruption_margin"])
                    - true["raw_residual"][eligible]
                    + corrupt["raw_residual"][eligible]
                ).mean()
            )
            null_terms.append(
                F.relu(
                    float(train["null_margin"])
                    - corrupt["null_mass"][eligible]
                    + true["null_mass"][eligible]
                ).mean()
            )
        score_loss = torch.stack(score_terms).mean()
        null_loss = torch.stack(null_terms).mean()

    exact = batch["history_mask"].any(dim=1) & batch["exact_match"].any(dim=1)
    exact_loss = logits.new_zeros(())
    if exact.any():
        exact_loss = F.relu(
            float(train["exact_mass_floor"]) - true["trusted_mass"][exact]
        ).mean()
    no_history = ~batch["history_mask"].any(dim=1)
    no_history_loss = (
        true["raw_residual"][no_history].abs().sum() if no_history.any() else logits.new_zeros(())
    )
    total = (
        float(train["ranking_weight"]) * ranking_loss
        + float(train["corruption_score_weight"]) * score_loss
        + float(train["corruption_null_weight"]) * null_loss
        + float(train["exact_mass_weight"]) * exact_loss
        + no_history_loss
    )
    return total, {
        "total": float(total.detach()),
        "ranking": float(ranking_loss.detach()),
        "corruption_score": float(score_loss.detach()),
        "corruption_null": float(null_loss.detach()),
        "exact_mass": float(exact_loss.detach()),
        "eligible": int(eligible.sum().item()),
    }


@torch.no_grad()
def _mean_validation_loss(
    model: TriadicTransportRanker,
    loader: DataLoader[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        loss, _ = _training_loss(model, batch, config)
        batch_count = int(batch["label"].numel())
        total += float(loss) * batch_count
        count += batch_count
    return total / max(count, 1)


def command_train(config: dict[str, Any], config_path: Path) -> None:
    device = require_assigned_gpu(config)
    assert_manifest(config)
    assert_run_id(config)
    set_determinism(int(config["candidate"]["seed"]))
    path = feature_path(config)
    if not path.exists():
        raise FileNotFoundError(f"prepare must run first: {path}")
    fit = ProbeCandidateDataset(path, validation=False)
    validation = ProbeCandidateDataset(path, validation=True)
    generator = torch.Generator().manual_seed(int(config["candidate"]["seed"]))
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "collate_fn": collate_candidates,
        "pin_memory": True,
    }
    fit_loader = DataLoader(fit, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation, shuffle=False, **loader_kwargs)
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    start = time.monotonic()
    epochs: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_validation = math.inf
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        running: defaultdict[str, float] = defaultdict(float)
        examples = 0
        for raw_batch in fit_loader:
            batch = to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, components = _training_loss(model, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(f"non-finite/missing gradients at epoch {epoch}")
            nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            batch_count = int(batch["label"].numel())
            examples += batch_count
            for key, value in components.items():
                running[key] += float(value) * batch_count
        validation_loss = _mean_validation_loss(model, validation_loader, config, device)
        epoch_row = {
            "epoch": epoch,
            "fit_examples": examples,
            **{key: value / max(examples, 1) for key, value in running.items()},
            "validation_loss": validation_loss,
        }
        epochs.append(epoch_row)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    seconds = time.monotonic() - start
    checkpoint = checkpoint_path(config)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "config": config,
            "best_validation_loss": best_validation,
            "epochs": epochs,
            "candidate_source_hash": candidate_source_hash(),
            "config_sha256": sha256_file(config_path),
            "seed": int(config["candidate"]["seed"]),
            "qrels_read": False,
            "test_read": False,
        },
        checkpoint,
    )
    _record_gpu_time(config, "train_internal_probe", seconds)
    write_json(
        repo_path(config["paths"]["output_dir"]) / "training_report.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "best_validation_loss": best_validation,
            "epochs": epochs,
            "seconds": seconds,
            "fit_candidates": len(fit),
            "validation_candidates": len(validation),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "qrels_read": False,
            "test_read": False,
        },
    )


def load_trained_model(config: dict[str, Any], device: torch.device) -> TriadicTransportRanker:
    checkpoint = checkpoint_path(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config, device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def _empty_diagnostic_accumulator() -> dict[str, Any]:
    return {
        operator: {
            corruption: {"score_drop_sum": 0.0, "null_increase_sum": 0.0, "count": 0}
            for corruption in CORRUPTIONS
        }
        for operator in DIAGNOSTIC_OPERATORS
    }


@torch.no_grad()
def internal_diagnostics(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    validation = ProbeCandidateDataset(feature_path(config), validation=True)
    loader = DataLoader(
        validation,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_candidates,
    )
    model = load_trained_model(config, device)
    accumulator = _empty_diagnostic_accumulator()
    exact_mass_sum = 0.0
    exact_residual_by_request: defaultdict[str, list[tuple[float, bool]]] = defaultdict(list)
    exact_count = 0
    no_history_max = 0.0
    max_marginal_error = 0.0
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        corruptions = _corrupted_inputs(batch)
        eligible = (
            batch["label"].gt(0.5)
            & batch["history_mask"].any(dim=1)
            & ~batch["exact_match"].any(dim=1)
        )
        for operator in DIAGNOSTIC_OPERATORS:
            true = _forward(model, batch, operator=operator)
            if operator == "cycle_null":
                max_marginal_error = max(
                    max_marginal_error, float(true["marginal_error"].max().cpu())
                )
                exact = batch["exact_match"].any(dim=1) & batch["history_mask"].any(dim=1)
                if exact.any():
                    exact_mass_sum += float(true["trusted_mass"][exact].sum().cpu())
                    exact_count += int(exact.sum().item())
                no_history = ~batch["history_mask"].any(dim=1)
                if no_history.any():
                    no_history_max = max(
                        no_history_max, float(true["raw_residual"][no_history].abs().max().cpu())
                    )
                for request_id, residual, is_exact in zip(
                    batch["request_id"],
                    true["raw_residual"].cpu().tolist(),
                    batch["exact_match"].any(dim=1).cpu().tolist(),
                ):
                    exact_residual_by_request[request_id].append((float(residual), bool(is_exact)))
            if not eligible.any():
                continue
            for corruption in CORRUPTIONS:
                changed = _forward(
                    model,
                    batch,
                    operator=operator,
                    override=corruptions[corruption],
                )
                row = accumulator[operator][corruption]
                row["score_drop_sum"] += float(
                    (true["raw_residual"][eligible] - changed["raw_residual"][eligible])
                    .sum()
                    .cpu()
                )
                row["null_increase_sum"] += float(
                    (changed["null_mass"][eligible] - true["null_mass"][eligible]).sum().cpu()
                )
                row["count"] += int(eligible.sum().item())

    exact_centered: list[float] = []
    for rows in exact_residual_by_request.values():
        raw = torch.tensor([row[0] for row in rows])
        centered = center_request_residual(raw, float(config["model"]["final_residual_scale"]))
        exact_centered.extend(
            float(value) for value, row in zip(centered.tolist(), rows) if row[1]
        )
    summary: dict[str, Any] = {}
    for operator, by_corruption in accumulator.items():
        summary[operator] = {}
        for corruption, values in by_corruption.items():
            count = max(int(values["count"]), 1)
            summary[operator][corruption] = {
                "count": int(values["count"]),
                "mean_score_drop": values["score_drop_sum"] / count,
                "mean_null_increase": values["null_increase_sum"] / count,
            }
    main = summary["cycle_null"]
    null_required = float(config["gate"]["null_increase_required"])
    required_count = int(config["gate"]["corruptions_required"])
    main_min_drop = min(main[name]["mean_score_drop"] for name in CORRUPTIONS)
    softmax_min_drop = min(
        summary["softmax"][name]["mean_score_drop"] for name in CORRUPTIONS
    )
    no_null_min_drop = min(
        summary["no_null"][name]["mean_score_drop"] for name in CORRUPTIONS
    )
    advantage = float(config["gate"]["control_advantage_required"])
    cycle_selectivity = max(
        main[name]["mean_score_drop"] - summary["no_cycle"][name]["mean_score_drop"]
        for name in ("shuffle", "query_mask")
    )
    checks = {
        "all_primary_score_drops_positive": all(
            main[name]["mean_score_drop"] > 0 for name in CORRUPTIONS
        ),
        "three_null_increases_at_least_threshold": sum(
            main[name]["mean_null_increase"] >= null_required for name in CORRUPTIONS
        )
        >= required_count,
        "all_null_increases_nonnegative": all(
            main[name]["mean_null_increase"] >= 0 for name in CORRUPTIONS
        ),
        "beats_softmax_min_drop": main_min_drop - softmax_min_drop >= advantage,
        "beats_no_null_min_drop": main_min_drop - no_null_min_drop >= advantage,
        "cycle_adds_shuffle_or_query_selectivity": cycle_selectivity >= advantage,
        "exact_trusted_mass_at_least_floor": (
            exact_mass_sum / max(exact_count, 1)
            >= float(config["training"]["exact_mass_floor"])
        ),
        "exact_centered_residual_nonnegative": (
            sum(exact_centered) / max(len(exact_centered), 1) >= 0
        ),
        "no_history_residual_exact_zero": no_history_max == 0.0,
        "marginal_error_within_tolerance": (
            max_marginal_error <= float(config["gate"]["float32_mass_tolerance"])
        ),
    }
    return {
        "operators": summary,
        "exact": {
            "count": exact_count,
            "mean_trusted_mass": exact_mass_sum / max(exact_count, 1),
            "mean_centered_residual": sum(exact_centered) / max(len(exact_centered), 1),
        },
        "no_history_max_abs_residual": no_history_max,
        "max_marginal_error": max_marginal_error,
        "checks": checks,
        "passed": all(checks.values()),
        "qrels_read": False,
        "test_read": False,
    }


def _embedding_lookup(
    texts: list[str], embeddings: Tensor, index: dict[str, int]
) -> Tensor:
    try:
        return torch.stack([embeddings[index[text]] for text in texts])
    except KeyError as exc:
        raise KeyError(f"dev text missing from frozen store: {exc.args[0][:120]!r}") from exc


def _event_type(event: object) -> int:
    return 2 if str(event) == "purchase" else 1


def _record_model_inputs(
    record: dict[str, Any],
    embeddings: Tensor,
    index: dict[str, int],
    max_history: int,
) -> dict[str, Any]:
    history_items = list(record.get("history", []))[-max_history:]
    candidates = list(record.get("candidates", []))
    history = torch.zeros((max_history, embeddings.shape[1]), dtype=torch.float32)
    coarse = torch.zeros_like(history)
    mask = torch.zeros(max_history, dtype=torch.bool)
    event_types = torch.zeros(max_history, dtype=torch.long)
    history_ids: list[str] = []
    if history_items:
        count = len(history_items)
        history[:count] = _embedding_lookup(
            [format_item(item) for item in history_items], embeddings, index
        ).float()
        coarse[:count] = _embedding_lookup(
            [format_coarse(item) for item in history_items], embeddings, index
        ).float()
        mask[:count] = True
        event_types[:count] = torch.tensor(
            [_event_type(item.get("event")) for item in history_items], dtype=torch.long
        )
        history_ids = [str(item.get("item_id", "")) for item in history_items]
    candidate_ids = [str(item.get("item_id", "")) for item in candidates]
    exact = torch.zeros((len(candidates), max_history), dtype=torch.bool)
    for candidate_index, candidate_id in enumerate(candidate_ids):
        for history_index, history_id in enumerate(history_ids):
            exact[candidate_index, history_index] = candidate_id == history_id
    return {
        "request_id": str(record["request_id"]),
        "user_id": str(record.get("user_id", "")),
        "query": embeddings[index[format_query(record.get("query", ""))]].float(),
        "history": history,
        "coarse_history": coarse,
        "history_mask": mask,
        "event_types": event_types,
        "history_ids": history_ids,
        "candidates": _embedding_lookup(
            [format_item(item) for item in candidates], embeddings, index
        ).float(),
        "candidate_ids": candidate_ids,
        "exact_match": exact,
    }


def _load_base_scores(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    path = assert_safe_input(
        config["paths"]["d2p_dev_scores"], config["integrity"]["reject_path_tokens"]
    )
    scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        item_id = str(row["candidate_item_id"])
        if item_id in scores[request_id]:
            raise ValueError(f"duplicate D2p score: {request_id}/{item_id}")
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError("non-finite D2p score")
        scores[request_id][item_id] = score
    return dict(scores)


@torch.no_grad()
def _score_one_record(
    model: TriadicTransportRanker,
    prepared: dict[str, Any],
    base_scores: dict[str, float],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_count = int(prepared["candidates"].shape[0])
    raw_parts: list[Tensor] = []
    trusted_parts: list[Tensor] = []
    null_parts: list[Tensor] = []
    cycle_parts: list[Tensor] = []
    max_error = 0.0
    batch_size = int(config["training"]["batch_size"])
    for start in range(0, candidate_count, batch_size):
        stop = min(start + batch_size, candidate_count)
        count = stop - start
        history = prepared["history"].unsqueeze(0).expand(count, -1, -1).to(device)
        history_mask = prepared["history_mask"].unsqueeze(0).expand(count, -1).to(device)
        event_types = prepared["event_types"].unsqueeze(0).expand(count, -1).to(device)
        output = model(
            prepared["query"].unsqueeze(0).expand(count, -1).to(device),
            prepared["candidates"][start:stop].to(device),
            history,
            history_mask,
            prepared["exact_match"][start:stop].to(device),
            event_types=event_types,
            operator="cycle_null",
        )
        raw_parts.append(output["raw_residual"].cpu())
        trusted_parts.append(output["trusted_mass"].cpu())
        null_parts.append(output["null_mass"].cpu())
        cycle_parts.append(output["cycle_gap"].cpu())
        max_error = max(max_error, float(output["marginal_error"].max().cpu()))
    raw = torch.cat(raw_parts)
    has_history = bool(prepared["history_mask"].any())
    centered = (
        center_request_residual(raw, float(config["model"]["final_residual_scale"]))
        if has_history
        else torch.zeros_like(raw)
    )
    rows: list[dict[str, Any]] = []
    no_history_mismatches = 0
    for item_id, residual in zip(prepared["candidate_ids"], centered.tolist()):
        if item_id not in base_scores:
            raise ValueError(f"D2p missing candidate {prepared['request_id']}/{item_id}")
        base = float(base_scores[item_id])
        score = base if not has_history else base + float(residual)
        if not has_history and score != base:
            no_history_mismatches += 1
        if not math.isfinite(score):
            raise FloatingPointError("C03 emitted non-finite score")
        rows.append(
            {
                "request_id": prepared["request_id"],
                "candidate_item_id": item_id,
                "score": score,
                "method_id": config["candidate"]["method_id"],
            }
        )
    trusted = torch.cat(trusted_parts)
    null = torch.cat(null_parts)
    cycle = torch.cat(cycle_parts)
    return rows, {
        "request_id": prepared["request_id"],
        "history_present": has_history,
        "repeat_present": bool(prepared["exact_match"].any()),
        "candidate_count": candidate_count,
        "mean_trusted_mass": float(trusted.mean()),
        "mean_null_mass": float(null.mean()),
        "mean_cycle_gap": float(cycle.mean()),
        "max_abs_raw_residual": float(raw.abs().max()),
        "max_marginal_error": max_error,
        "no_history_score_mismatches": no_history_mismatches,
    }


def _git_metadata() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _score_records(
    config: dict[str, Any],
    config_path: Path,
    device: torch.device,
    *,
    scores_path: Path,
    diagnostics_path: Path,
    limit_requests: int | None,
) -> dict[str, Any]:
    assert_manifest(config)
    reject = config["integrity"]["reject_path_tokens"]
    dev_path = assert_safe_input(config["paths"]["records_dev"], reject)
    embeddings, index, store_metadata = load_dev_text_store(config)
    if store_metadata["records_dev_sha256"] != sha256_file(dev_path):
        raise ValueError("frozen dev text store no longer matches label-free records")
    model = load_trained_model(config, device)
    base = _load_base_scores(config)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    request_count = 0
    candidate_count = 0
    no_history_mismatches = 0
    max_marginal_error = 0.0
    with scores_path.open("w", encoding="utf-8") as score_handle, diagnostics_path.open(
        "w", encoding="utf-8"
    ) as diagnostic_handle:
        for record in iter_jsonl(dev_path):
            if limit_requests is not None and request_count >= limit_requests:
                break
            prepared = _record_model_inputs(
                record, embeddings, index, int(config["data"]["max_history"])
            )
            request_id = prepared["request_id"]
            if request_id not in base:
                raise ValueError(f"D2p missing request: {request_id}")
            rows, diagnostic = _score_one_record(
                model, prepared, base[request_id], config, device
            )
            for row in rows:
                score_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            diagnostic_handle.write(
                json.dumps(diagnostic, ensure_ascii=False, sort_keys=True) + "\n"
            )
            request_count += 1
            candidate_count += len(rows)
            no_history_mismatches += int(diagnostic["no_history_score_mismatches"])
            max_marginal_error = max(
                max_marginal_error, float(diagnostic["max_marginal_error"])
            )
    return {
        "requests": request_count,
        "candidate_rows": candidate_count,
        "no_history_score_mismatches": no_history_mismatches,
        "max_marginal_error": max_marginal_error,
        "scores_sha256": sha256_file(scores_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path(config)),
        "candidate_source_hash": candidate_source_hash(),
        "qrels_read": False,
        "test_read": False,
    }


def command_determinism(config: dict[str, Any], config_path: Path) -> None:
    device = require_assigned_gpu(config)
    set_determinism(int(config["candidate"]["seed"]))
    output_dir = repo_path(config["paths"]["output_dir"]) / "determinism"
    start = time.monotonic()
    reports = []
    for suffix in ("a", "b"):
        reports.append(
            _score_records(
                config,
                config_path,
                device,
                scores_path=output_dir / f"scores_{suffix}.jsonl",
                diagnostics_path=output_dir / f"diagnostics_{suffix}.jsonl",
                limit_requests=int(config["data"]["deterministic_requests"]),
            )
        )
    seconds = time.monotonic() - start
    score_equal = (output_dir / "scores_a.jsonl").read_bytes() == (
        output_dir / "scores_b.jsonl"
    ).read_bytes()
    diagnostic_equal = (output_dir / "diagnostics_a.jsonl").read_bytes() == (
        output_dir / "diagnostics_b.jsonl"
    ).read_bytes()
    report = {
        "runs": reports,
        "score_files_byte_identical": score_equal,
        "diagnostic_files_byte_identical": diagnostic_equal,
        "passed": score_equal
        and diagnostic_equal
        and all(row["no_history_score_mismatches"] == 0 for row in reports),
        "seconds": seconds,
        "qrels_read": False,
        "test_read": False,
    }
    _record_gpu_time(config, "deterministic_rescore_2x", seconds)
    write_json(output_dir / "report.json", report)
    if not report["passed"]:
        raise RuntimeError("deterministic rescore gate failed")


def _select_dev_diagnostic_records(
    path: Path, *, count: int, seed: int
) -> list[dict[str, Any]]:
    heap: list[tuple[int, str, dict[str, Any]]] = []
    for record in iter_jsonl(path):
        request_id = str(record["request_id"])
        priority = stable_u64("c03-dev-diagnostic", seed, request_id)
        entry = (-priority, request_id, record)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    selected = [(-value, request_id, record) for value, request_id, record in heap]
    selected.sort(key=lambda row: (row[0], row[1]))
    return [record for _, _, record in selected]


@torch.no_grad()
def dev_corruption_diagnostics(
    config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    reject = config["integrity"]["reject_path_tokens"]
    dev_path = assert_safe_input(config["paths"]["records_dev"], reject)
    embeddings, index, _ = load_dev_text_store(config)
    selected = _select_dev_diagnostic_records(
        dev_path,
        count=int(config["data"]["dev_diagnostic_requests"]),
        seed=int(config["candidate"]["seed"]),
    )
    prepared = [
        _record_model_inputs(row, embeddings, index, int(config["data"]["max_history"]))
        for row in selected
    ]
    model = load_trained_model(config, device)
    accumulator = _empty_diagnostic_accumulator()
    for request_index, primary in enumerate(prepared):
        if not primary["history_mask"].any():
            continue
        donor_index = (request_index + 1) % len(prepared)
        while (
            donor_index != request_index
            and prepared[donor_index]["user_id"] == primary["user_id"]
        ):
            donor_index = (donor_index + 1) % len(prepared)
        donor = prepared[donor_index]
        candidate_count = len(primary["candidate_ids"])
        query = primary["query"].unsqueeze(0).expand(candidate_count, -1).to(device)
        candidate = primary["candidates"].to(device)

        def expand_history(source: dict[str, Any], *, coarse: bool = False) -> dict[str, Tensor]:
            values = source["coarse_history"] if coarse else source["history"]
            source_ids = [] if coarse else source["history_ids"]
            exact = torch.zeros((candidate_count, values.shape[0]), dtype=torch.bool)
            for c_index, candidate_id in enumerate(primary["candidate_ids"]):
                for h_index, history_id in enumerate(source_ids):
                    exact[c_index, h_index] = candidate_id == history_id
            return {
                "history": values.unsqueeze(0).expand(candidate_count, -1, -1).to(device),
                "history_mask": source["history_mask"]
                .unsqueeze(0)
                .expand(candidate_count, -1)
                .to(device),
                "event_types": source["event_types"]
                .unsqueeze(0)
                .expand(candidate_count, -1)
                .to(device),
                "exact_match": exact.to(device),
            }

        true_history = expand_history(primary)
        wrong_history = expand_history(donor)
        coarse_history = expand_history(primary, coarse=True)
        shuffled_history = {
            key: _reverse_valid(value, true_history["history_mask"])
            if key in {"history", "event_types", "exact_match"}
            else value
            for key, value in true_history.items()
        }
        variants = {
            "wrong_user": (query, wrong_history),
            "shuffle": (query, shuffled_history),
            "query_mask": (torch.zeros_like(query), true_history),
            "coarse_only": (query, coarse_history),
        }
        for operator in DIAGNOSTIC_OPERATORS:
            true = model(
                query,
                candidate,
                true_history["history"],
                true_history["history_mask"],
                true_history["exact_match"],
                event_types=true_history["event_types"],
                operator=operator,
            )
            for name, (variant_query, variant) in variants.items():
                changed = model(
                    variant_query,
                    candidate,
                    variant["history"],
                    variant["history_mask"],
                    variant["exact_match"],
                    event_types=variant["event_types"],
                    operator=operator,
                )
                row = accumulator[operator][name]
                row["score_drop_sum"] += float(
                    (true["raw_residual"] - changed["raw_residual"]).sum().cpu()
                )
                row["null_increase_sum"] += float(
                    (changed["null_mass"] - true["null_mass"]).sum().cpu()
                )
                row["count"] += candidate_count
    summary: dict[str, Any] = {}
    for operator, rows in accumulator.items():
        summary[operator] = {}
        for name, values in rows.items():
            count = max(int(values["count"]), 1)
            summary[operator][name] = {
                "count": int(values["count"]),
                "mean_score_drop": values["score_drop_sum"] / count,
                "mean_null_increase": values["null_increase_sum"] / count,
            }
    main = summary["cycle_null"]
    threshold = float(config["gate"]["null_increase_required"])
    checks = {
        "three_null_increases_at_least_threshold": sum(
            main[name]["mean_null_increase"] >= threshold for name in CORRUPTIONS
        )
        >= int(config["gate"]["corruptions_required"]),
        "none_decreases_null_beyond_tolerance": all(
            main[name]["mean_null_increase"] >= -0.005 for name in CORRUPTIONS
        ),
    }
    return {
        "selected_requests": len(selected),
        "operators": summary,
        "checks": checks,
        "passed": all(checks.values()),
        "selection_uses_labels": False,
        "qrels_read": False,
        "test_read": False,
    }


def command_diagnose(config: dict[str, Any]) -> None:
    device = require_assigned_gpu(config)
    set_determinism(int(config["candidate"]["seed"]))
    start = time.monotonic()
    internal = internal_diagnostics(config, device)
    dev = dev_corruption_diagnostics(config, device)
    seconds = time.monotonic() - start
    output_dir = repo_path(config["paths"]["output_dir"])
    write_json(output_dir / "internal_diagnostics.json", internal)
    write_json(output_dir / "dev_corruption_diagnostics.json", dev)
    _record_gpu_time(config, "internal_and_label_free_dev_diagnostics", seconds)


def command_score(config: dict[str, Any], config_path: Path) -> None:
    device = require_assigned_gpu(config)
    set_determinism(int(config["candidate"]["seed"]))
    run_id = assert_run_id(config)
    run_dir = repo_path(config["paths"]["run_dir"])
    expected_run_dir = REPO_ROOT / "runs" / run_id
    if run_dir.resolve() != expected_run_dir.resolve():
        raise ValueError("run_dir does not match frozen C03 run ID")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "run.lock"
    if lock.exists():
        raise FileExistsError(f"run lock already exists: {lock}")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    start = time.monotonic()
    try:
        report = _score_records(
            config,
            config_path,
            device,
            scores_path=run_dir / "scores.jsonl",
            diagnostics_path=run_dir / "transport_diagnostics.jsonl",
            limit_requests=None,
        )
        seconds = time.monotonic() - start
        commit, dirty = _git_metadata()
        metadata = {
            "run_id": run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_commit": commit,
            "git_dirty": dirty,
            "dataset_id": "kuaisearch",
            "dataset_version": "v0_lite",
            "split_id": "time_80_10_10_seed20260708",
            "candidate_manifest_sha256": config["integrity"]["candidate_manifest_sha256"],
            "method_id": config["candidate"]["method_id"],
            "method_group": "system-03",
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "config_sha256": sha256_file(config_path),
            "seed": int(config["candidate"]["seed"]),
            "env_group": "system-03",
            "env_name": config["candidate"]["environment"],
            "python": sys.version.split()[0],
            "packages": {
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "program_device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "hostname": socket.gethostname(),
            "command": "run_probe.py --config configs/c03_screening.yaml score",
            "checkpoint_sha256": report["checkpoint_sha256"],
            "candidate_source_hash": report["candidate_source_hash"],
            "scores_sha256": report["scores_sha256"],
            "transport_diagnostics_sha256": report["diagnostics_sha256"],
            "score_seconds": seconds,
            "score_rows": report["candidate_rows"],
            "requests": report["requests"],
            "no_history_score_mismatches": report["no_history_score_mismatches"],
            "qrels_read_by_training_or_scoring": False,
            "test_read": False,
        }
        write_json(run_dir / "metadata.json", metadata)
        _record_gpu_time(config, "full_dev_label_free_scoring", seconds)
        if report["no_history_score_mismatches"]:
            raise RuntimeError("no-history D2p score contract failed")
    finally:
        lock.unlink(missing_ok=True)


def _load_per_request(path: Path) -> dict[str, float]:
    return {str(row["request_id"]): float(row["ndcg@10"]) for row in iter_jsonl(path)}


def _paired_bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch = 128
    for start in range(0, samples, batch):
        stop = min(start + batch, samples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _load_score_subset(path: Path, request_ids: set[str]) -> dict[str, dict[str, float]]:
    result: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in request_ids:
            result[request_id][str(row["candidate_item_id"])] = float(row["score"])
    return dict(result)


def _rank_items(request_id: str, scores: dict[str, float]) -> list[str]:
    import hashlib

    return sorted(
        scores,
        key=lambda item_id: (
            -scores[item_id],
            hashlib.sha256((request_id + item_id + "20260708").encode()).hexdigest(),
        ),
    )


def command_adjudicate(config: dict[str, Any]) -> None:
    run_dir = repo_path(config["paths"]["run_dir"])
    metrics_path = run_dir / "metrics.json"
    per_request_path = run_dir / "per_request_metrics.jsonl"
    if not metrics_path.exists() or not per_request_path.exists():
        raise FileNotFoundError("shared evaluator outputs are required before adjudication")
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("candidate_manifest_sha256") != config["integrity"][
        "candidate_manifest_sha256"
    ]:
        raise ValueError("evaluator candidate hash mismatch")
    reject = config["integrity"]["reject_path_tokens"]
    dev_path = assert_safe_input(config["paths"]["records_dev"], reject)
    repeat_ids: set[str] = set()
    nonrepeat_ids: set[str] = set()
    no_history_ids: set[str] = set()
    for record in iter_jsonl(dev_path):
        request_id = str(record["request_id"])
        history_ids = {str(item.get("item_id", "")) for item in record.get("history", [])}
        if not history_ids:
            no_history_ids.add(request_id)
        elif any(
            str(candidate.get("item_id", "")) in history_ids
            for candidate in record.get("candidates", [])
        ):
            repeat_ids.add(request_id)
        else:
            nonrepeat_ids.add(request_id)
    subset_counts = {
        "repeat_present": len(repeat_ids),
        "nonrepeat_history_present": len(nonrepeat_ids),
        "no_history": len(no_history_ids),
    }
    expected_counts = {
        "repeat_present": 3442,
        "nonrepeat_history_present": 4677,
        "no_history": 4110,
    }
    primary = _load_per_request(per_request_path)
    item = _load_per_request(
        assert_safe_input(config["paths"]["item_dev_per_request"], reject)
    )
    d2p = _load_per_request(
        assert_safe_input(config["paths"]["d2p_dev_per_request"], reject)
    )

    repeat_deltas = np.asarray([primary[key] - item[key] for key in sorted(repeat_ids)])
    nonrepeat_deltas = np.asarray([primary[key] - d2p[key] for key in sorted(nonrepeat_ids)])
    bootstrap = _paired_bootstrap(
        nonrepeat_deltas,
        int(config["gate"]["bootstrap_samples"]),
        int(config["gate"]["bootstrap_seed"]),
    )
    primary_scores = _load_score_subset(run_dir / "scores.jsonl", no_history_ids)
    d2p_scores = _load_score_subset(
        assert_safe_input(config["paths"]["d2p_dev_scores"], reject), no_history_ids
    )
    score_mismatches = 0
    rank_mismatches = 0
    for request_id in no_history_ids:
        if primary_scores.get(request_id) != d2p_scores.get(request_id):
            left = primary_scores.get(request_id, {})
            right = d2p_scores.get(request_id, {})
            score_mismatches += sum(
                left.get(item_id) != right.get(item_id) for item_id in set(left) | set(right)
            )
        if _rank_items(request_id, primary_scores.get(request_id, {})) != _rank_items(
            request_id, d2p_scores.get(request_id, {})
        ):
            rank_mismatches += 1

    output_dir = repo_path(config["paths"]["output_dir"])
    with (output_dir / "internal_diagnostics.json").open("r", encoding="utf-8") as handle:
        internal = json.load(handle)
    with (output_dir / "dev_corruption_diagnostics.json").open(
        "r", encoding="utf-8"
    ) as handle:
        corruption = json.load(handle)
    with (output_dir / "determinism" / "report.json").open("r", encoding="utf-8") as handle:
        determinism = json.load(handle)
    dev_log_count = sum(
        1
        for row in iter_jsonl(REPO_ROOT / "reports" / "dev_eval_log.jsonl")
        if str(row.get("run_id")) == config["candidate"]["run_id"]
    )
    checks = {
        "subset_counts_match_frozen": subset_counts == expected_counts,
        "manifest_hash_matches": metrics["candidate_manifest_sha256"]
        == config["integrity"]["candidate_manifest_sha256"],
        "internal_gate_passed": bool(internal["passed"]),
        "label_free_corruption_gate_passed": bool(corruption["passed"]),
        "determinism_passed": bool(determinism["passed"]),
        "one_dev_evaluator_call": dev_log_count == 1,
        "overall_at_least_item_seed08": float(metrics["ndcg@10"])
        >= float(config["gate"]["item_seed08_overall_ndcg"]),
        "repeat_not_below_item_only": float(repeat_deltas.mean())
        >= float(config["gate"]["repeat_delta_min"]),
        "nonrepeat_delta_at_least_min": float(nonrepeat_deltas.mean())
        >= float(config["gate"]["nonrepeat_delta_min"]),
        "nonrepeat_bootstrap_lower_positive": bootstrap[0] > 0.0,
        "no_history_score_equivalent": score_mismatches == 0,
        "no_history_rank_equivalent": rank_mismatches == 0,
    }
    passed = all(checks.values())
    report = {
        "run_id": config["candidate"]["run_id"],
        "metrics_generated_by": metrics.get("generated_by"),
        "overall_ndcg@10": metrics["ndcg@10"],
        "item_seed08_waterline": config["gate"]["item_seed08_overall_ndcg"],
        "subset_counts": subset_counts,
        "repeat_primary_minus_item_mean_ndcg@10": float(repeat_deltas.mean()),
        "nonrepeat_primary_minus_d2p_mean_ndcg@10": float(nonrepeat_deltas.mean()),
        "nonrepeat_bootstrap_ci95": list(bootstrap),
        "no_history_score_mismatches": score_mismatches,
        "no_history_rank_mismatches": rank_mismatches,
        "dev_eval_log_count": dev_log_count,
        "checks": checks,
        "passed": passed,
        "recommendation": "advance-to-full-gate" if passed else "stop",
        "full_gate_authorized": False,
        "qrels_read_by_candidate_code": False,
        "test_read": False,
    }
    write_json(output_dir / "screening_adjudication.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    config, config_path = load_config(args.config)
    assert_run_id(config)
    if args.command == "prepare":
        command_prepare(config)
    elif args.command == "train":
        command_train(config, config_path)
    elif args.command == "diagnose":
        command_diagnose(config)
    elif args.command == "determinism":
        command_determinism(config, config_path)
    elif args.command == "score":
        command_score(config, config_path)
    elif args.command == "adjudicate":
        command_adjudicate(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
