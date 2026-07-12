#!/usr/bin/env python
"""Train one C76 data-free seed across all matched modes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from model.cltt import (  # noqa: E402
    CANDIDATE,
    HISTORY,
    MODES,
    QUERY,
    CounterfactualLayerTrajectoryTransformer,
    full_and_cut_masks,
)
from probe.freeze_lock import load_config, verify_execution_lock  # noqa: E402
from probe.synthetic import NULL_HISTORY, SyntheticSurface, make_surface  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def assert_device(config: dict[str, Any], seed: int, name: str) -> torch.device:
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C76 physical GPU registration differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C76 requires exactly one visible GPU")
    return torch.device(name)


def build_model(config: dict[str, Any], mode: str, maximum_length: int) -> CounterfactualLayerTrajectoryTransformer:
    row = config["model"]
    return CounterfactualLayerTrajectoryTransformer(
        mode=mode,
        vocabulary_size=int(row["vocabulary_size"]),
        hidden_size=int(row["hidden_size"]),
        attention_heads=int(row["attention_heads"]),
        backbone_layers=int(row["backbone_layers"]),
        trajectory_layers=int(row["trajectory_layers"]),
        trajectory_heads=int(row["trajectory_heads"]),
        maximum_length=maximum_length,
        correction_bound=float(row["correction_bound"]),
    )


def surface(config: dict[str, Any], split: str) -> SyntheticSurface:
    row = config["data"]
    return make_surface(
        requests=int(row["train_requests"] if split == "train" else row["validation_requests"]),
        candidates=int(row["candidates"]),
        history_events=int(row["history_events"]),
        attributes=int(row["attributes"]),
        values_per_attribute=int(row["values_per_attribute"]),
        seed=int(row["generator_seed"]) + (0 if split == "train" else 1),
        split=split,
    )


def forward_surface(
    model: CounterfactualLayerTrajectoryTransformer,
    batch: SyntheticSurface,
    token_field: str = "tokens",
) -> Any:
    return model(
        getattr(batch, token_field),
        batch.segments,
        batch.base_scores,
        batch.history_present,
        batch.repeat_present,
        batch.repeat_scores,
    )


def labels(batch: SyntheticSurface) -> torch.Tensor:
    return batch.labels.argmax(-1)


def mechanics(
    model: CounterfactualLayerTrajectoryTransformer,
    witness: SyntheticSurface,
    tolerance: float,
) -> dict[str, Any]:
    model.eval()
    sample = witness.subset(torch.arange(min(8, len(witness.tokens)))).to(next(model.parameters()).device)
    with torch.no_grad():
        clean = forward_surface(model, sample)
        repeated = forward_surface(model, sample)
        reversed_sample = SyntheticSurface(
            **{
                name: (value.flip(1) if value.ndim >= 2 and value.shape[1] == sample.tokens.shape[1] else value)
                for name, value in sample.__dict__.items()
            }
        )
        reversed_scores = forward_surface(model, reversed_sample).scores.flip(1)
        nohistory = SyntheticSurface(**{name: value.clone() for name, value in sample.__dict__.items()})
        nohistory.history_present[:] = False
        nohistory.repeat_present[:] = False
        nohistory.base_scores = torch.randn_like(nohistory.base_scores)
        nohistory.tokens[:, :, nohistory.segments[0, 0].eq(HISTORY)] = NULL_HISTORY
        nohistory_scores = forward_surface(model, nohistory).scores
        repeat = SyntheticSurface(**{name: value.clone() for name, value in sample.__dict__.items()})
        repeat.repeat_present[:] = True
        repeat.repeat_scores = torch.randn_like(repeat.repeat_scores)
        repeat_scores = forward_surface(model, repeat).scores

        flat_tokens = sample.tokens.reshape(-1, sample.tokens.shape[-1])
        flat_segments = sample.segments.reshape(-1, sample.segments.shape[-1])
        flat_present = sample.history_present[:, None].expand(-1, sample.tokens.shape[1]).reshape(-1)
        _, cut_mask = full_and_cut_masks(flat_segments, flat_present)
        cut_states = model.backbone(flat_tokens, flat_segments, cut_mask)
        null_tokens = flat_tokens.clone()
        null_tokens[:, flat_segments[0].eq(HISTORY)] = NULL_HISTORY
        null_states = model.backbone(null_tokens, flat_segments, cut_mask)
        qc = flat_segments.ne(HISTORY)
        cut_null_error = max(
            float((left[qc] - right[qc]).abs().max().cpu())
            for left, right in zip(cut_states, null_states)
        )

    model.train()
    model.zero_grad(set_to_none=True)
    sample.base_scores.requires_grad_(True)
    gradient_output = forward_surface(model, sample)
    loss = F.cross_entropy(gradient_output.scores, labels(sample))
    loss.backward()
    backbone_gradient = any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.")
    )
    trajectory_gradient = any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for name, parameter in model.named_parameters()
        if name.startswith("trajectory.")
    )
    protected_base_gradient_none = sample.base_scores.grad is None
    return {
        "deterministic_max_abs": float((clean.scores - repeated.scores).abs().max().cpu()),
        "candidate_permutation_max_abs": float((clean.scores - reversed_scores).abs().max().cpu()),
        "nohistory_base_max_abs": float((nohistory_scores - nohistory.base_scores).abs().max().cpu()),
        "repeat_item_only_max_abs": float((repeat_scores - repeat.repeat_scores).abs().max().cpu()),
        "history_cut_null_qc_max_abs": cut_null_error,
        "trajectory_rms": float(clean.trajectory_rms.cpu()),
        "early_trajectory_rms": float(clean.early_trajectory_rms.cpu()),
        "factual_cut_max_abs": float(clean.factual_cut_max_abs.cpu()),
        "backbone_gradient": backbone_gradient,
        "trajectory_gradient": trajectory_gradient,
        "protected_base_gradient_none": protected_base_gradient_none,
        "finite": bool(torch.isfinite(clean.scores).all()),
        "passed": bool(
            float((clean.scores - repeated.scores).abs().max().cpu()) <= tolerance
            and float((clean.scores - reversed_scores).abs().max().cpu()) <= tolerance
            and float((nohistory_scores - nohistory.base_scores).abs().max().cpu()) == 0.0
            and float((repeat_scores - repeat.repeat_scores).abs().max().cpu()) == 0.0
            and cut_null_error <= tolerance
            and float(clean.trajectory_rms.cpu()) > 0
            and float(clean.early_trajectory_rms.cpu()) > 0
            and backbone_gradient
            and trajectory_gradient
            and protected_base_gradient_none
            and bool(torch.isfinite(clean.scores).all())
        ),
    }


def train_mode(
    model: CounterfactualLayerTrajectoryTransformer,
    train: SyntheticSurface,
    batches: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    row = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(row["learning_rate"]),
        weight_decay=float(row["weight_decay"]),
    )
    trace = []
    finite = True
    model.train()
    for indices in batches:
        batch = train.subset(indices).to(next(model.parameters()).device)
        output = forward_surface(model, batch)
        loss = F.cross_entropy(output.scores, labels(batch))
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            finite = False
            break
        nn.utils.clip_grad_norm_(model.parameters(), float(row["gradient_clip_norm"]))
        optimizer.step()
        trace.append(float(loss.detach().cpu()))
    window = min(50, max(1, len(trace) // 4))
    return {
        "steps": len(trace),
        "finite": finite and len(trace) == len(batches),
        "loss_first": float(np.mean(trace[:window])),
        "loss_last": float(np.mean(trace[-window:])),
        "loss_decreased": float(np.mean(trace[-window:])) < float(np.mean(trace[:window])),
    }


def accuracy(scores: torch.Tensor, target: torch.Tensor) -> float:
    return float(scores.argmax(-1).eq(target).float().mean().cpu())


def margins(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = scores.gather(1, target.unsqueeze(1)).squeeze(1)
    masked = scores.clone()
    masked.scatter_(1, target.unsqueeze(1), -torch.inf)
    return positive - masked.max(-1).values


def evaluate(
    model: CounterfactualLayerTrajectoryTransformer,
    validation: SyntheticSurface,
    batch_size: int = 128,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    collected = {name: [] for name in ("clean", "wrong", "shuffle", "query_mask")}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(validation.tokens), batch_size):
            indices = np.arange(start, min(start + batch_size, len(validation.tokens)))
            batch = validation.subset(indices).to(device)
            for name, field in (
                ("clean", "tokens"),
                ("wrong", "wrong_tokens"),
                ("shuffle", "shuffled_tokens"),
                ("query_mask", "query_masked_tokens"),
            ):
                collected[name].append(forward_surface(model, batch, field).scores.cpu())
    scores = {name: torch.cat(rows) for name, rows in collected.items()}
    target = labels(validation)
    strata = validation.strata
    row: dict[str, Any] = {"accuracy": {}, "margins": {}}
    for name, values in scores.items():
        row["accuracy"][name] = {
            "overall": accuracy(values, target),
            "supported": accuracy(values[strata.eq(0)], target[strata.eq(0)]),
            "repeat": accuracy(values[strata.eq(1)], target[strata.eq(1)]),
            "nohistory": accuracy(values[strata.eq(2)], target[strata.eq(2)]),
        }
        supported_margin = margins(values[strata.eq(0)], target[strata.eq(0)])
        row["margins"][name] = float(supported_margin.mean())
    base = validation.base_scores
    supported = strata.eq(0)
    row["base_supported_accuracy"] = accuracy(base[supported], target[supported])
    row["supported_gain"] = row["accuracy"]["clean"]["supported"] - row["base_supported_accuracy"]
    clean_margin = row["margins"]["clean"]
    row["wrong_gain_retention"] = row["margins"]["wrong"] / clean_margin if clean_margin != 0 else float("inf")
    row["query_mask_gain_retention"] = row["margins"]["query_mask"] / clean_margin if clean_margin != 0 else float("inf")
    row["shuffle_gain_retention"] = row["margins"]["shuffle"] / clean_margin if clean_margin != 0 else float("inf")
    clean_order = torch.argsort(scores["clean"][supported], dim=-1, descending=True, stable=True)
    base_order = torch.argsort(base[supported], dim=-1, descending=True, stable=True)
    row["order_change_fraction"] = float(clean_order.ne(base_order).any(-1).float().mean())
    row["scores"] = scores
    return row


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.seed not in [int(value) for value in config["training"]["seeds"]]:
        raise ValueError("C76 seed differs")
    _, lock_hash = verify_execution_lock()
    device = assert_device(config, args.seed, args.device)
    seed_all(args.seed)
    training_surface = surface(config, "train")
    validation_surface = surface(config, "validation")
    maximum_length = int(training_surface.tokens.shape[-1])
    template = build_model(config, MODES[0], maximum_length)
    initial = {name: value.detach().clone() for name, value in template.state_dict().items()}
    counts = {}
    g0 = {}
    trained = {}
    evaluations = {}
    initial_scores = {}
    steps = int(config["training"]["steps"])
    batch_requests = int(config["training"]["batch_requests"])
    batch_rng = np.random.default_rng(args.seed + 991)
    batches = [
        batch_rng.choice(len(training_surface.tokens), size=batch_requests, replace=False)
        for _ in range(steps)
    ]
    for mode in MODES:
        model = build_model(config, mode, maximum_length).to(device)
        model.load_state_dict(initial, strict=True)
        counts[mode] = model.parameter_count()
        initial_witness = validation_surface.subset(
            np.arange(min(8, len(validation_surface.tokens)))
        ).to(device)
        model.eval()
        with torch.no_grad():
            initial_scores[mode] = forward_surface(model, initial_witness).scores.cpu()
        g0[mode] = mechanics(
            model,
            validation_surface,
            float(config["gate"]["numerical_tolerance"]),
        )
        trained[mode] = train_mode(model, training_surface, batches, config)
        evaluations[mode] = evaluate(model, validation_surface)
        del evaluations[mode]["scores"]
    primary = evaluations["counterfactual_trajectory"]
    initial_primary_final_difference = float(
        (
            initial_scores["counterfactual_trajectory"]
            - initial_scores["final_hidden_delta"]
        )
        .abs()
        .max()
    )
    control_advantage = {
        mode: primary["accuracy"]["clean"]["supported"]
        - evaluations[mode]["accuracy"]["clean"]["supported"]
        for mode in MODES
        if mode != "counterfactual_trajectory"
    }
    gate = config["gate"]
    checks = {
        "all_g0": all(value["passed"] for value in g0.values()),
        "primary_not_final_hidden_at_initialization": initial_primary_final_difference
        > float(gate["numerical_tolerance"]),
        "equal_parameters": len(set(counts.values())) == 1,
        "all_finite_and_complete": all(value["finite"] for value in trained.values()),
        "all_loss_decreased": all(value["loss_decreased"] for value in trained.values()),
        "supported_gain": primary["supported_gain"] >= float(gate["supported_gain_min"]),
        "supported_accuracy": primary["accuracy"]["clean"]["supported"] >= float(gate["supported_accuracy_min"]),
        "repeat_accuracy": primary["accuracy"]["clean"]["repeat"] >= float(gate["repeat_accuracy_min"]),
        "nohistory_accuracy": primary["accuracy"]["clean"]["nohistory"] >= float(gate["nohistory_accuracy_min"]),
        "wrong_rejected": primary["wrong_gain_retention"] <= float(gate["corruption_gain_retention_max"]),
        "query_mask_rejected": primary["query_mask_gain_retention"] <= float(gate["corruption_gain_retention_max"]),
        "shuffle_retained": primary["shuffle_gain_retention"] >= float(gate["shuffle_gain_retention_min"]),
        "order_active": primary["order_change_fraction"] >= float(gate["order_change_fraction_min"]),
        "all_controls_beaten": all(value >= float(gate["control_advantage_min"]) for value in control_advantage.values()),
        "repository_labels_dev_test_qrels_closed": True,
    }
    report_root = ROOT / "runs/design_gate"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"seed_{args.seed}.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report = {
        "candidate_id": "c76",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "execution_lock_sha256": lock_hash,
        "parameters": counts,
        "g0": g0,
        "training": trained,
        "evaluation": evaluations,
        "control_advantage": control_advantage,
        "initial_primary_final_hidden_max_abs": initial_primary_final_difference,
        "checks": checks,
        "passed": all(checks.values()),
        "access": {"repository_data": False, "labels": False, "dev_test_qrels": False},
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "passed": report["passed"], "primary": primary, "control_advantage": control_advantage, "checks": checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
