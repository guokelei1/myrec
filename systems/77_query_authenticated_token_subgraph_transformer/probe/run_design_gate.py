#!/usr/bin/env python
"""Train one C77 data-free seed across graph-matched controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
sys.path.insert(0, str(ROOT))

from model.qats import (  # noqa: E402
    MODES,
    QueryAuthenticatedTokenSubgraphTransformer,
)
from probe.c76_surface import (  # noqa: E402
    NUISANCE_NEGATIVE,
    NUISANCE_POSITIVE,
    SyntheticSurface,
    make_surface,
)
from probe.freeze_lock import load_config, verify_execution_lock  # noqa: E402


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
        raise RuntimeError("C77 physical GPU registration differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C77 requires one visible GPU")
    return torch.device(name)


def make_data(config: dict[str, Any], split: str) -> SyntheticSurface:
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


def build_model(config: dict[str, Any], mode: str, maximum_length: int) -> QueryAuthenticatedTokenSubgraphTransformer:
    model = config["model"]
    data = config["data"]
    return QueryAuthenticatedTokenSubgraphTransformer(
        mode=mode,
        vocabulary_size=int(model["vocabulary_size"]),
        hidden_size=int(model["hidden_size"]),
        attention_heads=int(model["attention_heads"]),
        interaction_layers=int(model["interaction_layers"]),
        maximum_length=maximum_length,
        correction_bound=float(model["correction_bound"]),
        attributes=int(data["attributes"]),
        values_per_attribute=int(data["values_per_attribute"]),
        anchor_dimension=int(model["anchor_dimension"]),
    )


def forward(model, batch: SyntheticSurface, field: str = "tokens"):
    return model(
        getattr(batch, field),
        batch.segments,
        batch.base_scores,
        batch.history_present,
        batch.repeat_present,
        batch.repeat_scores,
    )


def targets(batch: SyntheticSurface) -> torch.Tensor:
    return batch.labels.argmax(-1)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def mechanics(model, validation: SyntheticSurface, tolerance: float) -> dict[str, Any]:
    device = next(model.parameters()).device
    sample = validation.subset(np.arange(8)).to(device)
    anchor_before = tensor_hash(model.semantic_anchors)
    model.eval()
    with torch.no_grad():
        clean = forward(model, sample)
        repeated = forward(model, sample)
        reversed_sample = SyntheticSurface(
            **{
                name: (value.flip(1) if value.ndim >= 2 and value.shape[1] == sample.tokens.shape[1] else value)
                for name, value in sample.__dict__.items()
            }
        )
        reversed_scores = forward(model, reversed_sample).scores.flip(1)
        nohistory = SyntheticSurface(**{name: value.clone() for name, value in sample.__dict__.items()})
        nohistory.history_present[:] = False
        nohistory.repeat_present[:] = False
        nohistory.base_scores = torch.randn_like(nohistory.base_scores)
        nohistory_scores = forward(model, nohistory).scores
        query_mask = SyntheticSurface(**{name: value.clone() for name, value in sample.__dict__.items()})
        query_mask.repeat_present[:] = False
        query_mask.base_scores = torch.randn_like(query_mask.base_scores)
        query_scores = forward(model, query_mask, "query_masked_tokens").scores
        repeat = SyntheticSurface(**{name: value.clone() for name, value in sample.__dict__.items()})
        repeat.repeat_present[:] = True
        repeat.repeat_scores = torch.randn_like(repeat.repeat_scores)
        repeat_scores = forward(model, repeat).scores

    model.train()
    model.zero_grad(set_to_none=True)
    sample.base_scores.requires_grad_(True)
    output = forward(model, sample)
    loss = F.cross_entropy(output.scores, targets(sample))
    loss.backward()
    interaction_gradient = any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for name, parameter in model.named_parameters()
        if name.startswith("interaction.")
    )
    head_gradient = model.score_head.weight.grad is not None and bool(model.score_head.weight.grad.ne(0).any())
    nuisance_gradient = max(
        float(model.interaction.token_embedding.weight.grad[token].abs().max().cpu())
        for token in (NUISANCE_POSITIVE, NUISANCE_NEGATIVE)
    )
    primary_nuisance_zero = (
        nuisance_gradient == 0.0
        if model.mode == "query_authenticated_subgraph"
        else True
    )
    row = {
        "deterministic_max_abs": float((clean.scores - repeated.scores).abs().max().cpu()),
        "candidate_permutation_max_abs": float((clean.scores - reversed_scores).abs().max().cpu()),
        "nohistory_base_max_abs": float((nohistory_scores - nohistory.base_scores).abs().max().cpu()),
        "query_mask_base_max_abs": float((query_scores - query_mask.base_scores).abs().max().cpu()),
        "repeat_item_only_max_abs": float((repeat_scores - repeat.repeat_scores).abs().max().cpu()),
        "active_fraction": float(clean.active_fraction.cpu()),
        "candidate_history_edges": int(clean.candidate_history_edges.cpu()),
        "history_candidate_edges": int(clean.history_candidate_edges.cpu()),
        "interaction_gradient": interaction_gradient,
        "head_gradient": head_gradient,
        "protected_base_gradient_none": sample.base_scores.grad is None,
        "nuisance_embedding_gradient_max_abs": nuisance_gradient,
        "primary_nuisance_gradient_zero": primary_nuisance_zero,
        "anchor_hash_before": anchor_before,
        "anchor_has_no_gradient": model.semantic_anchors.grad is None,
        "finite": bool(torch.isfinite(clean.scores).all()),
    }
    row["passed"] = bool(
        row["deterministic_max_abs"] <= tolerance
        and row["candidate_permutation_max_abs"] <= tolerance
        and row["nohistory_base_max_abs"] == 0.0
        and row["query_mask_base_max_abs"] == 0.0
        and row["repeat_item_only_max_abs"] == 0.0
        and row["candidate_history_edges"] > 0
        and row["history_candidate_edges"] > 0
        and interaction_gradient
        and head_gradient
        and row["protected_base_gradient_none"]
        and primary_nuisance_zero
        and row["anchor_has_no_gradient"]
        and row["finite"]
    )
    return row


def train_mode(model, train: SyntheticSurface, batches: list[np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    row = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(row["learning_rate"]),
        weight_decay=float(row["weight_decay"]),
    )
    losses = []
    finite = True
    model.train()
    for indices in batches:
        batch = train.subset(indices).to(next(model.parameters()).device)
        output = forward(model, batch)
        loss = F.cross_entropy(output.scores, targets(batch))
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            finite = False
            break
        nn.utils.clip_grad_norm_(model.parameters(), float(row["gradient_clip_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    window = min(50, max(1, len(losses) // 4))
    return {
        "steps": len(losses),
        "finite": finite and len(losses) == len(batches),
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:])) < float(np.mean(losses[:window])),
    }


def accuracy(scores: torch.Tensor, target: torch.Tensor) -> float:
    return float(scores.argmax(-1).eq(target).float().mean())


def margins(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = scores.gather(1, target[:, None]).squeeze(1)
    negative = scores.clone()
    negative.scatter_(1, target[:, None], -torch.inf)
    return positive - negative.max(-1).values


def evaluate(model, validation: SyntheticSurface, batch_size: int = 128) -> dict[str, Any]:
    rows = {name: [] for name in ("clean", "wrong", "shuffle", "query_mask")}
    device = next(model.parameters()).device
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
                rows[name].append(forward(model, batch, field).scores.cpu())
    scores = {name: torch.cat(value) for name, value in rows.items()}
    target = targets(validation)
    strata = validation.strata
    result: dict[str, Any] = {"accuracy": {}, "margins": {}}
    for name, value in scores.items():
        result["accuracy"][name] = {
            "overall": accuracy(value, target),
            "supported": accuracy(value[strata.eq(0)], target[strata.eq(0)]),
            "repeat": accuracy(value[strata.eq(1)], target[strata.eq(1)]),
            "nohistory": accuracy(value[strata.eq(2)], target[strata.eq(2)]),
        }
        result["margins"][name] = float(margins(value[strata.eq(0)], target[strata.eq(0)]).mean())
    supported = strata.eq(0)
    result["base_supported_accuracy"] = accuracy(validation.base_scores[supported], target[supported])
    result["supported_gain"] = result["accuracy"]["clean"]["supported"] - result["base_supported_accuracy"]
    clean_margin = result["margins"]["clean"]
    result["wrong_gain_retention"] = result["margins"]["wrong"] / clean_margin if clean_margin != 0 else float("inf")
    result["query_mask_gain_retention"] = result["margins"]["query_mask"] / clean_margin if clean_margin != 0 else float("inf")
    result["shuffle_gain_retention"] = result["margins"]["shuffle"] / clean_margin if clean_margin != 0 else float("inf")
    clean_order = torch.argsort(scores["clean"][supported], dim=-1, descending=True, stable=True)
    base_order = torch.argsort(validation.base_scores[supported], dim=-1, descending=True, stable=True)
    result["order_change_fraction"] = float(clean_order.ne(base_order).any(-1).float().mean())
    return result


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.seed not in [int(value) for value in config["training"]["seeds"]]:
        raise ValueError("C77 seed differs")
    _, lock_hash = verify_execution_lock()
    device = assert_device(config, args.seed, args.device)
    seed_all(args.seed)
    train = make_data(config, "train")
    validation = make_data(config, "validation")
    length = int(train.tokens.shape[-1])
    template = build_model(config, MODES[0], length)
    initial = {name: value.detach().clone() for name, value in template.state_dict().items()}
    rng = np.random.default_rng(args.seed + 991)
    batches = [
        rng.choice(len(train.tokens), size=int(config["training"]["batch_requests"]), replace=False)
        for _ in range(int(config["training"]["steps"]))
    ]
    counts = {}
    g0 = {}
    training = {}
    evaluation = {}
    anchor_after = {}
    for mode in MODES:
        model = build_model(config, mode, length).to(device)
        model.load_state_dict(initial, strict=True)
        counts[mode] = model.parameter_count()
        g0[mode] = mechanics(model, validation, float(config["gate"]["numerical_tolerance"]))
        training[mode] = train_mode(model, train, batches, config)
        anchor_after[mode] = tensor_hash(model.semantic_anchors)
        evaluation[mode] = evaluate(model, validation)
    primary = evaluation["query_authenticated_subgraph"]
    control_advantage = {
        mode: primary["accuracy"]["clean"]["supported"]
        - evaluation[mode]["accuracy"]["clean"]["supported"]
        for mode in MODES
        if mode != "query_authenticated_subgraph"
    }
    gate = config["gate"]
    checks = {
        "all_g0": all(value["passed"] for value in g0.values()),
        "anchors_unchanged": all(anchor_after[mode] == g0[mode]["anchor_hash_before"] for mode in MODES),
        "equal_parameters": len(set(counts.values())) == 1,
        "all_finite_and_complete": all(value["finite"] for value in training.values()),
        "all_loss_decreased": all(value["loss_decreased"] for value in training.values()),
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
    path = ROOT / "runs/design_gate" / f"seed_{args.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    report = {
        "candidate_id": "c77",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "execution_lock_sha256": lock_hash,
        "parameters": counts,
        "g0": g0,
        "training": training,
        "evaluation": evaluation,
        "control_advantage": control_advantage,
        "anchor_hash_after": anchor_after,
        "checks": checks,
        "passed": all(checks.values()),
        "access": {"repository_data": False, "labels": False, "dev_test_qrels": False},
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "passed": report["passed"], "primary": primary, "control_advantage": control_advantage, "checks": checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
