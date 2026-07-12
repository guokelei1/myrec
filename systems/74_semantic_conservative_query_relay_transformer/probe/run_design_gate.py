"""Train and aggregate the locked C74 data-free GPU gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for value in (str(SYSTEM_ROOT), str(REPO_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from model.semantic_relay import (  # noqa: E402
    MODES,
    PRIMARY,
    SemanticConservativeQueryRelayTransformer,
    listwise_loss,
)
from probe.locking import atomic_json, load_config, verify_lock  # noqa: E402
from probe.synthetic import (  # noqa: E402
    SyntheticData,
    coarse_history,
    make_dataset,
    query_masked,
    shuffled_history,
    wrong_history,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def make_model(
    config: Mapping[str, Any], *, mode: str
) -> SemanticConservativeQueryRelayTransformer:
    row = config["model"]
    return SemanticConservativeQueryRelayTransformer(
        dim=int(config["data"]["dimension"]),
        route_rank=int(row["route_rank"]),
        max_history=int(config["data"]["history_events"]),
        mode=mode,
        temperature=float(row["temperature"]),
        profile_scale=float(row["profile_scale"]),
        correction_scale=float(row["correction_scale"]),
        route_init_std=float(row["route_init_std"]),
    )


def gradient_groups(
    model: SemanticConservativeQueryRelayTransformer,
) -> dict[str, bool]:
    active = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(parameter.grad.ne(0).any())
    }
    return {
        "history_route_down": "history_route.down.weight" in active,
        "history_route_up": "history_route.up.weight" in active,
        "candidate_route_down": "candidate_route.down.weight" in active,
        "candidate_route_up": "candidate_route.up.weight" in active,
        "chronology_bias": "chronology_bias" in active,
    }


def train_mode(
    config: Mapping[str, Any],
    train: SyntheticData,
    *,
    mode: str,
    seed: int,
    device: torch.device,
) -> tuple[SemanticConservativeQueryRelayTransformer, dict[str, Any]]:
    seed_all(seed)
    model = make_model(config, mode=mode).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batch_size = int(config["training"]["batch_size"])
    steps = int(config["training"]["steps"])
    generator = torch.Generator().manual_seed(seed + 74)
    order = torch.empty(0, dtype=torch.long)
    cursor = 0
    losses: list[float] = []
    union = {name: False for name in (
        "history_route_down",
        "history_route_up",
        "candidate_route_down",
        "candidate_route_up",
        "chronology_bias",
    )}
    model.train()
    for _ in range(steps):
        if cursor + batch_size > len(order):
            order = torch.randperm(len(train), generator=generator)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        batch = train.index(indices).to(device)
        output = model(**batch.forward_kwargs())
        loss = listwise_loss(output, batch.labels, batch.candidate_mask)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"C74 {mode} nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, passed in gradient_groups(model).items():
            union[name] = union[name] or passed
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"C74 {mode} nonfinite gradient: {name}")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip_norm"])
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    window = min(50, max(1, len(losses) // 2))
    return model, {
        "steps": steps,
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:]))
        < float(np.mean(losses[:window])),
        "finite": bool(np.isfinite(losses).all()),
        "gradient_groups": union,
        "all_gradient_groups": all(union.values()),
        "parameters": model.parameter_count(),
        "chronology_bias": model.chronology_bias.detach().float().cpu().tolist(),
    }


def score(
    model: SemanticConservativeQueryRelayTransformer,
    data: SyntheticData,
    *,
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data), 256):
            indices = torch.arange(start, min(start + 256, len(data)))
            batch = data.index(indices).to(device)
            rows.append(model(**batch.forward_kwargs()).scores.float().cpu())
    return torch.cat(rows)


def positive_ndcg(
    scores: torch.Tensor,
    labels: torch.Tensor,
    rows: torch.Tensor,
    *,
    k: int,
) -> float:
    subset = scores[rows]
    positive = labels[rows].argmax(-1)
    order = torch.argsort(subset, dim=-1, descending=True, stable=True)
    rank = (order == positive[:, None]).nonzero(as_tuple=False)[:, 1]
    value = torch.where(
        rank < k,
        1.0 / torch.log2(rank.float() + 2.0),
        torch.zeros_like(rank, dtype=torch.float32),
    )
    return float(value.mean())


def evaluate_seed(
    config: Mapping[str, Any], *, seed: int, device: torch.device
) -> dict[str, Any]:
    generator_seed = int(config["data"]["generator_seed"])
    train = make_dataset(
        config,
        examples=int(config["data"]["train_examples"]),
        seed=generator_seed,
        split="train",
    )
    validation = make_dataset(
        config,
        examples=int(config["data"]["validation_examples"]),
        seed=generator_seed + 1,
        split="validation",
    )
    models: dict[str, SemanticConservativeQueryRelayTransformer] = {}
    traces: dict[str, Any] = {}
    scores: dict[str, torch.Tensor] = {}
    for mode in MODES:
        model, trace = train_mode(
            config, train, mode=mode, seed=seed, device=device
        )
        models[mode] = model
        traces[mode] = trace
        scores[mode] = score(model, validation, device=device)

    k = int(config["evaluation"]["ndcg_k"])
    supported = validation.supported_request
    base_ndcg = positive_ndcg(
        validation.base_scores, validation.labels, supported, k=k
    )
    mode_ndcg = {
        mode: positive_ndcg(value, validation.labels, supported, k=k)
        for mode, value in scores.items()
    }
    clean_gain = mode_ndcg[PRIMARY] - base_ndcg
    primary = models[PRIMARY]
    corrupted = {
        "wrong": wrong_history(validation),
        "shuffled": shuffled_history(validation),
        "coarse": coarse_history(validation, config),
        "query_masked": query_masked(validation),
    }
    corruption_ndcg = {
        name: positive_ndcg(
            score(primary, data, device=device), validation.labels, supported, k=k
        )
        for name, data in corrupted.items()
    }
    retention = {
        name: (value - base_ndcg) / clean_gain if clean_gain > 1e-12 else float("inf")
        for name, value in corruption_ndcg.items()
    }

    primary_scores = scores[PRIMARY]
    no_history_error = float(
        (
            primary_scores[validation.no_history_request]
            - validation.base_scores[validation.no_history_request]
        ).abs().max()
    )
    repeat_error = float(
        (
            primary_scores[validation.repeat_request]
            - validation.item_only_scores[validation.repeat_request]
        ).abs().max()
    )
    no_history_ndcg = positive_ndcg(
        primary_scores, validation.labels, validation.no_history_request, k=k
    )
    repeat_ndcg = positive_ndcg(
        primary_scores, validation.labels, validation.repeat_request, k=k
    )

    first = validation.index(torch.arange(128)).to(device)
    primary.eval()
    with torch.inference_mode():
        original = primary(**first.forward_kwargs()).scores
        repeated = primary(**first.forward_kwargs()).scores
        reverse = torch.arange(
            first.candidate_tokens.shape[1] - 1, -1, -1, device=device
        )
        reversed_data = replace(
            first,
            candidate_tokens=first.candidate_tokens[:, reverse],
            candidate_mask=first.candidate_mask[:, reverse],
            base_scores=first.base_scores[:, reverse],
            item_only_scores=first.item_only_scores[:, reverse],
            labels=first.labels[:, reverse],
        )
        reversed_scores = primary(**reversed_data.forward_kwargs()).scores[:, reverse]
    deterministic_error = float((original - repeated).abs().max().cpu())
    permutation_error = float((original - reversed_scores).abs().max().cpu())

    margins = {
        "coupled": mode_ndcg[PRIMARY] - mode_ndcg["coupled_value_relay"],
        "pooled": mode_ndcg[PRIMARY] - mode_ndcg["pooled_semantic_relay"],
        "factual": mode_ndcg[PRIMARY] - mode_ndcg["factual_semantic_relay"],
    }
    e = config["evaluation"]
    checks = {
        "all_losses_finite": all(row["finite"] for row in traces.values()),
        "all_losses_decreased": all(row["loss_decreased"] for row in traces.values()),
        "all_gradient_groups": all(row["all_gradient_groups"] for row in traces.values()),
        "matched_parameter_count": len({row["parameters"] for row in traces.values()}) == 1,
        "supported_primary": mode_ndcg[PRIMARY] >= float(e["supported_primary_min"]),
        "supported_gain_over_base": clean_gain >= float(e["supported_gain_over_base_min"]),
        "primary_over_coupled": margins["coupled"] >= float(e["primary_minus_coupled_min"]),
        "primary_over_pooled": margins["pooled"] >= float(e["primary_minus_pooled_min"]),
        "primary_over_factual": margins["factual"] >= float(e["primary_minus_factual_min"]),
        "wrong_rejected": retention["wrong"] <= float(e["wrong_gain_retention_max"]),
        "shuffle_rejected": retention["shuffled"] <= float(e["shuffled_gain_retention_max"]),
        "coarse_rejected": retention["coarse"] <= float(e["coarse_gain_retention_max"]),
        "query_mask_rejected": retention["query_masked"] <= float(e["query_mask_gain_retention_max"]),
        "repeat_ndcg": repeat_ndcg >= float(e["repeat_ndcg_min"]),
        "no_history_ndcg": no_history_ndcg >= float(e["no_history_ndcg_min"]),
        "repeat_exact": repeat_error <= float(e["exact_fallback_tolerance"]),
        "no_history_exact": no_history_error <= float(e["exact_fallback_tolerance"]),
        "deterministic": deterministic_error <= float(e["deterministic_tolerance"]),
        "candidate_permutation": permutation_error <= float(e["candidate_permutation_tolerance"]),
    }
    return {
        "seed": seed,
        "device": str(device),
        "repository_data_read": False,
        "repository_labels_read": False,
        "dev_test_qrels_read": False,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "validation_counts": {
            "supported": int(validation.supported_request.sum()),
            "no_history": int(validation.no_history_request.sum()),
            "repeat": int(validation.repeat_request.sum()),
        },
        "training": traces,
        "supported_ndcg_at_10": {"base": base_ndcg, **mode_ndcg},
        "primary_gain_over_base": clean_gain,
        "primary_margins": margins,
        "corruption_ndcg_at_10": corruption_ndcg,
        "corruption_gain_retention": retention,
        "repeat_ndcg_at_10": repeat_ndcg,
        "no_history_ndcg_at_10": no_history_ndcg,
        "repeat_max_abs_error": repeat_error,
        "no_history_max_abs_error": no_history_error,
        "deterministic_max_abs_error": deterministic_error,
        "candidate_permutation_max_abs_error": permutation_error,
        "checks": checks,
        "passed": all(checks.values()),
    }


def aggregate(config: Mapping[str, Any], lock_hash: str) -> dict[str, Any]:
    root = REPO_ROOT / config["paths"]["artifact_root"]
    rows = [
        json.loads((root / f"seed_{int(seed)}.json").read_text(encoding="utf-8"))
        for seed in config["training"]["seeds"]
    ]
    passed = all(row["passed"] for row in rows)
    return {
        "candidate_id": "c74",
        "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "pass_authorize_pretrained_probe" if passed else "failed_design_gate_terminal",
        "passed": passed,
        "proposal_lock_sha256": lock_hash,
        "repository_data_read": False,
        "repository_labels_read": False,
        "dev_test_qrels_read": False,
        "shared_evaluator_calls": 0,
        "seeds": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if (args.seed is None) == (not args.aggregate):
        raise ValueError("choose exactly one of --seed or --aggregate")
    config = load_config()
    _, lock_hash = verify_lock(config)
    if args.seed is not None:
        seed = int(args.seed)
        gpu = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
        if not torch.cuda.is_available():
            raise RuntimeError("C74 requires CUDA")
        torch.cuda.set_device(gpu)
        result = evaluate_seed(config, seed=seed, device=torch.device(f"cuda:{gpu}"))
        result["proposal_lock_sha256"] = lock_hash
        path = REPO_ROOT / config["paths"]["artifact_root"] / f"seed_{seed}.json"
        atomic_json(path, result)
        print(json.dumps({"seed": seed, "passed": result["passed"], "path": str(path)}))
        return
    report = aggregate(config, lock_hash)
    path = REPO_ROOT / config["paths"]["promoted_report"]
    atomic_json(path, report)
    print(json.dumps({"decision": report["decision"], "path": str(path)}))


if __name__ == "__main__":
    main()
