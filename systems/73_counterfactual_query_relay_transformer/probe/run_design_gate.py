"""Train and aggregate the locked C73 data-free GPU gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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

from model.query_relay import (  # noqa: E402
    MODES,
    PRIMARY,
    CounterfactualQueryRelayTransformer,
    listwise_loss,
)
from probe.locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    verify_lock,
)
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


def make_model(config: Mapping[str, Any], *, mode: str) -> CounterfactualQueryRelayTransformer:
    return CounterfactualQueryRelayTransformer(
        input_dim=int(config["data"]["dimension"]),
        hidden_dim=int(config["model"]["hidden_dimension"]),
        heads=int(config["model"]["heads"]),
        ffn_dim=int(config["model"]["ffn_dimension"]),
        max_history=int(config["data"]["history_events"]),
        mode=mode,
        dropout=float(config["model"]["dropout"]),
        correction_cap=float(config["model"]["correction_cap"]),
    )


def gradient_groups(model: CounterfactualQueryRelayTransformer) -> dict[str, bool]:
    active = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(parameter.grad.ne(0).any())
    }
    return {
        "input_projection": any(name.startswith("input_projection.") for name in active),
        "query_history_attention": any(
            name.startswith("query_history_attention.") for name in active
        ),
        "candidate_relay_attention": any(
            name.startswith("candidate_relay_attention.") for name in active
        ),
        "relay_ffn": any(name.startswith("relay_ffn.") for name in active),
        "output_head": any(name.startswith("output_head.") for name in active),
    }


def train_mode(
    config: Mapping[str, Any],
    train: SyntheticData,
    *,
    mode: str,
    seed: int,
    device: torch.device,
) -> tuple[CounterfactualQueryRelayTransformer, dict[str, Any]]:
    seed_all(seed)
    model = make_model(config, mode=mode).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batch_size = int(config["training"]["batch_size"])
    steps = int(config["training"]["steps"])
    generator = torch.Generator().manual_seed(seed + 73)
    losses: list[float] = []
    group_union = {name: False for name in (
        "input_projection",
        "query_history_attention",
        "candidate_relay_attention",
        "relay_ffn",
        "output_head",
    )}
    order = torch.empty(0, dtype=torch.long)
    cursor = 0
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
            raise RuntimeError(f"C73 {mode} nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, passed in gradient_groups(model).items():
            group_union[name] = group_union[name] or passed
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"C73 {mode} nonfinite gradient: {name}")
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
        "gradient_groups": group_union,
        "all_gradient_groups": all(group_union.values()),
        "parameters": model.parameter_count(),
    }


def _positive_ndcg(scores: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, k: int) -> float:
    if not bool(rows.any()):
        return float("nan")
    subset_scores = scores[rows]
    positive = labels[rows].argmax(-1)
    order = torch.argsort(subset_scores, dim=-1, descending=True, stable=True)
    rank = (order == positive[:, None]).nonzero(as_tuple=False)[:, 1]
    value = torch.where(
        rank < k,
        1.0 / torch.log2(rank.float() + 2.0),
        torch.zeros_like(rank, dtype=torch.float32),
    )
    return float(value.mean().cpu())


def score(model: CounterfactualQueryRelayTransformer, data: SyntheticData, *, device: torch.device) -> torch.Tensor:
    values: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data), 256):
            batch = data.index(torch.arange(start, min(start + 256, len(data)))).to(device)
            values.append(model(**batch.forward_kwargs()).scores.float().cpu())
    return torch.cat(values)


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
    k = int(config["evaluation"]["ndcg_k"])
    models: dict[str, CounterfactualQueryRelayTransformer] = {}
    training: dict[str, Any] = {}
    mode_scores: dict[str, torch.Tensor] = {}
    for mode in MODES:
        model, trace = train_mode(
            config, train, mode=mode, seed=seed, device=device
        )
        models[mode] = model
        training[mode] = trace
        mode_scores[mode] = score(model, validation, device=device)

    base = validation.base_scores
    supported = validation.supported_request
    mode_ndcg = {
        mode: _positive_ndcg(values, validation.labels, supported, k)
        for mode, values in mode_scores.items()
    }
    base_ndcg = _positive_ndcg(base, validation.labels, supported, k)
    primary = models[PRIMARY]
    corruptions = {
        "wrong": wrong_history(validation),
        "shuffled": shuffled_history(validation),
        "coarse": coarse_history(validation, config),
        "query_masked": query_masked(validation),
    }
    corruption_scores = {
        name: score(primary, value, device=device)
        for name, value in corruptions.items()
    }
    corruption_ndcg = {
        name: _positive_ndcg(values, validation.labels, supported, k)
        for name, values in corruption_scores.items()
    }
    clean_gain = mode_ndcg[PRIMARY] - base_ndcg
    retention = {
        name: (
            (value - base_ndcg) / clean_gain
            if clean_gain > 1e-12
            else float("inf")
        )
        for name, value in corruption_ndcg.items()
    }

    first = validation.index(torch.arange(128)).to(device)
    primary.eval()
    with torch.inference_mode():
        original = primary(**first.forward_kwargs()).scores
        repeated = primary(**first.forward_kwargs()).scores
        reverse = torch.arange(first.candidate_tokens.shape[1] - 1, -1, -1, device=device)
        reversed_data = SyntheticData(
            query_tokens=first.query_tokens,
            history_tokens=first.history_tokens,
            candidate_tokens=first.candidate_tokens[:, reverse],
            history_mask=first.history_mask,
            candidate_mask=first.candidate_mask[:, reverse],
            base_scores=first.base_scores[:, reverse],
            item_only_scores=first.item_only_scores[:, reverse],
            repeat_request=first.repeat_request,
            query_present=first.query_present,
            labels=first.labels[:, reverse],
            supported_request=first.supported_request,
            no_history_request=first.no_history_request,
        )
        reversed_scores = primary(**reversed_data.forward_kwargs()).scores[:, reverse]
    deterministic_error = float((original - repeated).abs().max().cpu())
    permutation_error = float((original - reversed_scores).abs().max().cpu())

    primary_scores = mode_scores[PRIMARY]
    no_history_error = float(
        (primary_scores[validation.no_history_request] - base[validation.no_history_request])
        .abs()
        .max()
    )
    repeat_error = float(
        (
            primary_scores[validation.repeat_request]
            - validation.item_only_scores[validation.repeat_request]
        )
        .abs()
        .max()
    )
    no_history_ndcg = _positive_ndcg(
        primary_scores, validation.labels, validation.no_history_request, k
    )
    repeat_ndcg = _positive_ndcg(
        primary_scores, validation.labels, validation.repeat_request, k
    )

    e = config["evaluation"]
    checks = {
        "all_losses_finite": all(row["finite"] for row in training.values()),
        "all_losses_decreased": all(row["loss_decreased"] for row in training.values()),
        "all_gradient_groups": all(row["all_gradient_groups"] for row in training.values()),
        "matched_parameter_count": len({row["parameters"] for row in training.values()}) == 1,
        "supported_primary": mode_ndcg[PRIMARY] >= float(e["supported_primary_min"]),
        "supported_gain_over_base": clean_gain >= float(e["supported_gain_over_base_min"]),
        "primary_over_late": mode_ndcg[PRIMARY] - mode_ndcg["late_state_difference"]
        >= float(e["primary_minus_late_min"]),
        "primary_over_pooled": mode_ndcg[PRIMARY] - mode_ndcg["pooled_query_relay"]
        >= float(e["primary_minus_pooled_min"]),
        "primary_over_factual": mode_ndcg[PRIMARY] - mode_ndcg["factual_query_relay"]
        >= float(e["primary_minus_factual_min"]),
        "wrong_rejected": retention["wrong"] <= float(e["wrong_gain_retention_max"]),
        "shuffle_rejected": retention["shuffled"] <= float(e["shuffled_gain_retention_max"]),
        "coarse_rejected": retention["coarse"] <= float(e["coarse_gain_retention_max"]),
        "query_mask_rejected": retention["query_masked"]
        <= float(e["query_mask_gain_retention_max"]),
        "repeat_ndcg": repeat_ndcg >= float(e["repeat_ndcg_min"]),
        "no_history_ndcg": no_history_ndcg >= float(e["no_history_ndcg_min"]),
        "repeat_exact": repeat_error <= float(e["exact_fallback_tolerance"]),
        "no_history_exact": no_history_error <= float(e["exact_fallback_tolerance"]),
        "deterministic": deterministic_error <= float(e["deterministic_tolerance"]),
        "candidate_permutation": permutation_error
        <= float(e["candidate_permutation_tolerance"]),
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
        "training": training,
        "supported_ndcg_at_10": {"base": base_ndcg, **mode_ndcg},
        "primary_gain_over_base": clean_gain,
        "primary_margins": {
            "late": mode_ndcg[PRIMARY] - mode_ndcg["late_state_difference"],
            "pooled": mode_ndcg[PRIMARY] - mode_ndcg["pooled_query_relay"],
            "factual": mode_ndcg[PRIMARY] - mode_ndcg["factual_query_relay"],
        },
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
    rows = []
    for seed in config["training"]["seeds"]:
        path = root / f"seed_{int(seed)}.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    passed = all(row["passed"] for row in rows)
    return {
        "candidate_id": "c73",
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
            raise RuntimeError("C73 requires CUDA")
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
