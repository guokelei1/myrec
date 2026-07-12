"""Run C62's locked data-free multi-interest synthetic falsifier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    timestamp,
    verify_g0_lock,
)
from model.write_once_memory import (  # noqa: E402
    MODES,
    WriteOncePreferenceMemoryTransformer,
    listwise_training_loss,
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


def make_model(config: Mapping[str, Any], input_dim: int) -> WriteOncePreferenceMemoryTransformer:
    row = config["model"]
    return WriteOncePreferenceMemoryTransformer(
        input_dim=input_dim,
        hidden_dim=int(row["hidden_dim"]),
        heads=int(row["heads"]),
        ffn_dim=int(row["ffn_dim"]),
        history_layers=int(row["history_layers"]),
        candidate_layers=int(row["candidate_layers"]),
        memory_slots=int(row["memory_slots"]),
        max_history=int(row["max_history"]),
        dropout=float(row["dropout"]),
        zero_initial_output=bool(row["zero_initial_output"]),
    )


def synthetic_batch(
    *,
    seed: int,
    batch_size: int,
    input_dim: int,
    history_events: int,
    candidates: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if input_dim < 6 or history_events % 2 or candidates < 4:
        raise ValueError("C62 synthetic shape is invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    content_dim = input_dim - 2
    interests = F.normalize(
        torch.randn(batch_size, 2, content_dim, generator=generator), dim=-1
    )
    target_interest = torch.randint(0, 2, (batch_size,), generator=generator)

    events_per_interest = history_events // 2
    history = torch.zeros(batch_size, history_events, input_dim)
    for row in range(batch_size):
        events = []
        for interest in range(2):
            for _ in range(events_per_interest):
                value = torch.zeros(input_dim)
                value[interest] = 2.0
                value[2:] = interests[row, interest] + 0.04 * torch.randn(
                    content_dim, generator=generator
                )
                events.append(value)
        order = torch.randperm(history_events, generator=generator)
        history[row] = torch.stack(events)[order]

    query = torch.zeros(batch_size, input_dim)
    query[torch.arange(batch_size), target_interest] = 2.0
    query[:, 2:] = 0.02 * torch.randn(
        batch_size, content_dim, generator=generator
    )

    candidate_values = torch.zeros(batch_size, candidates, input_dim)
    labels = torch.zeros(batch_size, candidates)
    for row in range(batch_size):
        values = []
        for interest in range(2):
            value = torch.zeros(input_dim)
            value[2:] = interests[row, interest]
            values.append(value)
        distractors = F.normalize(
            torch.randn(candidates - 2, content_dim, generator=generator), dim=-1
        )
        for distractor in distractors:
            value = torch.zeros(input_dim)
            value[2:] = distractor
            values.append(value)
        permutation = torch.randperm(candidates, generator=generator)
        stacked = torch.stack(values)[permutation]
        candidate_values[row] = stacked
        original_target = int(target_interest[row])
        labels[row, int(torch.where(permutation == original_target)[0][0])] = 1.0

    mean_history = interests.mean(dim=1)
    base_scores = 0.2 * torch.einsum(
        "bcd,bd->bc", candidate_values[:, :, 2:], mean_history
    )
    candidate_mask = torch.ones(batch_size, candidates, dtype=torch.bool)
    history_mask = torch.ones(batch_size, history_events, dtype=torch.bool)
    wrong_history = history.roll(1, dims=0)
    return {
        "query": query.to(device),
        "candidates": candidate_values.to(device),
        "history": history.to(device),
        "wrong_history": wrong_history.to(device),
        "history_mask": history_mask.to(device),
        "candidate_mask": candidate_mask.to(device),
        "base_scores": base_scores.to(device),
        "item_only_scores": base_scores.to(device),
        "repeat_request": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "query_present": torch.ones(batch_size, dtype=torch.bool, device=device),
        "labels": labels.to(device),
    }


def forward_kwargs(batch: Mapping[str, torch.Tensor], *, wrong: bool = False) -> dict[str, torch.Tensor]:
    return {
        "query": batch["query"],
        "candidates": batch["candidates"],
        "history": batch["wrong_history"] if wrong else batch["history"],
        "history_mask": batch["history_mask"],
        "candidate_mask": batch["candidate_mask"],
        "base_scores": batch["base_scores"],
        "item_only_scores": batch["item_only_scores"],
        "repeat_request": batch["repeat_request"],
        "query_present": batch["query_present"],
    }


def gradient_groups(names: set[str]) -> dict[str, bool]:
    prefixes = {
        "history_encoder": "history_transformer.",
        "slot_writer": "slot_write_attention.",
        "memory_reader": "memory_read_attention.",
        "candidate_set": "candidate_transformer.",
        "score_head": "score_head.",
    }
    return {
        group: any(name.startswith(prefix) for name in names)
        for group, prefix in prefixes.items()
    }


def train_mode(
    config: Mapping[str, Any],
    *,
    seed: int,
    mode: str,
    device: torch.device,
) -> tuple[WriteOncePreferenceMemoryTransformer, dict[str, Any]]:
    g0 = config["synthetic_g0"]
    training = config["training"]
    seed_all(seed)
    model = make_model(config, int(g0["input_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(g0["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    losses: list[float] = []
    active_gradients: set[str] = set()
    model.train()
    for step in range(int(g0["steps"])):
        batch = synthetic_batch(
            seed=seed * 1000 + step,
            batch_size=int(g0["batch_size"]),
            input_dim=int(g0["input_dim"]),
            history_events=int(g0["history_events"]),
            candidates=int(g0["candidates"]),
            device=device,
        )
        output = model(**forward_kwargs(batch), mode=mode)
        wrong = model(**forward_kwargs(batch, wrong=True), mode=mode)
        loss, _ = listwise_training_loss(
            output,
            batch["labels"],
            batch["candidate_mask"],
            wrong_output=wrong,
            base_scores=batch["base_scores"],
            listwise_weight=float(training["listwise_loss_weight"]),
            wrong_base_kl_weight=float(training["wrong_history_base_kl_weight"]),
            correction_l2_weight=float(training["correction_l2_weight"]),
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("C62 synthetic loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"C62 synthetic gradient is nonfinite: {name}")
                if bool(parameter.grad.ne(0).any()):
                    active_gradients.add(name)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    window = min(50, len(losses) // 2)
    groups = gradient_groups(active_gradients)
    return model, {
        "steps": len(losses),
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:]))
        < float(np.mean(losses[:window])),
        "finite": bool(np.isfinite(losses).all()),
        "gradient_groups": groups,
        "all_gradient_groups": all(groups.values()),
    }


def accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return float((scores.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean().cpu())


def order_change(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        (first.argsort(dim=-1, descending=True) != second.argsort(dim=-1, descending=True))
        .any(dim=-1)
        .float()
        .mean()
        .cpu()
    )


def evaluate_mode(
    model: WriteOncePreferenceMemoryTransformer,
    config: Mapping[str, Any],
    *,
    seed: int,
    mode: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    g0 = config["synthetic_g0"]
    batch = synthetic_batch(
        seed=seed + 9_000_000,
        batch_size=int(g0["users"]),
        input_dim=int(g0["input_dim"]),
        history_events=int(g0["history_events"]),
        candidates=int(g0["candidates"]),
        device=device,
    )
    with torch.inference_mode():
        clean = model(**forward_kwargs(batch), mode=mode)
        again = model(**forward_kwargs(batch), mode=mode)
        wrong = model(**forward_kwargs(batch, wrong=True), mode=mode)
    metrics = {
        "clean_accuracy": accuracy(clean.scores, batch["labels"]),
        "wrong_accuracy": accuracy(wrong.scores, batch["labels"]),
        "wrong_accuracy_drop": accuracy(clean.scores, batch["labels"])
        - accuracy(wrong.scores, batch["labels"]),
        "clean_wrong_order_change_fraction": order_change(clean.scores, wrong.scores),
        "deterministic_max_abs": float((clean.scores - again.scores).abs().max().cpu()),
        "finite": bool(torch.isfinite(clean.scores).all() and torch.isfinite(wrong.scores).all()),
    }
    return metrics, batch


def structural_checks(
    model: WriteOncePreferenceMemoryTransformer,
    config: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, float | bool]:
    g0 = config["synthetic_g0"]
    with torch.inference_mode():
        first_memory, _, _ = model.build_memory(
            history=batch["history"],
            history_mask=batch["history_mask"],
            query=batch["query"],
            mode="write_once_memory",
        )
        altered_query = batch["query"].roll(1, dims=0) * 17.0
        second_memory, _, _ = model.build_memory(
            history=batch["history"],
            history_mask=batch["history_mask"],
            query=altered_query,
            mode="write_once_memory",
        )
        conditioned, _, _ = model.build_memory(
            history=batch["history"],
            history_mask=batch["history_mask"],
            query=altered_query,
            mode="query_conditioned_writer",
        )
        clean = model(**forward_kwargs(batch), mode="write_once_memory")
        permutation = torch.arange(batch["candidates"].shape[1] - 1, -1, -1, device=clean.scores.device)
        permuted = dict(batch)
        for name in ("candidates", "candidate_mask", "base_scores", "item_only_scores", "labels"):
            permuted[name] = batch[name][:, permutation]
        reversed_output = model(**forward_kwargs(permuted), mode="write_once_memory")
        permutation_error = float(
            (clean.scores - reversed_output.scores[:, permutation]).abs().max().cpu()
        )

        nohistory = dict(batch)
        nohistory["history_mask"] = torch.zeros_like(batch["history_mask"])
        empty_output = model(**forward_kwargs(nohistory), mode="write_once_memory")
        nohistory_error = float(
            (empty_output.scores - batch["base_scores"]).abs().max().cpu()
        )

        repeat = dict(batch)
        repeat["repeat_request"] = torch.ones_like(batch["repeat_request"])
        repeat["item_only_scores"] = torch.randn_like(batch["item_only_scores"])
        repeat_output = model(**forward_kwargs(repeat), mode="write_once_memory")
        repeat_error = float(
            (repeat_output.scores - repeat["item_only_scores"]).abs().max().cpu()
        )
        slot_variance = float(first_memory.var(dim=1).mean().cpu())
    return {
        "memory_query_candidate_invariance_max_abs": float(
            (first_memory - second_memory).abs().max().cpu()
        ),
        "query_conditioned_writer_changes_memory_max_abs": float(
            (first_memory - conditioned).abs().max().cpu()
        ),
        "candidate_permutation_max_abs": permutation_error,
        "nohistory_max_abs": nohistory_error,
        "repeat_max_abs": repeat_error,
        "slot_variance": slot_variance,
        "memory_is_finite": bool(torch.isfinite(first_memory).all()),
        "query_conditioned_memory_is_active": float(
            (first_memory - conditioned).abs().max().cpu()
        )
        > 1e-6,
        "slots_are_distinct": slot_variance > 1e-5,
        "invariance_pass": float((first_memory - second_memory).abs().max().cpu())
        <= float(g0["memory_query_candidate_invariance_tolerance"]),
    }


def run_seed(
    config: Mapping[str, Any], seed: int, device: torch.device
) -> dict[str, Any]:
    _, g0_lock_hash = verify_g0_lock(config)
    g0 = config["synthetic_g0"]
    if seed not in [int(value) for value in g0["seeds"]]:
        raise ValueError("C62 synthetic seed is not registered")
    expected_gpu = int(config["resources"]["synthetic_seed_to_physical_gpu"][str(seed)])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(expected_gpu):
        raise RuntimeError("C62 synthetic physical GPU registration differs")
    if str(device) != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C62 synthetic G0 requires one registered visible GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C62 deterministic CUBLAS setting is absent")

    trained: dict[str, WriteOncePreferenceMemoryTransformer] = {}
    training: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    evaluation_batch: dict[str, torch.Tensor] | None = None
    for mode in ("write_once_memory", "single_pooled_slot"):
        model, training[mode] = train_mode(
            config, seed=seed, mode=mode, device=device
        )
        trained[mode] = model
        metrics[mode], evaluation_batch = evaluate_mode(
            model, config, seed=seed, mode=mode, device=device
        )
    assert evaluation_batch is not None
    structural = structural_checks(
        trained["write_once_memory"], config, evaluation_batch
    )
    primary = metrics["write_once_memory"]
    single = metrics["single_pooled_slot"]
    parameter_counts = {
        mode: make_model(config, int(g0["input_dim"])).parameter_count()
        for mode in MODES
    }
    checks = {
        "primary_accuracy": primary["clean_accuracy"]
        >= float(g0["primary_accuracy_min"]),
        "multi_slot_rent": primary["clean_accuracy"] - single["clean_accuracy"]
        >= float(g0["primary_minus_single_slot_accuracy_min"]),
        "wrong_history_specificity": primary["wrong_accuracy_drop"]
        >= float(g0["wrong_history_accuracy_drop_min"]),
        "wrong_history_changes_orders": primary["clean_wrong_order_change_fraction"]
        >= 0.05,
        "primary_loss_decreased": training["write_once_memory"]["loss_decreased"],
        "primary_all_gradient_groups": training["write_once_memory"]["all_gradient_groups"],
        "all_training_finite": all(value["finite"] for value in training.values()),
        "all_scores_finite": all(value["finite"] for value in metrics.values()),
        "deterministic": primary["deterministic_max_abs"]
        <= float(config["evaluation"]["deterministic_tolerance"]),
        "memory_query_candidate_invariant": structural["invariance_pass"],
        "query_conditioned_control_active": structural[
            "query_conditioned_memory_is_active"
        ],
        "candidate_permutation": structural["candidate_permutation_max_abs"]
        <= float(g0["candidate_permutation_tolerance"]),
        "nohistory_exact": structural["nohistory_max_abs"]
        <= float(g0["exact_fallback_tolerance"]),
        "repeat_exact": structural["repeat_max_abs"]
        <= float(g0["exact_fallback_tolerance"]),
        "slots_distinct": structural["slots_are_distinct"],
        "parameter_counts_equal": len(set(parameter_counts.values())) == 1,
        "repository_records_fit_fresh_dev_test_qrels_closed": True,
    }
    root = REPO_ROOT / config["paths"]["artifact_root"] / "g0"
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"] / "g0"
    report_path = root / f"seed_{seed}.json"
    checkpoint_path = checkpoint_root / f"seed_{seed}.pt"
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError(report_path if report_path.exists() else checkpoint_path)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate_id": "c62",
            "seed": seed,
            "g0_lock_sha256": g0_lock_hash,
            "primary_state_dict": trained["write_once_memory"].state_dict(),
            "single_state_dict": trained["single_pooled_slot"].state_dict(),
        },
        checkpoint_path,
    )
    report = {
        "candidate_id": "c62",
        "created_at": timestamp(),
        "stage": "synthetic_G0",
        "status": "passed" if all(checks.values()) else "failed_terminal",
        "seed": seed,
        "physical_gpu": expected_gpu,
        "g0_lock_sha256": g0_lock_hash,
        "parameter_counts": parameter_counts,
        "training": training,
        "metrics": metrics,
        "structural": structural,
        "checks": checks,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(checkpoint_path),
        },
        "fresh_features_scores_labels_opened": False,
        "dev_test_qrels_opened": False,
    }
    atomic_json(report_path, report)
    return report


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, g0_lock_hash = verify_g0_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"] / "g0"
    target = root / "g0_report.json"
    reports = []
    for seed in config["synthetic_g0"]["seeds"]:
        path = root / f"seed_{int(seed)}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["g0_lock_sha256"] != g0_lock_hash:
            raise RuntimeError("C62 synthetic reports use different locks")
        reports.append((path, report))
    checks = {
        "three_registered_seeds": len(reports) == 3,
        "every_seed_passed": all(report["status"] == "passed" for _, report in reports),
        "same_lock": all(report["g0_lock_sha256"] == g0_lock_hash for _, report in reports),
        "fresh_features_scores_labels_closed": all(
            report["fresh_features_scores_labels_opened"] is False
            for _, report in reports
        ),
        "dev_test_qrels_closed": all(
            report["dev_test_qrels_opened"] is False for _, report in reports
        ),
    }
    value = {
        "candidate_id": "c62",
        "created_at": timestamp(),
        "stage": "synthetic_G0_aggregate",
        "status": "passed" if all(checks.values()) else "failed_terminal",
        "decision": "authorize_exposed_fit_dual_domain_gate"
        if all(checks.values())
        else "close_c62_before_repository_data",
        "g0_lock_sha256": g0_lock_hash,
        "checks": checks,
        "seed_reports": {
            str(report["seed"]): {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "status": report["status"],
            }
            for path, report in reports
        },
        "repository_records_fit_fresh_dev_test_qrels_opened": False,
    }
    atomic_json(target, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=SYSTEM_ROOT / "configs/train_gate.yaml"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.aggregate:
        value = aggregate(config)
    else:
        if args.seed is None:
            parser.error("--seed is required unless --aggregate is used")
        value = run_seed(config, args.seed, torch.device("cuda:0"))
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
