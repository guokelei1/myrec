"""Run C63's locked four-interest plus nuisance synthetic falsifier."""

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
from model.finite_evidence_memory import (  # noqa: E402
    MODES,
    FiniteEvidenceMemoryTransformer,
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


def make_model(config: Mapping[str, Any], input_dim: int) -> FiniteEvidenceMemoryTransformer:
    row = config["model"]
    return FiniteEvidenceMemoryTransformer(
        input_dim=input_dim,
        hidden_dim=int(row["hidden_dim"]),
        heads=int(row["heads"]),
        ffn_dim=int(row["ffn_dim"]),
        history_layers=int(row["history_layers"]),
        candidate_layers=int(row["candidate_layers"]),
        memory_slots=int(row["memory_slots"]),
        max_history=int(row["max_history"]),
        sinkhorn_iterations=int(row["sinkhorn_iterations"]),
        dropout=float(row["dropout"]),
        zero_initial_output=bool(row["zero_initial_output"]),
    )


def synthetic_batch(
    config: Mapping[str, Any], *, seed: int, batch_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    row = config["synthetic_G0"]
    interests_count = int(row["interests"])
    useful_per_interest = int(row["useful_events_per_interest"])
    nuisance_count = int(row["nuisance_events"])
    candidate_count = int(row["candidates"])
    input_dim = int(row["input_dim"])
    content_dim = input_dim - interests_count
    if content_dim < 8 or candidate_count < interests_count:
        raise ValueError("C63 synthetic dimensions are invalid")
    history_count = interests_count * useful_per_interest + nuisance_count
    generator = torch.Generator(device="cpu").manual_seed(seed)
    interests = F.normalize(
        torch.randn(batch_size, interests_count, content_dim, generator=generator),
        dim=-1,
    )
    target_interest = torch.randint(
        0, interests_count, (batch_size,), generator=generator
    )
    history = torch.zeros(batch_size, history_count, input_dim)
    nuisance_mask = torch.zeros(batch_size, history_count, dtype=torch.bool)
    for batch_index in range(batch_size):
        events: list[torch.Tensor] = []
        nuisance: list[bool] = []
        for interest in range(interests_count):
            for _ in range(useful_per_interest):
                value = torch.zeros(input_dim)
                value[interest] = 2.0
                value[interests_count:] = interests[batch_index, interest] + 0.04 * torch.randn(
                    content_dim, generator=generator
                )
                events.append(value)
                nuisance.append(False)
        for _ in range(nuisance_count):
            value = torch.zeros(input_dim)
            value[:interests_count] = -1.0
            value[interests_count:] = F.normalize(
                torch.randn(content_dim, generator=generator), dim=0
            )
            events.append(value)
            nuisance.append(True)
        permutation = torch.randperm(history_count, generator=generator)
        history[batch_index] = torch.stack(events)[permutation]
        nuisance_mask[batch_index] = torch.tensor(nuisance, dtype=torch.bool)[permutation]

    query = torch.zeros(batch_size, input_dim)
    query[torch.arange(batch_size), target_interest] = 2.0
    query[:, interests_count:] = 0.02 * torch.randn(
        batch_size, content_dim, generator=generator
    )
    candidates = torch.zeros(batch_size, candidate_count, input_dim)
    labels = torch.zeros(batch_size, candidate_count)
    for batch_index in range(batch_size):
        values: list[torch.Tensor] = []
        for interest in range(interests_count):
            value = torch.zeros(input_dim)
            value[interests_count:] = interests[batch_index, interest]
            values.append(value)
        distractors = F.normalize(
            torch.randn(
                candidate_count - interests_count,
                content_dim,
                generator=generator,
            ),
            dim=-1,
        )
        for distractor in distractors:
            value = torch.zeros(input_dim)
            value[interests_count:] = distractor
            values.append(value)
        permutation = torch.randperm(candidate_count, generator=generator)
        candidates[batch_index] = torch.stack(values)[permutation]
        target = int(target_interest[batch_index])
        labels[batch_index, int(torch.where(permutation == target)[0][0])] = 1.0

    mean_interest = interests.mean(dim=1)
    base_scores = 0.2 * torch.einsum(
        "bcd,bd->bc", candidates[:, :, interests_count:], mean_interest
    )
    history_mask = torch.ones(batch_size, history_count, dtype=torch.bool)
    candidate_mask = torch.ones(batch_size, candidate_count, dtype=torch.bool)
    return {
        "query": query.to(device),
        "candidates": candidates.to(device),
        "history": history.to(device),
        "wrong_history": history.roll(1, dims=0).to(device),
        "history_mask": history_mask.to(device),
        "candidate_mask": candidate_mask.to(device),
        "base_scores": base_scores.to(device),
        "item_only_scores": base_scores.to(device),
        "repeat_request": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "query_present": torch.ones(batch_size, dtype=torch.bool, device=device),
        "labels": labels.to(device),
        "nuisance_mask": nuisance_mask.to(device),
    }


def forward_kwargs(
    batch: Mapping[str, torch.Tensor], *, wrong: bool = False
) -> dict[str, torch.Tensor]:
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
        "event_key": "event_key_projection.",
        "event_value": "event_value_projection.",
        "slot_key": "slot_key_projection.",
        "memory_reader": "memory_read_attention.",
        "candidate_set": "candidate_transformer.",
        "score_head": "score_head.",
    }
    output = {
        group: any(name.startswith(prefix) for name in names)
        for group, prefix in prefixes.items()
    }
    output["break_bias"] = "break_bias" in names
    return output


def train_mode(
    config: Mapping[str, Any], *, seed: int, mode: str, device: torch.device
) -> tuple[FiniteEvidenceMemoryTransformer, dict[str, Any]]:
    g0, training = config["synthetic_G0"], config["training"]
    seed_all(seed)
    model = make_model(config, int(g0["input_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(g0["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    losses: list[float] = []
    active: set[str] = set()
    model.train()
    for step in range(int(g0["steps"])):
        batch = synthetic_batch(
            config,
            seed=seed * 1000 + step,
            batch_size=int(g0["batch_size"]),
            device=device,
        )
        clean = model(**forward_kwargs(batch), mode=mode)
        wrong = model(**forward_kwargs(batch, wrong=True), mode=mode)
        loss, _ = listwise_training_loss(
            clean,
            batch["labels"],
            batch["candidate_mask"],
            wrong_output=wrong,
            base_scores=batch["base_scores"],
            listwise_weight=float(training["listwise_loss_weight"]),
            wrong_base_kl_weight=float(training["wrong_history_base_kl_weight"]),
            correction_l2_weight=float(training["correction_l2_weight"]),
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"C63 {mode} loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"C63 nonfinite gradient: {name}")
                if bool(parameter.grad.ne(0).any()):
                    active.add(name)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    window = min(50, len(losses) // 2)
    groups = gradient_groups(active)
    return model, {
        "steps": len(losses),
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:]))
        < float(np.mean(losses[:window])),
        "finite": bool(np.isfinite(losses).all()),
        "gradient_groups": groups,
        "all_gradient_groups": all(groups.values()) if mode == "finite_evidence_memory" else None,
    }


def accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return float((scores.argmax(-1) == labels.argmax(-1)).float().mean().cpu())


def order_change(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        (first.argsort(-1, descending=True) != second.argsort(-1, descending=True))
        .any(-1)
        .float()
        .mean()
        .cpu()
    )


def evaluate_mode(
    model: FiniteEvidenceMemoryTransformer,
    config: Mapping[str, Any],
    *,
    seed: int,
    mode: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], Any]:
    batch = synthetic_batch(
        config,
        seed=seed + 9_000_000,
        batch_size=int(config["synthetic_G0"]["users"]),
        device=device,
    )
    with torch.inference_mode():
        clean = model(**forward_kwargs(batch), mode=mode)
        again = model(**forward_kwargs(batch), mode=mode)
        wrong = model(**forward_kwargs(batch, wrong=True), mode=mode)
    clean_accuracy = accuracy(clean.scores, batch["labels"])
    wrong_accuracy = accuracy(wrong.scores, batch["labels"])
    metrics = {
        "clean_accuracy": clean_accuracy,
        "wrong_accuracy": wrong_accuracy,
        "wrong_accuracy_drop": clean_accuracy - wrong_accuracy,
        "clean_wrong_order_change_fraction": order_change(clean.scores, wrong.scores),
        "deterministic_max_abs": float((clean.scores - again.scores).abs().max().cpu()),
        "finite": bool(torch.isfinite(clean.scores).all()),
    }
    if mode == "finite_evidence_memory":
        useful = ~batch["nuisance_mask"]
        total = clean.allocation.sum(-1) + clean.null_mass
        conservation = (
            total - batch["history_mask"].to(total.dtype)
        ).abs().max()
        total_real_mass = clean.slot_mass.sum().clamp_min(1e-8)
        slot_fraction = clean.slot_mass.sum(0) / total_real_mass
        metrics.update(
            {
                "useful_null_mass": float(clean.null_mass[useful].mean().cpu()),
                "nuisance_null_mass": float(
                    clean.null_mass[batch["nuisance_mask"]].mean().cpu()
                ),
                "nuisance_minus_useful_null_mass": float(
                    clean.null_mass[batch["nuisance_mask"]].mean().cpu()
                    - clean.null_mass[useful].mean().cpu()
                ),
                "evidence_conservation_max_abs": float(conservation.cpu()),
                "minimum_real_slot_mass_fraction": float(slot_fraction.min().cpu()),
                "real_slot_mass_fractions": [float(value) for value in slot_fraction.cpu()],
            }
        )
    return metrics, batch, clean


def structural_checks(
    model: FiniteEvidenceMemoryTransformer,
    config: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, float | bool]:
    with torch.inference_mode():
        first = model.build_memory(
            history=batch["history"],
            history_mask=batch["history_mask"],
            query=batch["query"],
            mode="finite_evidence_memory",
        )
        second = model.build_memory(
            history=batch["history"],
            history_mask=batch["history_mask"],
            query=batch["query"].roll(1, 0) * 19.0,
            mode="finite_evidence_memory",
        )
        invariance = max(float((a - b).abs().max().cpu()) for a, b in zip(first, second))
        clean = model(**forward_kwargs(batch), mode="finite_evidence_memory")
        permutation = torch.arange(
            batch["candidates"].shape[1] - 1,
            -1,
            -1,
            device=batch["candidates"].device,
        )
        reversed_batch = dict(batch)
        for name in ("candidates", "candidate_mask", "base_scores", "item_only_scores", "labels"):
            reversed_batch[name] = batch[name][:, permutation]
        reversed_output = model(
            **forward_kwargs(reversed_batch), mode="finite_evidence_memory"
        )
        permutation_error = float(
            (clean.scores - reversed_output.scores[:, permutation]).abs().max().cpu()
        )
        empty = dict(batch)
        empty["history_mask"] = torch.zeros_like(batch["history_mask"])
        empty_output = model(**forward_kwargs(empty), mode="finite_evidence_memory")
        nohistory_error = float(
            (empty_output.scores - batch["base_scores"]).abs().max().cpu()
        )
        repeat = dict(batch)
        repeat["repeat_request"] = torch.ones_like(batch["repeat_request"])
        repeat["item_only_scores"] = torch.randn_like(batch["item_only_scores"])
        repeat_output = model(**forward_kwargs(repeat), mode="finite_evidence_memory")
        repeat_error = float(
            (repeat_output.scores - repeat["item_only_scores"]).abs().max().cpu()
        )
    return {
        "memory_query_invariance_max_abs": invariance,
        "candidate_permutation_max_abs": permutation_error,
        "nohistory_max_abs": nohistory_error,
        "repeat_max_abs": repeat_error,
        "finite": bool(
            torch.isfinite(first[0]).all()
            and torch.isfinite(clean.scores).all()
        ),
    }


def run_seed(
    config: Mapping[str, Any], seed: int, device: torch.device
) -> dict[str, Any]:
    _, g0_lock_hash = verify_g0_lock(config)
    g0 = config["synthetic_G0"]
    if seed not in [int(value) for value in g0["seeds"]]:
        raise ValueError("C63 seed is not registered")
    expected_gpu = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(expected_gpu):
        raise RuntimeError("C63 physical GPU registration differs")
    if str(device) != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C63 requires one registered visible GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C63 deterministic CUBLAS setting is absent")

    models: dict[str, FiniteEvidenceMemoryTransformer] = {}
    training: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    evaluation_batch: dict[str, torch.Tensor] | None = None
    for mode in MODES:
        model, training[mode] = train_mode(
            config, seed=seed, mode=mode, device=device
        )
        models[mode] = model
        metrics[mode], evaluation_batch, _ = evaluate_mode(
            model, config, seed=seed, mode=mode, device=device
        )
    assert evaluation_batch is not None
    structural = structural_checks(
        models["finite_evidence_memory"], config, evaluation_batch
    )
    primary = metrics["finite_evidence_memory"]
    parameter_counts = {
        mode: make_model(config, int(g0["input_dim"])).parameter_count()
        for mode in MODES
    }
    checks = {
        "primary_accuracy": primary["clean_accuracy"]
        >= float(g0["primary_accuracy_min"]),
        "primary_beats_single_pool": primary["clean_accuracy"]
        - metrics["single_pooled_memory"]["clean_accuracy"]
        >= float(g0["primary_minus_single_pool_min"]),
        "primary_beats_standard_slots": primary["clean_accuracy"]
        - metrics["standard_slot_memory"]["clean_accuracy"]
        >= float(g0["primary_minus_standard_slots_min"]),
        "primary_beats_slot_competition": primary["clean_accuracy"]
        - metrics["slot_competition_memory"]["clean_accuracy"]
        >= float(g0["primary_minus_slot_competition_min"]),
        "primary_beats_balanced_transport": primary["clean_accuracy"]
        - metrics["balanced_transport_memory"]["clean_accuracy"]
        >= float(g0["primary_minus_balanced_transport_min"]),
        "wrong_history_specificity": primary["wrong_accuracy_drop"]
        >= float(g0["wrong_history_accuracy_drop_min"]),
        "wrong_history_changes_orders": primary["clean_wrong_order_change_fraction"] >= 0.10,
        "nuisance_abstention": primary["nuisance_minus_useful_null_mass"]
        >= float(g0["nuisance_minus_useful_null_mass_min"]),
        "all_real_slots_used": primary["minimum_real_slot_mass_fraction"]
        >= float(g0["minimum_real_slot_mass_fraction"]),
        "evidence_conserved": primary["evidence_conservation_max_abs"]
        <= float(g0["evidence_conservation_tolerance"]),
        "primary_all_gradient_groups": training["finite_evidence_memory"]["all_gradient_groups"],
        "all_losses_decreased": all(value["loss_decreased"] for value in training.values()),
        "all_training_finite": all(value["finite"] for value in training.values()),
        "all_scores_finite": all(value["finite"] for value in metrics.values()),
        "deterministic": primary["deterministic_max_abs"] == 0.0,
        "memory_query_invariant": structural["memory_query_invariance_max_abs"]
        <= float(g0["memory_query_invariance_tolerance"]),
        "candidate_permutation": structural["candidate_permutation_max_abs"]
        <= float(g0["candidate_permutation_tolerance"]),
        "nohistory_exact": structural["nohistory_max_abs"]
        <= float(g0["exact_fallback_tolerance"]),
        "repeat_exact": structural["repeat_max_abs"]
        <= float(g0["exact_fallback_tolerance"]),
        "parameter_counts_equal": len(set(parameter_counts.values())) == 1,
        "repository_data_and_all_labels_closed": True,
    }
    root = REPO_ROOT / config["paths"]["artifact_root"]
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    report_path = root / f"seed_{seed}.json"
    checkpoint_path = checkpoint_root / f"seed_{seed}.pt"
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError(report_path if report_path.exists() else checkpoint_path)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate_id": "c63",
            "seed": seed,
            "g0_lock_sha256": g0_lock_hash,
            "state_dicts": {mode: model.state_dict() for mode, model in models.items()},
        },
        checkpoint_path,
    )
    report = {
        "candidate_id": "c63",
        "created_at": timestamp(),
        "stage": "data_free_synthetic_G0",
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
        "repository_data_fit_fresh_dev_test_qrels_opened": False,
    }
    atomic_json(report_path, report)
    return report


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, g0_lock_hash = verify_g0_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    target = root / "g0_report.json"
    reports = []
    for seed in config["synthetic_G0"]["seeds"]:
        path = root / f"seed_{int(seed)}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append((path, report))
    checks = {
        "three_registered_seeds": len(reports) == 3,
        "every_seed_passed": all(report["status"] == "passed" for _, report in reports),
        "same_lock": all(report["g0_lock_sha256"] == g0_lock_hash for _, report in reports),
        "repository_data_fit_fresh_dev_test_qrels_closed": all(
            report["repository_data_fit_fresh_dev_test_qrels_opened"] is False
            for _, report in reports
        ),
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c63",
        "created_at": timestamp(),
        "stage": "data_free_synthetic_G0_aggregate",
        "status": "passed" if passed else "failed_terminal",
        "decision": "authorize_separate_dual_domain_design"
        if passed
        else "close_c63_before_repository_data",
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
        "repository_data_fit_fresh_dev_test_qrels_opened": False,
    }
    atomic_json(target, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/g0.yaml")
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
