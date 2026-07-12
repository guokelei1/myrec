"""One-shot, hash-locked C18 synthetic GPU falsifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.ecot import ECOTRanker, protected_margin_violation, soft_constraint_penalty
from train.synthetic import (
    NO_HISTORY,
    REPEAT_CONFLICT,
    SUPPORTED_NONREPEAT,
    SyntheticBatch,
    batch_schedule,
    corrupt_supported,
    generate_split,
    permute_candidates,
)


TRAINABLE_MODES = ("projection", "direct", "soft_penalty")
CORRUPTIONS = ("wrong_history", "shuffled_event", "query_mask", "coarse_only")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_lock() -> dict[str, Any]:
    lock_path = SYSTEM_ROOT / "notes" / "proposal_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("proposal lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    aggregate_lines: list[str] = []
    for relative, expected in sorted(lock["files_sha256"].items()):
        path = SYSTEM_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            failures.append(relative)
        aggregate_lines.append(f"{expected}  {relative}\n")
    aggregate = hashlib.sha256("".join(aggregate_lines).encode("utf-8")).hexdigest()
    if aggregate != lock["aggregate_sha256"]:
        failures.append("aggregate_sha256")
    if failures:
        raise RuntimeError(f"proposal-lock mismatch: {failures}")
    return lock


def model_from_config(config: dict[str, Any], mode: str) -> ECOTRanker:
    data = config["data"]
    model = config["model"]
    return ECOTRanker(
        input_dim=int(data["input_dim"]),
        d_model=int(model["d_model"]),
        nhead=int(model["nhead"]),
        layers=int(model["layers"]),
        ffn_dim=int(model["ffn_dim"]),
        history_slots=int(data["history_slots"]),
        dropout=float(model["dropout"]),
        proposal_radius=float(model["proposal_radius"]),
        repeat_bonus=float(model["repeat_bonus"]),
        projection_bisection_steps=int(model["projection_bisection_steps"]),
        mode=mode,
    )


def generate(config: dict[str, Any], seed: int, split: str) -> SyntheticBatch:
    data = config["data"]
    requests = int(data[f"{split}_requests"])
    return generate_split(
        seed=seed,
        split=split,
        requests=requests,
        candidates=int(data["candidates"]),
        history_slots=int(data["history_slots"]),
        input_dim=int(data["input_dim"]),
        topics=int(data["topics"]),
        strata_weights=[float(value) for value in data["strata_weights"]],
    )


def score(
    model: ECOTRanker,
    batch: SyntheticBatch,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    model.eval()
    keys = ("scores", "base_scores", "anchor_scores", "proposal_scores", "raw_transfer")
    outputs: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(batch)))
            values = batch.subset(indices).to(device)
            result = model(**values.model_inputs())
            for key in keys:
                outputs[key].append(getattr(result, key).detach().cpu())
    return {key: torch.cat(values) for key, values in outputs.items()}


def winner(scores: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    # Candidate-ID tie break remains invariant to input permutation.
    adjusted = scores.to(torch.float64) - candidate_ids.to(torch.float64) * 1e-12
    return adjusted.argmax(dim=1)


def accuracy(
    scores: torch.Tensor, batch: SyntheticBatch, mask: torch.Tensor
) -> float:
    if not bool(mask.any()):
        return float("nan")
    predicted = winner(scores[mask], batch.candidate_ids[mask])
    return float(predicted.eq(batch.target_index[mask]).to(torch.float64).mean())


def target_margin(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
    other = scores.clone()
    other.scatter_(1, targets.unsqueeze(1), -torch.inf)
    return target - other.amax(dim=1)


def train_model(
    model: ECOTRanker,
    batch: SyntheticBatch,
    schedule: torch.Tensor,
    config: dict[str, Any],
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
    ce_losses: list[float] = []
    penalty_losses: list[float] = []
    finite = True
    for step, indices in enumerate(schedule):
        values = batch.subset(indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**values.model_inputs())
        ce = F.cross_entropy(output.scores, values.target_index)
        penalty = output.scores.sum() * 0.0
        if model.mode == "soft_penalty":
            penalty = soft_constraint_penalty(
                output.scores,
                output.anchor_scores,
                values.repeat_mask,
                values.candidate_mask,
            )
        loss = ce + float(settings["soft_penalty_weight"]) * penalty
        if not bool(torch.isfinite(loss)):
            finite = False
            raise RuntimeError(f"non-finite loss for {model.mode} at step {step}")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
            finite = False
            raise RuntimeError(f"invalid gradients for {model.mode} at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
        optimizer.step()
        losses.append(float(loss.detach()))
        ce_losses.append(float(ce.detach()))
        penalty_losses.append(float(penalty.detach()))
    return {
        "steps": len(losses),
        "finite": finite,
        "loss_first_50_mean": sum(losses[:50]) / min(50, len(losses)),
        "loss_last_50_mean": sum(losses[-50:]) / min(50, len(losses)),
        "ce_last_50_mean": sum(ce_losses[-50:]) / min(50, len(ce_losses)),
        "penalty_last_50_mean": sum(penalty_losses[-50:]) / min(50, len(penalty_losses)),
    }


def evaluate_seed(
    *,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed * 10_000 + 5501)
    initial = model_from_config(config, "projection")
    initial_state = initial.state_dict()
    models: dict[str, ECOTRanker] = {}
    initial_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    for mode in TRAINABLE_MODES:
        model = model_from_config(config, mode)
        model.load_state_dict(initial_state)
        models[mode] = model
        initial_hashes[mode] = state_sha256(model)
        parameter_counts[mode] = sum(value.numel() for value in model.parameters())

    train = generate(config, seed, "train")
    evaluation = generate(config, seed, "eval")
    settings = config["training"]
    schedule = batch_schedule(
        seed=seed,
        requests=len(train),
        steps=int(settings["steps"]),
        batch_size=int(settings["batch_size"]),
    )
    training: dict[str, Any] = {}
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for mode in TRAINABLE_MODES:
        training[mode] = train_model(models[mode], train, schedule, config, device)
        outputs[mode] = score(models[mode], evaluation, device)

    no_history = evaluation.stratum.eq(NO_HISTORY)
    repeated = evaluation.stratum.eq(REPEAT_CONFLICT)
    supported = evaluation.stratum.eq(SUPPORTED_NONREPEAT)
    history_present = ~no_history
    metrics: dict[str, dict[str, float]] = {}
    for mode in TRAINABLE_MODES:
        values = outputs[mode]
        metrics[mode] = {
            "no_history_accuracy": accuracy(values["scores"], evaluation, no_history),
            "repeat_accuracy": accuracy(values["scores"], evaluation, repeated),
            "supported_accuracy": accuracy(values["scores"], evaluation, supported),
            "repeat_target_margin": float(
                target_margin(values["scores"][repeated], evaluation.target_index[repeated]).mean()
            ),
            "supported_target_margin": float(
                target_margin(values["scores"][supported], evaluation.target_index[supported]).mean()
            ),
        }

    primary = outputs["projection"]
    base_supported_accuracy = accuracy(primary["base_scores"], evaluation, supported)
    anchor_repeat_accuracy = accuracy(primary["anchor_scores"], evaluation, repeated)
    clean_margin_gain = float(
        (
            target_margin(primary["scores"][supported], evaluation.target_index[supported])
            - target_margin(primary["base_scores"][supported], evaluation.target_index[supported])
        ).mean()
    )

    supported_batch = evaluation.subset(torch.nonzero(supported).view(-1))
    corruption_results: dict[str, Any] = {}
    for corruption in CORRUPTIONS:
        corrupted = corrupt_supported(supported_batch, seed=seed, corruption=corruption)
        corrupted_scores = score(models["projection"], corrupted, device)
        gain = float(
            (
                target_margin(corrupted_scores["scores"], corrupted.target_index)
                - target_margin(corrupted_scores["base_scores"], corrupted.target_index)
            ).mean()
        )
        retention = gain / clean_margin_gain if clean_margin_gain > 0 else float("inf")
        corruption_results[corruption] = {
            "target_margin_gain_over_base": gain,
            "retention": retention,
            "accuracy": accuracy(
                corrupted_scores["scores"],
                corrupted,
                torch.ones(len(corrupted), dtype=torch.bool),
            ),
        }

    repeat_projection_delta = (
        primary["scores"][repeated] - primary["proposal_scores"][repeated]
    )
    active_fraction = float(
        repeat_projection_delta.abs().amax(dim=1).gt(1e-6).to(torch.float64).mean()
    )
    history_delta_range = (
        primary["scores"][history_present] - primary["base_scores"][history_present]
    ).amax(dim=1) - (
        primary["scores"][history_present] - primary["base_scores"][history_present]
    ).amin(dim=1)
    thresholds = config["thresholds"]
    load_bearing_fraction = float(
        history_delta_range
        .ge(float(thresholds["load_bearing_delta_range"]))
        .to(torch.float64)
        .mean()
    )
    violation = protected_margin_violation(
        primary["scores"],
        primary["anchor_scores"],
        evaluation.repeat_mask,
        evaluation.candidate_mask,
    )
    maximum_violation = float(violation.max())
    no_history_bitwise = torch.equal(
        primary["scores"][no_history], primary["base_scores"][no_history]
    )

    permutation_source = evaluation.subset(torch.arange(min(64, len(evaluation))))
    permutation = torch.arange(config["data"]["candidates"] - 1, -1, -1)
    permuted = permute_candidates(permutation_source, permutation)
    original_scores = score(models["projection"], permutation_source, device)["scores"]
    permuted_scores = score(models["projection"], permuted, device)["scores"]
    permutation_error = float((permuted_scores - original_scores[:, permutation]).abs().max())
    deterministic_scores = score(models["projection"], permutation_source, device)["scores"]
    deterministic_equal = torch.equal(original_scores, deterministic_scores)

    projection_repeat = metrics["projection"]["repeat_accuracy"]
    projection_supported = metrics["projection"]["supported_accuracy"]
    direct_repeat = metrics["direct"]["repeat_accuracy"]
    direct_supported = metrics["direct"]["supported_accuracy"]
    best_supported_control = max(
        metrics["direct"]["supported_accuracy"], metrics["soft_penalty"]["supported_accuracy"]
    )
    projection_worst = min(projection_repeat, projection_supported)
    best_control_worst = max(
        min(metrics[mode]["repeat_accuracy"], metrics[mode]["supported_accuracy"])
        for mode in ("direct", "soft_penalty")
    )
    all_finite = all(
        bool(torch.isfinite(value).all())
        for output in outputs.values()
        for value in output.values()
    ) and all(value["finite"] for value in training.values())

    conditions = {
        "repeat_accuracy": projection_repeat >= float(thresholds["repeat_accuracy_min"]),
        "repeat_anchor_noninferiority": projection_repeat
        >= anchor_repeat_accuracy - float(thresholds["repeat_anchor_noninferiority"]),
        "supported_accuracy": projection_supported
        >= float(thresholds["supported_accuracy_min"]),
        "supported_gain_over_base": projection_supported - base_supported_accuracy
        >= float(thresholds["supported_gain_over_base_min"]),
        "supported_control_noninferiority": projection_supported
        >= best_supported_control - float(thresholds["supported_control_noninferiority"]),
        "worst_subset_advantage": projection_worst - best_control_worst
        >= float(thresholds["worst_subset_advantage_min"]),
        "repeat_advantage_over_direct": projection_repeat - direct_repeat
        >= float(thresholds["repeat_advantage_over_direct_min"]),
        "supported_noninferior_to_direct": projection_supported
        >= direct_supported - float(thresholds["supported_loss_vs_direct_max"]),
        "clean_margin_gain_positive": clean_margin_gain > 0.0,
        "all_corruption_retention": all(
            value["retention"] <= float(thresholds["corruption_retention_max"])
            for value in corruption_results.values()
        ),
        "projection_active": active_fraction
        >= float(thresholds["projection_active_fraction_min"]),
        "load_bearing": load_bearing_fraction
        >= float(thresholds["load_bearing_fraction_min"]),
        "protected_margin": maximum_violation
        <= float(thresholds["max_margin_violation"]),
        "no_history_bitwise": no_history_bitwise,
        "candidate_permutation": permutation_error
        <= float(thresholds["max_candidate_permutation_error"]),
        "deterministic_rescore": deterministic_equal,
        "finite": all_finite,
        "matched_parameter_count": len(set(parameter_counts.values())) == 1,
        "matched_initialization": len(set(initial_hashes.values())) == 1,
    }

    return {
        "seed": seed,
        "parameter_counts": parameter_counts,
        "initial_state_sha256": initial_hashes,
        "training": training,
        "metrics": metrics,
        "base_supported_accuracy": base_supported_accuracy,
        "anchor_repeat_accuracy": anchor_repeat_accuracy,
        "clean_supported_margin_gain_over_base": clean_margin_gain,
        "corruptions": corruption_results,
        "projection_active_repeat_fraction": active_fraction,
        "load_bearing_history_present_fraction": load_bearing_fraction,
        "maximum_protected_margin_violation": maximum_violation,
        "no_history_bitwise_base": no_history_bitwise,
        "candidate_permutation_max_abs_error": permutation_error,
        "deterministic_rescore_equal": deterministic_equal,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(SYSTEM_ROOT / "configs" / "synthetic_gate.yaml")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    if config_path != (SYSTEM_ROOT / "configs" / "synthetic_gate.yaml").resolve():
        raise RuntimeError("only the locked candidate-local config is accepted")
    lock = verify_lock()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config["physical_gpu"]):
        raise RuntimeError("physical GPU binding mismatch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("deterministic CUBLAS workspace is required")
    if args.device != config["device"] or not torch.cuda.is_available():
        raise RuntimeError("locked CUDA device is unavailable")
    if int(config["attempts"]["learned_runs"]) != 1:
        raise RuntimeError("one-shot attempt contract changed")

    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output_root = REPOSITORY_ROOT / "artifacts" / "c18_evidence_constrained_order_transformer"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "synthetic_gate_v1.json"
    if output_path.exists():
        raise RuntimeError(f"one-shot outcome already exists: {output_path}")

    started = time.time()
    seed_results: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        result = evaluate_seed(seed=int(seed), config=config, device=device)
        seed_results.append(result)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "passed": result["passed"],
                    "projection": result["metrics"]["projection"],
                    "failed": [name for name, passed in result["conditions"].items() if not passed],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report = {
        "candidate_id": "c18",
        "gate_id": config["gate_id"],
        "status": "passed" if all(value["passed"] for value in seed_results) else "failed_stop",
        "passed": all(value["passed"] for value in seed_results),
        "decision": (
            "eligible for separately frozen train-internal real gate design"
            if all(value["passed"] for value in seed_results)
            else "stop C18 before repository data, dev, or test"
        ),
        "proposal_lock_sha256": sha256_file(SYSTEM_ROOT / "notes" / "proposal_lock.json"),
        "lock_aggregate_sha256": lock["aggregate_sha256"],
        "config_sha256": sha256_file(config_path),
        "execution": {
            "elapsed_seconds": time.time() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "physical_gpu": int(config["physical_gpu"]),
            "visible_device": args.device,
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "access": config["access"],
        "seed_results": seed_results,
    }
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(output_path), "passed": report["passed"]}, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
