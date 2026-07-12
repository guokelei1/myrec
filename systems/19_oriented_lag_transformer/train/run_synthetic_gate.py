"""One-shot hash-locked C19 synthetic GPU gate."""

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

from model.olt import OLTRanker
from train.synthetic import (
    EXACT_REPEAT,
    NO_HISTORY,
    SUPPORTED_SUCCESSOR,
    SyntheticBatch,
    batch_schedule,
    corrupt_supported,
    generate_split,
    permute_candidates,
)


MODES = ("oriented", "diagonal", "forward", "symmetric", "free_signed")
PRIMARY_CORRUPTIONS = ("wrong_history", "shuffled_event", "query_mask", "coarse_only")


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
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_lock() -> dict[str, Any]:
    path = SYSTEM_ROOT / "notes" / "proposal_lock.json"
    if not path.is_file():
        raise RuntimeError("proposal lock missing")
    lock = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    lines: list[str] = []
    for relative, expected in sorted(lock["files_sha256"].items()):
        target = SYSTEM_ROOT / relative
        if not target.is_file() or sha256_file(target) != expected:
            failures.append(relative)
        lines.append(f"{expected}  {relative}\n")
    aggregate = hashlib.sha256("".join(lines).encode()).hexdigest()
    if aggregate != lock["aggregate_sha256"]:
        failures.append("aggregate_sha256")
    if failures:
        raise RuntimeError(f"proposal lock mismatch: {failures}")
    return lock


def model_from_config(config: dict[str, Any], mode: str) -> OLTRanker:
    data, model = config["data"], config["model"]
    return OLTRanker(
        input_dim=int(data["input_dim"]),
        d_model=int(model["d_model"]),
        nhead=int(model["nhead"]),
        layers=int(model["layers"]),
        ffn_dim=int(model["ffn_dim"]),
        history_slots=int(data["history_slots"]),
        dropout=float(model["dropout"]),
        affinity_dim=int(model["affinity_dim"]),
        temperature=float(model["temperature"]),
        identity_bias=float(model["identity_bias"]),
        evidence_scale=float(model["evidence_scale"]),
        lag_scale_max=float(model["lag_scale_max"]),
        mode=mode,
    )


def generate(config: dict[str, Any], seed: int, split: str) -> SyntheticBatch:
    data = config["data"]
    return generate_split(
        seed=seed,
        split=split,
        requests=int(data[f"{split}_requests"]),
        candidates=int(data["candidates"]),
        history_slots=int(data["history_slots"]),
        input_dim=int(data["input_dim"]),
        semantic_items=int(data["semantic_items"]),
        strata_weights=[float(value) for value in data["strata_weights"]],
    )


def score(
    model: OLTRanker,
    batch: SyntheticBatch,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    keys = (
        "scores",
        "base_scores",
        "evidence",
        "diagonal",
        "forward",
        "reverse",
        "oriented_component",
    )
    values: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(batch)))
            output = model(**batch.subset(indices).to(device).model_inputs())
            for key in keys:
                values[key].append(getattr(output, key).detach().cpu())
    return {key: torch.cat(chunks) for key, chunks in values.items()}


def winner(scores: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    adjusted = scores.double() - candidate_ids.double() * 1e-12
    return adjusted.argmax(dim=1)


def accuracy(scores: torch.Tensor, batch: SyntheticBatch, mask: torch.Tensor) -> float:
    predicted = winner(scores[mask], batch.candidate_ids[mask])
    return float(predicted.eq(batch.target_index[mask]).double().mean())


def target_margin(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    selected = scores.gather(1, target[:, None]).squeeze(1)
    others = scores.clone()
    others.scatter_(1, target[:, None], -torch.inf)
    return selected - others.amax(1)


def correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    x = first.double().flatten()
    y = second.double().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denominator == 0:
        return float("nan")
    return float(torch.dot(x, y) / denominator)


def train_model(
    model: OLTRanker,
    train: SyntheticBatch,
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
    for step, indices in enumerate(schedule):
        batch = train.subset(indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch.model_inputs())
        loss = F.cross_entropy(output.scores, batch.target_index)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss {model.mode} step {step}")
        loss.backward()
        gradients = [value.grad for value in model.parameters() if value.grad is not None]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
            raise RuntimeError(f"invalid gradients {model.mode} step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "steps": len(losses),
        "finite": True,
        "loss_first_50_mean": sum(losses[:50]) / min(50, len(losses)),
        "loss_last_50_mean": sum(losses[-50:]) / min(50, len(losses)),
    }


def evaluate_seed(seed: int, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed * 10_000 + 5601)
    template = model_from_config(config, "oriented")
    initial_state = template.state_dict()
    models: dict[str, OLTRanker] = {}
    initial_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for mode in MODES:
        model = model_from_config(config, mode)
        model.load_state_dict(initial_state)
        models[mode] = model
        initial_hashes[mode] = state_sha256(model)
        counts[mode] = sum(value.numel() for value in model.parameters())

    train = generate(config, seed, "train")
    evaluation = generate(config, seed, "eval")
    schedule = batch_schedule(
        seed=seed,
        requests=len(train),
        steps=int(config["training"]["steps"]),
        batch_size=int(config["training"]["batch_size"]),
    )
    training: dict[str, Any] = {}
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for mode in MODES:
        training[mode] = train_model(models[mode], train, schedule, config, device)
        outputs[mode] = score(models[mode], evaluation, device)

    no_history = evaluation.stratum.eq(NO_HISTORY)
    repeated = evaluation.stratum.eq(EXACT_REPEAT)
    supported = evaluation.stratum.eq(SUPPORTED_SUCCESSOR)
    history_present = ~no_history
    metrics: dict[str, dict[str, float]] = {}
    for mode in MODES:
        output = outputs[mode]
        metrics[mode] = {
            "no_history_accuracy": accuracy(output["scores"], evaluation, no_history),
            "repeat_accuracy": accuracy(output["scores"], evaluation, repeated),
            "supported_accuracy": accuracy(output["scores"], evaluation, supported),
            "supported_target_margin": float(
                target_margin(output["scores"][supported], evaluation.target_index[supported]).mean()
            ),
        }

    primary = outputs["oriented"]
    base_supported = accuracy(primary["base_scores"], evaluation, supported)
    clean_margin_gain = float(
        (
            target_margin(primary["scores"][supported], evaluation.target_index[supported])
            - target_margin(primary["base_scores"][supported], evaluation.target_index[supported])
        ).mean()
    )
    supported_batch = evaluation.subset(torch.nonzero(supported).view(-1))
    corruptions: dict[str, Any] = {}
    for name in (*PRIMARY_CORRUPTIONS, "reversed_event"):
        changed = corrupt_supported(supported_batch, seed=seed, corruption=name)
        changed_output = score(models["oriented"], changed, device)
        gain = float(
            (
                target_margin(changed_output["scores"], changed.target_index)
                - target_margin(changed_output["base_scores"], changed.target_index)
            ).mean()
        )
        corruptions[name] = {
            "accuracy": accuracy(
                changed_output["scores"], changed, torch.ones(len(changed), dtype=torch.bool)
            ),
            "target_margin_gain_over_base": gain,
            "retention": gain / clean_margin_gain if clean_margin_gain > 0 else float("inf"),
        }
        if name == "reversed_event":
            corruptions[name]["oriented_component_correlation"] = correlation(
                primary["oriented_component"][supported], changed_output["oriented_component"]
            )

    delta = primary["scores"][history_present] - primary["base_scores"][history_present]
    delta_range = delta.amax(1) - delta.amin(1)
    thresholds = config["thresholds"]
    load_bearing = float(
        delta_range.ge(float(thresholds["load_bearing_delta_range"])).double().mean()
    )
    no_history_equal = torch.equal(
        primary["scores"][no_history], primary["base_scores"][no_history]
    )
    source = evaluation.subset(torch.arange(min(64, len(evaluation))))
    permutation = torch.arange(config["data"]["candidates"] - 1, -1, -1)
    changed = permute_candidates(source, permutation)
    original_scores = score(models["oriented"], source, device)["scores"]
    changed_scores = score(models["oriented"], changed, device)["scores"]
    permutation_error = float((changed_scores - original_scores[:, permutation]).abs().max())
    deterministic = torch.equal(
        original_scores, score(models["oriented"], source, device)["scores"]
    )

    oriented_repeat = metrics["oriented"]["repeat_accuracy"]
    oriented_supported = metrics["oriented"]["supported_accuracy"]
    structured_modes = ("diagonal", "forward", "symmetric")
    best_structured_supported = max(metrics[name]["supported_accuracy"] for name in structured_modes)
    oriented_worst = min(oriented_repeat, oriented_supported)
    best_structured_worst = max(
        min(metrics[name]["repeat_accuracy"], metrics[name]["supported_accuracy"])
        for name in structured_modes
    )
    all_finite = all(
        bool(torch.isfinite(value).all())
        for output in outputs.values()
        for value in output.values()
    )
    conditions = {
        "repeat_accuracy": oriented_repeat >= float(thresholds["repeat_accuracy_min"]),
        "supported_accuracy": oriented_supported >= float(thresholds["supported_accuracy_min"]),
        "supported_gain_over_base": oriented_supported - base_supported
        >= float(thresholds["supported_gain_over_base_min"]),
        "structured_control_advantage": oriented_supported - best_structured_supported
        >= float(thresholds["structured_control_advantage_min"]),
        "worst_subset_advantage": oriented_worst - best_structured_worst
        >= float(thresholds["worst_subset_advantage_min"]),
        "free_signed_noninferiority": oriented_supported
        >= metrics["free_signed"]["supported_accuracy"]
        - float(thresholds["free_signed_noninferiority"]),
        "margin_advantage_over_forward": metrics["oriented"]["supported_target_margin"]
        - metrics["forward"]["supported_target_margin"]
        >= float(thresholds["margin_advantage_over_forward_min"]),
        "clean_margin_gain_positive": clean_margin_gain > 0,
        "primary_corruption_retention": all(
            corruptions[name]["retention"] <= float(thresholds["corruption_retention_max"])
            for name in PRIMARY_CORRUPTIONS
        ),
        "reverse_correlation": corruptions["reversed_event"]["oriented_component_correlation"]
        <= float(thresholds["reverse_correlation_max"]),
        "reverse_retention": corruptions["reversed_event"]["retention"]
        <= float(thresholds["reverse_retention_max"]),
        "load_bearing": load_bearing >= float(thresholds["load_bearing_fraction_min"]),
        "no_history_bitwise": no_history_equal,
        "candidate_permutation": permutation_error
        <= float(thresholds["candidate_permutation_error_max"]),
        "deterministic_rescore": deterministic,
        "finite": all_finite and all(value["finite"] for value in training.values()),
        "matched_parameter_count": len(set(counts.values())) == 1,
        "matched_initialization": len(set(initial_hashes.values())) == 1,
    }
    return {
        "seed": seed,
        "parameter_counts": counts,
        "initial_state_sha256": initial_hashes,
        "training": training,
        "metrics": metrics,
        "base_supported_accuracy": base_supported,
        "clean_supported_margin_gain_over_base": clean_margin_gain,
        "corruptions": corruptions,
        "load_bearing_fraction": load_bearing,
        "no_history_bitwise_base": no_history_equal,
        "candidate_permutation_max_abs_error": permutation_error,
        "deterministic_rescore_equal": deterministic,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SYSTEM_ROOT / "configs" / "synthetic_gate.yaml"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    expected_config = (SYSTEM_ROOT / "configs" / "synthetic_gate.yaml").resolve()
    if config_path != expected_config:
        raise RuntimeError("only locked candidate config accepted")
    lock = verify_lock()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config["physical_gpu"]):
        raise RuntimeError("physical GPU binding mismatch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("deterministic CUBLAS workspace required")
    if args.device != config["device"] or not torch.cuda.is_available():
        raise RuntimeError("locked CUDA device unavailable")
    if int(config["attempts"]["learned_runs"]) != 1:
        raise RuntimeError("one-shot contract changed")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    output_root = REPOSITORY_ROOT / "artifacts" / "c19_oriented_lag_transformer"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "synthetic_gate_v1.json"
    if output_path.exists():
        raise RuntimeError(f"one-shot output exists: {output_path}")
    started = time.time()
    seed_results: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        result = evaluate_seed(int(seed), config, device)
        seed_results.append(result)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "passed": result["passed"],
                    "oriented": result["metrics"]["oriented"],
                    "failed": [name for name, passed in result["conditions"].items() if not passed],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    passed = all(value["passed"] for value in seed_results)
    report = {
        "candidate_id": "c19",
        "gate_id": config["gate_id"],
        "status": "passed" if passed else "failed_stop",
        "passed": passed,
        "decision": (
            "eligible for separately frozen train-internal real-gate design"
            if passed
            else "stop C19 before repository data, dev, or test"
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
    print(json.dumps({"output": str(output_path), "passed": passed}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
