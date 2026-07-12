"""One-shot hash-locked C22 synthetic GPU falsifier."""

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
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.eft import EvidenceFiltrationRanker, MODES
from train.synthetic import (
    EXACT_RECURRENCE,
    NO_HISTORY,
    SUPPORTED_TRANSFER,
    SyntheticBatch,
    batch_schedule,
    corrupt_supported,
    generate_split,
    permute_candidates,
    remove_identity,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("candidate_id") != "c22":
        raise ValueError("unexpected C22 config")
    if config.get("gate_id") != "c22_evidence_filtration_synthetic_v1":
        raise ValueError("unexpected C22 gate")
    if tuple(config["training"]["modes"]) != MODES:
        raise ValueError("C22 modes changed")
    if config["training"]["seeds"] != [20260730, 20260731, 20260732]:
        raise ValueError("C22 seeds changed")
    if int(config["training"]["attempts"]) != 1:
        raise ValueError("C22 attempt budget changed")
    if int(config["physical_gpu"]) != 2:
        raise ValueError("C22 GPU allocation changed")
    authorization = config["authorization"]
    if authorization.get("synthetic_gpu") is not True:
        raise PermissionError("C22 synthetic GPU gate is not authorized")
    if any(
        authorization[name] is not False
        for name in ("repository_data", "real_train", "dev", "test", "full_training")
    ):
        raise ValueError("C22 authorization expanded")
    return config


def verify_lock() -> tuple[dict[str, Any], str]:
    path = SYSTEM_ROOT / "notes" / "proposal_lock.json"
    if not path.is_file():
        raise PermissionError("C22 proposal lock is missing")
    with path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("status") != "locked_before_learned_outcome":
        raise ValueError("unexpected C22 lock status")
    lines: list[str] = []
    failures: list[str] = []
    for relative, expected in sorted(lock["files_sha256"].items()):
        target = SYSTEM_ROOT / relative
        if not target.is_file() or sha256_file(target) != expected:
            failures.append(relative)
        lines.append(f"{expected}  {relative}\n")
    aggregate = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    if aggregate != lock.get("aggregate_sha256"):
        failures.append("aggregate_sha256")
    if failures:
        raise RuntimeError(f"C22 proposal lock mismatch: {failures}")
    return lock, sha256_file(path)


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_model(config: Mapping[str, Any], mode: str) -> EvidenceFiltrationRanker:
    data, model = config["data"], config["model"]
    return EvidenceFiltrationRanker(
        input_dim=int(data["input_dim"]),
        anchor_dim=int(model["anchor_dim"]),
        recurrence_dim=int(model["recurrence_dim"]),
        transfer_dim=int(model["transfer_dim"]),
        history_slots=int(data["history_slots"]),
        layers=int(model["layers"]),
        heads_per_block=int(model["heads_per_block"]),
        ffn_multiplier=int(model["ffn_multiplier"]),
        dropout=float(model["dropout"]),
        transfer_delta_max=float(model["transfer_delta_max"]),
        recurrence_scale_min=float(model["recurrence_scale_min"]),
        mode=mode,
    )


def generate(config: Mapping[str, Any], seed: int, split: str) -> SyntheticBatch:
    data = config["data"]
    return generate_split(
        seed=seed,
        split=split,
        requests=int(data[f"{split}_requests"]),
        candidates=int(data["candidates"]),
        history_slots=int(data["history_slots"]),
        input_dim=int(data["input_dim"]),
        strata_weights=[float(value) for value in data["strata_weights"]],
    )


def score(
    model: EvidenceFiltrationRanker,
    batch: SyntheticBatch,
    device: torch.device,
    *,
    query_present: bool = True,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    outputs = {name: [] for name in ("scores", "base_scores", "recurrence_delta", "transfer_delta")}
    model.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(batch)))
            piece = batch.subset(indices).to(device)
            inputs = piece.model_inputs()
            inputs["query_present"] = torch.full(
                (len(piece),), query_present, dtype=torch.bool, device=device
            )
            result = model(**inputs)
            for name in outputs:
                outputs[name].append(getattr(result, name).detach().cpu())
    return {name: torch.cat(values) for name, values in outputs.items()}


def winner(scores: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    return (scores.double() - candidate_ids.double() * 1e-12).argmax(dim=-1)


def accuracy(scores: torch.Tensor, batch: SyntheticBatch, mask: torch.Tensor) -> float:
    return float(
        winner(scores[mask], batch.candidate_ids[mask])
        .eq(batch.target_index[mask])
        .double()
        .mean()
    )


def target_margin(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    selected = scores.gather(1, target[:, None]).squeeze(1)
    other = scores.clone()
    other.scatter_(1, target[:, None], -torch.inf)
    return selected - other.amax(dim=-1)


def train_model(
    model: EvidenceFiltrationRanker,
    train: SyntheticBatch,
    schedule: torch.Tensor,
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
    for step, indices in enumerate(schedule):
        piece = train.subset(indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**piece.model_inputs())
        loss = F.cross_entropy(output.scores, piece.target_index)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"C22 nonfinite loss {model.mode}/{step}")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"C22 nonfinite gradient {model.mode}/{name}")
                if bool(parameter.grad.ne(0).any()):
                    gradient_names.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "steps": len(losses),
        "finite": True,
        "loss_first_50_mean": sum(losses[:50]) / min(50, len(losses)),
        "loss_last_50_mean": sum(losses[-50:]) / min(50, len(losses)),
        "nonzero_gradient_parameters": sorted(gradient_names),
    }


def jacobian_audit(model: EvidenceFiltrationRanker, device: torch.device) -> dict[str, float]:
    layer = model.layers[0]
    dims = model.block_dims
    slices = (
        slice(0, dims[0]),
        slice(dims[0], dims[0] + dims[1]),
        slice(dims[0] + dims[1], sum(dims)),
    )

    def norm(output_block: int, input_block: int) -> float:
        value = torch.randn(1, 4, sum(dims), device=device, requires_grad=True)
        output = layer(value, torch.ones(1, 4, dtype=torch.bool, device=device), model.mode)
        gradient = torch.autograd.grad(output[..., slices[output_block]].sum(), value)[0]
        return float(gradient[..., slices[input_block]].norm().detach().cpu())

    return {
        "anchor_from_recurrence": norm(0, 1),
        "anchor_from_transfer": norm(0, 2),
        "recurrence_from_transfer": norm(1, 2),
        "transfer_from_recurrence": norm(2, 1),
    }


def order_change_fraction(scores: torch.Tensor, base: torch.Tensor, candidate_ids: torch.Tensor) -> float:
    score_order = torch.argsort(scores.double() - candidate_ids.double() * 1e-12, dim=-1, descending=True)
    base_order = torch.argsort(base.double() - candidate_ids.double() * 1e-12, dim=-1, descending=True)
    return float(score_order.ne(base_order).any(dim=-1).double().mean())


def evaluate_seed(seed: int, config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed * 10_000 + 2201)
    torch.cuda.manual_seed_all(seed * 10_000 + 2201)
    template = make_model(config, "filtration")
    initial_state = template.state_dict()
    models: dict[str, EvidenceFiltrationRanker] = {}
    initial_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    for mode in MODES:
        model = make_model(config, mode)
        model.load_state_dict(initial_state)
        models[mode] = model
        initial_hashes[mode] = state_sha256(model)
        parameter_counts[mode] = sum(value.numel() for value in model.parameters())

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

    masks = {
        "nohistory": evaluation.stratum.eq(NO_HISTORY),
        "repeat": evaluation.stratum.eq(EXACT_RECURRENCE),
        "supported": evaluation.stratum.eq(SUPPORTED_TRANSFER),
    }
    metrics: dict[str, dict[str, float]] = {}
    for mode in MODES:
        metrics[mode] = {
            f"{name}_accuracy": accuracy(outputs[mode]["scores"], evaluation, mask)
            for name, mask in masks.items()
        }
        metrics[mode]["minimum_repeat_supported_accuracy"] = min(
            metrics[mode]["repeat_accuracy"], metrics[mode]["supported_accuracy"]
        )

    primary = outputs["filtration"]
    supported = masks["supported"]
    repeated = masks["repeat"]
    base_supported = accuracy(primary["base_scores"], evaluation, supported)
    clean_supported_gain = float(
        (
            target_margin(primary["scores"][supported], evaluation.target_index[supported])
            - target_margin(primary["base_scores"][supported], evaluation.target_index[supported])
        ).mean()
    )
    supported_batch = evaluation.subset(torch.nonzero(supported).flatten())
    corruptions: dict[str, Any] = {}
    for name in ("wrong_history", "shuffled_event", "coarse_only"):
        changed = corrupt_supported(supported_batch, seed=seed, corruption=name)
        changed_output = score(models["filtration"], changed, device)
        gain = float(
            (
                target_margin(changed_output["scores"], changed.target_index)
                - target_margin(changed_output["base_scores"], changed.target_index)
            ).mean()
        )
        corruptions[name] = {
            "target_margin_gain": gain,
            "gain_retention": gain / clean_supported_gain if clean_supported_gain > 0 else 1.0e9,
            "accuracy": accuracy(
                changed_output["scores"], changed, torch.ones(len(changed), dtype=torch.bool)
            ),
        }
    query_masked = score(models["filtration"], supported_batch, device, query_present=False)
    query_gain = float(
        (
            target_margin(query_masked["scores"], supported_batch.target_index)
            - target_margin(query_masked["base_scores"], supported_batch.target_index)
        ).mean()
    )
    corruptions["query_mask"] = {
        "target_margin_gain": query_gain,
        "gain_retention": query_gain / clean_supported_gain if clean_supported_gain > 0 else 1.0e9,
        "bitwise_base": torch.equal(query_masked["scores"], query_masked["base_scores"]),
    }

    identity_removed = remove_identity(evaluation)
    removed_output = score(models["filtration"], identity_removed, device)
    clean_repeat_margin = target_margin(primary["scores"][repeated], evaluation.target_index[repeated])
    removed_repeat_margin = target_margin(
        removed_output["scores"][repeated], evaluation.target_index[repeated]
    )
    clean_supported_margin = target_margin(primary["scores"][supported], evaluation.target_index[supported])
    removed_supported_margin = target_margin(
        removed_output["scores"][supported], evaluation.target_index[supported]
    )
    identity_audit = {
        "repeat_target_margin_drop": float((clean_repeat_margin - removed_repeat_margin).mean()),
        "supported_target_margin_absolute_change": float(
            (clean_supported_margin - removed_supported_margin).abs().mean()
        ),
    }

    nohistory_exact = torch.equal(
        primary["scores"][masks["nohistory"]], primary["base_scores"][masks["nohistory"]]
    )
    repeated_score = score(models["filtration"], evaluation, device)
    deterministic = torch.equal(primary["scores"], repeated_score["scores"])
    permutation = torch.randperm(evaluation.candidates.shape[1], generator=torch.Generator().manual_seed(seed + 99))
    audit_batch = evaluation.subset(torch.arange(min(128, len(evaluation))))
    permuted_batch = permute_candidates(audit_batch, permutation)
    original_audit = score(models["filtration"], audit_batch, device)
    permuted_audit = score(models["filtration"], permuted_batch, device)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    permutation_error = float(
        (original_audit["scores"] - permuted_audit["scores"][:, inverse]).abs().max()
    )
    jacobians = {
        mode: jacobian_audit(models[mode], device) for mode in ("filtration", "dense")
    }
    control_margins = {
        mode: {
            "repeat": metrics["filtration"]["repeat_accuracy"] - metrics[mode]["repeat_accuracy"],
            "supported": metrics["filtration"]["supported_accuracy"] - metrics[mode]["supported_accuracy"],
            "minimum": metrics["filtration"]["minimum_repeat_supported_accuracy"]
            - metrics[mode]["minimum_repeat_supported_accuracy"],
        }
        for mode in MODES
        if mode != "filtration"
    }
    return {
        "seed": seed,
        "training": training,
        "metrics": metrics,
        "base_supported_accuracy": base_supported,
        "clean_supported_target_margin_gain": clean_supported_gain,
        "corruptions": corruptions,
        "identity_audit": identity_audit,
        "order_change_fraction": order_change_fraction(
            primary["scores"][~masks["nohistory"]],
            primary["base_scores"][~masks["nohistory"]],
            evaluation.candidate_ids[~masks["nohistory"]],
        ),
        "nohistory_bitwise_base": nohistory_exact,
        "deterministic_rescore": deterministic,
        "permutation_max_abs_error": permutation_error,
        "jacobians": jacobians,
        "control_margins": control_margins,
        "initial_state_hashes": initial_hashes,
        "parameter_counts": parameter_counts,
        "matched_initialization": len(set(initial_hashes.values())) == 1,
        "matched_parameters": len(set(parameter_counts.values())) == 1,
        "state_sha256": {mode: state_sha256(model) for mode, model in models.items()},
    }


def adjudicate(results: list[dict[str, Any]], config: Mapping[str, Any]) -> tuple[dict[str, bool], list[dict[str, bool]]]:
    gate = config["gate"]
    rows: list[dict[str, bool]] = []
    for result in results:
        primary = result["metrics"]["filtration"]
        filtration_jacobian = result["jacobians"]["filtration"]
        dense_jacobian = result["jacobians"]["dense"]
        row = {
            "nohistory_accuracy": primary["nohistory_accuracy"] >= float(gate["nohistory_accuracy_min"]),
            "repeat_accuracy": primary["repeat_accuracy"] >= float(gate["repeat_accuracy_min"]),
            "supported_accuracy": primary["supported_accuracy"] >= float(gate["supported_accuracy_min"]),
            "supported_gain": result["clean_supported_target_margin_gain"] >= float(gate["supported_gain_over_base_min"]),
            "base_is_blind": result["base_supported_accuracy"] <= float(gate["base_supported_accuracy_max"]),
            "control_repeat_margins": all(
                values["repeat"] >= float(gate["each_stratum_control_margin_min"])
                for values in result["control_margins"].values()
            ),
            "control_supported_margins": all(
                values["supported"] >= float(gate["each_stratum_control_margin_min"])
                for values in result["control_margins"].values()
            ),
            "control_minimum_margins": all(
                values["minimum"] >= float(gate["minimum_stratum_control_margin_min"])
                for values in result["control_margins"].values()
            ),
            "corruption_retention": all(
                values["gain_retention"] <= float(gate["corruption_retention_max"])
                for values in result["corruptions"].values()
            ),
            "identity_repeat_load_bearing": result["identity_audit"]["repeat_target_margin_drop"]
            >= float(gate["identity_repeat_margin_drop_min"]),
            "identity_does_not_leak_supported": result["identity_audit"]["supported_target_margin_absolute_change"]
            <= float(gate["identity_supported_margin_change_max"]),
            "protected_jacobians": max(
                filtration_jacobian["anchor_from_recurrence"],
                filtration_jacobian["anchor_from_transfer"],
                filtration_jacobian["recurrence_from_transfer"],
            )
            <= float(gate["protected_jacobian_max"]),
            "one_way_jacobian_active": filtration_jacobian["transfer_from_recurrence"] > 0.0,
            "dense_counterexample_active": dense_jacobian["anchor_from_transfer"] > 0.0,
            "order_changes": result["order_change_fraction"] >= float(gate["order_change_fraction_min"]),
            "candidate_permutation": result["permutation_max_abs_error"] <= float(gate["permutation_max_abs_error"]),
            "nohistory_bitwise_base": result["nohistory_bitwise_base"],
            "deterministic_rescore": result["deterministic_rescore"],
            "matched_initialization": result["matched_initialization"],
            "matched_parameters": result["matched_parameters"],
            "all_training_finite": all(value["finite"] for value in result["training"].values()),
        }
        rows.append(row)
    summary = {name: all(row[name] for row in rows) for name in rows[0]}
    summary["all_seeds_pass_every_check"] = all(summary.values())
    return summary, rows


def save_checkpoints(
    results: list[dict[str, Any]], config: Mapping[str, Any], lock_sha256: str
) -> dict[str, Any]:
    # State hashes are sufficient for the synthetic gate.  Checkpoints are not
    # retained here because evaluate_seed intentionally releases model objects
    # only after all diagnostics; the raw report binds final state hashes.
    return {
        "retained": False,
        "reason": "synthetic gate retains final state hashes; no downstream run is authorized",
        "proposal_lock_sha256": lock_sha256,
        "state_sha256": {str(row["seed"]): row["state_sha256"] for row in results},
    }


def formal(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    lock, lock_sha256 = verify_lock()
    artifact = REPOSITORY_ROOT / "artifacts" / "c22_evidence_filtration_transformer" / "synthetic_gate_v1.json"
    marker = artifact.with_name("formal_attempt.json")
    if artifact.exists() or marker.exists():
        raise FileExistsError("C22 one-shot synthetic attempt already exists")
    atomic_json(
        marker,
        {
            "candidate_id": "c22",
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "proposal_lock_sha256": lock_sha256,
        },
    )
    started = time.monotonic()
    results = [evaluate_seed(int(seed), config, device) for seed in config["training"]["seeds"]]
    checks, seed_checks = adjudicate(results, config)
    report = {
        "candidate_id": "c22",
        "gate_id": config["gate_id"],
        "status": "passed" if checks["all_seeds_pass_every_check"] else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "proposal_lock_sha256": lock_sha256,
        "proposal_lock_aggregate_sha256": lock["aggregate_sha256"],
        "results": results,
        "seed_checks": {
            str(result["seed"]): row for result, row in zip(results, seed_checks)
        },
        "checks": checks,
        "checkpoints": save_checkpoints(results, config, lock_sha256),
        "boundaries": {
            "repository_data_read": False,
            "standardized_records_read": False,
            "qrels_read": False,
            "dev_evaluator_calls": 0,
            "test_access": False,
            "real_training_authorized": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    atomic_json(artifact, report)
    atomic_json(
        marker,
        {
            "candidate_id": "c22",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "proposal_lock_sha256": lock_sha256,
            "report_sha256": sha256_file(artifact),
        },
    )
    return report


def smoke(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    batch = generate_split(
        seed=91,
        split="audit",
        requests=32,
        candidates=int(config["data"]["candidates"]),
        history_slots=int(config["data"]["history_slots"]),
        input_dim=int(config["data"]["input_dim"]),
        strata_weights=[0.2, 0.4, 0.4],
    )
    model = make_model(config, "filtration").to(device)
    piece = batch.to(device)
    output = model(**piece.model_inputs())
    loss = F.cross_entropy(output.scores, piece.target_index)
    loss.backward()
    return {
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
        "nohistory_bitwise_base": torch.equal(
            output.scores[piece.stratum.eq(NO_HISTORY)],
            output.base_scores[piece.stratum.eq(NO_HISTORY)],
        ),
        "repository_data_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("C22 is locked to physical GPU 2")
    if not torch.cuda.is_available():
        raise RuntimeError("C22 requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    result = smoke(config, device) if args.mode == "smoke" else formal(config, device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
