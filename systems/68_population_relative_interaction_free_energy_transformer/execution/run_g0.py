"""Run one locked C68 data-free GPU seed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import (  # noqa: E402
    load_config,
    sha256_file,
    timestamp,
    verify_g0_lock,
)
from execution.synthetic import synthetic_batch  # noqa: E402
from model import MODES, PopulationRelativeFreeEnergyRanker  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def make_model(config: Mapping[str, Any], mode: str) -> PopulationRelativeFreeEnergyRanker:
    row = config["model"]
    return PopulationRelativeFreeEnergyRanker(
        input_dim=int(row["input_dim"]),
        hidden_dim=int(row["hidden_dim"]),
        heads=int(row["heads"]),
        layers=int(row["layers"]),
        ffn_dim=int(row["ffn_dim"]),
        dropout=float(row["dropout"]),
        temperature=float(row["temperature"]),
        correction_scale=float(row["correction_scale"]),
        mode=mode,
    )


def model_inputs(batch: Mapping[str, torch.Tensor], *, wrong: bool = False) -> dict[str, torch.Tensor]:
    return {
        "query": batch["query"],
        "candidates": batch["candidates"],
        "history": batch["wrong_history"] if wrong else batch["history"],
        "reference": batch["reference"],
        "base_scores": batch["base_scores"],
        "history_present": batch["history_present"],
        "query_present": batch["query_present"],
    }


def gradient_groups(active_names: set[str]) -> dict[str, bool]:
    groups = {
        "input_projection": False,
        "token_type": "token_type" in active_names,
        "null_candidate": "null_candidate" in active_names,
        "triplet_transformer": False,
        "output_norm": False,
        "energy_head": False,
    }
    for name in active_names:
        for prefix in ("input_projection", "triplet_transformer", "output_norm", "energy_head"):
            if name.startswith(prefix + "."):
                groups[prefix] = True
    return groups


def train_mode(
    config: Mapping[str, Any], *, seed: int, mode: str, device: torch.device
) -> tuple[PopulationRelativeFreeEnergyRanker, dict[str, Any]]:
    g0 = config["synthetic_G0"]
    training = config["training"]
    seed_all(seed)
    model = make_model(config, mode).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(g0["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    losses: list[float] = []
    active_names: set[str] = set()
    model.train()
    for step in range(int(g0["train_steps"])):
        batch = synthetic_batch(
            config,
            seed=seed * 10000 + step,
            batch_size=int(g0["batch_size"]),
            device=device,
        )
        true = model(**model_inputs(batch))
        wrong = model(**model_inputs(batch, wrong=True))
        loss = F.cross_entropy(true.scores, batch["labels"])
        if bool(batch["unsupported"].any()):
            loss = loss + float(training["unsupported_neutrality_weight"]) * (
                true.correction[batch["unsupported"]].square().mean()
            )
        loss = loss + float(training["wrong_history_neutrality_weight"]) * (
            wrong.correction.square().mean()
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"C68 {mode} nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"C68 {mode} nonfinite gradient: {name}")
            if bool(parameter.grad.ne(0).any()):
                active_names.add(name)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    window = min(50, max(1, len(losses) // 2))
    groups = gradient_groups(active_names)
    model.eval()
    return model, {
        "steps": len(losses),
        "finite": bool(np.isfinite(losses).all()),
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:])) < float(np.mean(losses[:window])),
        "gradient_groups": groups,
        "all_gradient_groups": all(groups.values()),
        "parameter_count": model.parameter_count,
    }


def accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return float((scores.argmax(dim=1) == labels).float().mean().cpu())


def evaluate_mode(
    model: PopulationRelativeFreeEnergyRanker,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    count = int(config["synthetic_G0"]["evaluation_requests"])
    clean = synthetic_batch(
        config,
        seed=seed * 100000 + 501,
        batch_size=count,
        device=device,
        unsupported_fraction=0.0,
    )
    unsupported = synthetic_batch(
        config,
        seed=seed * 100000 + 502,
        batch_size=count,
        device=device,
        unsupported_fraction=1.0,
    )
    with torch.inference_mode():
        true = model(**model_inputs(clean))
        wrong = model(**model_inputs(clean, wrong=True))
        neutral = model(**model_inputs(unsupported))
    true_accuracy = accuracy(true.scores, clean["labels"])
    return {
        "base_accuracy": accuracy(clean["base_scores"], clean["labels"]),
        "clean_accuracy": true_accuracy,
        "wrong_history_accuracy": accuracy(wrong.scores, clean["labels"]),
        "wrong_history_accuracy_drop": true_accuracy - accuracy(wrong.scores, clean["labels"]),
        "unsupported_correction_rms": float(neutral.correction.square().mean().sqrt().cpu()),
    }


def mechanics(
    model: PopulationRelativeFreeEnergyRanker,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, float | bool]:
    row = config["synthetic_G0"]
    batch = synthetic_batch(
        config,
        seed=seed * 100000 + 601,
        batch_size=64,
        device=device,
        unsupported_fraction=0.0,
    )
    with torch.inference_mode():
        direct = model(**model_inputs(batch))
        again = model(**model_inputs(batch))

        perm = torch.arange(batch["candidates"].shape[1] - 1, -1, -1, device=device)
        permuted = model(
            query=batch["query"],
            candidates=batch["candidates"][:, perm],
            history=batch["history"],
            reference=batch["reference"],
            base_scores=batch["base_scores"][:, perm],
            history_present=batch["history_present"],
            query_present=batch["query_present"],
        )
        restored = permuted.scores[:, perm]

        off = torch.zeros_like(batch["history_present"])
        no_history_inputs = model_inputs(batch)
        no_history_inputs["history_present"] = off
        no_history = model(**no_history_inputs)
        no_query_inputs = model_inputs(batch)
        no_query_inputs["query_present"] = off
        no_query = model(**no_query_inputs)
        repeat_scores = torch.flip(batch["base_scores"], dims=(1,))
        repeat_inputs = model_inputs(batch)
        repeat_inputs["repeat_mask"] = torch.ones_like(off)
        repeat_inputs["repeat_scores"] = repeat_scores
        repeat = model(**repeat_inputs)

        equal_set = model(
            query=batch["query"], candidates=batch["candidates"],
            history=batch["history"], reference=batch["history"],
            base_scores=batch["base_scores"],
        )
        fixed_carrier_history = torch.zeros_like(batch["history"])
        fixed_carrier = model(
            query=batch["query"], candidates=batch["candidates"],
            history=fixed_carrier_history, reference=batch["reference"],
            base_scores=batch["base_scores"],
        )
        reference_only_history = batch["reference"][:, : batch["history"].shape[1]]
        reference_only = model(
            query=batch["query"], candidates=batch["candidates"],
            history=reference_only_history, reference=batch["reference"],
            base_scores=batch["base_scores"],
        )
    return {
        "determinism_max_abs": float((direct.scores - again.scores).abs().max().cpu()),
        "candidate_permutation_max_abs": float((direct.scores - restored).abs().max().cpu()),
        "no_history_max_abs": float((no_history.scores - batch["base_scores"]).abs().max().cpu()),
        "query_mask_max_abs": float((no_query.scores - batch["base_scores"]).abs().max().cpu()),
        "repeat_fallback_max_abs": float((repeat.scores - repeat_scores).abs().max().cpu()),
        "equal_set_correction_max_abs": float(equal_set.correction.abs().max().cpu()),
        "fixed_carrier_correction_rms": float(fixed_carrier.correction.square().mean().sqrt().cpu()),
        "reference_only_correction_rms": float(reference_only.correction.square().mean().sqrt().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/g0.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    lock, lock_hash = verify_g0_lock(config)
    seeds = [int(value) for value in config["synthetic_G0"]["seeds"]]
    if args.seed not in seeds:
        raise RuntimeError("C68 seed is not registered")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C68 expects exactly one visible GPU per seed")
    expected_physical = int(config["resources"]["seed_to_physical_gpu"][str(args.seed)])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(expected_physical):
        raise RuntimeError(f"C68 seed {args.seed} requires physical GPU {expected_physical}, got {visible}")
    device = torch.device("cuda:0")
    output = REPO_ROOT / args.output_root / f"seed_{args.seed}_report.json"
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    fits: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    models: dict[str, PopulationRelativeFreeEnergyRanker] = {}
    for mode in MODES:
        model, fit = train_mode(config, seed=args.seed, mode=mode, device=device)
        fits[mode] = fit
        evaluations[mode] = evaluate_mode(model, config, seed=args.seed, device=device)
        checkpoint = checkpoint_root / f"seed_{args.seed}_{mode}.pt"
        torch.save({"seed": args.seed, "mode": mode, "state_dict": model.state_dict()}, checkpoint)
        models[mode] = model

    primary = models["interaction_free_energy"]
    mechanical = mechanics(primary, config, seed=args.seed, device=device)
    row = config["synthetic_G0"]
    primary_eval = evaluations["interaction_free_energy"]
    controls = [mode for mode in MODES if mode != "interaction_free_energy"]
    checks: dict[str, bool] = {
        "primary_clean_accuracy": primary_eval["clean_accuracy"] >= float(row["primary_clean_accuracy_min"]),
        "wrong_history_accuracy_drop": primary_eval["wrong_history_accuracy_drop"] >= float(row["wrong_history_accuracy_drop_min"]),
        "unsupported_neutrality": primary_eval["unsupported_correction_rms"] <= float(row["unsupported_correction_rms_max"]),
        "equal_set_neutrality": mechanical["equal_set_correction_max_abs"] <= float(row["equal_set_correction_max"]),
        "fixed_carrier_neutrality": mechanical["fixed_carrier_correction_rms"] <= float(row["fixed_carrier_correction_rms_max"]),
        "reference_only_neutrality": mechanical["reference_only_correction_rms"] <= float(row["reference_only_correction_rms_max"]),
        "candidate_permutation": mechanical["candidate_permutation_max_abs"] <= float(row["candidate_permutation_tolerance"]),
        "determinism": mechanical["determinism_max_abs"] <= float(row["deterministic_tolerance"]),
        "no_history": mechanical["no_history_max_abs"] <= float(row["exact_fallback_tolerance"]),
        "query_mask": mechanical["query_mask_max_abs"] <= float(row["exact_fallback_tolerance"]),
        "repeat_fallback": mechanical["repeat_fallback_max_abs"] <= float(row["exact_fallback_tolerance"]),
        "all_fits_finite": all(value["finite"] for value in fits.values()),
        "all_losses_decreased": all(value["loss_decreased"] for value in fits.values()),
        "all_gradient_groups": all(value["all_gradient_groups"] for value in fits.values()),
        "equal_parameter_count": len({value["parameter_count"] for value in fits.values()}) == 1,
    }
    for control in controls:
        checks[f"primary_beats_{control}"] = (
            primary_eval["clean_accuracy"] - evaluations[control]["clean_accuracy"]
            >= float(row["primary_minus_each_control_min"])
        )
    report = {
        "schema": "myrec.c68.g0_seed.v1",
        "candidate_id": "c68",
        "created_at": timestamp(),
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "visible_physical_gpu": visible,
        "g0_lock_sha256": lock_hash,
        "proposal_lock_sha256": lock["proposal_lock_sha256"],
        "config_sha256": sha256_file(args.config),
        "fits": fits,
        "evaluations": evaluations,
        "mechanics": mechanical,
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": sorted(name for name, value in checks.items() if not value),
        "elapsed_seconds": time.monotonic() - started,
        "isolation": {
            "repository_data_opened": False,
            "labels_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "passed": report["passed"], "failed_checks": report["failed_checks"], "output": str(output.relative_to(REPO_ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
