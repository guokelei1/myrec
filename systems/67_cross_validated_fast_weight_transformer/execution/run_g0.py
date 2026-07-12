"""Run C67's locked shared-law plus nuisance data-free falsifier."""

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
from model.cross_validated_fast_weight import (  # noqa: E402
    MODES,
    CrossValidatedFastWeightTransformer,
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


def make_model(config: Mapping[str, Any]) -> CrossValidatedFastWeightTransformer:
    model = config["model"]
    synthetic = config["synthetic_G0"]
    return CrossValidatedFastWeightTransformer(
        input_dim=int(synthetic["input_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        projection_ffn_dim=int(model["projection_ffn_dim"]),
        heads=int(model["heads"]),
        dropout=float(model["dropout"]),
        initial_inner_step=float(model["initial_inner_step"]),
    )


def _householder(value: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    return value - 2.0 * (value * axis).sum(dim=-1, keepdim=True) * axis


def synthetic_batch(
    config: Mapping[str, Any],
    *,
    seed: int,
    batch_size: int,
    regime: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if regime not in {"clean", "noisy", "unsupported"}:
        raise ValueError("C67 synthetic regime differs")
    row = config["synthetic_G0"]
    dimension = int(row["input_dim"])
    events = int(row["history_events"])
    nuisance_count = int(row["nuisance_events"]) if regime == "noisy" else 0
    candidates_count = int(row["candidates"])
    noise = float(row["observation_noise"])
    generator = torch.Generator(device="cpu").manual_seed(seed)

    task_axis = F.normalize(
        torch.randn(batch_size, dimension, generator=generator), dim=-1
    )
    nuisance_axis = F.normalize(
        torch.randn(batch_size, dimension, generator=generator), dim=-1
    )
    history_keys = F.normalize(
        torch.randn(batch_size, events, dimension, generator=generator), dim=-1
    )
    if regime == "unsupported":
        event_axis = F.normalize(
            torch.randn(batch_size, events, dimension, generator=generator), dim=-1
        )
        history_values = _householder(history_keys, event_axis)
    else:
        history_values = _householder(history_keys, task_axis[:, None])
        if nuisance_count:
            history_values[:, -nuisance_count:] = _householder(
                history_keys[:, -nuisance_count:], nuisance_axis[:, None]
            )
    history_values = F.normalize(
        history_values
        + noise
        * torch.randn(history_values.shape, generator=generator),
        dim=-1,
    )
    nuisance_mask = torch.zeros(batch_size, events, dtype=torch.bool)
    if nuisance_count:
        nuisance_mask[:, -nuisance_count:] = True

    permutations = torch.stack(
        [torch.randperm(events, generator=generator) for _ in range(batch_size)]
    )
    history_keys = torch.gather(
        history_keys, 1, permutations[..., None].expand(-1, -1, dimension)
    )
    history_values = torch.gather(
        history_values, 1, permutations[..., None].expand(-1, -1, dimension)
    )
    nuisance_mask = torch.gather(nuisance_mask, 1, permutations)

    useful_positions = (~nuisance_mask).float()
    first_two = torch.topk(useful_positions, k=2, dim=1).indices
    selected = torch.gather(
        history_keys, 1, first_two[..., None].expand(-1, -1, dimension)
    )
    query = F.normalize(
        0.65 * selected[:, 0]
        + 0.35 * selected[:, 1]
        + 0.05 * torch.randn(batch_size, dimension, generator=generator),
        dim=-1,
    )
    target_axis = (
        F.normalize(torch.randn(batch_size, dimension, generator=generator), dim=-1)
        if regime == "unsupported"
        else task_axis
    )
    target = F.normalize(_householder(query, target_axis), dim=-1)
    distractors = F.normalize(
        torch.randn(
            batch_size, candidates_count - 1, dimension, generator=generator
        ),
        dim=-1,
    )
    candidates = torch.cat((target[:, None], distractors), dim=1)
    candidate_keys = torch.arange(candidates_count, dtype=torch.int64)[None].expand(
        batch_size, -1
    ).clone()
    labels = torch.zeros(batch_size, candidates_count)
    labels[:, 0] = 1.0
    candidate_permutations = torch.stack(
        [
            torch.randperm(candidates_count, generator=generator)
            for _ in range(batch_size)
        ]
    )
    candidates = torch.gather(
        candidates,
        1,
        candidate_permutations[..., None].expand(-1, -1, dimension),
    )
    candidate_keys = torch.gather(candidate_keys, 1, candidate_permutations)
    labels = torch.gather(labels, 1, candidate_permutations)
    base_scores = torch.zeros(batch_size, candidates_count)
    history_mask = torch.ones(batch_size, events, dtype=torch.bool)
    candidate_mask = torch.ones(batch_size, candidates_count, dtype=torch.bool)
    return {
        "query": query.to(device),
        "candidates": candidates.to(device),
        "candidate_keys": candidate_keys.to(device),
        "history_keys": history_keys.to(device),
        "history_values": history_values.to(device),
        "wrong_history_keys": history_keys.roll(1, dims=0).to(device),
        "wrong_history_values": history_values.roll(1, dims=0).to(device),
        "history_mask": history_mask.to(device),
        "candidate_mask": candidate_mask.to(device),
        "base_scores": base_scores.to(device),
        "item_only_scores": base_scores.to(device),
        "repeat_request": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "query_present": torch.ones(batch_size, dtype=torch.bool, device=device),
        "labels": labels.to(device),
        "nuisance_mask": nuisance_mask.to(device),
    }


def forward_values(
    batch: Mapping[str, torch.Tensor], *, wrong: bool = False
) -> dict[str, torch.Tensor]:
    return {
        "query": batch["query"],
        "candidates": batch["candidates"],
        "candidate_keys": batch["candidate_keys"],
        "history_keys": batch["wrong_history_keys"]
        if wrong
        else batch["history_keys"],
        "history_values": batch["wrong_history_values"]
        if wrong
        else batch["history_values"],
        "history_mask": batch["history_mask"],
        "candidate_mask": batch["candidate_mask"],
        "base_scores": batch["base_scores"],
        "item_only_scores": batch["item_only_scores"],
        "repeat_request": batch["repeat_request"],
        "query_present": batch["query_present"],
    }


def gradient_groups(names: set[str]) -> dict[str, bool]:
    return {
        "input_projection": any(name.startswith("input_projection.") for name in names),
        "pair_transformer": any(name.startswith("pair_transformer.") for name in names),
        "key_head": any(name.startswith("key_head.") for name in names),
        "value_head": any(name.startswith("value_head.") for name in names),
        "initial_weight": "initial_weight" in names,
        "inner_step": "inner_step_logit" in names,
        "role_tokens": "key_role" in names and "value_role" in names,
    }


def train_mode(
    config: Mapping[str, Any],
    *,
    seed: int,
    mode: str,
    device: torch.device,
) -> tuple[CrossValidatedFastWeightTransformer, dict[str, Any]]:
    synthetic = config["synthetic_G0"]
    training = config["training"]
    seed_all(seed)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(synthetic["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    losses: list[float] = []
    active_names: set[str] = set()
    model.train()
    for step in range(int(synthetic["train_steps"])):
        batch = synthetic_batch(
            config,
            seed=seed * 1000 + step,
            batch_size=int(synthetic["batch_size"]),
            regime="noisy",
            device=device,
        )
        clean = model(**forward_values(batch), mode=mode)
        wrong = model(**forward_values(batch, wrong=True), mode=mode)
        loss, _ = listwise_training_loss(
            clean,
            batch["labels"],
            batch["candidate_mask"],
            wrong_output=wrong,
            correction_l2_weight=float(training["correction_l2_weight"]),
            wrong_history_neutrality_weight=float(
                training["wrong_history_neutrality_weight"]
            ),
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"C67 {mode} loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"C67 nonfinite gradient: {name}")
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
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:]))
        < float(np.mean(losses[:window])),
        "finite": bool(np.isfinite(losses).all()),
        "gradient_groups": groups,
        "all_gradient_groups": all(groups.values()),
        "inner_step": float(model.inner_step.detach().cpu()),
    }


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    return float((output.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean().cpu())


def evaluate_mode(
    model: CrossValidatedFastWeightTransformer,
    config: Mapping[str, Any],
    *,
    seed: int,
    mode: str,
    device: torch.device,
) -> dict[str, Any]:
    count = int(config["synthetic_G0"]["evaluation_requests"])
    batches = {
        regime: synthetic_batch(
            config,
            seed=seed * 100000 + offset,
            batch_size=count,
            regime=regime,
            device=device,
        )
        for offset, regime in enumerate(("clean", "noisy", "unsupported"), start=91)
    }
    with torch.inference_mode():
        clean = model(**forward_values(batches["clean"]), mode=mode)
        noisy = model(**forward_values(batches["noisy"]), mode=mode)
        wrong = model(**forward_values(batches["noisy"], wrong=True), mode=mode)
        unsupported = model(**forward_values(batches["unsupported"]), mode=mode)
    nuisance = batches["noisy"]["nuisance_mask"]
    useful_weight = float(noisy.event_weight[~nuisance].mean().cpu())
    nuisance_weight = float(noisy.event_weight[nuisance].mean().cpu())
    return {
        "clean_accuracy": accuracy(clean.scores, batches["clean"]["labels"]),
        "noisy_accuracy": accuracy(noisy.scores, batches["noisy"]["labels"]),
        "wrong_history_accuracy": accuracy(
            wrong.scores, batches["noisy"]["labels"]
        ),
        "wrong_history_accuracy_drop": accuracy(
            noisy.scores, batches["noisy"]["labels"]
        )
        - accuracy(wrong.scores, batches["noisy"]["labels"]),
        "useful_event_weight_mean": useful_weight,
        "nuisance_event_weight_mean": nuisance_weight,
        "nuisance_minus_useful_weight": nuisance_weight - useful_weight,
        "unsupported_correction_rms": float(
            unsupported.correction.square().mean().sqrt().cpu()
        ),
    }


def mechanics(
    model: CrossValidatedFastWeightTransformer,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    batch = synthetic_batch(
        config,
        seed=seed * 100000 + 199,
        batch_size=32,
        regime="noisy",
        device=device,
    )
    with torch.inference_mode():
        first = model(**forward_values(batch), mode="cross_validated_write")
        again = model(**forward_values(batch), mode="cross_validated_write")
        reverse = torch.arange(
            batch["candidate_mask"].shape[1] - 1, -1, -1, device=device
        )
        reversed_batch = dict(batch)
        for name in (
            "candidates",
            "candidate_keys",
            "candidate_mask",
            "base_scores",
            "item_only_scores",
            "labels",
        ):
            reversed_batch[name] = batch[name][:, reverse]
        reversed_output = model(
            **forward_values(reversed_batch), mode="cross_validated_write"
        )
        permutation_error = float(
            (first.scores - reversed_output.scores[:, reverse]).abs().max().cpu()
        )
        permutation_exact = torch.equal(
            first.scores, reversed_output.scores[:, reverse]
        )
        empty_batch = dict(batch)
        empty_batch["history_mask"] = torch.zeros_like(batch["history_mask"])
        empty = model(**forward_values(empty_batch), mode="cross_validated_write")
        nohistory_error = float(
            (empty.scores - batch["base_scores"]).abs().max().cpu()
        )
        repeat_batch = dict(batch)
        repeat_batch["repeat_request"] = torch.ones_like(batch["repeat_request"])
        repeat_batch["item_only_scores"] = torch.randn_like(
            batch["item_only_scores"]
        )
        repeat = model(**forward_values(repeat_batch), mode="cross_validated_write")
        repeat_error = float(
            (repeat.scores - repeat_batch["item_only_scores"]).abs().max().cpu()
        )
        changed = dict(batch)
        changed["query"] = torch.randn_like(batch["query"])
        changed["candidates"] = torch.randn_like(batch["candidates"])
        changed_output = model(
            **forward_values(changed), mode="cross_validated_write"
        )
        write_invariance_error = float(
            (first.fast_weight - changed_output.fast_weight).abs().max().cpu()
        )
    return {
        "deterministic_max_abs": float((first.scores - again.scores).abs().max().cpu()),
        "candidate_permutation_max_abs": permutation_error,
        "candidate_permutation_bit_exact": bool(permutation_exact),
        "nohistory_max_abs": nohistory_error,
        "repeat_max_abs": repeat_error,
        "write_query_candidate_invariance_max_abs": write_invariance_error,
    }


def run_seed(
    config: Mapping[str, Any], *, seed: int, device: torch.device
) -> dict[str, Any]:
    _, g0_lock_hash = verify_g0_lock(config)
    if seed not in [int(value) for value in config["synthetic_G0"]["seeds"]]:
        raise ValueError("C67 seed is not registered")
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C67 physical GPU registration differs")
    if (
        str(device) != "cuda:0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("C67 requires one registered visible GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C67 deterministic CUBLAS setting is absent")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    checkpoints = REPO_ROOT / config["paths"]["checkpoint_root"]
    root.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    training_reports: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    parameter_counts: dict[str, int] = {}
    checkpoint_rows: dict[str, Any] = {}
    primary_model: CrossValidatedFastWeightTransformer | None = None
    for mode in MODES:
        model, training = train_mode(
            config, seed=seed, mode=mode, device=device
        )
        training_reports[mode] = training
        parameter_counts[mode] = model.parameter_count()
        evaluations[mode] = evaluate_mode(
            model, config, seed=seed, mode=mode, device=device
        )
        checkpoint = checkpoints / f"seed_{seed}_{mode}.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        torch.save(
            {
                "candidate_id": "c67",
                "seed": seed,
                "mode": mode,
                "g0_lock_sha256": g0_lock_hash,
                "state_dict": model.state_dict(),
            },
            checkpoint,
        )
        checkpoint_rows[mode] = {
            "path": str(checkpoint.relative_to(REPO_ROOT)),
            "sha256": sha256_file(checkpoint),
        }
        print(f"C67 seed={seed} mode={mode} complete", flush=True)
        if mode == "cross_validated_write":
            primary_model = model
        else:
            del model
        torch.cuda.empty_cache()
    if primary_model is None:
        raise RuntimeError("C67 primary model is absent")
    mechanics_report = mechanics(
        primary_model, config, seed=seed, device=device
    )
    evaluation = config["synthetic_G0"]
    primary = evaluations["cross_validated_write"]
    checks = {
        "all_modes_same_parameters": len(set(parameter_counts.values())) == 1,
        "all_training_finite": all(value["finite"] for value in training_reports.values()),
        "all_loss_decreased": all(
            value["loss_decreased"] for value in training_reports.values()
        ),
        "all_gradient_groups": all(
            value["all_gradient_groups"] for value in training_reports.values()
        ),
        "primary_clean_accuracy": primary["clean_accuracy"]
        >= float(evaluation["primary_clean_accuracy_min"]),
        "primary_noisy_accuracy": primary["noisy_accuracy"]
        >= float(evaluation["primary_noisy_accuracy_min"]),
        "primary_beats_standard_ttt": primary["noisy_accuracy"]
        - evaluations["standard_ttt_write"]["noisy_accuracy"]
        >= float(evaluation["primary_minus_standard_ttt_min"]),
        "primary_beats_self_validated": primary["noisy_accuracy"]
        - evaluations["self_validated_write"]["noisy_accuracy"]
        >= float(evaluation["primary_minus_self_validated_min"]),
        "primary_beats_gradient_agreement": primary["noisy_accuracy"]
        - evaluations["gradient_agreement_write"]["noisy_accuracy"]
        >= float(evaluation["primary_minus_gradient_agreement_min"]),
        "wrong_history_accuracy_drop": primary["wrong_history_accuracy_drop"]
        >= float(evaluation["wrong_history_accuracy_drop_min"]),
        "nuisance_rejected": primary["nuisance_minus_useful_weight"]
        <= float(evaluation["nuisance_minus_useful_weight_max"]),
        "unsupported_abstention": primary["unsupported_correction_rms"]
        <= float(evaluation["unsupported_correction_rms_max"]),
        "deterministic": mechanics_report["deterministic_max_abs"]
        <= float(evaluation["deterministic_tolerance"]),
        "candidate_permutation": mechanics_report["candidate_permutation_max_abs"]
        <= float(evaluation["candidate_permutation_tolerance"])
        and mechanics_report["candidate_permutation_bit_exact"],
        "nohistory_exact": mechanics_report["nohistory_max_abs"]
        <= float(evaluation["exact_fallback_tolerance"]),
        "repeat_exact": mechanics_report["repeat_max_abs"]
        <= float(evaluation["exact_fallback_tolerance"]),
        "history_only_write": mechanics_report[
            "write_query_candidate_invariance_max_abs"
        ]
        == 0.0,
        "repository_data_labels_dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c67",
        "created_at": timestamp(),
        "stage": "data_free_cross_validated_fast_weight_G0_seed",
        "status": "passed" if all(checks.values()) else "failed_seed",
        "seed": seed,
        "physical_gpu": physical,
        "g0_lock_sha256": g0_lock_hash,
        "parameter_counts": parameter_counts,
        "training": training_reports,
        "evaluation": evaluations,
        "mechanics": mechanics_report,
        "checks": checks,
        "checkpoints": checkpoint_rows,
        "repository_data_opened": False,
        "labels_opened": False,
        "dev_test_qrels_opened": False,
    }
    target = root / f"seed_{seed}_report.json"
    atomic_json(target, report)
    return report


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, g0_lock_hash = verify_g0_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    seed_reports = []
    sources: dict[str, Any] = {}
    for seed in config["synthetic_G0"]["seeds"]:
        path = root / f"seed_{int(seed)}_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["g0_lock_sha256"] != g0_lock_hash:
            raise RuntimeError("C67 seed report uses another lock")
        seed_reports.append(report)
        sources[str(seed)] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
        }
    passed = len(seed_reports) == 3 and all(
        report["status"] == "passed" for report in seed_reports
    )
    result = {
        "candidate_id": "c67",
        "created_at": timestamp(),
        "gate": "data_free_cross_validated_fast_weight_G0",
        "result": "passed" if passed else "failed_terminal",
        "decision": "authorize_separate_real_data_implementation_review"
        if passed
        else "close_c67_before_repository_data",
        "g0_lock_sha256": g0_lock_hash,
        "seeds": {
            str(report["seed"]): {
                "status": report["status"],
                "primary": report["evaluation"]["cross_validated_write"],
                "control_noisy_accuracy": {
                    mode: report["evaluation"][mode]["noisy_accuracy"]
                    for mode in MODES
                    if mode != "cross_validated_write"
                },
                "failed_checks": [
                    name for name, value in report["checks"].items() if not value
                ],
            }
            for report in seed_reports
        },
        "source_seed_reports": sources,
        "freshness": {
            "repository_data_opened": False,
            "labels_opened": False,
            "dev_test_qrels_opened": False,
        },
        "stop_rule": {
            "rerun_or_generator_rescue": False,
            "real_data_authorized": passed,
        },
    }
    atomic_json(REPO_ROOT / config["paths"]["promoted_report"], result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/g0.yaml")
    parser.add_argument("--stage", choices=("seed", "aggregate"), required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "seed":
        if args.seed is None:
            parser.error("--seed is required for seed stage")
        value = run_seed(config, seed=args.seed, device=torch.device("cuda:0"))
    else:
        value = aggregate(config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
