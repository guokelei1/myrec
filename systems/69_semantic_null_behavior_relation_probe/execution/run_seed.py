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

from execution.locking import load_config, sha256_file, timestamp, verify_execution_lock  # noqa: E402
from execution.runtime import (  # noqa: E402
    DomainStore, batch_pairs, choose_negative_indices, flatten, materialize_pairs,
    score_request,
)
# C47's frozen lock helper unconditionally prepends its own system root while
# runtime dependencies load.  Restore C69's root immediately before importing
# the candidate-local model, and name the concrete submodule explicitly.
sys.path.insert(0, str(SYSTEM_ROOT))
from model.behavior_relation import BehaviorRelationTransformer  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def make_model(config: Mapping[str, Any], input_dim: int) -> BehaviorRelationTransformer:
    row = config["model"]
    return BehaviorRelationTransformer(
        input_dim=input_dim,
        width=int(row["width"]),
        heads=int(row["heads"]),
        layers=int(row["layers"]),
        ffn_dim=int(row["ffn_dim"]),
        dropout=float(row["dropout"]),
        score_bound=float(row["score_bound"]),
    )


def gradient_groups(names: set[str]) -> dict[str, bool]:
    groups = {name: False for name in (
        "input_projection", "relation_token", "role", "transformer", "output_norm", "score_head"
    )}
    for name in names:
        for group in groups:
            if name == group or name.startswith(group + "."):
                groups[group] = True
    return groups


def train_mode(
    config: Mapping[str, Any],
    *,
    store: DomainStore,
    sequences: np.ndarray,
    request_rows: np.ndarray,
    target_rows: np.ndarray,
    seed: int,
    mode: str,
    device: torch.device,
) -> tuple[BehaviorRelationTransformer, dict[str, Any]]:
    row = config["training"]
    seed_all(seed)
    model = make_model(config, store.input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(row["learning_rate"]), weight_decay=float(row["weight_decay"])
    )
    rng = np.random.default_rng(seed + 69)
    losses, pair_gaps, target_cosines = [], [], []
    active: set[str] = set()
    model.train()
    for _ in range(int(row["steps"])):
        sampled = rng.integers(0, len(request_rows), size=int(row["batch_size"]))
        source_np, target_np, request_np = batch_pairs(sequences, request_rows, target_rows, sampled)
        source = torch.from_numpy(source_np).to(device)
        target = torch.from_numpy(target_np).to(device)
        requests = torch.from_numpy(request_np).to(device)
        negative_indices, diagnostic = choose_negative_indices(
            source, target, requests, mode=mode,
            target_similarity_weight=float(row["target_similarity_weight"]),
        )
        positive_score = model.anchored_score(source, target)
        negative_score = model.anchored_score(source, target[negative_indices])
        loss = F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"C69 {store.domain}/{mode} nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"C69 nonfinite gradient: {name}")
            if bool(parameter.grad.ne(0).any()):
                active.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(row["gradient_clip_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        pair_gaps.append(diagnostic["pair_cosine_abs_gap"])
        target_cosines.append(diagnostic["target_cosine"])
    groups = gradient_groups(active)
    model.eval()
    return model, {
        "mode": mode,
        "steps": len(losses),
        "transition_examples": len(request_rows),
        "finite": bool(np.isfinite(losses).all()),
        "loss_first_50": float(np.mean(losses[:50])),
        "loss_last_50": float(np.mean(losses[-50:])),
        "loss_decreased": float(np.mean(losses[-50:])) < float(np.mean(losses[:50])),
        "negative_pair_cosine_abs_gap": float(np.mean(pair_gaps)),
        "negative_target_cosine": float(np.mean(target_cosines)),
        "gradient_groups": groups,
        "all_gradient_groups": all(groups.values()),
        "parameter_count": model.parameter_count,
    }


def score_a(
    config: Mapping[str, Any],
    *,
    store: DomainStore,
    primary: BehaviorRelationTransformer,
    random_control: BehaviorRelationTransformer,
    device: torch.device,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    row = config["scoring"]
    store.assert_candidate_hash()
    scores: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "primary_true", "primary_wrong", "random_true", "semantic_true", "semantic_wrong"
        )
    }
    determinism = 0.0
    permutation = 0.0
    nohistory = 0.0
    source_zero = 0.0
    for position, (index, donor) in enumerate(zip(store.a_indices(), store.donors())):
        query = store.query(index)
        candidates = store.candidates(index)
        true_history = store.history(index)
        wrong_history = store.history(index, donor)
        true_primary, true_semantic = score_request(
            primary, query, true_history, candidates,
            temperature=float(row["query_history_temperature"]),
            batch_size=int(row["pair_batch_size"]), device=device,
        )
        wrong_primary, wrong_semantic = score_request(
            primary, query, wrong_history, candidates,
            temperature=float(row["query_history_temperature"]),
            batch_size=int(row["pair_batch_size"]), device=device,
        )
        random_true, _ = score_request(
            random_control, query, true_history, candidates,
            temperature=float(row["query_history_temperature"]),
            batch_size=int(row["pair_batch_size"]), device=device,
        )
        scores["primary_true"].append(true_primary)
        scores["primary_wrong"].append(wrong_primary)
        scores["random_true"].append(random_true)
        scores["semantic_true"].append(true_semantic)
        scores["semantic_wrong"].append(wrong_semantic)
        if position < 32:
            repeat, _ = score_request(
                primary, query, true_history, candidates,
                temperature=float(row["query_history_temperature"]),
                batch_size=int(row["pair_batch_size"]), device=device,
            )
            reverse, _ = score_request(
                primary, query, true_history, candidates[::-1].copy(),
                temperature=float(row["query_history_temperature"]),
                batch_size=int(row["pair_batch_size"]), device=device,
            )
            empty, _ = score_request(
                primary, query, np.empty((0, candidates.shape[1]), dtype=np.float32), candidates,
                temperature=float(row["query_history_temperature"]),
                batch_size=int(row["pair_batch_size"]), device=device,
            )
            determinism = max(determinism, float(np.max(np.abs(true_primary - repeat))))
            permutation = max(permutation, float(np.max(np.abs(true_primary - reverse[::-1]))))
            nohistory = max(nohistory, float(np.max(np.abs(empty))))
            with torch.inference_mode():
                zero = torch.zeros(min(64, len(candidates)), candidates.shape[1], device=device)
                c = torch.from_numpy(np.asarray(candidates[: len(zero)], dtype=np.float32)).to(device)
                source_zero = max(source_zero, float(primary.anchored_score(zero, c).abs().max().cpu()))
    mechanics = {
        "candidate_hash_matches": True,
        "determinism_max_abs": determinism,
        "candidate_permutation_max_abs": permutation,
        "nohistory_max_abs": nohistory,
        "source_zero_max_abs": source_zero,
        "all_finite": all(np.isfinite(row).all() for values in scores.values() for row in values),
    }
    return scores, mechanics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/signal_gate.yaml")
    parser.add_argument("--domain", choices=("kuai", "amazon"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    lock, lock_hash = verify_execution_lock(config)
    registered = [int(v) for v in config["training"]["seeds"][args.domain]]
    if args.seed not in registered:
        raise RuntimeError("C69 seed is not registered for domain")
    expected_gpu = str(config["resources"]["seed_to_physical_gpu"][str(args.seed)])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
        raise RuntimeError(f"C69 seed requires physical GPU {expected_gpu}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C69 expects one visible GPU")
    device = torch.device("cuda:0")
    c47 = load_config(REPO_ROOT / config["paths"]["c47_config"])
    c38 = load_config(REPO_ROOT / config["paths"]["c38_config"])
    store = DomainStore(args.domain, c47, c38)
    sequences, request_rows, target_rows = materialize_pairs(store)
    started = time.monotonic()
    models, fits = {}, {}
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for mode in config["training"]["modes"]:
        model, fit = train_mode(
            config, store=store, sequences=sequences, request_rows=request_rows,
            target_rows=target_rows, seed=args.seed, mode=mode, device=device,
        )
        models[mode] = model
        fits[mode] = fit
        torch.save(
            {"domain": args.domain, "seed": args.seed, "mode": mode, "state_dict": model.state_dict()},
            checkpoint_root / f"{args.domain}_seed_{args.seed}_{mode}.pt",
        )
    scores, mechanics = score_a(
        config, store=store, primary=models["semantic_matched_negative"],
        random_control=models["random_negative"], device=device,
    )
    root = REPO_ROOT / config["paths"]["artifact_root"]
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / f"{args.domain}_seed_{args.seed}_scores.npz"
    report_path = root / f"{args.domain}_seed_{args.seed}_report.json"
    if score_path.exists() or report_path.exists():
        raise FileExistsError(score_path if score_path.exists() else report_path)
    offsets, _ = flatten(scores["primary_true"])
    with score_path.open("wb") as handle:
        np.savez(handle, offsets=offsets, **{name: flatten(rows)[1] for name, rows in scores.items()})
    scoring = config["scoring"]
    checks = {
        "fit_finite": all(fit["finite"] for fit in fits.values()),
        "loss_decreased": all(fit["loss_decreased"] for fit in fits.values()),
        "all_gradient_groups": all(fit["all_gradient_groups"] for fit in fits.values()),
        "equal_parameter_count": len({fit["parameter_count"] for fit in fits.values()}) == 1,
        "candidate_hash": mechanics["candidate_hash_matches"],
        "finite_scores": mechanics["all_finite"],
        "determinism": mechanics["determinism_max_abs"] <= float(scoring["deterministic_tolerance"]),
        "candidate_permutation": mechanics["candidate_permutation_max_abs"] <= float(scoring["candidate_permutation_tolerance"]),
        "nohistory_zero": mechanics["nohistory_max_abs"] <= float(scoring["exact_zero_tolerance"]),
        "source_zero": mechanics["source_zero_max_abs"] <= float(scoring["exact_zero_tolerance"]),
    }
    report = {
        "schema": "myrec.c69.seed.v1", "candidate_id": "c69", "created_at": timestamp(),
        "domain": args.domain, "seed": args.seed, "execution_lock_sha256": lock_hash,
        "proposal_lock_sha256": lock["proposal_lock_sha256"], "fits": fits,
        "mechanics": mechanics, "checks": checks, "passed_A0": all(checks.values()),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "score_artifact": {"path": str(score_path.relative_to(REPO_ROOT)), "sha256": sha256_file(score_path)},
        "elapsed_seconds": time.monotonic() - started,
        "isolation": {"A_labels_opened_during_fit_scoring": False, "fresh_reserve_opened": False,
                      "dev_test_qrels_opened": False},
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"domain": args.domain, "seed": args.seed, "passed_A0": report["passed_A0"],
                      "failed_checks": report["failed_checks"], "score": report["score_artifact"]}, sort_keys=True))


if __name__ == "__main__":
    main()
