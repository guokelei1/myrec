"""Train, score, and aggregate the locked C56 fit-internal gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for value in (str(SYSTEM_ROOT), str(REPO_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from model.query_complement import (  # noqa: E402
    MODES,
    QueryComplementTokenCompetitionTransformer,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from probe.locking import (  # noqa: E402
    load_config,
    read_json,
    sha256_file,
    verify_execution,
    write_once,
)
from train.data import C56Store, iter_batches, to_device  # noqa: E402


PRIMARY = "query_complement_token"
CONTROL_TO_SCORE = {
    "unprojected_token": "unprojected",
    "pooled_complement": "pooled",
    "raw_candidate": "raw",
}
SCORE_NAMES = (
    "base",
    "item_only",
    "primary",
    "wrong",
    "edge",
    "unprojected",
    "pooled",
    "raw",
    "primary_correction",
    "wrong_correction",
    "edge_correction",
    "unprojected_correction",
    "pooled_correction",
    "raw_correction",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def assert_cuda(config: Mapping[str, Any], seed: int, device_name: str) -> None:
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C56 seed/GPU binding differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C56 deterministic CUBLAS workspace absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C56 requires exactly one visible GPU")


def make_model(config: Mapping[str, Any]) -> QueryComplementTokenCompetitionTransformer:
    value = config["model"]
    encoding = config["encoding"]
    return QueryComplementTokenCompetitionTransformer(
        input_dim=int(encoding["input_dim"]),
        hidden_dim=int(value["hidden_dim"]),
        heads=int(value["heads"]),
        ffn_dim=int(value["ffn_dim"]),
        token_layers=int(value["token_layers"]),
        dropout=float(value["dropout"]),
        max_query_tokens=int(encoding["max_query_tokens"]),
        max_item_tokens=int(encoding["max_item_tokens"]),
        max_history=int(value["max_history"]),
        zero_initial_output=bool(value["zero_initial_output"]),
    )


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positive = labels.gt(0) & mask.bool()
    count = positive.sum(dim=-1, keepdim=True)
    if not bool(count.gt(0).all()):
        raise ValueError("C56 training row lacks positive label")
    target = positive.to(scores.dtype) / count.to(scores.dtype)
    negative = -torch.finfo(scores.dtype).max
    log_probability = torch.log_softmax(scores.masked_fill(~mask.bool(), negative), dim=-1)
    return -(target * log_probability).sum(dim=-1).mean()


def probability_residual(base: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shifted = base.astype(np.float64) - float(np.max(base))
    probability = np.exp(shifted)
    probability /= probability.sum()
    positive = labels > 0
    target = positive.astype(np.float64) / positive.sum()
    return (target - probability).astype(np.float32)


def batches(
    store: C56Store,
    indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
    shuffle: bool,
) -> list[np.ndarray]:
    value = config["training"]
    return list(
        iter_batches(
            store,
            indices,
            seed=seed,
            shuffle=shuffle,
            max_requests=int(value["max_requests_per_batch"]),
            max_transport_cells=int(value["max_transport_cells"]),
        )
    )


def train_mode(
    *,
    store: C56Store,
    indices: Sequence[int],
    schedules: Sequence[Sequence[np.ndarray]],
    config: Mapping[str, Any],
    mode: str,
    seed: int,
    device: torch.device,
) -> tuple[QueryComplementTokenCompetitionTransformer, dict[str, Any]]:
    seed_all(seed)
    model = make_model(config).to(device)
    initial = state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    gradient_names: set[str] = set()
    epoch_means: list[float] = []
    started = time.time()
    model.train()
    for epoch_batches in schedules:
        losses: list[float] = []
        for request_batch in epoch_batches:
            batch = store.collate(request_batch, with_labels=True)
            tensors = to_device(batch, device)
            labels = tensors.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            output = model(**tensors, mode=mode)
            loss = listwise_loss(output.scores, labels, tensors["candidate_mask"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite C56 loss: {mode}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C56 gradient: {mode}/{name}")
                    if bool(parameter.grad.ne(0).any()):
                        gradient_names.add(name)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_means.append(float(np.mean(losses)))
    return model.eval(), {
        "mode": mode,
        "initial_state_sha256": initial,
        "final_state_sha256": state_hash(model),
        "parameters": model.parameter_count(),
        "epoch_loss_means": epoch_means,
        "finite": bool(np.isfinite(epoch_means).all()),
        "loss_decreased": epoch_means[-1] < epoch_means[0],
        "active_gradient_parameters": sorted(gradient_names),
        "elapsed_seconds": time.time() - started,
    }


def score_model(
    *,
    model: QueryComplementTokenCompetitionTransformer,
    store: C56Store,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    mode: str,
    history_source: str = "true",
    edge_ablation: bool = False,
) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "request_ids": [],
        "item_ids": [],
        "base": [],
        "item_only": [],
        "scores": [],
        "correction": [],
    }
    model.eval()
    with torch.inference_mode():
        for request_batch in batches(store, indices, config, seed=0, shuffle=False):
            batch = store.collate(request_batch, history_source=history_source)
            tensors = to_device(batch, device)
            output = model(**tensors, mode=mode, edge_ablation=edge_ablation)
            arrays = {
                "base": tensors["base_scores"].cpu().numpy(),
                "item_only": tensors["item_only_scores"].cpu().numpy(),
                "scores": output.scores.cpu().numpy(),
                "correction": output.correction.cpu().numpy(),
            }
            mask = batch["candidate_mask"]
            for row in range(len(request_batch)):
                count = int(mask[row].sum())
                rows["request_ids"].append(batch["request_ids"][row])
                rows["item_ids"].append(
                    [str(value) for value in batch["candidate_item_ids"][row, :count]]
                )
                for name in arrays:
                    rows[name].append(np.asarray(arrays[name][row, :count], dtype=np.float32).copy())
    return rows


def maximum_difference(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    return max(float(np.max(np.abs(a - b))) for a, b in zip(first, second))


def ranked_items(request_id: str, item_ids: Sequence[str], scores: np.ndarray) -> list[str]:
    return [
        row.item_id
        for row in sort_candidates(
            request_id,
            [ScoredCandidate(str(item), float(score)) for item, score in zip(item_ids, scores)],
        )
    ]


def order_changes(reference: Mapping[str, Any], proposed: Mapping[str, Any]) -> dict[str, Any]:
    any_count = top_count = 0
    for request_id, items, first, second in zip(
        reference["request_ids"], reference["item_ids"], reference["scores"], proposed["scores"]
    ):
        a = ranked_items(request_id, items, first)
        b = ranked_items(request_id, items, second)
        any_count += int(a != b)
        top_count += int(set(a[:10]) != set(b[:10]))
    count = len(reference["request_ids"])
    return {
        "requests": count,
        "any_count": any_count,
        "any_fraction": any_count / count,
        "top10_count": top_count,
        "top10_fraction": top_count / count,
    }


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, dtype=np.int64), np.concatenate(rows).astype(np.float32, copy=False)


def save_score_rows(path: Path, rows: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    offsets, _ = flatten(rows["base"])
    payload = {"offsets": offsets}
    for name in SCORE_NAMES:
        payload[name] = flatten(rows[name])[1]
    np.savez(path, **payload)
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}


def permutation_audit(
    model: QueryComplementTokenCompetitionTransformer,
    store: C56Store,
    indices: Sequence[int],
    device: torch.device,
) -> float:
    batch = store.collate(indices[:2])
    tensors = to_device(batch, device)
    count = tensors["candidate_mask"].shape[1]
    permutation = torch.arange(count - 1, -1, -1, device=device)
    inverse = torch.argsort(permutation)
    changed = dict(tensors)
    for name in (
        "candidate_tokens",
        "candidate_token_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        changed[name] = tensors[name][:, permutation]
    with torch.inference_mode():
        first = model(**tensors, mode=PRIMARY).scores
        second = model(**changed, mode=PRIMARY).scores[:, inverse]
    return float((first - second).abs().max().cpu())


def structural_fallback_audit(
    model: QueryComplementTokenCompetitionTransformer,
    store: C56Store,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    nohistory = score_model(
        model=model,
        store=store,
        indices=store.role("structural_nohistory"),
        config=config,
        device=device,
        mode=PRIMARY,
    )
    repeat = score_model(
        model=model,
        store=store,
        indices=store.role("structural_repeat"),
        config=config,
        device=device,
        mode=PRIMARY,
    )
    return {
        "nohistory_max_abs_vs_base": maximum_difference(nohistory["scores"], nohistory["base"]),
        "repeat_max_abs_vs_item_only": maximum_difference(repeat["scores"], repeat["item_only"]),
    }


def run_seed(config_path: str | Path, seed: int, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_cuda(config, seed, device_name)
    store = C56Store(config, REPO_ROOT)
    train_indices, holdout = store.role("train"), store.role("holdout")
    expected_hash = store.selection["candidate_key_sha256"]["holdout"]
    actual_hash = store.candidate_hash(holdout)
    if actual_hash != expected_hash:
        raise RuntimeError("C56 holdout candidate hash differs")
    schedules = [
        batches(store, train_indices, config, seed=seed + epoch, shuffle=True)
        for epoch in range(int(config["training"]["epochs"]))
    ]
    device = torch.device(device_name)
    models: dict[str, QueryComplementTokenCompetitionTransformer] = {}
    training: dict[str, Any] = {}
    for mode in MODES:
        models[mode], training[mode] = train_mode(
            store=store,
            indices=train_indices,
            schedules=schedules,
            config=config,
            mode=mode,
            seed=seed,
            device=device,
        )
    initials = {row["initial_state_sha256"] for row in training.values()}
    parameters = {row["parameters"] for row in training.values()}
    if len(initials) != 1 or len(parameters) != 1:
        raise RuntimeError("C56 paired initialization/capacity differs")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for mode, model in models.items():
        path = checkpoint_root / f"seed_{seed}_{mode}.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save(
            {
                "candidate_id": "c56",
                "seed": seed,
                "mode": mode,
                "state_dict": model.state_dict(),
                "execution_lock_sha256": execution_hash,
            },
            path,
        )
        checkpoints[mode] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}

    primary = score_model(
        model=models[PRIMARY], store=store, indices=holdout, config=config, device=device, mode=PRIMARY
    )
    deterministic = score_model(
        model=models[PRIMARY], store=store, indices=holdout, config=config, device=device, mode=PRIMARY
    )
    wrong = score_model(
        model=models[PRIMARY], store=store, indices=holdout, config=config, device=device,
        mode=PRIMARY, history_source="wrong",
    )
    edge = score_model(
        model=models[PRIMARY], store=store, indices=holdout, config=config, device=device,
        mode=PRIMARY, edge_ablation=True,
    )
    controls = {
        score_name: score_model(
            model=models[mode], store=store, indices=holdout, config=config, device=device, mode=mode
        )
        for mode, score_name in CONTROL_TO_SCORE.items()
    }
    base_reference = dict(primary)
    base_reference["scores"] = primary["base"]
    scored_rows: dict[str, Any] = {
        "request_ids": primary["request_ids"],
        "item_ids": primary["item_ids"],
        "base": primary["base"],
        "item_only": primary["item_only"],
        "primary": primary["scores"],
        "wrong": wrong["scores"],
        "edge": edge["scores"],
        "unprojected": controls["unprojected"]["scores"],
        "pooled": controls["pooled"]["scores"],
        "raw": controls["raw"]["scores"],
        "primary_correction": primary["correction"],
        "wrong_correction": wrong["correction"],
        "edge_correction": edge["correction"],
        "unprojected_correction": controls["unprojected"]["correction"],
        "pooled_correction": controls["pooled"]["correction"],
        "raw_correction": controls["raw"]["correction"],
    }
    score_artifact = save_score_rows(root / f"seed_{seed}_scores.npz", scored_rows)
    fallback = structural_fallback_audit(models[PRIMARY], store, config, device)
    deterministic_error = maximum_difference(primary["scores"], deterministic["scores"])
    permutation_error = permutation_audit(models[PRIMARY], store, holdout, device)
    changes = {
        "primary_vs_base": order_changes(base_reference, primary),
        "primary_vs_wrong": order_changes(primary, wrong),
        "primary_vs_edge": order_changes(primary, edge),
    }
    tolerance = float(config["evaluation"]["exact_fallback_tolerance"])
    checks = {
        "train_holdout_disjoint": not (set(train_indices) & set(holdout)),
        "candidate_hash_asserted": actual_hash == expected_hash,
        "paired_initialization": len(initials) == 1,
        "equal_parameters": len(parameters) == 1,
        "finite_training": all(row["finite"] for row in training.values()),
        "all_mode_loss_decreased": all(row["loss_decreased"] for row in training.values()),
        "deterministic": deterministic_error <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": permutation_error <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "exact_nohistory_base": fallback["nohistory_max_abs_vs_base"] <= tolerance,
        "exact_repeat_item_only": fallback["repeat_max_abs_vs_item_only"] <= tolerance,
        "holdout_labels_closed": True,
        "C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c56",
        "stage": "seed",
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_lock_sha256": execution_hash,
        "train_requests": len(train_indices),
        "holdout_requests": len(holdout),
        "holdout_candidate_key_sha256": actual_hash,
        "training": training,
        "fallback": fallback,
        "deterministic_max_abs_difference": deterministic_error,
        "candidate_permutation_max_abs_difference": permutation_error,
        "changes": changes,
        "checks": checks,
        "checkpoints": checkpoints,
        "score_artifact": score_artifact,
        "train_fit_labels_read": True,
        "holdout_fit_labels_read": False,
        "C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / f"seed_{seed}_report.json", report)
    return report


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[row]) : int(offsets[row + 1])], dtype=np.float32).copy()
        for row in range(len(offsets) - 1)
    ]


def load_score_rows(report: Mapping[str, Any]) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C56 score artifact changed")
    with np.load(path, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        return {name: unflatten(offsets, source[name]) for name in SCORE_NAMES}


def reference_rows(store: C56Store, indices: Sequence[int], scores: Sequence[np.ndarray]) -> dict[str, Any]:
    return {
        "request_ids": [store.data.request_ids[int(index)] for index in indices],
        "item_ids": [store.data.candidate_ids(int(index)) for index in indices],
        "scores": scores,
    }


def average_rows(collections: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [np.mean(np.stack(rows), axis=0).astype(np.float32) for rows in zip(*collections)]


def aggregate_a0(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    store = C56Store(config, REPO_ROOT)
    holdout = store.role("holdout")
    actual_hash = store.candidate_hash(holdout)
    if actual_hash != store.selection["candidate_key_sha256"]["holdout"]:
        raise RuntimeError("C56 A0 candidate hash differs")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    seeds = list(map(int, config["training"]["seeds"]))
    reports = [read_json(root / f"seed_{seed}_report.json") for seed in seeds]
    rows = {seed: load_score_rows(report) for seed, report in zip(seeds, reports)}
    base = rows[seeds[0]]["base"]
    if any(not all(np.array_equal(a, b) for a, b in zip(base, rows[seed]["base"])) for seed in seeds[1:]):
        raise RuntimeError("C56 base differs across seeds")
    ensemble = {
        name: average_rows([rows[seed][name] for seed in seeds])
        for name in ("primary", "wrong", "edge", "unprojected", "pooled", "raw")
    }
    reference = reference_rows(store, holdout, base)
    primary = reference_rows(store, holdout, ensemble["primary"])
    changes = {
        "primary_vs_base": order_changes(reference, primary),
        "primary_vs_wrong": order_changes(primary, reference_rows(store, holdout, ensemble["wrong"])),
        "primary_vs_edge": order_changes(primary, reference_rows(store, holdout, ensemble["edge"])),
    }
    per_seed_changes = {
        str(seed): {
            "primary_vs_base": order_changes(
                reference, reference_rows(store, holdout, rows[seed]["primary"])
            ),
            "primary_vs_wrong": order_changes(
                reference_rows(store, holdout, rows[seed]["primary"]),
                reference_rows(store, holdout, rows[seed]["wrong"]),
            ),
            "primary_vs_edge": order_changes(
                reference_rows(store, holdout, rows[seed]["primary"]),
                reference_rows(store, holdout, rows[seed]["edge"]),
            ),
        }
        for seed in seeds
    }
    ev = config["evaluation"]
    checks = {
        "all_seed_execution_checks": all(all(report["checks"].values()) for report in reports),
        "ensemble_base_activity": changes["primary_vs_base"]["any_fraction"] >= float(ev["active_order_change_fraction_min"])
        and changes["primary_vs_base"]["top10_fraction"] >= float(ev["active_top10_change_fraction_min"]),
        "ensemble_wrong_history_load_bearing": changes["primary_vs_wrong"]["any_fraction"] >= float(ev["wrong_order_change_fraction_min"])
        and changes["primary_vs_wrong"]["top10_fraction"] >= float(ev["wrong_top10_change_fraction_min"]),
        "ensemble_candidate_edges_load_bearing": changes["primary_vs_edge"]["any_fraction"] >= float(ev["edge_order_change_fraction_min"])
        and changes["primary_vs_edge"]["top10_fraction"] >= float(ev["edge_top10_change_fraction_min"]),
        "all_seed_wrong_history_load_bearing": all(
            row["primary_vs_wrong"]["any_fraction"] >= float(ev["wrong_order_change_fraction_min"])
            and row["primary_vs_wrong"]["top10_fraction"] >= float(ev["wrong_top10_change_fraction_min"])
            for row in per_seed_changes.values()
        ),
        "all_seed_candidate_edges_load_bearing": all(
            row["primary_vs_edge"]["any_fraction"] >= float(ev["edge_order_change_fraction_min"])
            and row["primary_vs_edge"]["top10_fraction"] >= float(ev["edge_top10_change_fraction_min"])
            for row in per_seed_changes.values()
        ),
        "holdout_labels_closed": True,
        "C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c56",
        "gate": "A0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_label_free_mechanics" if passed else "failed_label_free_mechanics_terminal",
        "execution_lock_sha256": execution_hash,
        "holdout_requests": len(holdout),
        "holdout_candidate_key_sha256": actual_hash,
        "changes": changes,
        "per_seed_changes": per_seed_changes,
        "checks": checks,
        "train_fit_labels_read": True,
        "holdout_fit_labels_read": False,
        "C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / "a0_report.json", value)
    if not passed:
        terminal = dict(value)
        terminal.update(
            {
                "gate_id": config["gate_id"],
                "decision": "close_query_complement_token_competition_on_mechanics",
                "claims": {"architecture_signal": False, "fresh_result": False, "novelty": False},
            }
        )
        write_once(REPO_ROOT / config["paths"]["promoted_report"], terminal)
    return value


def paired_interval(difference: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    difference = np.asarray(difference, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        positions = rng.integers(0, len(difference), size=(count, len(difference)))
        draws[start : start + count] = difference[positions].mean(axis=1)
    return {
        "mean": float(difference.mean()),
        "percentile_95_ci": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "requests": len(difference),
        "samples": samples,
        "seed": seed,
    }


def metric_rows(
    store: C56Store,
    indices: Sequence[int],
    scores: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    ndcg = {name: [] for name in scores}
    mse = {"zero": []}
    mse.update({name: [] for name in scores if name != "base"})
    for row, index in enumerate(indices):
        request_id = store.data.request_ids[int(index)]
        item_ids = store.data.candidate_ids(int(index))
        labels = store.label(int(index))
        positives = {item for item, label in zip(item_ids, labels) if label > 0}
        base = np.asarray(scores["base"][row], dtype=np.float32)
        target = probability_residual(base, labels)
        mse["zero"].append(float(np.mean(target**2)))
        for name, values in scores.items():
            value = np.asarray(values[row], dtype=np.float32)
            ranking = ranked_items(request_id, item_ids, value)
            ndcg[name].append(ndcg_at_k(ranking, positives, 10))
            if name != "base":
                correction = value - base
                mse[name].append(float(np.mean((correction - target) ** 2)))
    return {
        "ndcg": {name: np.asarray(value, dtype=np.float64) for name, value in ndcg.items()},
        "mse": {name: np.asarray(value, dtype=np.float64) for name, value in mse.items()},
    }


def fold_means(values: np.ndarray, request_ids: Sequence[str], folds: int) -> list[float]:
    groups = np.asarray(
        [int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % folds for value in request_ids]
    )
    return [float(values[groups == fold].mean()) for fold in range(folds)]


def aggregate_a1(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    a0 = read_json(root / "a0_report.json")
    if a0.get("status") != "passed_label_free_mechanics":
        raise PermissionError("C56 A1 requires passed label-free A0")
    store = C56Store(config, REPO_ROOT)
    holdout = store.role("holdout")
    actual_hash = store.candidate_hash(holdout)
    if actual_hash != store.selection["candidate_key_sha256"]["holdout"]:
        raise RuntimeError("C56 A1 candidate hash differs")
    seeds = list(map(int, config["training"]["seeds"]))
    reports = [read_json(root / f"seed_{seed}_report.json") for seed in seeds]
    rows = {seed: load_score_rows(report) for seed, report in zip(seeds, reports)}
    base = rows[seeds[0]]["base"]
    score_names = ("primary", "wrong", "edge", "unprojected", "pooled", "raw")
    ensemble = {"base": base}
    ensemble.update(
        {name: average_rows([rows[seed][name] for seed in seeds]) for name in score_names}
    )
    metrics = metric_rows(store, holdout, ensemble)
    ev = config["evaluation"]
    samples, bootstrap_seed = int(ev["bootstrap_samples"]), int(ev["bootstrap_seed"])
    comparisons: dict[str, Any] = {}
    control_names = ("unprojected", "pooled", "raw")
    comparisons["ndcg_primary_minus_base"] = paired_interval(
        metrics["ndcg"]["primary"] - metrics["ndcg"]["base"], samples=samples, seed=bootstrap_seed
    )
    for offset, name in enumerate((*control_names, "wrong", "edge"), start=1):
        comparisons[f"ndcg_primary_minus_{name}"] = paired_interval(
            metrics["ndcg"]["primary"] - metrics["ndcg"][name],
            samples=samples,
            seed=bootstrap_seed + offset,
        )
    comparisons["mse_zero_minus_primary"] = paired_interval(
        metrics["mse"]["zero"] - metrics["mse"]["primary"],
        samples=samples,
        seed=bootstrap_seed + 10,
    )
    for offset, name in enumerate((*control_names, "wrong"), start=11):
        comparisons[f"mse_{name}_minus_primary"] = paired_interval(
            metrics["mse"][name] - metrics["mse"]["primary"],
            samples=samples,
            seed=bootstrap_seed + offset,
        )
    mean_mse = {name: float(value.mean()) for name, value in metrics["mse"].items()}
    relative_mse = {
        "over_zero": (mean_mse["zero"] - mean_mse["primary"]) / mean_mse["zero"],
        **{
            f"over_{name}": (mean_mse[name] - mean_mse["primary"]) / mean_mse[name]
            for name in (*control_names, "wrong")
        },
    }
    seed_directions = {}
    for seed in seeds:
        seed_scores = {"base": rows[seed]["base"]}
        seed_scores.update({name: rows[seed][name] for name in score_names})
        seed_metric = metric_rows(store, holdout, seed_scores)
        seed_directions[str(seed)] = {
            f"ndcg_over_{name}": float(
                (seed_metric["ndcg"]["primary"] - seed_metric["ndcg"][name]).mean()
            )
            for name in ("base", *control_names, "wrong")
        }
        seed_directions[str(seed)].update(
            {
                f"mse_over_{name}": float(
                    (seed_metric["mse"][name] - seed_metric["mse"]["primary"]).mean()
                )
                for name in ("zero", *control_names, "wrong")
            }
        )
    request_ids = [store.data.request_ids[index] for index in holdout]
    fold_directions = {
        f"ndcg_over_{name}": fold_means(
            metrics["ndcg"]["primary"] - metrics["ndcg"][name],
            request_ids,
            int(ev["hash_folds"]),
        )
        for name in ("base", *control_names, "wrong")
    }
    fold_directions.update(
        {
            f"mse_over_{name}": fold_means(
                metrics["mse"][name] - metrics["mse"]["primary"],
                request_ids,
                int(ev["hash_folds"]),
            )
            for name in ("zero", *control_names, "wrong")
        }
    )
    checks = {
        "A0_passed": True,
        "candidate_hash_asserted": actual_hash == a0["holdout_candidate_key_sha256"],
        "ndcg_gain_over_base": comparisons["ndcg_primary_minus_base"]["mean"] >= float(ev["ndcg_primary_minus_base_min"])
        and comparisons["ndcg_primary_minus_base"]["percentile_95_ci"][0] > 0,
        "ndcg_gain_over_controls": all(
            comparisons[f"ndcg_primary_minus_{name}"]["mean"] >= float(ev["ndcg_primary_minus_each_control_min"])
            and comparisons[f"ndcg_primary_minus_{name}"]["percentile_95_ci"][0] > 0
            for name in control_names
        ),
        "ndcg_gain_over_wrong": comparisons["ndcg_primary_minus_wrong"]["mean"] >= float(ev["ndcg_primary_minus_wrong_min"])
        and comparisons["ndcg_primary_minus_wrong"]["percentile_95_ci"][0] > 0,
        "mse_gain_over_zero": relative_mse["over_zero"] >= float(ev["residual_mse_relative_gain_over_zero_min"])
        and comparisons["mse_zero_minus_primary"]["percentile_95_ci"][0] > 0,
        "mse_gain_over_controls": all(
            relative_mse[f"over_{name}"] >= float(ev["residual_mse_relative_gain_over_each_control_min"])
            and comparisons[f"mse_{name}_minus_primary"]["percentile_95_ci"][0] > 0
            for name in control_names
        ),
        "mse_gain_over_wrong": relative_mse["over_wrong"] >= float(ev["residual_mse_relative_gain_over_each_control_min"])
        and comparisons["mse_wrong_minus_primary"]["percentile_95_ci"][0] > 0,
        "all_seed_directions_positive": all(all(value > 0 for value in row.values()) for row in seed_directions.values()),
        "all_fold_directions_positive": all(all(value > 0 for value in row) for row in fold_directions.values()),
        "C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c56",
        "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_exposed_token_foundation" if passed else "failed_exposed_token_foundation_terminal",
        "decision": "authorize_fresh_dual_domain_proposal" if passed else "close_query_complement_token_competition",
        "execution_lock_sha256": execution_hash,
        "holdout_requests": len(holdout),
        "holdout_candidate_key_sha256": actual_hash,
        "mean_ndcg10": {name: float(row.mean()) for name, row in metrics["ndcg"].items()},
        "mean_residual_mse": mean_mse,
        "relative_mse_gain": relative_mse,
        "comparisons": comparisons,
        "seed_directions": seed_directions,
        "fold_directions": fold_directions,
        "checks": checks,
        "claims": {"architecture_signal": passed, "fresh_result": False, "novelty": False},
        "train_and_holdout_fit_labels_read": True,
        "C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / "signal_gate_report.json", value)
    write_once(REPO_ROOT / config["paths"]["promoted_report"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("seed", "a0", "a1"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "seed":
        if args.seed is None:
            raise ValueError("C56 seed stage requires --seed")
        value = run_seed(args.config, args.seed, args.device)
    elif args.stage == "a0":
        value = aggregate_a0(args.config)
    else:
        value = aggregate_a1(args.config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
