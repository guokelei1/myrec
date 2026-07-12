"""Execute C57's locked fit-internal mechanics and utility gates."""

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
C56_ROOT = REPO_ROOT / "systems/56_query_complement_token_competition_transformer"
for value in (str(C56_ROOT), str(REPO_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)
from train.data import C56Store, iter_batches, to_device  # noqa: E402
if str(C56_ROOT) in sys.path:
    sys.path.remove(str(C56_ROOT))
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import (  # noqa: E402
    load_config,
    read_json,
    sha256_file,
    verify_execution,
    write_once,
)
from model.candidate_budget import (  # noqa: E402
    MODES,
    CandidateBudgetAttentionTransformer,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402


PRIMARY = "candidate_budget"
SCORE_NAMES = (
    "base",
    "item_only",
    "primary",
    "wrong",
    "axis",
    "slot_no_null",
    "history_softmax",
    "pooled_history",
    "raw_candidate",
)
CONTROL_MODES = {
    "slot_budget_no_null": "slot_no_null",
    "history_softmax": "history_softmax",
    "pooled_history": "pooled_history",
    "raw_candidate": "raw_candidate",
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def assert_sources(config: Mapping[str, Any]) -> None:
    for path_name, hash_name in (
        ("selection", "c56_v2_selection_sha256"),
        ("contextual_manifest", "c56_contextual_manifest_sha256"),
        ("c56_data_source", "c56_data_source_sha256"),
        ("c56_report", "c56_report_sha256"),
    ):
        if sha256_file(REPO_ROOT / config["paths"][path_name]) != config["integrity"][hash_name]:
            raise RuntimeError(f"C57 registered source changed: {path_name}")
    source = read_json(REPO_ROOT / config["paths"]["c56_report"])
    if source.get("status") != "failed_label_free_mechanics_terminal":
        raise RuntimeError("C57 C56 terminal boundary differs")
    if source.get("holdout_fit_labels_read") is not False:
        raise PermissionError("C57 source holdout labels are not closed")


def assert_cuda(config: Mapping[str, Any], seed: int, device_name: str) -> None:
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C57 GPU binding differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C57 deterministic CUBLAS workspace absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C57 requires exactly one visible GPU")


def make_model(config: Mapping[str, Any]) -> CandidateBudgetAttentionTransformer:
    value, encoding = config["model"], config["encoding"]
    return CandidateBudgetAttentionTransformer(
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
        raise ValueError("C57 row lacks positive label")
    target = positive.to(scores.dtype) / count.to(scores.dtype)
    negative = -torch.finfo(scores.dtype).max
    log_probability = torch.log_softmax(scores.masked_fill(~mask.bool(), negative), dim=-1)
    return -(target * log_probability).sum(dim=-1).mean()


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
    schedules: Sequence[Sequence[np.ndarray]],
    config: Mapping[str, Any],
    mode: str,
    seed: int,
    device: torch.device,
) -> tuple[CandidateBudgetAttentionTransformer, dict[str, Any]]:
    seed_all(seed)
    model = make_model(config).to(device)
    initial = state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epoch_means: list[float] = []
    gradient_names: set[str] = set()
    started = time.time()
    model.train()
    for schedule in schedules:
        losses: list[float] = []
        for request_batch in schedule:
            batch = store.collate(request_batch, with_labels=True)
            tensors = to_device(batch, device)
            labels = tensors.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            output = model(**tensors, mode=mode)
            loss = listwise_loss(output.scores, labels, tensors["candidate_mask"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite C57 loss: {mode}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C57 gradient: {mode}/{name}")
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
    model: CandidateBudgetAttentionTransformer,
    store: C56Store,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    mode: str,
    history_source: str = "true",
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
            output = model(**tensors, mode=mode)
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
                for name, value in arrays.items():
                    rows[name].append(np.asarray(value[row, :count], dtype=np.float32).copy())
    return rows


def max_difference(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    return max(float(np.max(np.abs(a - b))) for a, b in zip(first, second))


def ranked(request_id: str, items: Sequence[str], scores: np.ndarray) -> list[str]:
    return [
        row.item_id
        for row in sort_candidates(
            request_id,
            [ScoredCandidate(str(item), float(score)) for item, score in zip(items, scores)],
        )
    ]


def changes(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    any_count = top_count = 0
    for request_id, items, a, b in zip(
        first["request_ids"], first["item_ids"], first["scores"], second["scores"]
    ):
        ra, rb = ranked(request_id, items, a), ranked(request_id, items, b)
        any_count += int(ra != rb)
        top_count += int(set(ra[:10]) != set(rb[:10]))
    count = len(first["request_ids"])
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


def save_rows(path: Path, rows: Mapping[str, Sequence[np.ndarray]]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    offsets, _ = flatten(rows["base"])
    payload = {"offsets": offsets}
    for name in SCORE_NAMES:
        payload[name] = flatten(rows[name])[1]
    np.savez(path, **payload)
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}


def permutation_audit(
    model: CandidateBudgetAttentionTransformer,
    store: C56Store,
    indices: Sequence[int],
    device: torch.device,
) -> float:
    tensors = to_device(store.collate(indices[:4]), device)
    count = tensors["candidate_mask"].shape[1]
    order = torch.arange(count - 1, -1, -1, device=device)
    inverse = torch.argsort(order)
    moved = dict(tensors)
    for name in (
        "candidate_tokens",
        "candidate_token_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        moved[name] = tensors[name][:, order]
    with torch.inference_mode():
        first = model(**tensors, mode=PRIMARY).scores
        second = model(**moved, mode=PRIMARY).scores[:, inverse]
    return float((first - second).abs().max().cpu())


def fallback_audit(
    model: CandidateBudgetAttentionTransformer,
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
        "nohistory_max_abs_vs_base": max_difference(nohistory["scores"], nohistory["base"]),
        "repeat_max_abs_vs_item_only": max_difference(repeat["scores"], repeat["item_only"]),
    }


def run_seed(config_path: str | Path, seed: int, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_sources(config)
    assert_cuda(config, seed, device_name)
    store = C56Store(config, REPO_ROOT)
    train, holdout = store.role("train"), store.role("holdout")
    expected_hash = store.selection["candidate_key_sha256"]["holdout"]
    actual_hash = store.candidate_hash(holdout)
    if actual_hash != expected_hash:
        raise RuntimeError("C57 holdout candidate hash differs")
    schedules = [
        batches(store, train, config, seed=seed + epoch, shuffle=True)
        for epoch in range(int(config["training"]["epochs"]))
    ]
    device = torch.device(device_name)
    models: dict[str, CandidateBudgetAttentionTransformer] = {}
    training = {}
    for mode in MODES:
        models[mode], training[mode] = train_mode(
            store=store,
            schedules=schedules,
            config=config,
            mode=mode,
            seed=seed,
            device=device,
        )
    initials = {row["initial_state_sha256"] for row in training.values()}
    parameters = {row["parameters"] for row in training.values()}
    if len(initials) != 1 or len(parameters) != 1:
        raise RuntimeError("C57 paired initialization/capacity differs")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    checkpoints_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for mode, model in models.items():
        path = checkpoints_root / f"seed_{seed}_{mode}.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save(
            {
                "candidate_id": "c57",
                "seed": seed,
                "mode": mode,
                "execution_lock_sha256": execution_hash,
                "state_dict": model.state_dict(),
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
    axis = score_model(
        model=models[PRIMARY], store=store, indices=holdout, config=config, device=device,
        mode="history_softmax",
    )
    controls = {
        score_name: score_model(
            model=models[mode], store=store, indices=holdout, config=config, device=device, mode=mode
        )
        for mode, score_name in CONTROL_MODES.items()
    }
    base = dict(primary)
    base["scores"] = primary["base"]
    score_rows = {
        "base": primary["base"],
        "item_only": primary["item_only"],
        "primary": primary["scores"],
        "wrong": wrong["scores"],
        "axis": axis["scores"],
        **{name: row["scores"] for name, row in controls.items()},
    }
    artifact = save_rows(root / f"seed_{seed}_scores.npz", score_rows)
    fallback = fallback_audit(models[PRIMARY], store, config, device)
    deterministic_error = max_difference(primary["scores"], deterministic["scores"])
    permutation_error = permutation_audit(models[PRIMARY], store, holdout, device)
    order_changes = {
        "primary_vs_base": changes(base, primary),
        "primary_vs_wrong": changes(primary, wrong),
        "primary_vs_axis": changes(primary, axis),
    }
    tolerance = float(config["evaluation"]["exact_fallback_tolerance"])
    checks = {
        "train_holdout_disjoint": not (set(train) & set(holdout)),
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
    value = {
        "candidate_id": "c57",
        "stage": "seed",
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_lock_sha256": execution_hash,
        "training": training,
        "changes": order_changes,
        "fallback": fallback,
        "deterministic_max_abs_difference": deterministic_error,
        "candidate_permutation_max_abs_difference": permutation_error,
        "checks": checks,
        "checkpoints": checkpoints,
        "score_artifact": artifact,
        "holdout_candidate_key_sha256": actual_hash,
        "train_fit_labels_read": True,
        "holdout_fit_labels_read": False,
        "C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / f"seed_{seed}_report.json", value)
    return value


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[i]) : int(offsets[i + 1])], dtype=np.float32).copy()
        for i in range(len(offsets) - 1)
    ]


def load_rows(report: Mapping[str, Any]) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C57 score artifact changed")
    with np.load(path, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        return {name: unflatten(offsets, source[name]) for name in SCORE_NAMES}


def average_rows(collections: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [np.mean(np.stack(rows), axis=0).astype(np.float32) for rows in zip(*collections)]


def reference(store: C56Store, indices: Sequence[int], scores: Sequence[np.ndarray]) -> dict[str, Any]:
    return {
        "request_ids": [store.data.request_ids[index] for index in indices],
        "item_ids": [store.data.candidate_ids(index) for index in indices],
        "scores": scores,
    }


def aggregate_a0(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_sources(config)
    store = C56Store(config, REPO_ROOT)
    holdout = store.role("holdout")
    actual_hash = store.candidate_hash(holdout)
    if actual_hash != store.selection["candidate_key_sha256"]["holdout"]:
        raise RuntimeError("C57 A0 candidate hash differs")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    seeds = list(map(int, config["training"]["seeds"]))
    reports = [read_json(root / f"seed_{seed}_report.json") for seed in seeds]
    rows = {seed: load_rows(report) for seed, report in zip(seeds, reports)}
    base = rows[seeds[0]]["base"]
    ensemble = {
        name: average_rows([rows[seed][name] for seed in seeds])
        for name in ("primary", "wrong", "axis")
    }
    base_ref = reference(store, holdout, base)
    primary_ref = reference(store, holdout, ensemble["primary"])
    order_changes = {
        "primary_vs_base": changes(base_ref, primary_ref),
        "primary_vs_wrong": changes(primary_ref, reference(store, holdout, ensemble["wrong"])),
        "primary_vs_axis": changes(primary_ref, reference(store, holdout, ensemble["axis"])),
    }
    per_seed = {
        str(seed): {
            "primary_vs_base": changes(base_ref, reference(store, holdout, rows[seed]["primary"])),
            "primary_vs_wrong": changes(
                reference(store, holdout, rows[seed]["primary"]),
                reference(store, holdout, rows[seed]["wrong"]),
            ),
            "primary_vs_axis": changes(
                reference(store, holdout, rows[seed]["primary"]),
                reference(store, holdout, rows[seed]["axis"]),
            ),
        }
        for seed in seeds
    }
    ev = config["evaluation"]
    def above(row: Mapping[str, Any], prefix: str) -> bool:
        return row["any_fraction"] >= float(ev[f"{prefix}_order_change_fraction_min"]) and row[
            "top10_fraction"
        ] >= float(ev[f"{prefix}_top10_change_fraction_min"])
    checks = {
        "all_seed_execution_checks": all(all(report["checks"].values()) for report in reports),
        "ensemble_base_activity": above(order_changes["primary_vs_base"], "active"),
        "ensemble_wrong_history_load_bearing": above(order_changes["primary_vs_wrong"], "wrong"),
        "ensemble_axis_load_bearing": above(order_changes["primary_vs_axis"], "axis"),
        "all_seed_wrong_history_load_bearing": all(above(row["primary_vs_wrong"], "wrong") for row in per_seed.values()),
        "all_seed_axis_load_bearing": all(above(row["primary_vs_axis"], "axis") for row in per_seed.values()),
        "holdout_labels_closed": True,
        "C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c57",
        "gate": "A0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_label_free_mechanics" if passed else "failed_label_free_mechanics_terminal",
        "execution_lock_sha256": execution_hash,
        "holdout_candidate_key_sha256": actual_hash,
        "holdout_requests": len(holdout),
        "changes": order_changes,
        "per_seed_changes": per_seed,
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
                "decision": "close_candidate_budget_attention_on_mechanics",
                "claims": {"architecture_signal": False, "fresh_result": False, "novelty": False},
            }
        )
        write_once(REPO_ROOT / config["paths"]["promoted_report"], terminal)
    return value


def paired_interval(values: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        positions = rng.integers(0, len(values), size=(count, len(values)))
        draws[start : start + count] = values[positions].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "percentile_95_ci": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "requests": len(values),
        "samples": samples,
        "seed": seed,
    }


def ndcg_rows(
    store: C56Store, indices: Sequence[int], scores: Mapping[str, Sequence[np.ndarray]]
) -> dict[str, np.ndarray]:
    output = {name: [] for name in scores}
    for row, index in enumerate(indices):
        request_id = store.data.request_ids[index]
        items = store.data.candidate_ids(index)
        labels = store.label(index)
        positive = {item for item, label in zip(items, labels) if label > 0}
        for name, values in scores.items():
            output[name].append(ndcg_at_k(ranked(request_id, items, values[row]), positive, 10))
    return {name: np.asarray(value, dtype=np.float64) for name, value in output.items()}


def fold_means(values: np.ndarray, request_ids: Sequence[str], folds: int) -> list[float]:
    group = np.asarray(
        [int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % folds for value in request_ids]
    )
    return [float(values[group == fold].mean()) for fold in range(folds)]


def aggregate_a1(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_sources(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    a0 = read_json(root / "a0_report.json")
    if a0.get("status") != "passed_label_free_mechanics":
        raise PermissionError("C57 A1 requires passed A0")
    store = C56Store(config, REPO_ROOT)
    holdout = store.role("holdout")
    actual_hash = store.candidate_hash(holdout)
    if actual_hash != a0["holdout_candidate_key_sha256"]:
        raise RuntimeError("C57 A1 candidate hash differs")
    seeds = list(map(int, config["training"]["seeds"]))
    reports = [read_json(root / f"seed_{seed}_report.json") for seed in seeds]
    rows = {seed: load_rows(report) for seed, report in zip(seeds, reports)}
    names = ("primary", "wrong", "slot_no_null", "history_softmax", "pooled_history", "raw_candidate")
    ensemble = {"base": rows[seeds[0]]["base"]}
    ensemble.update({name: average_rows([rows[seed][name] for seed in seeds]) for name in names})
    metric = ndcg_rows(store, holdout, ensemble)
    ev = config["evaluation"]
    samples, base_seed = int(ev["bootstrap_samples"]), int(ev["bootstrap_seed"])
    comparisons = {}
    compare_names = ("base", "wrong", "slot_no_null", "history_softmax", "pooled_history", "raw_candidate")
    for offset, name in enumerate(compare_names):
        comparisons[f"primary_minus_{name}"] = paired_interval(
            metric["primary"] - metric[name], samples=samples, seed=base_seed + offset
        )
    seed_directions = {}
    for seed in seeds:
        seed_scores = {"base": rows[seed]["base"]}
        seed_scores.update({name: rows[seed][name] for name in names})
        seed_metric = ndcg_rows(store, holdout, seed_scores)
        seed_directions[str(seed)] = {
            name: float((seed_metric["primary"] - seed_metric[name]).mean()) for name in compare_names
        }
    request_ids = [store.data.request_ids[index] for index in holdout]
    folds = {
        name: fold_means(
            metric["primary"] - metric[name], request_ids, int(ev["hash_folds"])
        )
        for name in compare_names
    }
    controls = ("slot_no_null", "history_softmax", "pooled_history", "raw_candidate")
    checks = {
        "A0_passed": True,
        "candidate_hash_asserted": actual_hash == a0["holdout_candidate_key_sha256"],
        "gain_over_base": comparisons["primary_minus_base"]["mean"] >= float(ev["ndcg_primary_minus_base_min"])
        and comparisons["primary_minus_base"]["percentile_95_ci"][0] > 0,
        "gain_over_wrong": comparisons["primary_minus_wrong"]["mean"] >= float(ev["ndcg_primary_minus_wrong_min"])
        and comparisons["primary_minus_wrong"]["percentile_95_ci"][0] > 0,
        "gain_over_controls": all(
            comparisons[f"primary_minus_{name}"]["mean"] >= float(ev["ndcg_primary_minus_each_control_min"])
            and comparisons[f"primary_minus_{name}"]["percentile_95_ci"][0] > 0
            for name in controls
        ),
        "all_seed_directions_positive": all(all(value > 0 for value in row.values()) for row in seed_directions.values()),
        "all_fold_directions_positive": all(all(value > 0 for value in row) for row in folds.values()),
        "C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c57",
        "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_exposed_candidate_budget_foundation" if passed else "failed_exposed_candidate_budget_terminal",
        "decision": "authorize_fresh_dual_domain_candidate_budget" if passed else "close_candidate_budget_attention",
        "execution_lock_sha256": execution_hash,
        "holdout_candidate_key_sha256": actual_hash,
        "holdout_requests": len(holdout),
        "mean_ndcg10": {name: float(value.mean()) for name, value in metric.items()},
        "comparisons": comparisons,
        "seed_directions": seed_directions,
        "fold_directions": folds,
        "checks": checks,
        "claims": {"architecture_signal": passed, "fresh_result": False, "novelty": False},
        "train_and_holdout_fit_labels_read": True,
        "C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / "mechanism_gate_report.json", value)
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
            raise ValueError("C57 seed stage requires seed")
        value = run_seed(args.config, args.seed, args.device)
    elif args.stage == "a0":
        value = aggregate_a0(args.config)
    else:
        value = aggregate_a1(args.config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
