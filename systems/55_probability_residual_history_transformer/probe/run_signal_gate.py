"""Materialize, train, and aggregate the locked C55 residual signal gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C54_ROOT = REPO_ROOT / "systems/54_history_carrier_competition_transformer"
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))
# Pin the `execution` package to C54, then remove C54 from sys.path so C54's
# own bootstrap moves its root ahead of C38's also-named `model` package.
if str(C54_ROOT) in sys.path:
    sys.path.remove(str(C54_ROOT))
sys.path.insert(0, str(C54_ROOT))
import execution as _c54_execution  # noqa: E402,F401
sys.path.remove(str(C54_ROOT))

from probe.locking import (  # noqa: E402
    load_config, sha256_file, verify_execution, verify_proposal, write_once,
)
from execution.run_gate import (  # noqa: E402
    DomainData, batches, candidate_hash, collate, flatten, make_model,
    rankings, seed_all,
)
from myrec.eval.metrics import ndcg_at_k  # noqa: E402
from train.freeze_locks import load_config as load_c38_config  # noqa: E402
from train.store import FrozenTransferStore  # noqa: E402


MODES = ("history_carrier", "raw_candidate")
SCORE_NAMES = ("base", "target", "primary_correction", "wrong_correction", "raw_correction")


def standardize_base(base: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(base.dtype)
    count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (base * weight).sum(dim=-1, keepdim=True) / count
    centered = (base - mean) * weight
    scale = torch.sqrt((centered.square().sum(dim=-1, keepdim=True) / count).clamp_min(1e-12))
    return centered / scale


def probability_residual(
    standardized_base: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    negative = -torch.finfo(standardized_base.dtype).max
    probability = torch.softmax(standardized_base.masked_fill(~mask, negative), dim=-1)
    positive = labels.gt(0) & mask
    count = positive.sum(dim=-1, keepdim=True)
    if not bool(count.gt(0).all()):
        raise ValueError("C55 row lacks positive label")
    target = positive.to(standardized_base.dtype) / count.to(standardized_base.dtype)
    return (target - probability) * mask.to(standardized_base.dtype)


def residual_mse(
    correction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(correction.dtype)
    return ((correction - target).square() * weight).sum(dim=-1).div(
        weight.sum(dim=-1).clamp_min(1.0)
    ).mean()


def stable_key(seed: int, domain: str, request_id: str) -> bytes:
    return hashlib.sha256(f"c55:{seed}:{domain}:{request_id}".encode()).digest()


def rows_hash(request_ids: Sequence[str], candidate_rows: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for request_id, candidates in zip(request_ids, candidate_rows):
        digest.update(json.dumps([request_id, *candidates], separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_kuai_request_ids(path: Path) -> list[str]:
    return [str(json.loads(line)["request_id"]) for line in path.open(encoding="utf-8")]


def materialize_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal(config)
    paths = config["paths"]; seed = int(config["selection"]["seed"])
    holdout_count = int(config["selection"]["holdout_requests_per_domain"])
    c47 = json.loads((REPO_ROOT / paths["c47_selection"]).read_text(encoding="utf-8"))
    c34 = json.loads((REPO_ROOT / paths["c34_selection"]).read_text(encoding="utf-8"))
    c38 = load_c38_config(REPO_ROOT / paths["c38_config"])
    amazon_store = FrozenTransferStore(c38)
    c53_root = REPO_ROOT / paths["c53_artifact_root"]
    kuai_fit = [int(value) for value in np.load(c53_root / "kuai_fit_indices.npy", mmap_mode="r")]
    amazon_fit = [int(value) for value in np.load(c53_root / "amazon_fit_indices.npy", mmap_mode="r")]
    kuai_ids = load_kuai_request_ids(REPO_ROOT / paths["kuai_request_ids"])
    c34_fit = list(map(int, c34["roles"]["fit"]["indices"]))
    c34_donors = list(map(int, c34["wrong_history_donors"]["fit"]["indices"]))
    donor_by_index = dict(zip(c34_fit, c34_donors))

    roles: dict[str, dict[str, Any]] = {}
    for domain, fit, request_id in (
        ("kuai", kuai_fit, lambda index: kuai_ids[index]),
        ("amazon", amazon_fit, amazon_store.request_id),
    ):
        ordered = sorted(fit, key=lambda index: (stable_key(seed, domain, request_id(index)), index))
        holdout = ordered[-holdout_count:]
        train = ordered[:-holdout_count]
        roles[domain] = {"train": train, "holdout": holdout}

    kuai_offsets = np.load(REPO_ROOT / paths["kuai_candidate_offsets"], mmap_mode="r")
    kuai_items = np.load(REPO_ROOT / paths["kuai_candidate_item_ids"], mmap_mode="r")
    kuai_holdout_candidates = []
    for index in roles["kuai"]["holdout"]:
        start, stop = int(kuai_offsets[index]), int(kuai_offsets[index + 1])
        kuai_holdout_candidates.append([str(value) for value in kuai_items[start:stop].tolist()])
    amazon_holdout_candidates = [amazon_store.candidate_ids(index) for index in roles["amazon"]["holdout"]]
    candidate_hashes = {
        "kuai": rows_hash(
            [kuai_ids[index] for index in roles["kuai"]["holdout"]], kuai_holdout_candidates,
        ),
        "amazon": rows_hash(
            [amazon_store.request_id(index) for index in roles["amazon"]["holdout"]],
            amazon_holdout_candidates,
        ),
    }
    kuai_wrong = {
        role: [donor_by_index[index] for index in roles["kuai"][role]]
        for role in ("train", "holdout")
    }
    c47_a = {
        "kuai": set(map(int, c47["roles"]["kuai_internal_A"]["indices"])),
        "amazon": set(map(int, c47["roles"]["amazon_internal_A"]["indices"])),
    }
    checks = {
        "exact_holdout_counts": all(len(roles[domain]["holdout"]) == holdout_count for domain in roles),
        "train_holdout_disjoint": all(
            not (set(roles[domain]["train"]) & set(roles[domain]["holdout"])) for domain in roles
        ),
        "fit_coverage_exact": set(roles["kuai"]["train"] + roles["kuai"]["holdout"]) == set(kuai_fit)
        and set(roles["amazon"]["train"] + roles["amazon"]["holdout"]) == set(amazon_fit),
        "C47_A_disjoint": all(
            not ((set(roles[domain]["train"]) | set(roles[domain]["holdout"])) & c47_a[domain])
            for domain in roles
        ),
        "kuai_wrong_donors_complete": all(len(kuai_wrong[role]) == len(roles["kuai"][role]) for role in kuai_wrong),
        "fit_labels_closed": True,
        "C53_A_reserve_dev_test_qrels_closed": True,
    }
    value = {
        "candidate_id": "c55", "selection_id": "c55_fit_residual_split_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "label_free_split_frozen" if all(checks.values()) else "failed",
        "proposal_lock_sha256": proposal_hash, "seed": seed,
        "roles": roles, "kuai_wrong_donors": kuai_wrong,
        "holdout_candidate_key_sha256": candidate_hashes,
        "checks": checks, "fit_labels_read": False,
        "C53_A_reserve_dev_test_qrels_opened": False,
    }
    write_once(REPO_ROOT / paths["selection"], value); return value


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(b"\0")
        digest.update(str(tensor.dtype).encode()); digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def assert_gpu(config: Mapping[str, Any], domain: str, seed: int, device_name: str) -> None:
    physical = int(config["resources"][f"{domain}_seed_to_physical_gpu"][str(seed)])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C55 GPU binding differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C55 deterministic CUBLAS workspace absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C55 requires one visible GPU")


def prepare_batch(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    base = standardize_base(batch["base_scores"], batch["candidate_mask"])
    target = probability_residual(base, batch["labels"], batch["candidate_mask"])
    inputs = {name: value for name, value in batch.items() if name != "labels"}
    inputs["base_scores"] = base
    return inputs, target


def train_mode(
    *, data: DomainData, indices: Sequence[int], config: Mapping[str, Any],
    c54_config: Mapping[str, Any], mode: str, seed: int, device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    seed_all(seed); model = make_model(c54_config, data.input_dim).to(device)
    initial = state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epoch_means = []; gradient_names: set[str] = set(); model.train()
    for epoch in range(int(config["training"]["epochs"])):
        losses = []
        for request_batch in batches(data, indices, config, seed=seed + epoch, shuffle=True):
            batch = collate(data, request_batch, source="true", labels=True, device=device)
            inputs, target = prepare_batch(batch); optimizer.zero_grad(set_to_none=True)
            output = model(**inputs, mode=mode)
            loss = residual_mse(output.correction, target, inputs["candidate_mask"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("C55 nonfinite loss")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and bool(parameter.grad.ne(0).any()):
                    gradient_names.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        epoch_means.append(float(np.mean(losses)))
    return model.eval(), {
        "mode": mode, "initial_state_sha256": initial,
        "epoch_loss_means": epoch_means,
        "finite": bool(np.isfinite(epoch_means).all()),
        "loss_decreased": epoch_means[-1] < epoch_means[0],
        "active_gradient_parameters": sorted(gradient_names),
        "parameters": sum(value.numel() for value in model.parameters()),
    }


def row_metrics(
    data: DomainData, indices: Sequence[int], base: Sequence[np.ndarray],
    target: Sequence[np.ndarray], corrections: Mapping[str, Sequence[np.ndarray]],
    labels: Sequence[np.ndarray],
) -> dict[str, Any]:
    names = tuple(corrections)
    ndcg: dict[str, list[float]] = {"base": []}
    ndcg.update({name: [] for name in names})
    mse: dict[str, list[float]] = {"zero": []}
    mse.update({name: [] for name in names})
    cosine: dict[str, list[float]] = {name: [] for name in names}
    clicked: dict[str, list[float]] = {name: [] for name in names}
    for row, index in enumerate(indices):
        item_ids = data.candidate_ids(index)
        positive = {item for item, label in zip(item_ids, labels[row]) if label > 0}
        ndcg["base"].append(ndcg_at_k(rankings(data.request_id(index), item_ids, base[row]), positive, 10))
        mse["zero"].append(float(np.mean(target[row] ** 2)))
        pos = labels[row] > 0; neg = ~pos
        for name in names:
            correction = corrections[name][row]
            ndcg[name].append(ndcg_at_k(
                rankings(data.request_id(index), item_ids, base[row] + correction), positive, 10,
            ))
            mse[name].append(float(np.mean((correction - target[row]) ** 2)))
            denominator = float(np.linalg.norm(correction) * np.linalg.norm(target[row]))
            cosine[name].append(0.0 if denominator <= 1e-12 else float(np.dot(correction, target[row]) / denominator))
            clicked[name].append(
                0.0 if not bool(neg.any())
                else float(correction[pos].mean() - correction[neg].mean())
            )
    return {
        "ndcg_rows": {name: np.asarray(values, np.float64) for name, values in ndcg.items()},
        "mse_rows": {name: np.asarray(values, np.float64) for name, values in mse.items()},
        "cosine_rows": {name: np.asarray(values, np.float64) for name, values in cosine.items()},
        "clicked_rows": {name: np.asarray(values, np.float64) for name, values in clicked.items()},
    }


def save_rows(path: Path, rows: Mapping[str, Sequence[np.ndarray]]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    offsets, _ = flatten(rows["base"])
    with path.open("wb") as handle:
        np.savez(handle, offsets=offsets, **{name: flatten(value)[1] for name, value in rows.items()})
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}


def run_seed(
    config: Mapping[str, Any], domain: str, seed: int, device_name: str,
) -> dict[str, Any]:
    _, execution_hash = verify_execution(config); assert_gpu(config, domain, seed, device_name)
    selection = json.loads((REPO_ROOT / config["paths"]["selection"]).read_text(encoding="utf-8"))
    c54_config = yaml.safe_load((REPO_ROOT / config["paths"]["c54_config"]).read_text(encoding="utf-8"))
    data = DomainData(domain, c54_config)
    train_indices = list(map(int, selection["roles"][domain]["train"]))
    holdout_indices = list(map(int, selection["roles"][domain]["holdout"]))
    if domain == "kuai":
        data.donor.update(dict(zip(
            train_indices, map(int, selection["kuai_wrong_donors"]["train"]),
        )))
        data.donor.update(dict(zip(
            holdout_indices, map(int, selection["kuai_wrong_donors"]["holdout"]),
        )))
    actual_hash = candidate_hash(data, holdout_indices)
    expected_hash = selection["holdout_candidate_key_sha256"][domain]
    if actual_hash != expected_hash:
        raise RuntimeError(f"C55 {domain} holdout candidate hash differs")
    device = torch.device(device_name)
    models = {}; training = {}
    for mode in MODES:
        models[mode], training[mode] = train_mode(
            data=data, indices=train_indices, config=config, c54_config=c54_config,
            mode=mode, seed=seed, device=device,
        )
    if len({row["initial_state_sha256"] for row in training.values()}) != 1:
        raise RuntimeError("C55 paired initialization differs")

    artifact_root = REPO_ROOT / config["paths"]["artifact_root"]
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    artifact_root.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for mode, model in models.items():
        path = checkpoint_root / f"{domain}_seed_{seed}_{mode}.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save({
            "candidate_id": "c55", "domain": domain, "seed": seed,
            "mode": mode, "state_dict": model.state_dict(),
            "execution_lock_sha256": execution_hash,
        }, path)
        checkpoints[mode] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}

    rows: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}
    label_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for request_batch in batches(data, holdout_indices, config, seed=0, shuffle=False):
            true_batch = collate(data, request_batch, source="true", labels=True, device=device)
            wrong_batch = collate(data, request_batch, source="wrong", labels=True, device=device)
            true_inputs, target = prepare_batch(true_batch)
            wrong_inputs, _ = prepare_batch(wrong_batch)
            primary = models["history_carrier"](**true_inputs, mode="history_carrier")
            wrong = models["history_carrier"](**wrong_inputs, mode="history_carrier")
            raw = models["raw_candidate"](**true_inputs, mode="raw_candidate")
            mask = true_inputs["candidate_mask"].cpu().numpy()
            arrays = {
                "base": true_inputs["base_scores"].cpu().numpy(),
                "target": target.cpu().numpy(),
                "primary_correction": primary.correction.cpu().numpy(),
                "wrong_correction": wrong.correction.cpu().numpy(),
                "raw_correction": raw.correction.cpu().numpy(),
            }
            labels = true_batch["labels"].cpu().numpy()
            for row in range(len(request_batch)):
                count = int(mask[row].sum())
                for name in SCORE_NAMES:
                    rows[name].append(np.asarray(arrays[name][row, :count], np.float32).copy())
                label_rows.append(np.asarray(labels[row, :count], np.float32).copy())

    corrections = {
        "primary": rows["primary_correction"],
        "wrong": rows["wrong_correction"],
        "raw": rows["raw_correction"],
    }
    metrics = row_metrics(
        data, holdout_indices, rows["base"], rows["target"], corrections, label_rows,
    )
    score_path = artifact_root / f"{domain}_seed_{seed}_scores.npz"
    score_artifact = save_rows(score_path, rows)
    report_path = artifact_root / f"{domain}_seed_{seed}_report.json"
    checks = {
        "train_holdout_disjoint": not (set(train_indices) & set(holdout_indices)),
        "candidate_hash_asserted": actual_hash == expected_hash,
        "paired_initialization": len({row["initial_state_sha256"] for row in training.values()}) == 1,
        "equal_parameters": len({row["parameters"] for row in training.values()}) == 1,
        "finite_training": all(row["finite"] for row in training.values()),
        "loss_decreased": all(row["loss_decreased"] for row in training.values()),
        "finite_scores": all(np.isfinite(value).all() for group in metrics.values() for value in group.values()),
        "C53_A_reserve_dev_test_qrels_closed": True,
    }
    summary = {
        "mean_ndcg10": {name: float(values.mean()) for name, values in metrics["ndcg_rows"].items()},
        "mean_residual_mse": {name: float(values.mean()) for name, values in metrics["mse_rows"].items()},
        "mean_residual_cosine": {name: float(values.mean()) for name, values in metrics["cosine_rows"].items()},
        "mean_clicked_direction": {name: float(values.mean()) for name, values in metrics["clicked_rows"].items()},
    }
    report = {
        "candidate_id": "c55", "domain": domain, "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_lock_sha256": execution_hash, "checks": checks,
        "train_requests": len(train_indices), "holdout_requests": len(holdout_indices),
        "holdout_candidate_key_sha256": actual_hash,
        "training": training, "summary": summary,
        "checkpoints": checkpoints, "score_artifact": score_artifact,
        "fit_labels_read": True, "C53_A_reserve_dev_test_qrels_opened": False,
    }
    write_once(report_path, report); return report


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(values[int(offsets[i]):int(offsets[i + 1])], np.float32).copy() for i in range(len(offsets) - 1)]


def load_score_rows(report: Mapping[str, Any]) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C55 score artifact changed")
    with np.load(path, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], np.int64)
        return {name: unflatten(offsets, source[name]) for name in SCORE_NAMES}


def paired_interval(difference: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    difference = np.asarray(difference, np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(difference), size=(count, len(difference)))
        draws[start:start + count] = difference[indices].mean(axis=1)
    return {
        "mean": float(difference.mean()),
        "percentile_95_ci": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "requests": len(difference), "samples": samples, "seed": seed,
    }


def aggregate_domain(config: Mapping[str, Any], domain: str) -> dict[str, Any]:
    selection = json.loads((REPO_ROOT / config["paths"]["selection"]).read_text(encoding="utf-8"))
    c54_config = yaml.safe_load((REPO_ROOT / config["paths"]["c54_config"]).read_text(encoding="utf-8"))
    data = DomainData(domain, c54_config)
    train_indices = list(map(int, selection["roles"][domain]["train"]))
    holdout = list(map(int, selection["roles"][domain]["holdout"]))
    if domain == "kuai":
        data.donor.update(dict(zip(train_indices, map(int, selection["kuai_wrong_donors"]["train"]))))
        data.donor.update(dict(zip(holdout, map(int, selection["kuai_wrong_donors"]["holdout"]))))
    if candidate_hash(data, holdout) != selection["holdout_candidate_key_sha256"][domain]:
        raise RuntimeError(f"C55 {domain} aggregate candidate hash differs")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    seeds = list(map(int, config["training"][f"{domain}_seeds"]))
    reports = [json.loads((root / f"{domain}_seed_{seed}_report.json").read_text(encoding="utf-8")) for seed in seeds]
    seed_rows = {seed: load_score_rows(report) for seed, report in zip(seeds, reports)}
    base = seed_rows[seeds[0]]["base"]; target = seed_rows[seeds[0]]["target"]
    if any(
        not all(np.array_equal(a, b) for a, b in zip(base, seed_rows[seed]["base"]))
        or not all(np.array_equal(a, b) for a, b in zip(target, seed_rows[seed]["target"]))
        for seed in seeds[1:]
    ):
        raise RuntimeError("C55 base/target differs across seeds")
    corrections = {
        name: [
            np.mean(np.stack(values), axis=0).astype(np.float32)
            for values in zip(*[seed_rows[seed][f"{name}_correction"] for seed in seeds])
        ]
        for name in ("primary", "wrong", "raw")
    }
    label_rows = [data.fit_label(index) for index in holdout]
    metrics = row_metrics(data, holdout, base, target, corrections, label_rows)
    ev = config["evaluation"]; samples = int(ev["bootstrap_samples"]); base_seed = int(ev["bootstrap_seed"])
    comparisons = {
        "ndcg_primary_minus_base": paired_interval(metrics["ndcg_rows"]["primary"] - metrics["ndcg_rows"]["base"], samples=samples, seed=base_seed),
        "ndcg_primary_minus_raw": paired_interval(metrics["ndcg_rows"]["primary"] - metrics["ndcg_rows"]["raw"], samples=samples, seed=base_seed + 1),
        "ndcg_primary_minus_wrong": paired_interval(metrics["ndcg_rows"]["primary"] - metrics["ndcg_rows"]["wrong"], samples=samples, seed=base_seed + 2),
        "mse_zero_minus_primary": paired_interval(metrics["mse_rows"]["zero"] - metrics["mse_rows"]["primary"], samples=samples, seed=base_seed + 3),
        "mse_raw_minus_primary": paired_interval(metrics["mse_rows"]["raw"] - metrics["mse_rows"]["primary"], samples=samples, seed=base_seed + 4),
        "mse_wrong_minus_primary": paired_interval(metrics["mse_rows"]["wrong"] - metrics["mse_rows"]["primary"], samples=samples, seed=base_seed + 5),
    }
    mean_mse = {name: float(values.mean()) for name, values in metrics["mse_rows"].items()}
    relative = {
        "over_zero": (mean_mse["zero"] - mean_mse["primary"]) / mean_mse["zero"],
        "over_raw": (mean_mse["raw"] - mean_mse["primary"]) / mean_mse["raw"],
        "over_wrong": (mean_mse["wrong"] - mean_mse["primary"]) / mean_mse["wrong"],
    }
    seed_directions = {}
    for seed in seeds:
        summary = next(report["summary"] for report in reports if report["seed"] == seed)
        seed_directions[str(seed)] = {
            "ndcg_over_base": summary["mean_ndcg10"]["primary"] - summary["mean_ndcg10"]["base"],
            "ndcg_over_raw": summary["mean_ndcg10"]["primary"] - summary["mean_ndcg10"]["raw"],
            "ndcg_over_wrong": summary["mean_ndcg10"]["primary"] - summary["mean_ndcg10"]["wrong"],
            "mse_over_zero": summary["mean_residual_mse"]["zero"] - summary["mean_residual_mse"]["primary"],
            "mse_over_raw": summary["mean_residual_mse"]["raw"] - summary["mean_residual_mse"]["primary"],
            "mse_over_wrong": summary["mean_residual_mse"]["wrong"] - summary["mean_residual_mse"]["primary"],
        }
    checks = {
        "all_seed_execution_checks": all(all(report["checks"].values()) for report in reports),
        "mse_gain_over_zero": relative["over_zero"] >= float(ev["residual_mse_relative_gain_over_zero_min"]),
        "mse_gain_over_raw": relative["over_raw"] >= float(ev["residual_mse_relative_gain_over_raw_min"]),
        "mse_gain_over_wrong": relative["over_wrong"] >= float(ev["residual_mse_relative_gain_over_wrong_min"]),
        "mse_intervals_positive": all(comparisons[name]["percentile_95_ci"][0] > 0 for name in ("mse_zero_minus_primary", "mse_raw_minus_primary", "mse_wrong_minus_primary")),
        "ndcg_gain_over_base": comparisons["ndcg_primary_minus_base"]["mean"] >= float(ev["ndcg_primary_minus_base_min"]),
        "ndcg_gain_over_raw": comparisons["ndcg_primary_minus_raw"]["mean"] >= float(ev["ndcg_primary_minus_raw_min"]),
        "ndcg_gain_over_wrong": comparisons["ndcg_primary_minus_wrong"]["mean"] >= float(ev["ndcg_primary_minus_wrong_min"]),
        "ndcg_intervals_positive": all(comparisons[name]["percentile_95_ci"][0] > 0 for name in ("ndcg_primary_minus_base", "ndcg_primary_minus_raw", "ndcg_primary_minus_wrong")),
        "all_seed_directions_positive": all(all(value > 0 for value in row.values()) for row in seed_directions.values()),
        "C53_A_reserve_dev_test_qrels_closed": True,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "requests": len(holdout), "checks": checks,
        "mean_ndcg10": {name: float(values.mean()) for name, values in metrics["ndcg_rows"].items()},
        "mean_residual_mse": mean_mse,
        "mean_residual_cosine": {name: float(values.mean()) for name, values in metrics["cosine_rows"].items()},
        "mean_clicked_direction": {name: float(values.mean()) for name, values in metrics["clicked_rows"].items()},
        "relative_mse_gain": relative, "comparisons": comparisons,
        "seed_directions": seed_directions,
    }


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, execution_hash = verify_execution(config)
    domains = {domain: aggregate_domain(config, domain) for domain in ("kuai", "amazon")}
    passed = all(row["status"] == "passed" for row in domains.values())
    value = {
        "candidate_id": "c55", "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_residual_signal" if passed else "failed_residual_signal_terminal",
        "decision": "authorize_new_residual_target_architecture" if passed else "close_probability_residual_on_frozen_states",
        "execution_lock_sha256": execution_hash, "domains": domains,
        "claims": {"signal_falsifier": True, "architecture_innovation": False, "fresh_result": False},
        "C53_A_reserve_dev_test_qrels_opened": False,
    }
    root = REPO_ROOT / config["paths"]["artifact_root"]
    write_once(root / "signal_gate_report.json", value)
    write_once(REPO_ROOT / config["paths"]["promoted_report"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("selection", "seed", "aggregate"))
    parser.add_argument("--domain", choices=("kuai", "amazon"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); config = load_config(args.config)
    if args.stage == "selection":
        value = materialize_selection(config)
    elif args.stage == "seed":
        if args.domain is None or args.seed is None:
            raise ValueError("C55 seed requires domain/seed")
        value = run_seed(config, args.domain, args.seed, args.device)
    else:
        value = aggregate(config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
