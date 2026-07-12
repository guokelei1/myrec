"""Train and evaluate C61 under its staged fit/fresh-A barrier."""

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
from torch.nn import functional as F


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for value in (str(SYSTEM_ROOT), str(REPO_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from execution.locking import (  # noqa: E402
    load_config,
    read_json,
    sha256_file,
    verify_execution,
    write_once,
)
from model.counterfactual_edge import (  # noqa: E402
    MODES,
    CounterfactualEdgeLikelihoodTransformer,
    adjacent_pair_targets,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from train.data import C61Store, iter_batches, to_device  # noqa: E402


PRIMARY = "counterfactual_edge"
SCORE_NAMES = (
    "base",
    "item_only",
    "primary",
    "wrong",
    "null_ablation",
    "factual_edge",
    "ordinary_candidate_attention",
    "candidate_only_edge",
    "fixed_c60",
)
CONTROL_MODES = (
    "factual_edge",
    "ordinary_candidate_attention",
    "candidate_only_edge",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def assert_sources(config: Mapping[str, Any]) -> None:
    for path_name, hash_name in (
        ("c26_config", "c26_config_sha256"),
        ("c26_selection", "c26_selection_sha256"),
        ("c26_g0_report", "c26_g0_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("c60_report", "c60_report_sha256"),
        ("c59_report", "c59_report_sha256"),
    ):
        if sha256_file(REPO_ROOT / config["paths"][path_name]) != config["integrity"][hash_name]:
            raise RuntimeError(f"C61 registered source changed: {path_name}")
    manifest = read_json(REPO_ROOT / config["paths"]["contextual_manifest"])
    if manifest.get("status") != "passed":
        raise RuntimeError("C61 contextual manifest differs")
    g0 = read_json(REPO_ROOT / config["paths"]["artifact_root"] / "g0_report.json")
    if g0.get("status") != "passed" or g0.get("fit_labels_read") is not False:
        raise RuntimeError("C61 G0 boundary differs")
    if g0.get("internal_A_delayed_B_escrow_dev_test_qrels_opened") is not False:
        raise PermissionError("C61 G0 fresh roles are not closed")


def assert_cuda(config: Mapping[str, Any], seed: int, device_name: str) -> None:
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C61 GPU binding differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C61 deterministic CUBLAS workspace absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C61 requires exactly one visible GPU")


def make_model(config: Mapping[str, Any]) -> CounterfactualEdgeLikelihoodTransformer:
    value = config["encoding"]
    return CounterfactualEdgeLikelihoodTransformer(
        input_dim=int(value["input_dim"]),
        hidden_dim=int(value["hidden_dim"]),
        heads=int(value["heads"]),
        ffn_dim=int(value["ffn_dim"]),
        token_layers=int(value["token_layers"]),
        edge_layers=int(value["edge_layers"]),
        dropout=float(value["dropout"]),
        max_query_tokens=int(value["max_query_tokens"]),
        max_item_tokens=int(value["max_item_tokens"]),
        max_history=int(value["max_history"]),
        zero_initial_output=True,
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


def batches(
    store: C61Store,
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
            max_edge_cells=int(value["max_edge_cells"]),
        )
    )


def train_mode(
    *,
    store: C61Store,
    schedules: Sequence[Sequence[np.ndarray]],
    config: Mapping[str, Any],
    mode: str,
    seed: int,
    device: torch.device,
) -> tuple[CounterfactualEdgeLikelihoodTransformer, dict[str, Any]]:
    seed_all(seed)
    model = make_model(config).to(device)
    initial = state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epoch_means: list[float] = []
    epoch_edges: list[int] = []
    gradient_names: set[str] = set()
    started = time.time()
    model.train()
    for schedule in schedules:
        losses: list[float] = []
        edges = 0
        for request_batch in schedule:
            batch = store.collate(request_batch, label_role="fit")
            tensors = to_device(batch, device)
            labels = tensors.pop("labels")
            target, eligible = adjacent_pair_targets(
                labels, tensors["canonical_order"], tensors["candidate_mask"]
            )
            if not bool(eligible.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            output = model(**tensors, mode=mode)
            loss = F.binary_cross_entropy_with_logits(
                output.pair_logits[eligible], target[eligible]
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite C61 loss: {mode}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C61 gradient: {mode}/{name}")
                    if bool(parameter.grad.ne(0).any()):
                        gradient_names.add(name)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            edges += int(eligible.sum())
        if not losses:
            raise RuntimeError(f"C61 epoch has no eligible edges: {mode}")
        epoch_means.append(float(np.mean(losses)))
        epoch_edges.append(edges)
    return model.eval(), {
        "mode": mode,
        "initial_state_sha256": initial,
        "final_state_sha256": state_hash(model),
        "parameters": sum(value.numel() for value in model.parameters()),
        "epoch_loss_means": epoch_means,
        "epoch_eligible_edges": epoch_edges,
        "finite": bool(np.isfinite(epoch_means).all()),
        "loss_decreased": epoch_means[-1] < epoch_means[0],
        "active_gradient_parameters": sorted(gradient_names),
        "elapsed_seconds": time.time() - started,
    }


def _stable_zscore(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(values, dtype=torch.float32)
    for row in range(len(values)):
        positions = torch.where(mask[row])[0]
        if not len(positions):
            continue
        selected = values[row, positions].double()
        ordered = torch.sort(selected).values
        mean = ordered.sum(dtype=torch.float64) / len(ordered)
        scale = torch.sqrt(((ordered - mean) ** 2).sum(dtype=torch.float64) / len(ordered))
        if float(scale) > 1e-6:
            output[row, positions] = ((selected - mean) / scale).float()
    return output


def fixed_c60_scores(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Exact C59 semantic evidence passed through C60's fixed safe interface."""
    query = F.normalize(tensors["query_tokens"].float(), dim=-1)
    candidate = F.normalize(tensors["candidate_tokens"].float(), dim=-1)
    history = F.normalize(tensors["history_tokens"].float(), dim=-1)
    qmask = tensors["query_token_mask"].bool()
    cmask = tensors["candidate_mask"].bool()
    ctmask = tensors["candidate_token_mask"].bool()
    htmask = tensors["history_token_mask"].bool()
    event_mask = htmask.any(dim=-1)
    batch, candidates = cmask.shape
    feature = query.new_zeros((batch, candidates))
    for row in range(batch):
        cp = torch.where(cmask[row])[0]
        qp = torch.where(qmask[row])[0]
        hp = torch.where(event_mask[row])[0]
        if not len(cp) or not len(qp) or not len(hp):
            continue
        q = query[row, qp]
        c = candidate[row, cp]
        ctm = ctmask[row, cp]
        weights = tensors["event_weights"][row, hp].float().clamp_min(0)
        weights = weights / weights.sum().clamp_min(1e-12)
        logits = query.new_zeros((len(cp), len(hp)))
        for local, event_position in enumerate(hp):
            ep = torch.where(htmask[row, event_position])[0]
            h = history[row, event_position, ep]
            a = (q @ h.T).max(dim=-1).values.mean()
            similarity = torch.einsum("ctd,md->ctm", c, h).max(dim=-1).values
            b = (similarity * ctm).sum(dim=-1) / ctm.sum(dim=-1).clamp_min(1)
            logits[:, local] = a * b
        event_logits = logits.T.double()
        joined = torch.cat((event_logits, event_logits.new_zeros((len(hp), 1))), dim=-1)
        ordered = torch.sort(joined, dim=-1).values
        maximum = ordered[..., -1:]
        denominator = torch.exp(ordered - maximum).sum(dim=-1, keepdim=True)
        probability = torch.exp(joined - maximum) / denominator
        feature[row, cp] = (weights.double()[:, None] * probability[:, :-1]).sum(dim=0).float()
    evidence = _stable_zscore(feature, cmask)
    base = tensors["base_scores"].float()
    correction = torch.zeros_like(base)
    active = qmask.any(dim=-1) & event_mask.any(dim=-1) & ~tensors["repeat_request"].bool()
    for row in torch.where(active)[0]:
        count = int(cmask[row].sum())
        order = tensors["canonical_order"][row, :count].long()
        s = base[row, order].double()
        e = evidence[row, order].double()
        gap = (s[:-1] - s[1:]).clamp_min(0)
        p0 = torch.sigmoid(-gap)
        p1 = torch.sigmoid(-gap + e[1:] - e[:-1])
        rate = ((p1 - p0) / (1 - p0).clamp_min(1e-12)).clamp(0, 1)
        flow = rate * gap
        value = torch.zeros_like(s)
        value[:-1] -= flow
        value[1:] += flow
        correction[row, order] = value.float()
    score = base + correction
    score = torch.where(
        tensors["repeat_request"][:, None].bool(), tensors["item_only_scores"].float(), score
    )
    return score.masked_fill(~cmask, 0.0)


def score_model(
    *,
    model: CounterfactualEdgeLikelihoodTransformer,
    store: C61Store,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    mode: str,
    history_source: str = "true",
    include_fixed: bool = False,
) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "request_ids": [],
        "item_ids": [],
        "base": [],
        "item_only": [],
        "scores": [],
        "correction": [],
        "likelihood": [],
        "transport": [],
        "gap": [],
        "fixed_c60": [],
    }
    model.eval()
    with torch.inference_mode():
        for request_batch in batches(store, indices, config, seed=0, shuffle=False):
            batch = store.collate(request_batch, history_source=history_source)
            tensors = to_device(batch, device)
            output = model(**tensors, mode=mode)
            fixed = fixed_c60_scores(tensors) if include_fixed else None
            for row in range(len(request_batch)):
                count = int(batch["candidate_mask"][row].sum())
                edges = max(0, count - 1)
                rows["request_ids"].append(batch["request_ids"][row])
                rows["item_ids"].append(
                    [str(value) for value in batch["candidate_item_ids"][row, :count]]
                )
                for name, value in (
                    ("base", tensors["base_scores"]),
                    ("item_only", tensors["item_only_scores"]),
                    ("scores", output.scores),
                    ("correction", output.correction),
                ):
                    rows[name].append(value[row, :count].cpu().numpy().astype(np.float32, copy=True))
                rows["likelihood"].append(
                    output.likelihood_ratio[row, :edges].cpu().numpy().astype(np.float32, copy=True)
                )
                rows["transport"].append(
                    output.transport[row, :edges].cpu().numpy().astype(np.float32, copy=True)
                )
                rows["gap"].append(
                    output.base_gap[row, :edges].cpu().numpy().astype(np.float32, copy=True)
                )
                if fixed is not None:
                    rows["fixed_c60"].append(
                        fixed[row, :count].cpu().numpy().astype(np.float32, copy=True)
                    )
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


def fallback_audit(
    model: CounterfactualEdgeLikelihoodTransformer,
    store: C61Store,
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


def permutation_audit(
    model: CounterfactualEdgeLikelihoodTransformer,
    store: C61Store,
    index: int,
    device: torch.device,
) -> float:
    tensors = to_device(store.collate([index]), device)
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
    moved["canonical_order"] = inverse[tensors["canonical_order"]]
    with torch.inference_mode():
        first = model(**tensors).scores
        second = model(**moved).scores[:, inverse]
    return float((first - second).abs().max().cpu())


def run_seed(config_path: str | Path, seed: int, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_sources(config)
    assert_cuda(config, seed, device_name)
    store = C61Store(config, REPO_ROOT)
    fit, fresh = store.role("fit"), store.role("internal_A")
    expected_hash = store.selection["roles"]["internal_A"]["candidate_key_sha256"]
    actual_hash = store.candidate_hash(fresh)
    if actual_hash != expected_hash:
        raise RuntimeError("C61 fresh-A candidate hash differs")
    schedules = [
        batches(store, fit, config, seed=seed + epoch, shuffle=True)
        for epoch in range(int(config["training"]["epochs"]))
    ]
    device = torch.device(device_name)
    models: dict[str, CounterfactualEdgeLikelihoodTransformer] = {}
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
        raise RuntimeError("C61 paired initialization/capacity differs")
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
                "candidate_id": "c61",
                "seed": seed,
                "mode": mode,
                "execution_lock_sha256": execution_hash,
                "state_dict": model.state_dict(),
            },
            path,
        )
        checkpoints[mode] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
    primary = score_model(
        model=models[PRIMARY], store=store, indices=fresh, config=config, device=device,
        mode=PRIMARY, include_fixed=True,
    )
    deterministic = score_model(
        model=models[PRIMARY], store=store, indices=fresh, config=config, device=device, mode=PRIMARY
    )
    wrong = score_model(
        model=models[PRIMARY], store=store, indices=fresh, config=config, device=device,
        mode=PRIMARY, history_source="wrong",
    )
    null_ablation = score_model(
        model=models[PRIMARY], store=store, indices=fresh, config=config, device=device,
        mode="factual_edge",
    )
    controls = {
        mode: score_model(
            model=models[mode], store=store, indices=fresh, config=config, device=device, mode=mode
        )
        for mode in CONTROL_MODES
    }
    base = dict(primary)
    base["scores"] = primary["base"]
    score_rows = {
        "base": primary["base"],
        "item_only": primary["item_only"],
        "primary": primary["scores"],
        "wrong": wrong["scores"],
        "null_ablation": null_ablation["scores"],
        "fixed_c60": primary["fixed_c60"],
        **{name: value["scores"] for name, value in controls.items()},
    }
    artifact = save_rows(root / f"seed_{seed}_scores.npz", score_rows)
    fallback = fallback_audit(models[PRIMARY], store, config, device)
    deterministic_error = max_difference(primary["scores"], deterministic["scores"])
    permutation_error = permutation_audit(models[PRIMARY], store, fresh[0], device)
    conservation_error = max(float(abs(value.sum())) for value in primary["correction"])
    capacity_error = max(
        float(np.maximum(np.abs(value) - gap, 0).max(initial=0.0))
        for value, gap in zip(primary["transport"], primary["gap"])
    )
    order_changes = {
        "primary_vs_base": changes(base, primary),
        "primary_vs_wrong": changes(primary, wrong),
        "primary_vs_null_ablation": changes(primary, null_ablation),
    }
    tolerance = float(config["evaluation"]["exact_fallback_tolerance"])
    checks = {
        "fit_fresh_disjoint": not (set(fit) & set(fresh)),
        "fresh_candidate_hash_asserted": actual_hash == expected_hash,
        "paired_initialization": len(initials) == 1,
        "equal_parameters": len(parameters) == 1,
        "finite_training": all(row["finite"] for row in training.values()),
        "all_mode_loss_decreased": all(row["loss_decreased"] for row in training.values()),
        "deterministic": deterministic_error <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": permutation_error <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "score_conservation": conservation_error <= float(config["evaluation"]["conservation_tolerance"]),
        "edge_capacity": capacity_error <= float(config["evaluation"]["capacity_tolerance"]),
        "exact_nohistory_base": fallback["nohistory_max_abs_vs_base"] <= tolerance,
        "exact_repeat_item_only": fallback["repeat_max_abs_vs_item_only"] <= tolerance,
        "fresh_A_labels_closed": True,
        "delayed_B_escrow_dev_test_qrels_closed": True,
    }
    value = {
        "candidate_id": "c61",
        "stage": "seed",
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_lock_sha256": execution_hash,
        "training": training,
        "changes": order_changes,
        "fallback": fallback,
        "deterministic_max_abs_difference": deterministic_error,
        "candidate_permutation_max_abs_difference": permutation_error,
        "conservation_max_abs_error": conservation_error,
        "capacity_max_excess": capacity_error,
        "checks": checks,
        "checkpoints": checkpoints,
        "score_artifact": artifact,
        "internal_A_candidate_key_sha256": actual_hash,
        "fit_labels_read": True,
        "internal_A_labels_read": False,
        "delayed_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / f"seed_{seed}_report.json", value)
    return value


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[row]) : int(offsets[row + 1])], dtype=np.float32).copy()
        for row in range(len(offsets) - 1)
    ]


def load_rows(report: Mapping[str, Any]) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C61 score artifact changed")
    with np.load(path, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        return {name: unflatten(offsets, source[name]) for name in SCORE_NAMES}


def average_rows(collections: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [np.mean(np.stack(rows), axis=0).astype(np.float32) for rows in zip(*collections)]


def reference(store: C61Store, indices: Sequence[int], scores: Sequence[np.ndarray]) -> dict[str, Any]:
    return {
        "request_ids": [store.data.request_ids[index] for index in indices],
        "item_ids": [store.data.candidate_ids(index) for index in indices],
        "scores": scores,
    }


def aggregate_a0(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_sources(config)
    store = C61Store(config, REPO_ROOT)
    fresh = store.role("internal_A")
    actual_hash = store.candidate_hash(fresh)
    expected_hash = store.selection["roles"]["internal_A"]["candidate_key_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError("C61 A0 candidate hash differs")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    seeds = list(map(int, config["training"]["seeds"]))
    reports = [read_json(root / f"seed_{seed}_report.json") for seed in seeds]
    rows = {seed: load_rows(report) for seed, report in zip(seeds, reports)}
    base = rows[seeds[0]]["base"]
    ensemble = {
        name: average_rows([rows[seed][name] for seed in seeds])
        for name in ("primary", "wrong", "null_ablation")
    }
    base_ref = reference(store, fresh, base)
    primary_ref = reference(store, fresh, ensemble["primary"])
    order_changes = {
        "primary_vs_base": changes(base_ref, primary_ref),
        "primary_vs_wrong": changes(primary_ref, reference(store, fresh, ensemble["wrong"])),
        "primary_vs_null_ablation": changes(
            primary_ref, reference(store, fresh, ensemble["null_ablation"])
        ),
    }
    per_seed = {
        str(seed): {
            "primary_vs_base": changes(base_ref, reference(store, fresh, rows[seed]["primary"])),
            "primary_vs_wrong": changes(
                reference(store, fresh, rows[seed]["primary"]),
                reference(store, fresh, rows[seed]["wrong"]),
            ),
            "primary_vs_null_ablation": changes(
                reference(store, fresh, rows[seed]["primary"]),
                reference(store, fresh, rows[seed]["null_ablation"]),
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
        "ensemble_NULL_ablation_load_bearing": above(order_changes["primary_vs_null_ablation"], "null"),
        "all_seed_base_activity": all(above(row["primary_vs_base"], "active") for row in per_seed.values()),
        "all_seed_wrong_history_load_bearing": all(above(row["primary_vs_wrong"], "wrong") for row in per_seed.values()),
        "all_seed_NULL_ablation_load_bearing": all(
            above(row["primary_vs_null_ablation"], "null") for row in per_seed.values()
        ),
        "internal_A_labels_closed": True,
        "delayed_B_escrow_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c61",
        "gate": "A0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_label_free_mechanics" if passed else "failed_label_free_mechanics_terminal",
        "execution_lock_sha256": execution_hash,
        "internal_A_candidate_key_sha256": actual_hash,
        "internal_A_requests": len(fresh),
        "changes": order_changes,
        "per_seed_changes": per_seed,
        "checks": checks,
        "fit_labels_read": True,
        "internal_A_labels_read": False,
        "delayed_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / "a0_report.json", value)
    if not passed:
        terminal = dict(value)
        terminal.update(
            {
                "gate_id": config["gate_id"],
                "decision": "close_counterfactual_edge_likelihood_on_mechanics",
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
    store: C61Store, indices: Sequence[int], scores: Mapping[str, Sequence[np.ndarray]]
) -> dict[str, np.ndarray]:
    output = {name: [] for name in scores}
    for row, index in enumerate(indices):
        request_id = store.data.request_ids[index]
        items = store.data.candidate_ids(index)
        labels = store.labels(index, role="internal_A")
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
        raise PermissionError("C61 A1 requires passed A0")
    store = C61Store(config, REPO_ROOT)
    fresh = store.role("internal_A")
    actual_hash = store.candidate_hash(fresh)
    if actual_hash != a0["internal_A_candidate_key_sha256"]:
        raise RuntimeError("C61 A1 candidate hash differs")
    seeds = list(map(int, config["training"]["seeds"]))
    reports = [read_json(root / f"seed_{seed}_report.json") for seed in seeds]
    rows = {seed: load_rows(report) for seed, report in zip(seeds, reports)}
    names = (
        "primary",
        "wrong",
        "null_ablation",
        "factual_edge",
        "ordinary_candidate_attention",
        "candidate_only_edge",
        "fixed_c60",
    )
    ensemble = {"base": rows[seeds[0]]["base"]}
    ensemble.update({name: average_rows([rows[seed][name] for seed in seeds]) for name in names})
    metric = ndcg_rows(store, fresh, ensemble)
    compare_names = (
        "base",
        "wrong",
        "null_ablation",
        "factual_edge",
        "ordinary_candidate_attention",
        "candidate_only_edge",
        "fixed_c60",
    )
    ev = config["evaluation"]
    samples, base_seed = int(ev["bootstrap_samples"]), int(ev["bootstrap_seed"])
    comparisons = {
        f"primary_minus_{name}": paired_interval(
            metric["primary"] - metric[name], samples=samples, seed=base_seed + offset
        )
        for offset, name in enumerate(compare_names)
    }
    seed_directions = {}
    for seed in seeds:
        seed_scores = {"base": rows[seed]["base"]}
        seed_scores.update({name: rows[seed][name] for name in names})
        seed_metric = ndcg_rows(store, fresh, seed_scores)
        seed_directions[str(seed)] = {
            name: float((seed_metric["primary"] - seed_metric[name]).mean())
            for name in compare_names
        }
    request_ids = [store.data.request_ids[index] for index in fresh]
    folds = {
        name: fold_means(metric["primary"] - metric[name], request_ids, int(ev["hash_folds"]))
        for name in compare_names
    }
    controls = tuple(name for name in compare_names if name not in {"base", "wrong"})
    checks = {
        "A0_passed": True,
        "candidate_hash_asserted": actual_hash == a0["internal_A_candidate_key_sha256"],
        "gain_over_base": comparisons["primary_minus_base"]["mean"]
        >= float(ev["ndcg_primary_minus_base_min"])
        and comparisons["primary_minus_base"]["percentile_95_ci"][0] > 0,
        "gain_over_wrong": comparisons["primary_minus_wrong"]["mean"]
        >= float(ev["ndcg_primary_minus_wrong_min"])
        and comparisons["primary_minus_wrong"]["percentile_95_ci"][0] > 0,
        "gain_over_controls": all(
            comparisons[f"primary_minus_{name}"]["mean"]
            >= float(ev["ndcg_primary_minus_each_control_min"])
            and comparisons[f"primary_minus_{name}"]["percentile_95_ci"][0] > 0
            for name in controls
        ),
        "all_seed_directions_positive": all(
            all(value > 0 for value in row.values()) for row in seed_directions.values()
        ),
        "all_fold_directions_positive": all(all(value > 0 for value in row) for row in folds.values()),
        "delayed_B_escrow_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c61",
        "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_fresh_counterfactual_edge_foundation" if passed else "failed_fresh_counterfactual_edge_terminal",
        "decision": "authorize_delayed_B_and_cross_domain_confirmation" if passed else "close_counterfactual_edge_likelihood",
        "execution_lock_sha256": execution_hash,
        "internal_A_candidate_key_sha256": actual_hash,
        "internal_A_requests": len(fresh),
        "mean_ndcg10": {name: float(value.mean()) for name, value in metric.items()},
        "comparisons": comparisons,
        "seed_directions": seed_directions,
        "fold_directions": folds,
        "checks": checks,
        "claims": {"architecture_signal": passed, "fresh_result": passed, "novelty": False},
        "fit_and_internal_A_labels_read": True,
        "delayed_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / "train_gate_report.json", value)
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
            raise ValueError("C61 seed stage requires --seed")
        value = run_seed(args.config, args.seed, args.device)
    elif args.stage == "a0":
        value = aggregate_a0(args.config)
    else:
        value = aggregate_a1(args.config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
