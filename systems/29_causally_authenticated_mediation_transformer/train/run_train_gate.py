"""Parallel-seed, staged C29 train-only architecture gate."""

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
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from model.authenticated_mediation import (  # noqa: E402
    PRIMARY,
    AuthenticatedMediationTransformer,
)
from myrec.eval.metrics import ScoredCandidate, request_metrics, sort_candidates  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402
from train.locking import verify_execution_lock, verify_proposal_lock  # noqa: E402
from train.real_data import (  # noqa: E402
    CompactLabels,
    FrozenMediationStore,
    MediationSequenceBuilder,
    open_original_labels,
    state_sha256,
)
from train.structure import atomic_json, load_config, read_json, sha256_file  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def assert_cuda(config: Mapping[str, Any], seed: int, device: torch.device) -> int:
    seeds = [int(value) for value in config["training"]["seeds"]]
    physicals = [int(value) for value in config["resources"]["physical_gpus"]]
    if seed not in seeds:
        raise ValueError(f"unregistered C29 seed: {seed}")
    physical = physicals[seeds.index(seed)]
    if str(device) != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C29 seed/GPU registration mismatch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C29 deterministic CUBLAS setting absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C29 seed process requires exactly one visible GPU")
    return physical


def fit_labels(config: Mapping[str, Any]) -> CompactLabels:
    root = Path(config["paths"]["artifact_root"])
    return CompactLabels(
        request_indices=np.load(root / "fit_request_indices.npy", allow_pickle=False),
        offsets=np.load(root / "fit_label_offsets.npy", allow_pickle=False),
        values=np.load(root / "fit_labels.npy", allow_pickle=False),
    )


def make_model(config: Mapping[str, Any], mode: str = PRIMARY) -> AuthenticatedMediationTransformer:
    return AuthenticatedMediationTransformer(
        str(config["paths"]["bge_snapshot"]),
        mode=mode,
        correction_cap=float(config["model"]["correction_cap"]),
    )


def make_builder(store: FrozenMediationStore, config: Mapping[str, Any]) -> MediationSequenceBuilder:
    g0 = read_json(Path(config["paths"]["artifact_root"]) / "g0_report.json")
    token = g0["tokenization"]
    return MediationSequenceBuilder(
        store,
        cls_token_id=int(token["cls_token_id"]),
        sep_token_id=int(token["sep_token_id"]),
        pad_token_id=int(token["padding_idx"]),
    )


def training_examples(
    store: FrozenMediationStore, labels: CompactLabels
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    request_rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    mixed = single = 0
    candidates = 0
    for request_index in store.role_indices("fit"):
        count = store.candidate_count(request_index)
        target = labels.rows([request_index], [count])[0]
        if float(target.min(initial=0.0)) < 0.0 or float(target.max(initial=0.0)) > 1.0:
            raise ValueError("C29 BCE fit labels are outside [0,1]")
        positive = target > 0
        positives, negatives = int(positive.sum()), int((~positive).sum())
        if positives and negatives:
            weight = np.where(positive, 0.5 / positives, 0.5 / negatives).astype(np.float32)
            mixed += 1
        else:
            weight = np.full(count, 1.0 / max(count, 1), dtype=np.float32)
            single += 1
        request_rows.append(np.full(count, request_index, dtype=np.int64))
        positions.append(np.arange(count, dtype=np.int32))
        targets.append(np.asarray(positive, dtype=np.float32))
        weights.append(weight)
        bases.append(store.base_row(request_index))
        candidates += count
    return (
        np.concatenate(request_rows),
        np.concatenate(positions),
        np.concatenate(targets),
        np.concatenate(weights),
        np.concatenate(bases),
        {
            "fit_requests": len(request_rows),
            "all_candidates_used": candidates,
            "mixed_label_requests": mixed,
            "single_class_requests": single,
            "candidate_sampling": False,
        },
    )


def train_primary(
    model: AuthenticatedMediationTransformer,
    builder: MediationSequenceBuilder,
    store: FrozenMediationStore,
    labels: CompactLabels,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    request_indices, positions, targets, weights, bases, audit = training_examples(store, labels)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": float(training["encoder_learning_rate"])},
            {"params": model.head.parameters(), "lr": float(training["head_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    model.to(device).train()
    order = np.random.default_rng(seed).permutation(len(request_indices))
    batch_size = int(training["batch_size"])
    losses: list[float] = []
    true_losses: list[float] = []
    wrong_losses: list[float] = []
    contrast_losses: list[float] = []
    gradient_names: set[str] = set()
    weights_by_term = (
        float(training["true_ranking_weight"]),
        float(training["wrong_return_to_base_weight"]),
        float(training["true_minus_wrong_weight"]),
    )
    if any(value <= 0 for value in weights_by_term):
        raise ValueError("C29 loss weights must be positive")
    for epoch in range(int(training["epochs"])):
        epoch_order = (
            order
            if epoch == 0
            else np.random.default_rng(seed + epoch * 10003).permutation(len(request_indices))
        )
        for start in range(0, len(epoch_order), batch_size):
            selected = epoch_order[start : start + batch_size]
            examples = [
                (int(request_indices[index]), int(positions[index])) for index in selected
            ]
            true_ids, true_attention = builder.batch(
                examples,
                history_source="true",
                authenticated=True,
                device=device,
            )
            wrong_ids, wrong_attention = builder.batch(
                examples,
                history_source="wrong",
                authenticated=True,
                device=device,
            )
            true_delta = model.correction_from_paired_logits(model(true_ids, true_attention))
            wrong_delta = model.correction_from_paired_logits(model(wrong_ids, wrong_attention))
            target = torch.from_numpy(targets[selected]).to(device)
            weight = torch.from_numpy(weights[selected]).to(device)
            base = torch.from_numpy(bases[selected]).to(device)
            true_cell = F.binary_cross_entropy_with_logits(
                base + true_delta, target, reduction="none"
            )
            neutral_cell = F.binary_cross_entropy_with_logits(
                base + wrong_delta, torch.sigmoid(base).detach(), reduction="none"
            )
            contrast_cell = F.binary_cross_entropy_with_logits(
                true_delta - wrong_delta, target, reduction="none"
            )
            normalizer = weight.sum().clamp_min(1e-12)
            true_loss = (true_cell * weight).sum() / normalizer
            wrong_loss = (neutral_cell * weight).sum() / normalizer
            contrast_loss = (contrast_cell * weight).sum() / normalizer
            total_weight = sum(weights_by_term)
            loss = (
                weights_by_term[0] * true_loss
                + weights_by_term[1] * wrong_loss
                + weights_by_term[2] * contrast_loss
            ) / total_weight
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite C29 training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C29 gradient: {name}")
                    if bool(parameter.grad.ne(0).any()):
                        gradient_names.add(name)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            true_losses.append(float(true_loss.detach().cpu()))
            wrong_losses.append(float(wrong_loss.detach().cpu()))
            contrast_losses.append(float(contrast_loss.detach().cpu()))
    auth_nonempty = sum(
        bool(store.authenticated_history(index, "true").size)
        for index in store.role_indices("fit")
    )
    names = sorted(gradient_names)
    return {
        **audit,
        "steps": len(losses),
        "finite": bool(losses) and bool(np.isfinite(losses).all()),
        "loss_first_100_mean": float(np.mean(losses[:100])),
        "loss_last_100_mean": float(np.mean(losses[-100:])),
        "true_loss_last_100_mean": float(np.mean(true_losses[-100:])),
        "wrong_loss_last_100_mean": float(np.mean(wrong_losses[-100:])),
        "contrast_loss_last_100_mean": float(np.mean(contrast_losses[-100:])),
        "authenticated_fit_nonempty": auth_nonempty,
        "authenticated_fit_nonempty_fraction": auth_nonempty / len(store.role_indices("fit")),
        "nonzero_gradient_parameter_count": len(names),
        "nonzero_gradient_parameter_names_sha256": hashlib.sha256(
            "\n".join(names).encode()
        ).hexdigest(),
    }


def score(
    model: AuthenticatedMediationTransformer,
    builder: MediationSequenceBuilder,
    store: FrozenMediationStore,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    *,
    history_source: str = "true",
    query_present: bool = True,
    reverse_candidate_order: bool = False,
) -> dict[str, Any]:
    model.to(device).eval()
    request_indices = [int(value) for value in indices]
    base_rows = [store.base_row(index) for index in request_indices]
    item_only_rows = [store.item_only_row(index) for index in request_indices]
    item_ids = [store.candidate_item_ids(index) for index in request_indices]
    corrections = [np.zeros(len(row), dtype=np.float32) for row in base_rows]
    active = []
    examples: list[tuple[int, int]] = []
    locations: list[tuple[int, int]] = []
    for row, index in enumerate(request_indices):
        history = (
            store.authenticated_history(index, history_source)
            if model.uses_authentication
            else store.raw_history(index, history_source)
        )
        is_active = (
            query_present
            and bool(store.query_tokens(index, int(config["sequence"]["max_query_content"])))
            and not store.has_repeat(index)
            and bool(history.size)
        )
        active.append(is_active)
        if not is_active:
            continue
        positions = list(range(len(base_rows[row])))
        if reverse_candidate_order:
            positions.reverse()
        for position in positions:
            examples.append((index, position))
            locations.append((row, position))
    batch_size = int(config["training"]["batch_size"])
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            ids, attention = builder.batch(
                examples[start : start + batch_size],
                history_source=history_source,
                authenticated=model.uses_authentication,
                device=device,
            )
            values.append(model.correction_from_paired_logits(model(ids, attention)).cpu().numpy())
    flat = np.concatenate(values).astype(np.float32, copy=False) if values else np.empty(0, np.float32)
    for value, (row, position) in zip(flat, locations):
        corrections[row][position] = value
    scores: list[np.ndarray] = []
    for row, index in enumerate(request_indices):
        if store.has_repeat(index):
            corrections[row].fill(0.0)
            scores.append(item_only_rows[row].copy())
        elif not active[row]:
            corrections[row].fill(0.0)
            scores.append(base_rows[row].copy())
        else:
            corrections[row] -= float(np.asarray(corrections[row], dtype=np.float64).mean())
            scores.append((base_rows[row] + corrections[row]).astype(np.float32))
    return {
        "request_indices": request_indices,
        "request_ids": [store.data.request_ids[index] for index in request_indices],
        "item_ids": item_ids,
        "scores": scores,
        "base_scores": base_rows,
        "item_only_scores": item_only_rows,
        "corrections": corrections,
        "active": active,
    }


def max_difference(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    return max(float(np.max(np.abs(a - b))) for a, b in zip(first, second))


def change_fraction(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    return sum(int(bool(np.max(np.abs(a - b)) > 1e-7)) for a, b in zip(first, second)) / len(first)


def rankings(
    request_ids: Sequence[str], item_ids: Sequence[np.ndarray], scores: Sequence[np.ndarray]
) -> list[list[str]]:
    return [
        [
            row.item_id
            for row in sort_candidates(
                request_id,
                [ScoredCandidate(str(item), float(score)) for item, score in zip(items, values)],
            )
        ]
        for request_id, items, values in zip(request_ids, item_ids, scores)
    ]


def order_changes(reference: Mapping[str, Any], proposed: Mapping[str, Any]) -> dict[str, Any]:
    first = rankings(reference["request_ids"], reference["item_ids"], reference["scores"])
    second = rankings(proposed["request_ids"], proposed["item_ids"], proposed["scores"])
    any_count = sum(int(a != b) for a, b in zip(first, second))
    top_count = sum(int(set(a[:10]) != set(b[:10])) for a, b in zip(first, second))
    count = len(first)
    return {
        "requests": count,
        "any_count": any_count,
        "any_fraction": any_count / count,
        "top10_count": top_count,
        "top10_fraction": top_count / count,
    }


def average_rows(collections: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [np.mean(np.stack(rows), axis=0) for rows in zip(*collections)]


def flatten_rows(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, dtype=np.int64), np.concatenate(rows).astype(np.float32, copy=False)


def unflatten_rows(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[row]) : int(offsets[row + 1])], dtype=np.float32).copy()
        for row in range(len(offsets) - 1)
    ]


def write_npz_once(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"immutable C29 score artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def candidate_hashes(store: FrozenMediationStore) -> dict[str, str]:
    output: dict[str, str] = {}
    for role, row in store.selection["roles"].items():
        actual = store.candidate_hash(row["indices"])
        if actual != row["candidate_key_sha256"]:
            raise RuntimeError(f"C29 candidate hash differs: {role}")
        output[role] = actual
    return output


def save_checkpoint(
    model: AuthenticatedMediationTransformer,
    config: Mapping[str, Any],
    seed: int,
    proposal_hash: str,
    execution_hash: str,
) -> dict[str, Any]:
    root = Path(config["paths"]["checkpoint_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"seed_{seed}_{PRIMARY}.pt"
    if path.exists():
        raise FileExistsError(f"C29 checkpoint exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "candidate_id": "c29",
            "seed": seed,
            "mode": PRIMARY,
            "proposal_lock_sha256": proposal_hash,
            "execution_lock_sha256": execution_hash,
            "state_dict": model.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "state_sha256": state_sha256(model),
    }


def prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    root = Path(config["paths"]["artifact_root"])
    attempt = root / "formal_attempt.json"
    report = root / "train_gate_report.json"
    if attempt.exists() or report.exists():
        raise FileExistsError("C29 formal attempt already exists")
    store = FrozenMediationStore(config)
    hashes = candidate_hashes(store)
    value = {
        "candidate_id": "c29",
        "status": "prepared_before_training_or_internal_A_score",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "candidate_hashes": hashes,
        "authorized_seeds": [int(value) for value in config["training"]["seeds"]],
        "primary_mode_only": True,
        "internal_A_scores_opened": False,
        "internal_A_labels_opened": False,
        "delayed_B_features_labels_scores_opened": False,
        "escrow_dev_test_opened": False,
    }
    atomic_json(attempt, value)
    return value


def run_seed(
    config: Mapping[str, Any], seed: int, device: torch.device
) -> dict[str, Any]:
    physical = assert_cuda(config, seed, device)
    seed_all(seed)
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    root = Path(config["paths"]["artifact_root"])
    attempt = read_json(root / "formal_attempt.json")
    if attempt.get("status") != "prepared_before_training_or_internal_A_score":
        raise PermissionError("C29 formal attempt not prepared")
    report_path = root / f"seed_{seed}_report.json"
    score_path = root / f"seed_{seed}_internal_A_scores.npz"
    if report_path.exists() or score_path.exists():
        raise FileExistsError(f"C29 seed output exists: {seed}")
    started = time.time()
    store = FrozenMediationStore(config)
    if candidate_hashes(store) != attempt["candidate_hashes"]:
        raise RuntimeError("C29 candidate hashes changed before seed training")
    labels = fit_labels(config)
    model = make_model(config, PRIMARY)
    identity = model.identity()
    if identity["head_initialized_exact_zero"] is not True or identity["dropout_disabled"] is not True:
        raise RuntimeError("C29 neutral initialization differs")
    initial_hash = state_sha256(model)
    builder = make_builder(store, config)
    training = train_primary(model, builder, store, labels, config, seed, device)
    final_hash = state_sha256(model)
    checkpoint = save_checkpoint(model, config, seed, proposal_hash, execution_hash)
    A_indices = store.role_indices("internal_A")
    clean = score(model, builder, store, A_indices, config, device)
    repeated = score(model, builder, store, A_indices, config, device)
    wrong = score(
        model, builder, store, A_indices, config, device, history_source="wrong"
    )
    noauth = score(
        model, builder, store, A_indices, config, device, history_source="none"
    )
    query_absent = score(
        model, builder, store, A_indices, config, device, query_present=False
    )
    permuted = score(
        model,
        builder,
        store,
        A_indices,
        config,
        device,
        reverse_candidate_order=True,
    )
    repeat = score(
        model, builder, store, store.role_indices("structural_repeat"), config, device
    )
    nohistory = score(
        model, builder, store, store.role_indices("structural_nohistory"), config, device
    )
    offsets, clean_scores = flatten_rows(clean["scores"])
    _, clean_corrections = flatten_rows(clean["corrections"])
    _, wrong_scores = flatten_rows(wrong["scores"])
    _, wrong_corrections = flatten_rows(wrong["corrections"])
    _, base_scores = flatten_rows(clean["base_scores"])
    write_npz_once(
        score_path,
        request_indices=np.asarray(A_indices, dtype=np.int64),
        candidate_offsets=offsets,
        clean_scores=clean_scores,
        clean_corrections=clean_corrections,
        wrong_scores=wrong_scores,
        wrong_corrections=wrong_corrections,
        base_scores=base_scores,
    )
    output = {
        "candidate_id": "c29",
        "seed": seed,
        "mode": PRIMARY,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "physical_gpu": physical,
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "candidate_hashes": attempt["candidate_hashes"],
        "model_identity": identity,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "parameters_updated": final_hash != initial_hash,
        "training": training,
        "checkpoint": checkpoint,
        "score_artifact": {"path": str(score_path), "sha256": sha256_file(score_path)},
        "internal_A_active_requests": int(sum(clean["active"])),
        "internal_A_scores_opened": True,
        "internal_A_labels_opened": False,
        "delayed_B_features_labels_scores_opened": False,
        "deterministic_max_abs_difference": max_difference(
            clean["scores"], repeated["scores"]
        ),
        "candidate_permutation_max_abs_difference": max_difference(
            clean["scores"], permuted["scores"]
        ),
        "wrong_correction_change_fraction": change_fraction(
            clean["corrections"], wrong["corrections"]
        ),
        "wrong_order_changes": order_changes(wrong, clean),
        "noauth_exact_d2p": all(
            np.array_equal(a, b) for a, b in zip(noauth["scores"], noauth["base_scores"])
        ),
        "query_absent_exact_d2p": all(
            np.array_equal(a, b)
            for a, b in zip(query_absent["scores"], query_absent["base_scores"])
        ),
        "repeat_exact_item_only": all(
            np.array_equal(a, b)
            for a, b in zip(repeat["scores"], repeat["item_only_scores"])
        ),
        "nohistory_exact_d2p": all(
            np.array_equal(a, b)
            for a, b in zip(nohistory["scores"], nohistory["base_scores"])
        ),
        "c29_code_dev_test_qrels_metrics_read": False,
    }
    atomic_json(report_path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return output


def load_seed_scores(
    report: Mapping[str, Any], store: FrozenMediationStore
) -> dict[str, Any]:
    path = Path(report["score_artifact"]["path"])
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C29 seed score artifact changed")
    with np.load(path, allow_pickle=False) as data:
        indices = [int(value) for value in data["request_indices"]]
        offsets = np.asarray(data["candidate_offsets"])
        clean = unflatten_rows(offsets, np.asarray(data["clean_scores"]))
        correction = unflatten_rows(offsets, np.asarray(data["clean_corrections"]))
        wrong = unflatten_rows(offsets, np.asarray(data["wrong_scores"]))
        wrong_correction = unflatten_rows(offsets, np.asarray(data["wrong_corrections"]))
        base = unflatten_rows(offsets, np.asarray(data["base_scores"]))
    expected = store.role_indices("internal_A")
    if indices != expected:
        raise RuntimeError("C29 seed score request order differs")
    request_ids = [store.data.request_ids[index] for index in indices]
    item_ids = [store.candidate_item_ids(index) for index in indices]
    common = {"request_ids": request_ids, "item_ids": item_ids}
    return {
        "clean": {**common, "scores": clean, "corrections": correction, "base_scores": base},
        "wrong": {
            **common,
            "scores": wrong,
            "corrections": wrong_correction,
            "base_scores": base,
        },
    }


def metric_rows(
    scored: Mapping[str, Any], key: str, labels: CompactLabels, indices: Sequence[int]
) -> tuple[np.ndarray, list[np.ndarray]]:
    label_rows = labels.rows(indices, [len(row) for row in scored["item_ids"]])
    values = []
    for request_id, items, scores, target in zip(
        scored["request_ids"], scored["item_ids"], scored[key], label_rows
    ):
        positives = {str(item) for item, label in zip(items, target) if label > 0}
        row = request_metrics(
            request_id,
            [ScoredCandidate(str(item), float(score)) for item, score in zip(items, scores)],
            positives,
            set(),
        )
        values.append(float(row["ndcg@10"]))
    return np.asarray(values), label_rows


def utility_gate(
    *,
    store: FrozenMediationStore,
    outputs: Mapping[int, Mapping[str, Any]],
    labels: CompactLabels,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    seeds = [int(value) for value in config["training"]["seeds"]]
    indices = store.role_indices("internal_A")
    arrays: dict[int, dict[str, np.ndarray]] = {}
    label_rows = None
    for seed in seeds:
        clean = outputs[seed]["clean"]
        wrong = outputs[seed]["wrong"]
        primary, label_rows = metric_rows(clean, "scores", labels, indices)
        base, _ = metric_rows(clean, "base_scores", labels, indices)
        corrupted, _ = metric_rows(wrong, "scores", labels, indices)
        arrays[seed] = {"primary": primary, "d2p": base, "wrong": corrupted}
    averaged = {
        name: np.mean(np.stack([arrays[seed][name] for seed in seeds]), axis=0)
        for name in ("primary", "d2p", "wrong")
    }
    request_ids = outputs[seeds[0]]["clean"]["request_ids"]
    comparison = compare(
        request_ids,
        averaged["primary"],
        {"d2p": averaged["d2p"]},
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
        folds=int(config["evaluation"]["hash_folds"]),
    )["d2p"]
    seed_differences = {
        str(seed): float((arrays[seed]["primary"] - arrays[seed]["d2p"]).mean())
        for seed in seeds
    }
    true_wrong = bootstrap(
        averaged["primary"] - averaged["wrong"],
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 101,
    )
    assert label_rows is not None
    corrections = average_rows([outputs[seed]["clean"]["corrections"] for seed in seeds])
    clicked = bootstrap(
        clicked_direction(corrections, label_rows),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 201,
    )
    gate = config["gate"]
    checks = {
        "over_d2p_effect": comparison["mean"]
        >= float(gate["ndcg10_delta_over_d2p_min"]),
        "over_d2p_ci": comparison["percentile_95_ci"][0] > 0,
        "over_d2p_all_seeds": all(value > 0 for value in seed_differences.values()),
        "over_d2p_all_folds": all(
            row["mean_difference"] > 0 for row in comparison["hash_folds"]
        ),
        "true_over_wrong_ci": true_wrong["percentile_95_ci"][0] > 0,
        "clicked_direction_ci": clicked["percentile_95_ci"][0] > 0,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "seed_averaged_ndcg10": {
            name: float(value.mean()) for name, value in averaged.items()
        },
        "primary_minus_d2p": comparison,
        "seed_primary_minus_d2p": seed_differences,
        "true_minus_wrong": true_wrong,
        "clicked_minus_unclicked_correction": clicked,
    }


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    root = Path(config["paths"]["artifact_root"])
    report_path = root / "train_gate_report.json"
    if report_path.exists():
        raise FileExistsError("immutable C29 train-gate report exists")
    attempt_path = root / "formal_attempt.json"
    attempt = read_json(attempt_path)
    if attempt.get("status") != "prepared_before_training_or_internal_A_score":
        raise PermissionError("C29 aggregate attempt state differs")
    store = FrozenMediationStore(config)
    hashes = candidate_hashes(store)
    if hashes != attempt["candidate_hashes"]:
        raise RuntimeError("C29 candidate hashes changed before aggregate")
    seeds = [int(value) for value in config["training"]["seeds"]]
    seed_reports: dict[int, dict[str, Any]] = {}
    outputs: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        row = read_json(root / f"seed_{seed}_report.json")
        if int(row.get("seed", -1)) != seed or row.get("internal_A_labels_opened") is not False:
            raise ValueError("C29 seed report identity/boundary differs")
        if row.get("candidate_hashes") != hashes:
            raise RuntimeError("C29 seed candidate hash differs")
        seed_reports[seed] = row
        outputs[seed] = load_seed_scores(row, store)
    clean_average = average_rows([outputs[seed]["clean"]["scores"] for seed in seeds])
    base_rows = outputs[seeds[0]]["clean"]["base_scores"]
    common = {
        "request_ids": outputs[seeds[0]]["clean"]["request_ids"],
        "item_ids": outputs[seeds[0]]["clean"]["item_ids"],
    }
    activity = order_changes(
        {**common, "scores": base_rows}, {**common, "scores": clean_average}
    )
    gate = config["gate"]
    selection_checks = store.selection["checks"]
    g0 = read_json(root / "g0_report.json")
    a0_checks = {
        "training_finite": all(row["training"]["finite"] for row in seed_reports.values()),
        "gradients_active": all(
            int(row["training"]["nonzero_gradient_parameter_count"]) > 0
            for row in seed_reports.values()
        ),
        "parameters_updated": all(row["parameters_updated"] for row in seed_reports.values()),
        "parameter_count_identical": len(
            {int(row["model_identity"]["parameters"]) for row in seed_reports.values()}
        )
        == 1,
        "initialization_identical": len(
            {str(row["initial_state_sha256"]) for row in seed_reports.values()}
        )
        == 1,
        "head_zero_dropout_zero": all(
            row["model_identity"]["head_initialized_exact_zero"] is True
            and row["model_identity"]["dropout_disabled"] is True
            for row in seed_reports.values()
        ),
        "order_active": activity["any_fraction"] >= float(gate["order_change_fraction_min"]),
        "top10_active": activity["top10_fraction"] >= float(gate["top10_change_fraction_min"]),
        "wrong_changes_correction": all(
            float(row["wrong_correction_change_fraction"])
            >= float(gate["wrong_correction_change_fraction_min"])
            for row in seed_reports.values()
        ),
        "wrong_changes_order": all(
            float(row["wrong_order_changes"]["any_fraction"])
            >= float(gate["wrong_order_change_fraction_min"])
            for row in seed_reports.values()
        ),
        "wrong_changes_top10": all(
            float(row["wrong_order_changes"]["top10_fraction"])
            >= float(gate["wrong_top10_change_fraction_min"])
            for row in seed_reports.values()
        ),
        "deterministic": all(
            float(row["deterministic_max_abs_difference"])
            <= float(gate["deterministic_max_abs_difference"])
            for row in seed_reports.values()
        ),
        "candidate_permutation": all(
            float(row["candidate_permutation_max_abs_difference"])
            <= float(gate["candidate_permutation_max_abs_difference"])
            for row in seed_reports.values()
        ),
        "noauth_exact_d2p": all(row["noauth_exact_d2p"] for row in seed_reports.values()),
        "query_absent_exact_d2p": all(
            row["query_absent_exact_d2p"] for row in seed_reports.values()
        ),
        "repeat_exact_item_only": all(
            row["repeat_exact_item_only"] for row in seed_reports.values()
        ),
        "nohistory_exact_d2p": all(
            row["nohistory_exact_d2p"] for row in seed_reports.values()
        ),
        "authentication_G0": g0["authentication_gate"]["status"] == "passed",
        "selection_integrity": (
            selection_checks["c29_internal_A_features_labels_scores_opened"] is False
            and selection_checks["c29_delayed_B_features_labels_scores_opened"] is False
            and selection_checks["roles_pairwise_disjoint"] is True
            and selection_checks["strict_nonrepeat_fit_A_B_escrow"] is True
            and selection_checks["donor_candidate_overlap_zero"] is True
            and selection_checks["donor_user_overlap_zero"] is True
            and selection_checks["c29_code_dev_test_qrels_metrics_read"] is False
        ),
    }
    a0 = {
        "status": "passed" if all(a0_checks.values()) else "failed",
        "checks": a0_checks,
        "order_changes_vs_d2p": activity,
        "seed_wrong_correction_change_fraction": {
            str(seed): seed_reports[seed]["wrong_correction_change_fraction"] for seed in seeds
        },
        "seed_wrong_order_changes": {
            str(seed): seed_reports[seed]["wrong_order_changes"] for seed in seeds
        },
        "seed_deterministic_max_abs_difference": {
            str(seed): seed_reports[seed]["deterministic_max_abs_difference"] for seed in seeds
        },
        "seed_candidate_permutation_max_abs_difference": {
            str(seed): seed_reports[seed]["candidate_permutation_max_abs_difference"]
            for seed in seeds
        },
    }
    common_report = {
        "candidate_id": "c29",
        "candidate_hashes": hashes,
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "A0": a0,
        "seed_reports": {str(seed): seed_reports[seed] for seed in seeds},
        "internal_A_scores_opened": True,
        "delayed_B_features_labels_scores_opened": False,
        "escrow_dev_test_opened": False,
        "c29_code_dev_test_qrels_metrics_read": False,
    }
    if a0["status"] != "passed":
        report = {
            **common_report,
            "status": "failed_A0_terminal",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "internal_A_labels_opened": False,
        }
        atomic_json(report_path, report)
        atomic_json(attempt_path, {**attempt, "status": report["status"]})
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return report
    A_indices = store.role_indices("internal_A")
    if candidate_hashes(store) != hashes:
        raise RuntimeError("C29 candidate hashes changed before A1 label access")
    A_labels = open_original_labels(
        data=store.data,
        indices=A_indices,
        path=config["paths"]["train_candidate_labels"],
        selection_path=config["paths"]["selection"],
        selection_sha256=config["paths"]["selection_sha256"],
    )
    a1 = utility_gate(store=store, outputs=outputs, labels=A_labels, config=config)
    status = "passed_A1_controls_authorized" if a1["status"] == "passed" else "failed_A1_terminal"
    report = {
        **common_report,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "A1": a1,
        "internal_A_labels_opened": True,
        "controls_trained": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        attempt_path,
        {
            **attempt,
            "status": status,
            "internal_A_scores_opened": True,
            "internal_A_labels_opened": True,
            "report_sha256": sha256_file(report_path),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("prepare", "seed", "aggregate"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config, require_selection=True)
    if args.stage == "prepare":
        value = prepare(config)
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.stage == "seed":
        if args.seed is None:
            parser.error("--seed is required for seed stage")
        run_seed(config, args.seed, torch.device(args.device))
    else:
        aggregate(config)


if __name__ == "__main__":
    main()
