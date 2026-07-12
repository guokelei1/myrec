"""Run the frozen C43 KuaiSearch cross-domain architecture gate."""

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

from model.metric_coupled import (  # noqa: E402
    MODES,
    MULTIHEAD_COUPLED,
    SELECTION_ONLY,
    SHIFTED_LOOP,
    SINGLE_WIDE_COUPLED,
    MetricCoupledTransportTransformer,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402
from train.locking import verify_execution_lock, verify_proposal_lock  # noqa: E402
from train.real_data import (  # noqa: E402
    CompactLabels,
    FrozenTransferStore,
    open_original_labels,
    to_tensor,
)
from train.structure import atomic_json, load_config, read_json, sha256_file  # noqa: E402


PRIMARY = MULTIHEAD_COUPLED
MATCHED_CONTROLS = (SINGLE_WIDE_COUPLED, SELECTION_ONLY, SHIFTED_LOOP)
FUNCTIONAL_CONTROLS = ("fixed_semantic", "uniform_semantic")


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
    if seed not in seeds:
        raise ValueError(f"unregistered C43 seed: {seed}")
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if str(device) != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C43 seed/GPU registration mismatch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C43 deterministic CUBLAS setting absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C43 seed process requires exactly one visible GPU")
    return physical


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_model(
    config: Mapping[str, Any], seed: int, mode: str
) -> MetricCoupledTransportTransformer:
    row = config["model"]
    return MetricCoupledTransportTransformer(
        dim=int(row["embedding_dim"]),
        heads=int(row["heads"]),
        rank=int(row["rank"]),
        temperature=float(row["history_temperature"]),
        profile_scale=float(row["profile_scale"]),
        correction_scale=float(row["correction_scale"]),
        seed=seed,
        mode=mode,
        init_std=float(row["init_std"]),
    )


def load_fit_labels(config: Mapping[str, Any]) -> CompactLabels:
    root = Path(config["paths"]["artifact_root"])
    g0 = read_json(root / "g0_report.json")
    for name in ("fit_request_indices.npy", "fit_label_offsets.npy", "fit_labels.npy"):
        if sha256_file(root / name) != g0["outputs"][name]["sha256"]:
            raise RuntimeError(f"C43 fit-label artifact changed: {name}")
    return CompactLabels(
        request_indices=np.load(root / "fit_request_indices.npy", allow_pickle=False),
        offsets=np.load(root / "fit_label_offsets.npy", allow_pickle=False),
        values=np.load(root / "fit_labels.npy", allow_pickle=False),
    )


def model_inputs(
    store: FrozenTransferStore,
    index: int,
    history_source: str,
    device: torch.device,
    *,
    candidate_order: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = to_tensor(store.query(index), device)
    history = to_tensor(store.item_embeddings(store.history(index, history_source)), device)
    candidates = store.candidate_embedding_indices(index)
    if candidate_order is not None:
        candidates = candidates[candidate_order]
    return query, history, to_tensor(store.item_embeddings(candidates), device)


def train_mode(
    model: MetricCoupledTransportTransformer,
    store: FrozenTransferStore,
    labels: CompactLabels,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    row = config["training"]
    indices: list[int] = []
    targets: list[np.ndarray] = []
    for index in store.role_indices("fit"):
        target = labels.row(index, store.candidate_count(index)) > 0
        if not len(store.history(index, "true")) or store.has_repeat(index):
            raise ValueError("C43 fit selection violates strict nonrepeat history contract")
        if not target.any() or target.all():
            raise ValueError("C43 fit labels are not mixed")
        indices.append(index)
        targets.append(target)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(row["learning_rate"]),
        weight_decay=float(row["weight_decay"]),
    )
    model.to(device).train()
    losses: list[float] = []
    listwise_losses: list[float] = []
    direction_losses: list[float] = []
    gradients: set[str] = set()
    for epoch in range(int(row["epochs"])):
        order = np.random.default_rng(seed + epoch * 10003).permutation(len(indices))
        batch_size = int(row["max_requests_per_batch"])
        for start in range(0, len(order), batch_size):
            request_losses: list[torch.Tensor] = []
            request_listwise: list[torch.Tensor] = []
            request_direction: list[torch.Tensor] = []
            for raw in order[start : start + batch_size]:
                position = int(raw)
                index = indices[position]
                target = torch.from_numpy(targets[position]).to(device)
                query, history, candidates = model_inputs(store, index, "true", device)
                correction = model(query, history, candidates)
                score = to_tensor(store.base_row(index), device) + correction
                listwise = torch.logsumexp(score, dim=0) - score[target].mean()
                direction = F.softplus(-(correction[target].mean() - correction[~target].mean()))
                request_listwise.append(listwise)
                request_direction.append(direction)
                request_losses.append(
                    float(row["listwise_loss_weight"]) * listwise
                    + float(row["direction_loss_weight"]) * direction
                )
            loss = torch.stack(request_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite C43 training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C43 gradient: {name}")
                    if bool(parameter.grad.ne(0).any()):
                        gradients.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(row["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            listwise_losses.append(float(torch.stack(request_listwise).mean().detach().cpu()))
            direction_losses.append(float(torch.stack(request_direction).mean().detach().cpu()))
    return {
        "fit_requests": len(indices),
        "epochs": int(row["epochs"]),
        "steps": len(losses),
        "all_candidates_used": True,
        "candidate_sampling": False,
        "finite": bool(losses) and bool(np.isfinite(losses).all()),
        "loss_first_30": float(np.mean(losses[:30])),
        "loss_last_30": float(np.mean(losses[-30:])),
        "listwise_last_30": float(np.mean(listwise_losses[-30:])),
        "direction_last_30": float(np.mean(direction_losses[-30:])),
        "gradient_parameter_names": sorted(gradients),
    }


def fixed_semantic_correction(
    query: torch.Tensor,
    history: torch.Tensor,
    candidates: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    if len(history) == 0:
        return candidates.new_zeros(len(candidates))
    query = F.normalize(query, dim=-1, eps=1e-6)
    history = F.normalize(history, dim=-1, eps=1e-6)
    candidates = F.normalize(candidates, dim=-1, eps=1e-6)
    attention = torch.softmax(
        history.mv(query) / float(config["model"]["history_temperature"]), dim=0
    )
    profile = torch.einsum("j,jd->d", attention, history)
    transported = F.normalize(
        query + float(config["model"]["profile_scale"]) * profile, dim=-1, eps=1e-6
    )
    return float(config["model"]["correction_scale"]) * (
        candidates.mv(transported) - candidates.mv(query)
    )


def uniform_semantic_correction(
    query: torch.Tensor,
    history: torch.Tensor,
    candidates: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    if len(history) == 0:
        return candidates.new_zeros(len(candidates))
    query = F.normalize(query, dim=-1, eps=1e-6)
    history = F.normalize(history, dim=-1, eps=1e-6)
    candidates = F.normalize(candidates, dim=-1, eps=1e-6)
    profile = history.mean(dim=0)
    transported = F.normalize(
        query + float(config["model"]["profile_scale"]) * profile, dim=-1, eps=1e-6
    )
    return float(config["model"]["correction_scale"]) * (
        candidates.mv(transported) - candidates.mv(query)
    )


def score_callable(
    scorer: Any,
    store: FrozenTransferStore,
    indices: Sequence[int],
    history_source: str,
    device: torch.device,
    *,
    query_present: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    scores: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    with torch.inference_mode():
        for index in indices:
            query, history, candidates = model_inputs(store, int(index), history_source, device)
            if store.has_repeat(int(index)):
                correction = torch.zeros(len(candidates), device=device)
                score = store.item_only_row(int(index))
            elif not query_present or len(history) == 0:
                correction = torch.zeros(len(candidates), device=device)
                score = store.base_row(int(index))
            else:
                correction = scorer(query, history, candidates)
                score = store.base_row(int(index)) + correction.detach().cpu().numpy().astype(np.float32)
            corrections.append(correction.detach().cpu().numpy().astype(np.float32))
            scores.append(np.asarray(score, dtype=np.float32))
    return scores, corrections


def diagnose_model(
    model: MetricCoupledTransportTransformer,
    store: FrozenTransferStore,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    A_indices = store.role_indices("internal_A")[:32]
    deterministic = 0.0
    permutation_error = 0.0
    nohistory_error = 0.0
    query_absent_error = 0.0
    repeat_wrapper_error = 0.0
    finite = True
    assignment: list[int] | None = None
    model.eval()
    with torch.inference_mode():
        for index in A_indices:
            query, history, candidates = model_inputs(store, index, "true", device)
            first = model(query, history, candidates)
            second = model(query, history, candidates)
            deterministic = max(deterministic, float((first - second).abs().max().cpu()))
            permutation = np.random.default_rng(20262799 + index).permutation(len(candidates))
            tensor_order = torch.from_numpy(permutation).to(device)
            actual = model(query, history, candidates[tensor_order])
            permutation_error = max(
                permutation_error, float((first[tensor_order] - actual).abs().max().cpu())
            )
            nohistory_error = max(
                nohistory_error, float(model(query, history[:0], candidates).abs().max().cpu())
            )
            query_absent_error = max(
                query_absent_error,
                float(model(query, history, candidates, query_present=False).abs().max().cpu()),
            )
            repeat_wrapper_error = max(
                repeat_wrapper_error,
                float(model(query, history, candidates, repeat_present=True).abs().max().cpu()),
            )
            state = model.components(query, history, candidates)
            finite = finite and all(
                bool(torch.isfinite(value).all())
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
            current = [int(value) for value in state["loop_assignment"].cpu().tolist()]
            assignment = current if assignment is None else assignment
            if current != assignment:
                raise RuntimeError("C43 loop assignment changed across requests")
    repeat_indices = store.role_indices("structural_repeat")
    repeat_scores, repeat_corrections = score_callable(model, store, repeat_indices, "true", device)
    repeat_score_error = max(
        float(np.max(np.abs(score - store.item_only_row(index))))
        for index, score in zip(repeat_indices, repeat_scores)
    )
    repeat_correction_error = max(float(np.max(np.abs(row))) for row in repeat_corrections)
    nohistory_indices = store.role_indices("structural_nohistory")
    nohistory_scores, nohistory_corrections = score_callable(
        model, store, nohistory_indices, "true", device
    )
    nohistory_score_error = max(
        float(np.max(np.abs(score - store.base_row(index))))
        for index, score in zip(nohistory_indices, nohistory_scores)
    )
    nohistory_correction_error = max(float(np.max(np.abs(row))) for row in nohistory_corrections)
    return {
        "deterministic_max_abs": deterministic,
        "candidate_permutation_max_abs": permutation_error,
        "nohistory_wrapper_max_abs": nohistory_error,
        "query_absent_wrapper_max_abs": query_absent_error,
        "repeat_wrapper_max_abs": repeat_wrapper_error,
        "repeat_item_only_score_max_abs": repeat_score_error,
        "repeat_correction_max_abs": repeat_correction_error,
        "nohistory_base_score_max_abs": nohistory_score_error,
        "nohistory_correction_max_abs": nohistory_correction_error,
        "states_finite": finite,
        "loop_assignment": assignment,
    }


def diagnose_functional(
    scorer: Any,
    store: FrozenTransferStore,
    device: torch.device,
) -> dict[str, float]:
    deterministic = 0.0
    permutation_error = 0.0
    nohistory_error = 0.0
    with torch.inference_mode():
        for index in store.role_indices("internal_A")[:32]:
            query, history, candidates = model_inputs(store, index, "true", device)
            first = scorer(query, history, candidates)
            second = scorer(query, history, candidates)
            deterministic = max(deterministic, float((first - second).abs().max().cpu()))
            order = np.random.default_rng(20262899 + index).permutation(len(candidates))
            tensor_order = torch.from_numpy(order).to(device)
            actual = scorer(query, history, candidates[tensor_order])
            permutation_error = max(
                permutation_error, float((first[tensor_order] - actual).abs().max().cpu())
            )
            nohistory_error = max(
                nohistory_error, float(scorer(query, history[:0], candidates).abs().max().cpu())
            )
    return {
        "deterministic_max_abs": deterministic,
        "candidate_permutation_max_abs": permutation_error,
        "nohistory_max_abs": nohistory_error,
    }


def flatten(rows: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(row, dtype=np.float32) for row in rows])


def offsets(rows: Sequence[np.ndarray]) -> np.ndarray:
    values = [0]
    for row in rows:
        values.append(values[-1] + len(row))
    return np.asarray(values, dtype=np.int64)


def unflatten(row_offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(row_offsets[row]) : int(row_offsets[row + 1])], dtype=np.float32)
        for row in range(len(row_offsets) - 1)
    ]


def average_rows(groups: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [
        np.mean(np.stack([group[row] for group in groups]), axis=0).astype(np.float32)
        for row in range(len(groups[0]))
    ]


def run_seed(config: Mapping[str, Any], seed: int, device: torch.device) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    physical = assert_cuda(config, seed, device)
    seed_all(seed)
    store = FrozenTransferStore(config)
    labels = load_fit_labels(config)
    indices = store.role_indices("internal_A")
    artifact_root = Path(config["paths"]["artifact_root"])
    report_path = artifact_root / f"seed_{seed}_report.json"
    score_path = artifact_root / f"seed_{seed}_internal_A_scores.npz"
    if report_path.exists() or score_path.exists():
        raise FileExistsError(f"C43 seed output exists: {seed}")
    started = time.time()
    payload: dict[str, np.ndarray] = {}
    mode_reports: dict[str, Any] = {}
    initial_hashes: dict[str, str] = {}
    for mode in MODES:
        seed_all(seed)
        model = make_model(config, seed, mode)
        initial = state_sha256(model)
        initial_hashes[mode] = initial
        training = train_mode(model, store, labels, config, seed, device)
        final = state_sha256(model)
        checkpoint_root = Path(config["paths"]["checkpoint_root"])
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_root / f"seed_{seed}_{mode}.pt"
        if checkpoint_path.exists():
            raise FileExistsError(checkpoint_path)
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "candidate_id": "c43",
                "seed": seed,
                "mode": mode,
                "proposal_lock_sha256": proposal_hash,
                "execution_lock_sha256": execution_hash,
                "state_dict": model.state_dict(),
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        model.eval()
        true_scores, true_corrections = score_callable(model, store, indices, "true", device)
        wrong_scores, _ = score_callable(model, store, indices, "wrong", device)
        mode_reports[mode] = {
            "parameters": model.trainable_parameter_count(),
            "training": training,
            "initial_state_sha256": initial,
            "final_state_sha256": final,
            "parameters_updated": initial != final,
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "diagnostics": diagnose_model(model, store, config, device),
        }
        payload[f"{mode}_true"] = flatten(true_scores)
        payload[f"{mode}_wrong"] = flatten(wrong_scores)
        payload[f"{mode}_correction"] = flatten(true_corrections)

    fixed = lambda q, h, c: fixed_semantic_correction(q, h, c, config)
    uniform = lambda q, h, c: uniform_semantic_correction(q, h, c, config)
    fixed_true, _ = score_callable(fixed, store, indices, "true", device)
    fixed_wrong, _ = score_callable(fixed, store, indices, "wrong", device)
    uniform_true, _ = score_callable(uniform, store, indices, "true", device)
    base_rows = [store.base_row(index) for index in indices]
    payload.update(
        {
            "fixed_semantic_true": flatten(fixed_true),
            "fixed_semantic_wrong": flatten(fixed_wrong),
            "uniform_semantic_true": flatten(uniform_true),
            "base": flatten(base_rows),
            "offsets": offsets(base_rows),
        }
    )
    temporary_score = score_path.with_suffix(score_path.suffix + ".tmp")
    with temporary_score.open("wb") as handle:
        np.savez(handle, **payload)
    temporary_score.replace(score_path)
    report = {
        "candidate_id": "c43",
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "physical_gpu": physical,
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "mode_reports": mode_reports,
        "functional_control_reports": {
            "fixed_semantic": diagnose_functional(fixed, store, device),
            "uniform_semantic": diagnose_functional(uniform, store, device),
        },
        "paired_initialization": len(set(initial_hashes.values())) == 1,
        "seed_specific_initial_state_sha256": initial_hashes[PRIMARY],
        "score_artifact": {"path": str(score_path), "sha256": sha256_file(score_path)},
        "optimizer_steps": sum(mode_reports[mode]["training"]["steps"] for mode in MODES),
        "internal_A_scores_opened": True,
        "internal_A_labels_opened": False,
        "dev_test_qrels_read": False,
    }
    atomic_json(report_path, report)
    return report


def rankings(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    scores: Sequence[np.ndarray],
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


def order_changes(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    first_scores: Sequence[np.ndarray],
    second_scores: Sequence[np.ndarray],
) -> dict[str, Any]:
    first = rankings(request_ids, item_ids, first_scores)
    second = rankings(request_ids, item_ids, second_scores)
    any_count = sum(int(a != b) for a, b in zip(first, second))
    top10_count = sum(int(set(a[:10]) != set(b[:10])) for a, b in zip(first, second))
    return {
        "requests": len(first),
        "any_count": any_count,
        "any_fraction": any_count / len(first),
        "top10_count": top10_count,
        "top10_fraction": top10_count / len(first),
    }


def ndcg_rows(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
) -> np.ndarray:
    values = []
    for request_id, items, score, relevance in zip(request_ids, item_ids, scores, labels):
        ranked = rankings([request_id], [items], [score])[0]
        positives = {str(item) for item, value in zip(items, relevance) if value > 0}
        values.append(ndcg_at_k(ranked, positives, 10))
    return np.asarray(values, dtype=np.float64)


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    artifact_root = Path(config["paths"]["artifact_root"])
    output_path = artifact_root / "train_gate_report.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    seeds = [int(value) for value in config["training"]["seeds"]]
    reports = [read_json(artifact_root / f"seed_{seed}_report.json") for seed in seeds]
    store = FrozenTransferStore(config)
    indices = store.role_indices("internal_A")
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_item_ids(index).astype(str).tolist() for index in indices]
    names = [
        "base",
        "fixed_semantic_true",
        "fixed_semantic_wrong",
        "uniform_semantic_true",
        *[f"{mode}_{source}" for mode in MODES for source in ("true", "wrong", "correction")],
    ]
    score_rows: dict[int, dict[str, list[np.ndarray]]] = {}
    for seed, report in zip(seeds, reports):
        path = Path(report["score_artifact"]["path"])
        if sha256_file(path) != report["score_artifact"]["sha256"]:
            raise RuntimeError(f"C43 score artifact changed: {seed}")
        with np.load(path, allow_pickle=False) as values:
            row_offsets = np.asarray(values["offsets"], dtype=np.int64)
            score_rows[seed] = {name: unflatten(row_offsets, values[name]) for name in names}
    averaged = {
        name: average_rows([score_rows[seed][name] for seed in seeds]) for name in names
    }
    control_score_name = {
        **{mode: f"{mode}_true" for mode in MATCHED_CONTROLS},
        "fixed_semantic": "fixed_semantic_true",
        "uniform_semantic": "uniform_semantic_true",
    }
    activity = {
        "primary_vs_base": order_changes(
            request_ids, item_ids, averaged["base"], averaged[f"{PRIMARY}_true"]
        ),
        "true_vs_wrong": order_changes(
            request_ids,
            item_ids,
            averaged[f"{PRIMARY}_true"],
            averaged[f"{PRIMARY}_wrong"],
        ),
        **{
            f"primary_vs_{name}": order_changes(
                request_ids,
                item_ids,
                averaged[control_score_name[name]],
                averaged[f"{PRIMARY}_true"],
            )
            for name in (*MATCHED_CONTROLS, *FUNCTIONAL_CONTROLS)
        },
    }
    gate = config["gate"]
    selection = read_json(config["paths"]["selection"])
    g0 = read_json(artifact_root / "g0_report.json")
    A_candidate_hash = store.candidate_hash(indices)
    expected_assignments = {
        MULTIHEAD_COUPLED: [0, 1, 2, 3],
        SINGLE_WIDE_COUPLED: [0],
        SELECTION_ONLY: [-1, -1, -1, -1],
        SHIFTED_LOOP: [1, 2, 3, 0],
    }
    initial_hashes = [report["seed_specific_initial_state_sha256"] for report in reports]
    a0_checks = {
        "paired_initialization": all(report["paired_initialization"] for report in reports),
        "seed_specific_initialization": len(set(initial_hashes)) == len(seeds),
        "equal_capacity": all(
            {report["mode_reports"][mode]["parameters"] for mode in MODES} == {65536}
            for report in reports
        ),
        "parameters_updated": all(
            report["mode_reports"][mode]["parameters_updated"]
            for report in reports
            for mode in MODES
        ),
        "training_finite": all(
            report["mode_reports"][mode]["training"]["finite"]
            for report in reports
            for mode in MODES
        ),
        "all_parameters_receive_gradients": all(
            set(report["mode_reports"][mode]["training"]["gradient_parameter_names"])
            == {"down", "up"}
            for report in reports
            for mode in MODES
        ),
        "mode_loop_assignments": all(
            report["mode_reports"][mode]["diagnostics"]["loop_assignment"]
            == expected_assignments[mode]
            for report in reports
            for mode in MODES
        ),
        "model_contracts": all(
            report["mode_reports"][mode]["diagnostics"]["deterministic_max_abs"]
            <= float(gate["deterministic_max_abs"])
            and report["mode_reports"][mode]["diagnostics"]["candidate_permutation_max_abs"]
            <= float(gate["candidate_permutation_max_abs"])
            and report["mode_reports"][mode]["diagnostics"]["nohistory_wrapper_max_abs"] == 0.0
            and report["mode_reports"][mode]["diagnostics"]["query_absent_wrapper_max_abs"] == 0.0
            and report["mode_reports"][mode]["diagnostics"]["repeat_wrapper_max_abs"] == 0.0
            and report["mode_reports"][mode]["diagnostics"]["repeat_item_only_score_max_abs"] == 0.0
            and report["mode_reports"][mode]["diagnostics"]["repeat_correction_max_abs"] == 0.0
            and report["mode_reports"][mode]["diagnostics"]["nohistory_base_score_max_abs"] == 0.0
            and report["mode_reports"][mode]["diagnostics"]["nohistory_correction_max_abs"] == 0.0
            and report["mode_reports"][mode]["diagnostics"]["states_finite"]
            for report in reports
            for mode in MODES
        ),
        "functional_contracts": all(
            report["functional_control_reports"][name]["deterministic_max_abs"]
            <= float(gate["deterministic_max_abs"])
            and report["functional_control_reports"][name]["candidate_permutation_max_abs"]
            <= float(gate["candidate_permutation_max_abs"])
            and report["functional_control_reports"][name]["nohistory_max_abs"] == 0.0
            for report in reports
            for name in FUNCTIONAL_CONTROLS
        ),
        "primary_order_active": activity["primary_vs_base"]["any_fraction"]
        >= float(gate["primary_vs_base_order_fraction_min"]),
        "primary_top10_active": activity["primary_vs_base"]["top10_fraction"]
        >= float(gate["primary_vs_base_top10_fraction_min"]),
        "controls_order_distinct": all(
            activity[f"primary_vs_{name}"]["any_fraction"]
            >= float(gate["primary_vs_control_order_fraction_min"])
            for name in (*MATCHED_CONTROLS, *FUNCTIONAL_CONTROLS)
        ),
        "controls_top10_distinct": all(
            activity[f"primary_vs_{name}"]["top10_fraction"]
            >= float(gate["primary_vs_control_top10_fraction_min"])
            for name in (*MATCHED_CONTROLS, *FUNCTIONAL_CONTROLS)
        ),
        "wrong_order_distinct": activity["true_vs_wrong"]["any_fraction"]
        >= float(gate["true_vs_wrong_order_fraction_min"]),
        "wrong_top10_distinct": activity["true_vs_wrong"]["top10_fraction"]
        >= float(gate["true_vs_wrong_top10_fraction_min"]),
        "checkpoint_hashes_present": all(
            sha256_file(report["mode_reports"][mode]["checkpoint"]["path"])
            == report["mode_reports"][mode]["checkpoint"]["sha256"]
            for report in reports
            for mode in MODES
        ),
        "labels_closed_during_training_scoring": all(
            report["internal_A_labels_opened"] is False for report in reports
        ),
        "candidate_set_hash_asserted": A_candidate_hash
        == selection["roles"]["internal_A"]["candidate_key_sha256"]
        == g0["A_candidate_key_sha256"],
        "dev_test_closed": all(report["dev_test_qrels_read"] is False for report in reports),
    }
    report: dict[str, Any] = {
        "candidate_id": "c43",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "A0": {"status": "passed" if all(a0_checks.values()) else "failed", "checks": a0_checks, "activity": activity},
        "seed_reports": {str(seed): value for seed, value in zip(seeds, reports)},
        "internal_A_labels_opened": False,
        "dev_test_qrels_read": False,
        "primary_dev_evaluator_calls": 0,
        "A_candidate_key_sha256": A_candidate_hash,
    }
    if not all(a0_checks.values()):
        report["status"] = "failed_A0_terminal"
        atomic_json(output_path, report)
        return report

    labels = open_original_labels(
        data=store.data,
        indices=indices,
        path=config["paths"]["train_candidate_labels"],
        expected_sha256=config["integrity"]["train_candidate_labels_sha256"],
        selection_path=config["paths"]["selection"],
        selection_sha256=config["paths"]["selection_sha256"],
    )
    label_rows = labels.rows(indices, [store.candidate_count(index) for index in indices])
    ndcg: dict[int, dict[str, np.ndarray]] = {
        seed: {
            name: ndcg_rows(request_ids, item_ids, score_rows[seed][name], label_rows)
            for name in names
            if not name.endswith("_correction")
        }
        for seed in seeds
    }
    averaged_ndcg = {
        name: ndcg_rows(request_ids, item_ids, averaged[name], label_rows)
        for name in names
        if not name.endswith("_correction")
    }
    comparison_names = {
        "base": "base",
        **{mode: f"{mode}_true" for mode in MATCHED_CONTROLS},
        "fixed_semantic": "fixed_semantic_true",
        "wrong_history": f"{PRIMARY}_wrong",
    }
    comparisons = compare(
        request_ids,
        averaged_ndcg[f"{PRIMARY}_true"],
        {name: averaged_ndcg[value] for name, value in comparison_names.items()},
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
        folds=int(config["evaluation"]["hash_folds"]),
    )
    seed_differences = {
        name: {
            str(seed): float(
                (ndcg[seed][f"{PRIMARY}_true"] - ndcg[seed][score_name]).mean()
            )
            for seed in seeds
        }
        for name, score_name in comparison_names.items()
    }
    clicked = bootstrap(
        clicked_direction(averaged[f"{PRIMARY}_correction"], label_rows),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 100,
    )

    def all_seeds_positive(name: str) -> bool:
        return all(value > 0 for value in seed_differences[name].values())

    def all_folds_positive(name: str) -> bool:
        return all(row["mean_difference"] > 0 for row in comparisons[name]["hash_folds"])

    a1_checks: dict[str, bool] = {
        "over_base_effect": comparisons["base"]["mean"] >= float(gate["primary_minus_base_min"]),
        "over_base_ci": comparisons["base"]["percentile_95_ci"][0] > 0,
        "over_base_all_seeds": all_seeds_positive("base"),
        "over_base_all_folds": all_folds_positive("base"),
        "over_fixed_effect": comparisons["fixed_semantic"]["mean"]
        >= float(gate["primary_minus_fixed_min"]),
        "over_fixed_ci": comparisons["fixed_semantic"]["percentile_95_ci"][0] > 0,
        "over_fixed_all_seeds": all_seeds_positive("fixed_semantic"),
        "over_fixed_all_folds": all_folds_positive("fixed_semantic"),
        "true_over_wrong_ci": comparisons["wrong_history"]["percentile_95_ci"][0] > 0,
        "true_over_wrong_all_seeds": all_seeds_positive("wrong_history"),
        "true_over_wrong_all_folds": all_folds_positive("wrong_history"),
        "clicked_direction_ci": clicked["percentile_95_ci"][0] > 0,
    }
    for mode in MATCHED_CONTROLS:
        a1_checks[f"over_{mode}_effect"] = comparisons[mode]["mean"] >= float(
            gate["primary_minus_matched_min"]
        )
        a1_checks[f"over_{mode}_ci"] = comparisons[mode]["percentile_95_ci"][0] > 0
        a1_checks[f"over_{mode}_all_seeds"] = all_seeds_positive(mode)
        a1_checks[f"over_{mode}_all_folds"] = all_folds_positive(mode)
    report["A1"] = {
        "status": "passed" if all(a1_checks.values()) else "failed",
        "checks": a1_checks,
        "comparisons": comparisons,
        "seed_differences": seed_differences,
        "seed_averaged_ndcg10": {
            "base": float(averaged_ndcg["base"].mean()),
            "primary": float(averaged_ndcg[f"{PRIMARY}_true"].mean()),
            "primary_wrong": float(averaged_ndcg[f"{PRIMARY}_wrong"].mean()),
            "fixed_semantic": float(averaged_ndcg["fixed_semantic_true"].mean()),
            "uniform_semantic": float(averaged_ndcg["uniform_semantic_true"].mean()),
            **{
                mode: float(averaged_ndcg[f"{mode}_true"].mean())
                for mode in MATCHED_CONTROLS
            },
        },
        "clicked_direction": clicked,
    }
    report["internal_A_labels_opened"] = True
    report["status"] = "passed_A1_cross_domain" if all(a1_checks.values()) else "failed_A1_terminal"
    atomic_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("seed", "aggregate"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config, require_selection=True)
    if args.stage == "seed":
        if args.seed is None:
            raise ValueError("C43 seed stage requires --seed")
        value = run_seed(config, int(args.seed), torch.device(args.device))
    else:
        value = aggregate(config)
    print(json.dumps({"candidate_id": "c43", "stage": args.stage, "status": value.get("status", "complete")}, sort_keys=True))


if __name__ == "__main__":
    main()
