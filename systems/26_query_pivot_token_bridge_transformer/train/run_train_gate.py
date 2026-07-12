"""One-shot staged C26 token-bridge GPU gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from model.token_bridge import MODES, QueryPivotTokenBridgeTransformer  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, request_metrics, sort_candidates  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare, retention  # noqa: E402
from train.locking import verify_execution_lock, verify_proposal_lock  # noqa: E402
from train.losses import masked_listwise_loss  # noqa: E402
from train.real_data import (  # noqa: E402
    CompactLabels,
    FrozenTokenStore,
    iter_batches,
    open_original_labels,
    state_sha256,
    to_device,
)
from train.structure import atomic_json, load_config, read_json, sha256_file  # noqa: E402


PRIMARY = "token_bridge"
CONTROLS = tuple(mode for mode in MODES if mode != PRIMARY)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def assert_cuda(config: Mapping[str, Any], device: torch.device) -> None:
    if str(device) != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(
        config["resources"]["physical_gpu"]
    ):
        raise RuntimeError("C26 CUDA registration mismatch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C26 deterministic CUBLAS setting absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C26 requires one visible GPU")


def make_model(
    config: Mapping[str, Any],
    mode: str,
    embedding_weight: torch.Tensor,
    padding_idx: int,
) -> QueryPivotTokenBridgeTransformer:
    value = config["model"]
    return QueryPivotTokenBridgeTransformer(
        embedding_weight=embedding_weight,
        padding_idx=padding_idx,
        input_dim=int(value["input_dim"]),
        hidden_dim=int(value["hidden_dim"]),
        heads=int(value["heads"]),
        token_layers=int(value["token_layers"]),
        history_layers=int(value["history_layers"]),
        ffn_dim=int(value["ffn_dim"]),
        dropout=float(value["dropout"]),
        max_query_tokens=int(value["max_query_tokens"]),
        max_item_tokens=int(value["max_item_tokens"]),
        max_history=int(value["max_history"]),
        score_delta_max=float(value["score_delta_max"]),
        mode=mode,
    )


def fit_labels(config: Mapping[str, Any]) -> CompactLabels:
    root = Path(config["paths"]["artifact_root"])
    return CompactLabels(
        request_indices=np.load(root / "fit_request_indices.npy"),
        offsets=np.load(root / "fit_label_offsets.npy"),
        values=np.load(root / "fit_labels.npy"),
    )


def batches(
    store: FrozenTokenStore,
    indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
    shuffle: bool,
) -> list[np.ndarray]:
    training = config["training"]
    return list(
        iter_batches(
            store,
            indices,
            seed=seed,
            shuffle=shuffle,
            max_requests=int(training["max_requests_per_batch"]),
            max_bridge_cells=int(training["max_bridge_cells"]),
        )
    )


FORWARD_NAMES = (
    "query_token_ids",
    "query_attention_mask",
    "query_content_mask",
    "candidate_token_ids",
    "candidate_attention_mask",
    "candidate_content_mask",
    "history_token_ids",
    "history_attention_mask",
    "history_content_mask",
    "candidate_mask",
    "history_mask",
    "repeat_mask",
    "event_weights",
    "base_scores",
    "item_only_scores",
)


def forward_kwargs(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    values = {name: tensors[name] for name in FORWARD_NAMES if name != "query_token_ids"}
    values["query_ids"] = tensors["query_token_ids"]
    return values


def train_model(
    model: QueryPivotTokenBridgeTransformer,
    store: FrozenTokenStore,
    labels: CompactLabels,
    schedules: Sequence[Sequence[np.ndarray]],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    model.to(device).train()
    losses: list[float] = []
    gradient_names: set[str] = set()
    for epoch_batches in schedules:
        for indices in epoch_batches:
            batch = store.collate(indices, labels=labels)
            tensors = to_device(batch, device, beta=float(config["base"]["item_only_beta"]))
            optimizer.zero_grad(set_to_none=True)
            output = model(**forward_kwargs(tensors))
            loss = masked_listwise_loss(output.scores, tensors["labels"], tensors["candidate_mask"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite C26 loss: {model.mode}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C26 gradient: {model.mode}/{name}")
                    if bool(parameter.grad.ne(0).any()):
                        gradient_names.add(name)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(training["gradient_clip_norm"]),
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {
        "steps": len(losses),
        "finite": bool(losses) and bool(np.isfinite(losses).all()),
        "loss_first_30_mean": float(np.mean(losses[:30])),
        "loss_last_30_mean": float(np.mean(losses[-30:])),
        "nonzero_gradient_parameters": sorted(gradient_names),
    }


def score(
    model: QueryPivotTokenBridgeTransformer,
    store: FrozenTokenStore,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    *,
    history_source: str = "true",
    query_present: bool = True,
) -> dict[str, Any]:
    model.to(device).eval()
    rows: dict[str, list[Any]] = {
        "request_ids": [],
        "item_ids": [],
        "scores": [],
        "base_scores": [],
        "item_only_scores": [],
        "corrections": [],
    }
    centre = 0.0
    with torch.no_grad():
        for request_batch in batches(store, indices, config, seed=0, shuffle=False):
            batch = store.collate(request_batch, history_source=history_source)
            tensors = to_device(batch, device, beta=float(config["base"]["item_only_beta"]))
            present = torch.full(
                (len(request_batch),), query_present, dtype=torch.bool, device=device
            )
            result = model(**forward_kwargs(tensors), query_present=present)
            mask = batch["candidate_mask_numpy"]
            cpu = {
                "scores": result.scores.cpu().numpy(),
                "base_scores": tensors["base_scores"].cpu().numpy(),
                "item_only_scores": result.anchor_scores.cpu().numpy(),
                "corrections": result.correction.cpu().numpy(),
            }
            for row in range(len(request_batch)):
                count = int(mask[row].sum())
                rows["request_ids"].append(batch["request_ids"][row])
                rows["item_ids"].append(batch["candidate_item_ids"][row, :count].copy())
                for name in ("scores", "base_scores", "item_only_scores", "corrections"):
                    rows[name].append(cpu[name][row, :count].copy())
                centre = max(centre, float(abs(cpu["corrections"][row, :count].sum())))
    if rows["request_ids"] != [store.data.request_ids[int(index)] for index in indices]:
        raise ValueError("C26 score order differs")
    rows["maximum_abs_correction_sum"] = centre
    return rows


def max_difference(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    return max(float(np.max(np.abs(a - b))) for a, b in zip(first, second))


def change_fraction(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    return sum(int(bool(np.max(np.abs(a - b)) > 1e-7)) for a, b in zip(first, second)) / len(first)


def rankings(request_ids: Sequence[str], item_ids: Sequence[np.ndarray], scores: Sequence[np.ndarray]) -> list[list[str]]:
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


def average_rows(collections: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [np.mean(np.stack(rows), axis=0) for rows in zip(*collections)]


def permutation_audit(
    model: QueryPivotTokenBridgeTransformer,
    store: FrozenTokenStore,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
) -> float:
    batch = store.collate(indices[:4])
    tensors = to_device(batch, device, beta=float(config["base"]["item_only_beta"]))
    count = tensors["candidate_mask"].shape[1]
    permutation = torch.arange(count - 1, -1, -1, device=device)
    inverse = torch.argsort(permutation)
    changed = forward_kwargs(tensors)
    changed = dict(changed)
    for name in (
        "candidate_token_ids",
        "candidate_attention_mask",
        "candidate_content_mask",
        "candidate_mask",
        "repeat_mask",
        "base_scores",
        "item_only_scores",
    ):
        changed[name] = tensors[name][:, permutation]
    with torch.no_grad():
        clean = model(**forward_kwargs(tensors)).scores
        recovered = model(**changed).scores[:, inverse]
    return float((clean - recovered).abs().max().cpu())


def save_checkpoint(
    model: QueryPivotTokenBridgeTransformer,
    config: Mapping[str, Any],
    seed: int,
    mode: str,
    proposal_hash: str,
    execution_hash: str,
) -> dict[str, Any]:
    root = Path(config["paths"]["checkpoint_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"seed_{seed}_{mode}.pt"
    if path.exists():
        raise FileExistsError(f"C26 checkpoint exists: {path}")
    state = {
        name: value
        for name, value in model.state_dict().items()
        if name != "word_embeddings.weight"
    }
    torch.save(
        {
            "candidate_id": "c26",
            "seed": seed,
            "mode": mode,
            "proposal_lock_sha256": proposal_hash,
            "execution_lock_sha256": execution_hash,
            "frozen_word_embeddings_sha256": sha256_file(
                Path(config["paths"]["artifact_root"]) / "word_embeddings.npy"
            ),
            "state_dict_without_frozen_word_embeddings": state,
        },
        path,
    )
    return {"path": str(path), "sha256": sha256_file(path), "state_sha256": state_sha256(model)}


def candidate_hashes(store: FrozenTokenStore) -> dict[str, str]:
    output = {}
    for role, row in store.selection["roles"].items():
        actual = store.candidate_hash(row["indices"])
        if actual != row["candidate_key_sha256"]:
            raise RuntimeError(f"C26 candidate hash differs: {role}")
        output[role] = actual
    return output


def utility_gate(
    *,
    role: str,
    indices: Sequence[int],
    labels: CompactLabels,
    outputs: Mapping[int, Mapping[str, Any]],
    seeds: Sequence[int],
    config: Mapping[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    arrays: dict[int, dict[str, np.ndarray]] = {}
    label_rows = None
    for seed in seeds:
        role_output = outputs[seed][role]
        rows: dict[str, np.ndarray] = {}
        primary = role_output["modes"][PRIMARY]
        rows["d2p"], label_rows = metric_rows(primary, "base_scores", labels, indices)
        for mode in MODES:
            rows[mode], _ = metric_rows(role_output["modes"][mode], "scores", labels, indices)
        rows["wrong_history"], _ = metric_rows(role_output["wrong"], "scores", labels, indices)
        arrays[seed] = rows
    names = ("d2p", *MODES, "wrong_history")
    averaged = {
        name: np.mean(np.stack([arrays[seed][name] for seed in seeds]), axis=0)
        for name in names
    }
    request_ids = outputs[seeds[0]][role]["modes"][PRIMARY]["request_ids"]
    comparisons = compare(
        request_ids,
        averaged[PRIMARY],
        {"d2p": averaged["d2p"], **{mode: averaged[mode] for mode in CONTROLS}},
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + seed_offset,
        folds=int(config["evaluation"]["hash_folds"]),
    )
    seed_differences = {
        name: {
            str(seed): float((arrays[seed][PRIMARY] - arrays[seed][name]).mean())
            for seed in seeds
        }
        for name in ("d2p", *CONTROLS)
    }
    wrong_retention = retention(
        averaged[PRIMARY] - averaged["d2p"],
        averaged["wrong_history"] - averaged["d2p"],
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + seed_offset + 101,
    )
    true_wrong = bootstrap(
        averaged[PRIMARY] - averaged["wrong_history"],
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + seed_offset + 201,
    )
    assert label_rows is not None
    corrections = average_rows(
        [outputs[seed][role]["modes"][PRIMARY]["corrections"] for seed in seeds]
    )
    clicked = bootstrap(
        clicked_direction(corrections, label_rows),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + seed_offset + 301,
    )
    gate = config["gate"]
    checks = {
        "over_d2p_effect": comparisons["d2p"]["mean"] >= float(gate["ndcg10_delta_over_d2p_min"]),
        "over_d2p_ci": comparisons["d2p"]["percentile_95_ci"][0] > 0,
        "over_d2p_all_seeds": all(value > 0 for value in seed_differences["d2p"].values()),
        "over_d2p_all_folds": all(row["mean_difference"] > 0 for row in comparisons["d2p"]["hash_folds"]),
        "over_controls_effect": all(
            comparisons[mode]["mean"] >= float(gate["ndcg10_delta_over_each_control_min"])
            for mode in CONTROLS
        ),
        "over_controls_ci": all(comparisons[mode]["percentile_95_ci"][0] > 0 for mode in CONTROLS),
        "over_controls_all_seeds": all(
            all(value > 0 for value in seed_differences[mode].values()) for mode in CONTROLS
        ),
        "wrong_retention": bool(wrong_retention["applicable"])
        and float(wrong_retention["retention"]) <= float(gate["corruption_retention_max"]),
        "wrong_retention_ci": bool(wrong_retention["applicable"])
        and float(wrong_retention["percentile_95_ci"][1])
        <= float(gate["corruption_retention_ci_high_max"]),
        "true_over_wrong": true_wrong["percentile_95_ci"][0] > 0,
        "clicked_direction": clicked["percentile_95_ci"][0] > 0,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "seed_averaged_ndcg10": {name: float(value.mean()) for name, value in averaged.items()},
        "comparisons": comparisons,
        "seed_differences": seed_differences,
        "wrong_history_retention": wrong_retention,
        "true_minus_wrong": true_wrong,
        "clicked_minus_unclicked": clicked,
    }


def formal(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    root = Path(config["paths"]["artifact_root"])
    report_path, attempt_path = root / "train_gate_report.json", root / "formal_attempt.json"
    if report_path.exists() or attempt_path.exists():
        raise FileExistsError("C26 formal attempt exists")
    atomic_json(
        attempt_path,
        {
            "candidate_id": "c26",
            "status": "started_before_training_or_internal_A_access",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "internal_A_labels_opened": False,
            "delayed_B_labels_opened": False,
            "escrow_dev_test_opened": False,
        },
    )
    store = FrozenTokenStore(config)
    hashes = candidate_hashes(store)
    labels = fit_labels(config)
    fit_indices = store.role_indices("fit")
    probe_indices = store.role_indices("internal_A")
    delayed_indices = store.role_indices("delayed_B")
    seeds = [int(value) for value in config["training"]["seeds"]]
    g0 = read_json(root / "g0_report.json")
    padding_idx = int(g0["tokenization"]["padding_idx"])
    embedding_weight = torch.from_numpy(
        np.asarray(store.word_embeddings, dtype=np.float32).copy()
    )
    outputs: dict[int, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    parameter_counts: dict[str, dict[str, int]] = {}
    initial_hashes: dict[str, dict[str, str]] = {}
    deterministic_max = centre_max = permutation_max = 0.0
    wrong_correction: dict[str, float] = {}
    wrong_order: dict[str, dict[str, Any]] = {}
    query_exact: dict[str, bool] = {}
    repeat_exact: dict[str, bool] = {}
    nohistory_exact: dict[str, bool] = {}
    started = time.monotonic()
    for seed in seeds:
        seed_all(seed)
        template = make_model(config, PRIMARY, embedding_weight, padding_idx)
        initial = {name: value.detach().clone() for name, value in template.state_dict().items()}
        schedules = [
            batches(store, fit_indices, config, seed=seed + epoch * 10003, shuffle=True)
            for epoch in range(int(config["training"]["epochs"]))
        ]
        modes_A: dict[str, Any] = {}
        modes_B: dict[str, Any] = {}
        training_rows: dict[str, Any] = {}
        hashes_by_mode: dict[str, str] = {}
        wrong_A = wrong_B = None
        for mode in MODES:
            model = make_model(config, mode, embedding_weight, padding_idx)
            model.load_state_dict(initial, strict=True)
            hashes_by_mode[mode] = state_sha256(model)
            parameter_counts[mode] = {
                "total": model.parameter_count(),
                "trainable": model.parameter_count(trainable_only=True),
            }
            training_rows[mode] = train_model(model, store, labels, schedules, config, device)
            checkpoints[f"{seed}/{mode}"] = save_checkpoint(
                model, config, seed, mode, proposal_hash, execution_hash
            )
            clean_A = score(model, store, probe_indices, config, device)
            clean_B = score(model, store, delayed_indices, config, device)
            modes_A[mode], modes_B[mode] = clean_A, clean_B
            if mode == PRIMARY:
                repeated = score(model, store, probe_indices, config, device)
                deterministic_max = max(
                    deterministic_max, max_difference(clean_A["scores"], repeated["scores"])
                )
                centre_max = max(centre_max, float(clean_A["maximum_abs_correction_sum"]))
                wrong_A = score(model, store, probe_indices, config, device, history_source="wrong")
                wrong_B = score(model, store, delayed_indices, config, device, history_source="wrong")
                wrong_correction[str(seed)] = change_fraction(
                    clean_A["corrections"], wrong_A["corrections"]
                )
                wrong_order[str(seed)] = order_changes(wrong_A, clean_A)
                query_absent = score(model, store, probe_indices, config, device, query_present=False)
                repeat = score(model, store, store.role_indices("structural_repeat"), config, device)
                nohistory = score(model, store, store.role_indices("structural_nohistory"), config, device)
                query_exact[str(seed)] = all(
                    np.array_equal(a, b)
                    for a, b in zip(query_absent["scores"], query_absent["base_scores"])
                )
                repeat_exact[str(seed)] = all(
                    np.array_equal(a, b)
                    for a, b in zip(repeat["scores"], repeat["item_only_scores"])
                )
                nohistory_exact[str(seed)] = all(
                    np.array_equal(a, b)
                    for a, b in zip(nohistory["scores"], nohistory["base_scores"])
                )
                permutation_max = max(
                    permutation_max, permutation_audit(model, store, probe_indices, config, device)
                )
            del model
            torch.cuda.empty_cache()
        assert wrong_A is not None and wrong_B is not None
        initial_hashes[str(seed)] = hashes_by_mode
        outputs[seed] = {
            "internal_A": {"modes": modes_A, "wrong": wrong_A},
            "delayed_B": {"modes": modes_B, "wrong": wrong_B},
        }
        summaries[str(seed)] = {
            "training": training_rows,
            "matched_initialization": len(set(hashes_by_mode.values())) == 1,
            "wrong_correction_change_fraction": wrong_correction[str(seed)],
            "wrong_order_changes": wrong_order[str(seed)],
        }

    averaged_primary = average_rows(
        [outputs[seed]["internal_A"]["modes"][PRIMARY]["scores"] for seed in seeds]
    )
    base_rows = outputs[seeds[0]]["internal_A"]["modes"][PRIMARY]["base_scores"]
    base_reference = {**outputs[seeds[0]]["internal_A"]["modes"][PRIMARY], "scores": base_rows}
    primary_average = {
        **outputs[seeds[0]]["internal_A"]["modes"][PRIMARY],
        "scores": averaged_primary,
    }
    aggregate_order = order_changes(base_reference, primary_average)
    gate = config["gate"]
    checks = store.selection["checks"]
    a0_checks = {
        "training_finite": all(
            row["finite"] for seed_row in summaries.values() for row in seed_row["training"].values()
        ),
        "gradients_active": all(
            bool(row["nonzero_gradient_parameters"])
            for seed_row in summaries.values()
            for row in seed_row["training"].values()
        ),
        "matched_parameters": len({json.dumps(value, sort_keys=True) for value in parameter_counts.values()}) == 1,
        "matched_initialization": all(len(set(row.values())) == 1 for row in initial_hashes.values()),
        "candidate_centered": centre_max <= float(gate["correction_center_abs_max"]),
        "deterministic": deterministic_max <= float(gate["deterministic_max_abs_difference"]),
        "candidate_permutation": permutation_max <= float(gate["candidate_permutation_max_abs_difference"]),
        "order_active": aggregate_order["any_fraction"] >= float(gate["order_change_fraction_min"]),
        "top10_active": aggregate_order["top10_fraction"] >= float(gate["top10_change_fraction_min"]),
        "wrong_changes_correction": all(
            value >= float(gate["wrong_correction_change_fraction_min"])
            for value in wrong_correction.values()
        ),
        "wrong_changes_order": all(
            row["any_fraction"] >= float(gate["wrong_order_change_fraction_min"])
            for row in wrong_order.values()
        ),
        "query_absent_d2p": all(query_exact.values()),
        "repeat_item_only": all(repeat_exact.values()),
        "nohistory_d2p": all(nohistory_exact.values()),
        "selection_integrity": (
            checks["c25_internal_A_labels_opened"] is False
            and checks["c25_delayed_B_labels_opened"] is False
            and checks["c26_internal_A_labels_opened"] is False
            and checks["c26_delayed_B_labels_opened"] is False
            and checks["roles_pairwise_disjoint"] is True
            and checks["donor_candidate_overlap_zero"] is True
            and checks["dev_test_qrels_metrics_read"] is False
        ),
    }
    a0 = {
        "status": "passed" if all(a0_checks.values()) else "failed",
        "checks": a0_checks,
        "order_changes_vs_d2p": aggregate_order,
        "wrong_correction_change_fraction": wrong_correction,
        "wrong_order_changes": wrong_order,
        "maximum_abs_correction_sum": centre_max,
        "deterministic_max_abs_difference": deterministic_max,
        "candidate_permutation_max_abs_difference": permutation_max,
    }
    common = {
        "candidate_id": "c26",
        "A0": a0,
        "seed_summaries": summaries,
        "checkpoints": checkpoints,
        "candidate_hashes": hashes,
        "parameter_counts": parameter_counts,
        "elapsed_seconds": time.monotonic() - started,
    }
    if not all(a0_checks.values()):
        report = {
            **common,
            "status": "failed_A0_terminal",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "internal_A_labels_opened": False,
            "delayed_B_labels_opened": False,
            "escrow_dev_test_opened": False,
        }
        atomic_json(report_path, report)
        atomic_json(attempt_path, {**read_json(attempt_path), "status": report["status"]})
        return report

    internal_labels = open_original_labels(
        data=store.data,
        indices=probe_indices,
        path=config["paths"]["train_candidate_labels"],
        selection_path=config["paths"]["selection"],
        selection_sha256=config["paths"]["selection_sha256"],
    )
    if candidate_hashes(store) != hashes:
        raise RuntimeError("C26 candidate hashes changed before A1")
    a1 = utility_gate(
        role="internal_A",
        indices=probe_indices,
        labels=internal_labels,
        outputs=outputs,
        seeds=seeds,
        config=config,
        seed_offset=0,
    )
    if a1["status"] != "passed":
        report = {
            **common,
            "status": "failed_A1_terminal",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "A1": a1,
            "internal_A_labels_opened": True,
            "delayed_B_labels_opened": False,
            "escrow_dev_test_opened": False,
        }
        atomic_json(report_path, report)
        atomic_json(
            attempt_path,
            {
                **read_json(attempt_path),
                "status": report["status"],
                "internal_A_labels_opened": True,
                "report_sha256": sha256_file(report_path),
            },
        )
        return report

    delayed_labels = open_original_labels(
        data=store.data,
        indices=delayed_indices,
        path=config["paths"]["train_candidate_labels"],
        selection_path=config["paths"]["selection"],
        selection_sha256=config["paths"]["selection_sha256"],
    )
    if candidate_hashes(store) != hashes:
        raise RuntimeError("C26 candidate hashes changed before A2")
    a2 = utility_gate(
        role="delayed_B",
        indices=delayed_indices,
        labels=delayed_labels,
        outputs=outputs,
        seeds=seeds,
        config=config,
        seed_offset=1000,
    )
    report = {
        **common,
        "status": "passed" if a2["status"] == "passed" else "failed_A2_terminal",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "A1": a1,
        "A2": a2,
        "internal_A_labels_opened": True,
        "delayed_B_labels_opened": True,
        "escrow_dev_test_opened": False,
        "execution": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    atomic_json(report_path, report)
    atomic_json(
        attempt_path,
        {
            **read_json(attempt_path),
            "status": report["status"],
            "internal_A_labels_opened": True,
            "delayed_B_labels_opened": True,
            "report_sha256": sha256_file(report_path),
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config, require_selection=True)
    device = torch.device(args.device)
    assert_cuda(config, device)
    report = formal(config, device)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
