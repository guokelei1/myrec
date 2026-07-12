"""One-shot C24 GPU train-only competition gate."""

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

from model.competition import MODES, MultiRecurrenceCompetitionTransformer  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, request_metrics, sort_candidates  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare, retention  # noqa: E402
from train.locking import verify_execution_lock, verify_proposal_lock  # noqa: E402
from train.losses import masked_listwise_loss  # noqa: E402
from train.real_data import (  # noqa: E402
    CompactLabels,
    FrozenFeatureStore,
    iter_batches,
    open_original_labels,
    state_sha256,
    to_device,
)
from train.structure import atomic_json, load_config, read_json, sha256_file  # noqa: E402


PRIMARY = "set_attention"
CONTROLS = ("independent", "query_independent")


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
        raise RuntimeError("C24 CUDA registration mismatch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C24 deterministic CUBLAS setting absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C24 requires one visible GPU")


def make_model(config: Mapping[str, Any], mode: str) -> MultiRecurrenceCompetitionTransformer:
    value = config["model"]
    return MultiRecurrenceCompetitionTransformer(
        input_dim=int(value["input_dim"]),
        hidden_dim=int(value["hidden_dim"]),
        heads=int(value["heads"]),
        layers=int(value["layers"]),
        ffn_dim=int(value["ffn_dim"]),
        dropout=float(value["dropout"]),
        max_history=int(value["max_history"]),
        max_repeat_candidates=int(value["max_repeat_candidates"]),
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
    store: FrozenFeatureStore,
    indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
    shuffle: bool,
) -> list[np.ndarray]:
    training = config["training"]
    return list(
        iter_batches(
            store.data,
            indices,
            seed=seed,
            shuffle=shuffle,
            max_requests=int(training["max_requests_per_batch"]),
            max_padded_candidates=int(training["max_padded_candidates"]),
        )
    )


def train_model(
    model: MultiRecurrenceCompetitionTransformer,
    store: FrozenFeatureStore,
    labels: CompactLabels,
    schedules: Sequence[Sequence[np.ndarray]],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            output = model(
                query=tensors["query"],
                candidates=tensors["candidates"],
                candidate_mask=tensors["candidate_mask"],
                history_mask=tensors["history_mask"],
                repeat_mask=tensors["repeat_mask"],
                event_weights=tensors["event_weights"],
                base_scores=tensors["base_scores"],
                item_only_scores=tensors["item_only_scores"],
            )
            loss = masked_listwise_loss(output.scores, tensors["labels"], tensors["candidate_mask"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite C24 loss: {model.mode}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C24 gradient: {model.mode}/{name}")
                    if bool(parameter.grad.ne(0).any()):
                        gradient_names.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
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
    model: MultiRecurrenceCompetitionTransformer,
    store: FrozenFeatureStore,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    *,
    disable_cross: bool = False,
    query_present: bool = True,
) -> dict[str, Any]:
    model.to(device).eval()
    output_rows: dict[str, list[Any]] = {
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
            batch = store.collate(request_batch)
            tensors = to_device(batch, device, beta=float(config["base"]["item_only_beta"]))
            present = torch.full(
                (len(request_batch),), query_present, dtype=torch.bool, device=device
            )
            result = model(
                query=tensors["query"],
                candidates=tensors["candidates"],
                candidate_mask=tensors["candidate_mask"],
                history_mask=tensors["history_mask"],
                repeat_mask=tensors["repeat_mask"],
                event_weights=tensors["event_weights"],
                base_scores=tensors["base_scores"],
                item_only_scores=tensors["item_only_scores"],
                query_present=present,
                disable_cross_candidate=disable_cross,
            )
            mask = batch["candidate_mask_numpy"]
            cpu = {
                "scores": result.scores.cpu().numpy(),
                "base_scores": tensors["base_scores"].cpu().numpy(),
                "item_only_scores": result.anchor_scores.cpu().numpy(),
                "corrections": result.correction.cpu().numpy(),
            }
            for row in range(len(request_batch)):
                count = int(mask[row].sum())
                output_rows["request_ids"].append(batch["request_ids"][row])
                output_rows["item_ids"].append(batch["candidate_item_ids"][row, :count].copy())
                for name in ("scores", "base_scores", "item_only_scores", "corrections"):
                    output_rows[name].append(cpu[name][row, :count].copy())
                centre = max(centre, float(abs(cpu["corrections"][row, :count].sum())))
    if output_rows["request_ids"] != [store.data.request_ids[int(i)] for i in indices]:
        raise ValueError("C24 score order differs")
    output_rows["maximum_abs_correction_sum"] = centre
    return output_rows


def max_difference(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    return max(float(np.max(np.abs(a - b))) for a, b in zip(first, second))


def change_fraction(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    return sum(int(bool(np.max(np.abs(a - b)) > 1e-7)) for a, b in zip(first, second)) / len(first)


def rankings(
    request_ids: Sequence[str], item_ids: Sequence[np.ndarray], score_rows: Sequence[np.ndarray]
) -> list[list[str]]:
    return [
        [
            value.item_id
            for value in sort_candidates(
                request_id,
                [ScoredCandidate(str(item), float(score)) for item, score in zip(items, scores)],
            )
        ]
        for request_id, items, scores in zip(request_ids, item_ids, score_rows)
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
    model: MultiRecurrenceCompetitionTransformer,
    store: FrozenFeatureStore,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
) -> float:
    batch = store.collate(indices[:32])
    tensors = to_device(batch, device, beta=float(config["base"]["item_only_beta"]))
    count = tensors["candidate_mask"].shape[1]
    permutation = torch.arange(count - 1, -1, -1, device=device)
    inverse = torch.argsort(permutation)
    kwargs = dict(
        query=tensors["query"], history_mask=tensors["history_mask"],
        event_weights=tensors["event_weights"],
    )
    with torch.no_grad():
        clean = model(
            **kwargs, candidates=tensors["candidates"], candidate_mask=tensors["candidate_mask"],
            repeat_mask=tensors["repeat_mask"], base_scores=tensors["base_scores"],
            item_only_scores=tensors["item_only_scores"],
        ).scores
        changed = model(
            **kwargs, candidates=tensors["candidates"][:, permutation],
            candidate_mask=tensors["candidate_mask"][:, permutation],
            repeat_mask=tensors["repeat_mask"][:, permutation],
            base_scores=tensors["base_scores"][:, permutation],
            item_only_scores=tensors["item_only_scores"][:, permutation],
        ).scores[:, inverse]
    return float((clean - changed).abs().max().cpu())


def save_checkpoint(
    model: MultiRecurrenceCompetitionTransformer,
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
        raise FileExistsError(f"C24 checkpoint exists: {path}")
    torch.save(
        {
            "candidate_id": "c24",
            "seed": seed,
            "mode": mode,
            "proposal_lock_sha256": proposal_hash,
            "execution_lock_sha256": execution_hash,
            "state_dict": model.state_dict(),
        },
        path,
    )
    return {"path": str(path), "sha256": sha256_file(path), "state_sha256": state_sha256(model)}


def candidate_hashes(store: FrozenFeatureStore) -> dict[str, str]:
    output = {}
    for role, row in store.selection["roles"].items():
        actual = store.candidate_hash(row["indices"])
        if actual != row["candidate_key_sha256"]:
            raise RuntimeError(f"C24 candidate hash differs: {role}")
        output[role] = actual
    return output


def formal(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    root = Path(config["paths"]["artifact_root"])
    report_path, attempt_path = root / "train_gate_report.json", root / "formal_attempt.json"
    if report_path.exists() or attempt_path.exists():
        raise FileExistsError("C24 formal attempt exists")
    atomic_json(
        attempt_path,
        {
            "candidate_id": "c24",
            "status": "started_before_training_or_internal_A_access",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "internal_A_labels_opened": False,
            "escrow_dev_test_opened": False,
        },
    )
    store = FrozenFeatureStore(config)
    hashes = candidate_hashes(store)
    labels = fit_labels(config)
    fit_indices, probe_indices = store.role_indices("fit"), store.role_indices("internal_A")
    request_ids = [store.data.request_ids[index] for index in probe_indices]
    seeds = [int(value) for value in config["training"]["seeds"]]
    outputs: dict[int, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    parameter_counts: dict[str, int] = {}
    initial_hashes: dict[str, dict[str, str]] = {}
    deterministic_max = centre_max = permutation_max = 0.0
    cross_correction_fraction: dict[str, float] = {}
    cross_order: dict[str, dict[str, Any]] = {}
    query_exact: dict[str, bool] = {}
    single_exact: dict[str, bool] = {}
    nohistory_exact: dict[str, bool] = {}
    nonrepeat_exact: dict[str, bool] = {}
    started = time.monotonic()
    for seed in seeds:
        seed_all(seed)
        template = make_model(config, PRIMARY)
        initial = {name: value.detach().clone() for name, value in template.state_dict().items()}
        schedules = [
            batches(store, fit_indices, config, seed=seed + epoch * 10003, shuffle=True)
            for epoch in range(int(config["training"]["epochs"]))
        ]
        mode_outputs = {}
        training_rows = {}
        hashes_by_mode = {}
        edge_output = None
        for mode in MODES:
            model = make_model(config, mode)
            model.load_state_dict(initial, strict=True)
            hashes_by_mode[mode] = state_sha256(model)
            parameter_counts[mode] = model.parameter_count()
            training_rows[mode] = train_model(model, store, labels, schedules, config, device)
            checkpoints[f"{seed}/{mode}"] = save_checkpoint(
                model, config, seed, mode, proposal_hash, execution_hash
            )
            clean = score(model, store, probe_indices, config, device)
            repeated = score(model, store, probe_indices, config, device)
            deterministic_max = max(deterministic_max, max_difference(clean["scores"], repeated["scores"]))
            centre_max = max(centre_max, float(clean["maximum_abs_correction_sum"]))
            mode_outputs[mode] = clean
            if mode == PRIMARY:
                edge_output = score(
                    model, store, probe_indices, config, device, disable_cross=True
                )
                cross_correction_fraction[str(seed)] = change_fraction(
                    clean["corrections"], edge_output["corrections"]
                )
                cross_order[str(seed)] = order_changes(edge_output, clean)
                query_absent = score(
                    model, store, probe_indices, config, device, query_present=False
                )
                single = score(
                    model, store, store.role_indices("structural_single_repeat"), config, device
                )
                nohistory = score(
                    model, store, store.role_indices("structural_nohistory"), config, device
                )
                nonrepeat = score(
                    model, store, store.role_indices("structural_nonrepeat"), config, device
                )
                query_exact[str(seed)] = all(
                    np.array_equal(a, b)
                    for a, b in zip(query_absent["scores"], query_absent["item_only_scores"])
                )
                single_exact[str(seed)] = all(
                    np.array_equal(a, b) for a, b in zip(single["scores"], single["item_only_scores"])
                )
                nohistory_exact[str(seed)] = all(
                    np.array_equal(a, b) for a, b in zip(nohistory["scores"], nohistory["base_scores"])
                )
                nonrepeat_exact[str(seed)] = all(
                    np.array_equal(a, b) for a, b in zip(nonrepeat["scores"], nonrepeat["base_scores"])
                )
                permutation_max = max(
                    permutation_max,
                    permutation_audit(model, store, probe_indices, config, device),
                )
            del model
            torch.cuda.empty_cache()
        assert edge_output is not None
        initial_hashes[str(seed)] = hashes_by_mode
        outputs[seed] = {"modes": mode_outputs, "edge_ablation": edge_output}
        summaries[str(seed)] = {
            "training": training_rows,
            "matched_initialization": len(set(hashes_by_mode.values())) == 1,
            "cross_edge_correction_change_fraction": cross_correction_fraction[str(seed)],
            "cross_edge_order_changes": cross_order[str(seed)],
        }

    averaged_primary = average_rows([outputs[seed]["modes"][PRIMARY]["scores"] for seed in seeds])
    item_rows = outputs[seeds[0]]["modes"][PRIMARY]["item_only_scores"]
    item_reference = {**outputs[seeds[0]]["modes"][PRIMARY], "scores": item_rows}
    primary_average = {**outputs[seeds[0]]["modes"][PRIMARY], "scores": averaged_primary}
    aggregate_order = order_changes(item_reference, primary_average)
    gate = config["gate"]
    selection_checks = store.selection["checks"]
    a0_checks = {
        "training_finite": all(
            row["finite"] for seed_row in summaries.values() for row in seed_row["training"].values()
        ),
        "gradients_active": all(
            bool(row["nonzero_gradient_parameters"])
            for seed_row in summaries.values()
            for row in seed_row["training"].values()
        ),
        "matched_parameters": len(set(parameter_counts.values())) == 1,
        "matched_initialization": all(len(set(row.values())) == 1 for row in initial_hashes.values()),
        "candidate_centered": centre_max <= float(gate["correction_center_abs_max"]),
        "deterministic": deterministic_max <= float(gate["deterministic_max_abs_difference"]),
        "candidate_permutation": permutation_max <= float(gate["candidate_permutation_max_abs_difference"]),
        "order_active": aggregate_order["any_fraction"] >= float(gate["order_change_fraction_min"]),
        "top10_active": aggregate_order["top10_fraction"] >= float(gate["top10_change_fraction_min"]),
        "cross_edges_change_correction": all(
            value >= float(gate["cross_edge_correction_change_fraction_min"])
            for value in cross_correction_fraction.values()
        ),
        "cross_edges_change_order": all(
            row["any_fraction"] >= float(gate["cross_edge_order_change_fraction_min"])
            for row in cross_order.values()
        ),
        "query_absent_item_only": all(query_exact.values()),
        "single_repeat_item_only": all(single_exact.values()),
        "nohistory_d2p": all(nohistory_exact.values()),
        "nonrepeat_d2p": all(nonrepeat_exact.values()),
        "selection_integrity": (
            selection_checks["c23_delayed_escrow_labels_opened"] is False
            and selection_checks["c24_internal_A_labels_opened"] is False
            and selection_checks["roles_pairwise_disjoint"] is True
            and selection_checks["delayed_multi_pool_exhaustively_partitioned"] is True
            and selection_checks["dev_test_qrels_metrics_read"] is False
        ),
    }
    a0 = {
        "status": "passed" if all(a0_checks.values()) else "failed",
        "checks": a0_checks,
        "order_changes_vs_item_only": aggregate_order,
        "cross_edge_correction_change_fraction": cross_correction_fraction,
        "cross_edge_order_changes": cross_order,
        "maximum_abs_correction_sum": centre_max,
        "deterministic_max_abs_difference": deterministic_max,
        "candidate_permutation_max_abs_difference": permutation_max,
    }
    if not all(a0_checks.values()):
        report = {
            "candidate_id": "c24",
            "status": "failed_A0_terminal",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "A0": a0,
            "seed_summaries": summaries,
            "checkpoints": checkpoints,
            "candidate_hashes": hashes,
            "internal_A_labels_opened": False,
            "escrow_dev_test_opened": False,
            "elapsed_seconds": time.monotonic() - started,
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
        raise RuntimeError("C24 candidate hashes changed before A1")
    metric_arrays: dict[int, dict[str, np.ndarray]] = {}
    label_rows = None
    for seed in seeds:
        rows = {}
        primary = outputs[seed]["modes"][PRIMARY]
        rows["d2p"], label_rows = metric_rows(primary, "base_scores", internal_labels, probe_indices)
        rows["item_only"], _ = metric_rows(primary, "item_only_scores", internal_labels, probe_indices)
        for mode in MODES:
            rows[mode], _ = metric_rows(outputs[seed]["modes"][mode], "scores", internal_labels, probe_indices)
        rows["edge_ablation"], _ = metric_rows(
            outputs[seed]["edge_ablation"], "scores", internal_labels, probe_indices
        )
        metric_arrays[seed] = rows
        summaries[str(seed)]["ndcg10"] = {name: float(value.mean()) for name, value in rows.items()}
        np.savez_compressed(
            root / f"seed_{seed}_request_metrics.npz",
            request_indices=np.asarray(probe_indices),
            **rows,
        )
    averaged = {
        name: np.mean(np.stack([metric_arrays[seed][name] for seed in seeds]), axis=0)
        for name in ("d2p", "item_only", *MODES, "edge_ablation")
    }
    comparisons = compare(
        request_ids,
        averaged[PRIMARY],
        {"item_only": averaged["item_only"], **{mode: averaged[mode] for mode in CONTROLS}},
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
        folds=int(config["evaluation"]["hash_folds"]),
    )
    seed_differences = {
        name: {
            str(seed): float((metric_arrays[seed][PRIMARY] - metric_arrays[seed][name]).mean())
            for seed in seeds
        }
        for name in ("item_only", *CONTROLS)
    }
    edge_retention = retention(
        averaged[PRIMARY] - averaged["item_only"],
        averaged["edge_ablation"] - averaged["item_only"],
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 101,
    )
    assert label_rows is not None
    correction_rows = average_rows(
        [outputs[seed]["modes"][PRIMARY]["corrections"] for seed in seeds]
    )
    clicked = bootstrap(
        clicked_direction(correction_rows, label_rows),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 201,
    )
    a1_checks = {
        "over_item_effect": comparisons["item_only"]["mean"] >= float(gate["ndcg10_delta_over_item_only_min"]),
        "over_item_ci": comparisons["item_only"]["percentile_95_ci"][0] > 0,
        "over_item_all_seeds": all(value > 0 for value in seed_differences["item_only"].values()),
        "over_item_all_folds": all(row["mean_difference"] > 0 for row in comparisons["item_only"]["hash_folds"]),
        "over_controls_effect": all(
            comparisons[mode]["mean"] >= float(gate["ndcg10_delta_over_each_control_min"])
            for mode in CONTROLS
        ),
        "over_controls_ci": all(comparisons[mode]["percentile_95_ci"][0] > 0 for mode in CONTROLS),
        "over_controls_all_seeds": all(
            all(value > 0 for value in seed_differences[mode].values()) for mode in CONTROLS
        ),
        "edge_retention": bool(edge_retention["applicable"])
        and float(edge_retention["retention"]) <= float(gate["corruption_retention_max"]),
        "edge_retention_ci": bool(edge_retention["applicable"])
        and float(edge_retention["percentile_95_ci"][1])
        <= float(gate["corruption_retention_ci_high_max"]),
        "clicked_direction": clicked["percentile_95_ci"][0] > 0,
    }
    report = {
        "candidate_id": "c24",
        "status": "passed" if all(a1_checks.values()) else "failed_A1_terminal",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "A0": a0,
        "A1": {
            "checks": a1_checks,
            "seed_averaged_ndcg10": {name: float(value.mean()) for name, value in averaged.items()},
            "comparisons": comparisons,
            "seed_differences": seed_differences,
            "edge_ablation_retention": edge_retention,
            "clicked_minus_unclicked": clicked,
        },
        "seed_summaries": summaries,
        "checkpoints": checkpoints,
        "candidate_hashes": hashes,
        "internal_A_labels_opened": True,
        "escrow_dev_test_opened": False,
        "elapsed_seconds": time.monotonic() - started,
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
