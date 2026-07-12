"""Train and aggregate C64's exposed-fit end-to-end LM probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModel


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for value in (str(SYSTEM_ROOT), str(REPO_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from execution.locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    timestamp,
    verify_execution_lock,
)
from model.adaptive_joint_ranker import (  # noqa: E402
    AdaptiveJointLMRanker,
    MODES,
    listwise_loss,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from train.data import (  # noqa: E402
    C64Store,
    iter_training_batches,
    iter_validation_batches,
    to_device,
)
from train.gate_metrics import compare  # noqa: E402


FORWARD_NAMES = (
    "query_input_ids",
    "query_attention_mask",
    "query_content_mask",
    "candidate_input_ids",
    "candidate_attention_mask",
    "candidate_content_mask",
    "history_input_ids",
    "history_attention_mask",
    "history_content_mask",
    "history_event_mask",
    "candidate_mask",
    "base_scores",
    "item_only_scores",
    "repeat_request",
    "query_present",
)

SCORE_NAMES = (
    "base",
    "adaptive_history_lm",
    "adaptive_wrong_history",
    "adaptive_query_candidate_lm",
    "frozen_history_lm",
    "primary_correction",
    "wrong_correction",
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


def make_model(
    config: Mapping[str, Any], *, mode: str, device: torch.device
) -> AdaptiveJointLMRanker:
    backbone = AutoModel.from_pretrained(
        REPO_ROOT / config["paths"]["bge_snapshot"], local_files_only=True
    )
    row = config["model"]
    return AdaptiveJointLMRanker(
        backbone=backbone,
        mode=mode,
        trainable_last_lm_layers=int(row["trainable_last_lm_layers"]),
        input_dim=int(row["input_dim"]),
        hidden_dim=int(row["joint_hidden_dim"]),
        heads=int(row["joint_heads"]),
        layers=int(row["joint_layers"]),
        ffn_dim=int(row["joint_ffn_dim"]),
        dropout=float(row["dropout"]),
        max_history=int(config["selection"]["max_history"]),
        zero_initial_output=bool(row["zero_initial_output"]),
    ).to(device)


def forward_kwargs(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensors[name] for name in FORWARD_NAMES}


def gradient_groups(
    model: AdaptiveJointLMRanker, active_names: set[str]
) -> dict[str, bool]:
    layers = len(model._backbone_layers())
    first = layers - int(model.trainable_last_lm_layers)
    adaptive = model.mode != "frozen_history_lm"
    return {
        "first_adaptive_lm_layer": (
            any(name.startswith(f"backbone.encoder.layer.{first}.") for name in active_names)
            if adaptive
            else True
        ),
        "last_lm_layer": (
            any(name.startswith(f"backbone.encoder.layer.{layers - 1}.") for name in active_names)
            if adaptive
            else True
        ),
        "joint_transformer": any(
            name.startswith("joint_transformer.") for name in active_names
        ),
        "output_head": any(name.startswith("output_head.") for name in active_names),
        "frozen_earlier_lm_has_no_gradient": not any(
            name.startswith("backbone.encoder.layer.")
            and int(name.split(".")[3]) < first
            for name in active_names
        ),
        "frozen_mode_has_no_backbone_gradient": (
            not any(name.startswith("backbone.") for name in active_names)
            if not adaptive
            else True
        ),
    }


def train_model(
    model: AdaptiveJointLMRanker,
    store: C64Store,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    groups: list[dict[str, Any]] = []
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": float(training["backbone_learning_rate"]),
            }
        )
    groups.append(
        {"params": head_parameters, "lr": float(training["head_learning_rate"])}
    )
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(training["weight_decay"])
    )
    losses: list[float] = []
    active_names: set[str] = set()
    steps = 0
    for epoch in range(int(training["epochs"])):
        model.train()
        if model.mode == "frozen_history_lm":
            model.backbone.eval()
        sample_rng = np.random.default_rng(seed + epoch * 1009 + 64)
        for batch_indices in iter_training_batches(
            store.train_indices,
            seed=seed + epoch * 1009,
            batch_size=int(training["max_requests_per_batch"]),
        ):
            batch = store.collate(
                batch_indices,
                label_access=True,
                history_source="true",
                sampled_candidates=int(config["selection"]["sampled_candidates"]),
                rng=sample_rng,
            )
            tensors = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**forward_kwargs(tensors))
                loss, _ = listwise_loss(
                    output,
                    tensors["labels"],
                    tensors["candidate_mask"],
                    correction_l2_weight=float(training["correction_l2_weight"]),
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"C64 {model.mode} loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"C64 nonfinite gradient: {name}")
                    if bool(parameter.grad.ne(0).any()):
                        active_names.add(name)
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad],
                float(training["gradient_clip_norm"]),
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
    window = min(50, max(1, len(losses) // 2))
    groups_report = gradient_groups(model, active_names)
    return {
        "steps": steps,
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:]))
        < float(np.mean(losses[:window])),
        "finite": bool(np.isfinite(losses).all()),
        "gradient_groups": groups_report,
        "all_gradient_groups": all(groups_report.values()),
        "trainable_parameters": model.trainable_parameter_count(),
    }


def _reverse_candidates(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = dict(tensors)
    permutation = torch.arange(
        tensors["candidate_mask"].shape[1] - 1,
        -1,
        -1,
        device=tensors["candidate_mask"].device,
    )
    for name in (
        "candidate_input_ids",
        "candidate_attention_mask",
        "candidate_content_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
        "labels",
    ):
        output[name] = tensors[name][:, permutation]
    return output


def score_model(
    model: AdaptiveJointLMRanker,
    store: C64Store,
    config: Mapping[str, Any],
    *,
    device: torch.device,
    include_wrong: bool,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    model.eval()
    rows: dict[str, list[np.ndarray]] = {
        "true": [], "wrong": [], "base": [], "correction": [], "wrong_correction": []
    }
    deterministic_error = 0.0
    permutation_error = 0.0
    first_batch = True
    with torch.inference_mode():
        for batch_indices in iter_validation_batches(
            store,
            store.validation_indices,
            max_requests=int(config["training"]["validation_max_requests_per_batch"]),
            max_sequences=int(config["training"]["max_encoded_sequences_per_batch"]),
        ):
            batch = store.collate(
                batch_indices, label_access=False, history_source="true"
            )
            tensors = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**forward_kwargs(tensors))
            wrong_output = None
            if include_wrong:
                wrong_batch = store.collate(
                    batch_indices, label_access=False, history_source="wrong"
                )
                wrong_tensors = to_device(wrong_batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wrong_output = model(**forward_kwargs(wrong_tensors))
            if first_batch:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    repeated = model(**forward_kwargs(tensors))
                    reversed_tensors = _reverse_candidates(tensors)
                    reversed_output = model(**forward_kwargs(reversed_tensors))
                permutation = torch.arange(
                    output.scores.shape[1] - 1, -1, -1, device=device
                )
                deterministic_error = float(
                    (output.scores - repeated.scores).abs().max().cpu()
                )
                permutation_error = float(
                    (output.scores - reversed_output.scores[:, permutation])
                    .abs()
                    .max()
                    .cpu()
                )
                first_batch = False
            for row, count in enumerate(tensors["candidate_mask"].sum(-1).tolist()):
                count = int(count)
                rows["true"].append(
                    output.scores[row, :count].float().cpu().numpy()
                )
                rows["base"].append(
                    tensors["base_scores"][row, :count].float().cpu().numpy()
                )
                rows["correction"].append(
                    output.correction[row, :count].float().cpu().numpy()
                )
                if wrong_output is not None:
                    rows["wrong"].append(
                        wrong_output.scores[row, :count].float().cpu().numpy()
                    )
                    rows["wrong_correction"].append(
                        wrong_output.correction[row, :count].float().cpu().numpy()
                    )
    return rows, {
        "deterministic_max_abs": deterministic_error,
        "candidate_permutation_max_abs": permutation_error,
        "validation_labels_opened": False,
    }


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    values = np.concatenate(rows).astype(np.float32, copy=False)
    return np.asarray(offsets, dtype=np.int64), values


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[row]) : int(offsets[row + 1])], dtype=np.float32).copy()
        for row in range(len(offsets) - 1)
    ]


def ranking(request_id: str, item_ids: Sequence[str], values: np.ndarray) -> list[str]:
    return [
        row.item_id
        for row in sort_candidates(
            request_id,
            [ScoredCandidate(str(item), float(score)) for item, score in zip(item_ids, values)],
        )
    ]


def activity(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
) -> dict[str, Any]:
    order_changes = []
    top10_changes = []
    for request_id, items, left, right in zip(request_ids, item_ids, first, second):
        first_rank = ranking(request_id, items, left)
        second_rank = ranking(request_id, items, right)
        order_changes.append(first_rank != second_rank)
        top10_changes.append(set(first_rank[:10]) != set(second_rank[:10]))
    return {
        "requests": len(order_changes),
        "order_change_count": int(sum(order_changes)),
        "order_change_fraction": float(np.mean(order_changes)),
        "top10_change_count": int(sum(top10_changes)),
        "top10_change_fraction": float(np.mean(top10_changes)),
    }


def run_seed(
    config: Mapping[str, Any], *, seed: int, device: torch.device
) -> dict[str, Any]:
    _, execution_lock_hash = verify_execution_lock(config)
    if seed not in [int(value) for value in config["training"]["seeds"]]:
        raise ValueError("C64 seed is not registered")
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C64 physical GPU registration differs")
    if str(device) != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C64 requires one registered visible GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C64 deterministic CUBLAS setting is absent")
    store = C64Store(config, REPO_ROOT)
    expected_candidate_hash = json.loads(
        (REPO_ROOT / config["paths"]["artifact_root"] / "split_manifest.json").read_text(
            encoding="utf-8"
        )
    )["validation_candidate_hash"]
    if store.candidate_hash(store.validation_indices) != expected_candidate_hash:
        raise RuntimeError("C64 validation candidate hash differs before training")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[np.ndarray]] = {}
    training_reports: dict[str, Any] = {}
    score_reports: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    for mode in MODES:
        seed_all(seed)
        model = make_model(config, mode=mode, device=device)
        training_reports[mode] = train_model(
            model, store, config, seed=seed, device=device
        )
        print(f"C64 seed={seed} mode={mode} trained", flush=True)
        checkpoint = checkpoint_root / f"seed_{seed}_{mode}.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        torch.save(
            {
                "candidate_id": "c64",
                "seed": seed,
                "mode": mode,
                "execution_lock_sha256": execution_lock_hash,
                "state_dict": model.state_dict(),
            },
            checkpoint,
        )
        checkpoints[mode] = {
            "path": str(checkpoint.relative_to(REPO_ROOT)),
            "sha256": sha256_file(checkpoint),
        }
        rows, score_report = score_model(
            model,
            store,
            config,
            device=device,
            include_wrong=mode == "adaptive_history_lm",
        )
        score_reports[mode] = score_report
        all_rows[mode] = rows["true"]
        if mode == "adaptive_history_lm":
            all_rows["base"] = rows["base"]
            all_rows["adaptive_wrong_history"] = rows["wrong"]
            all_rows["primary_correction"] = rows["correction"]
            all_rows["wrong_correction"] = rows["wrong_correction"]
        print(f"C64 seed={seed} mode={mode} scored", flush=True)
        del model
        torch.cuda.empty_cache()
    if store.candidate_hash(store.validation_indices) != expected_candidate_hash:
        raise RuntimeError("C64 validation candidate hash differs after scoring")
    offsets, _ = flatten(all_rows["base"])
    score_path = root / f"seed_{seed}_scores.npz"
    seed_report_path = root / f"seed_{seed}_report.json"
    if score_path.exists() or seed_report_path.exists():
        raise FileExistsError(score_path if score_path.exists() else seed_report_path)
    with score_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=np.asarray(store.validation_indices, dtype=np.int64),
            offsets=offsets,
            **{name: flatten(all_rows[name])[1] for name in SCORE_NAMES},
        )
    request_ids = [store.data.request_ids[index] for index in store.validation_indices]
    item_ids = [store.data.candidate_ids(index) for index in store.validation_indices]
    activities = {
        "primary_vs_base": activity(
            request_ids, item_ids, all_rows["adaptive_history_lm"], all_rows["base"]
        ),
        "true_vs_wrong": activity(
            request_ids,
            item_ids,
            all_rows["adaptive_history_lm"],
            all_rows["adaptive_wrong_history"],
        ),
        "primary_vs_query_candidate": activity(
            request_ids,
            item_ids,
            all_rows["adaptive_history_lm"],
            all_rows["adaptive_query_candidate_lm"],
        ),
        "primary_vs_frozen_history": activity(
            request_ids,
            item_ids,
            all_rows["adaptive_history_lm"],
            all_rows["frozen_history_lm"],
        ),
    }
    mechanics = {
        "all_training_finite": all(value["finite"] for value in training_reports.values()),
        "all_loss_decreased": all(value["loss_decreased"] for value in training_reports.values()),
        "all_gradient_groups": all(value["all_gradient_groups"] for value in training_reports.values()),
        "deterministic": all(
            value["deterministic_max_abs"]
            <= float(config["evaluation"]["deterministic_tolerance"])
            for value in score_reports.values()
        ),
        "candidate_permutation": all(
            value["candidate_permutation_max_abs"]
            <= float(config["evaluation"]["candidate_permutation_tolerance"])
            for value in score_reports.values()
        ),
        "candidate_hash": True,
        "validation_labels_closed_during_scoring": True,
        "fresh_dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c64",
        "created_at": timestamp(),
        "stage": "exposed_fit_training_and_label_free_validation_scoring",
        "status": "scored" if all(mechanics.values()) else "failed_terminal",
        "seed": seed,
        "physical_gpu": physical,
        "execution_lock_sha256": execution_lock_hash,
        "validation_candidate_hash": expected_candidate_hash,
        "training": training_reports,
        "scoring": score_reports,
        "activity": activities,
        "mechanics": mechanics,
        "checkpoints": checkpoints,
        "scores": {
            "path": str(score_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(score_path),
        },
        "fit_train_labels_opened": True,
        "validation_labels_opened": False,
        "fresh_dev_test_qrels_opened": False,
    }
    atomic_json(seed_report_path, report)
    return report


def a0(config: Mapping[str, Any]) -> dict[str, Any]:
    _, execution_lock_hash = verify_execution_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    target = root / "a0_report.json"
    reports = []
    for seed in config["training"]["seeds"]:
        path = root / f"seed_{int(seed)}_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / report["scores"]["path"]) != report["scores"]["sha256"]:
            raise RuntimeError("C64 score artifact changed before A0")
        reports.append((path, report))
    evaluation = config["evaluation"]
    checks = {
        "three_registered_seed_reports": len(reports) == 3,
        "every_seed_scored": all(report["status"] == "scored" for _, report in reports),
        "same_execution_lock": all(
            report["execution_lock_sha256"] == execution_lock_hash for _, report in reports
        ),
        "same_candidate_hash": len(
            {report["validation_candidate_hash"] for _, report in reports}
        )
        == 1,
        "every_seed_primary_active": all(
            report["activity"]["primary_vs_base"]["order_change_fraction"]
            >= float(evaluation["wrong_order_change_fraction_min"])
            and report["activity"]["primary_vs_base"]["top10_change_fraction"]
            >= float(evaluation["wrong_top10_change_fraction_min"])
            for _, report in reports
        ),
        "every_seed_wrong_history_active": all(
            report["activity"]["true_vs_wrong"]["order_change_fraction"]
            >= float(evaluation["wrong_order_change_fraction_min"])
            and report["activity"]["true_vs_wrong"]["top10_change_fraction"]
            >= float(evaluation["wrong_top10_change_fraction_min"])
            for _, report in reports
        ),
        "validation_labels_closed_during_all_scoring": all(
            report["validation_labels_opened"] is False for _, report in reports
        ),
        "fresh_dev_test_qrels_closed": all(
            report["fresh_dev_test_qrels_opened"] is False for _, report in reports
        ),
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c64",
        "created_at": timestamp(),
        "stage": "A0_label_release_gate",
        "status": "passed" if passed else "failed_terminal",
        "decision": "authorize_exposed_validation_labels"
        if passed
        else "close_c64_before_validation_labels",
        "execution_lock_sha256": execution_lock_hash,
        "checks": checks,
        "seed_reports": {
            str(report["seed"]): {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }
            for path, report in reports
        },
        "validation_labels_opened": False,
        "fresh_dev_test_qrels_opened": False,
    }
    atomic_json(target, value)
    return value


def load_scores(path: Path) -> dict[str, list[np.ndarray]]:
    with np.load(path, allow_pickle=False) as values:
        offsets = np.asarray(values["offsets"], dtype=np.int64)
        return {name: unflatten(offsets, values[name]) for name in SCORE_NAMES}


def ndcg_rows(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
) -> np.ndarray:
    output = []
    for request_id, items, values, label in zip(request_ids, item_ids, scores, labels):
        ranked = ranking(request_id, items, values)
        positive = {str(item) for item, value in zip(items, label) if value > 0}
        output.append(ndcg_at_k(ranked, positive, 10))
    return np.asarray(output, dtype=np.float64)


def mean_rows(seed_rows: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [
        np.mean(np.stack([rows[position] for rows in seed_rows]), axis=0).astype(np.float32)
        for position in range(len(seed_rows[0]))
    ]


def a1(config: Mapping[str, Any]) -> dict[str, Any]:
    _, execution_lock_hash = verify_execution_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    gate = json.loads((root / "a0_report.json").read_text(encoding="utf-8"))
    if gate["status"] != "passed" or gate["execution_lock_sha256"] != execution_lock_hash:
        raise PermissionError("C64 validation labels are not authorized")
    store = C64Store(config, REPO_ROOT)
    expected_hash = json.loads(
        (root / "split_manifest.json").read_text(encoding="utf-8")
    )["validation_candidate_hash"]
    if store.candidate_hash(store.validation_indices) != expected_hash:
        raise RuntimeError("C64 candidate hash differs before A1")
    seeds = [int(value) for value in config["training"]["seeds"]]
    score_sets = [load_scores(root / f"seed_{seed}_scores.npz") for seed in seeds]
    for score in score_sets[1:]:
        if any(
            not np.array_equal(left, right)
            for left, right in zip(score_sets[0]["base"], score["base"])
        ):
            raise RuntimeError("C64 base differs across seeds")
    request_ids = [store.data.request_ids[index] for index in store.validation_indices]
    item_ids = [store.data.candidate_ids(index) for index in store.validation_indices]
    labels = [store.labels(index) for index in store.validation_indices]
    ensemble = {
        name: mean_rows([score[name] for score in score_sets])
        for name in SCORE_NAMES
    }
    ndcg = {
        name: ndcg_rows(request_ids, item_ids, values, labels)
        for name, values in ensemble.items()
        if name not in {"primary_correction", "wrong_correction"}
    }
    evaluation = config["evaluation"]
    comparisons = compare(
        request_ids,
        ndcg["adaptive_history_lm"],
        {
            "base": ndcg["base"],
            "adaptive_query_candidate_lm": ndcg["adaptive_query_candidate_lm"],
            "frozen_history_lm": ndcg["frozen_history_lm"],
            "wrong_history": ndcg["adaptive_wrong_history"],
        },
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]),
        folds=int(evaluation["hash_folds"]),
    )
    seed_differences: dict[str, dict[str, float]] = {}
    for seed, scores in zip(seeds, score_sets):
        seed_ndcg = {
            name: ndcg_rows(request_ids, item_ids, scores[name], labels)
            for name in (
                "base",
                "adaptive_history_lm",
                "adaptive_query_candidate_lm",
                "frozen_history_lm",
                "adaptive_wrong_history",
            )
        }
        seed_differences[str(seed)] = {
            "base": float(
                (seed_ndcg["adaptive_history_lm"] - seed_ndcg["base"]).mean()
            ),
            "adaptive_query_candidate_lm": float(
                (
                    seed_ndcg["adaptive_history_lm"]
                    - seed_ndcg["adaptive_query_candidate_lm"]
                ).mean()
            ),
            "frozen_history_lm": float(
                (
                    seed_ndcg["adaptive_history_lm"]
                    - seed_ndcg["frozen_history_lm"]
                ).mean()
            ),
            "wrong_history": float(
                (
                    seed_ndcg["adaptive_history_lm"]
                    - seed_ndcg["adaptive_wrong_history"]
                ).mean()
            ),
        }
    thresholds = {
        "base": float(evaluation["primary_minus_base_min"]),
        "adaptive_query_candidate_lm": float(
            evaluation["primary_minus_query_candidate_min"]
        ),
        "frozen_history_lm": float(evaluation["primary_minus_frozen_history_min"]),
        "wrong_history": float(evaluation["true_minus_wrong_min"]),
    }
    checks: dict[str, bool] = {
        "candidate_hash_asserted": store.candidate_hash(store.validation_indices)
        == expected_hash,
        "validation_labels_opened_only_after_A0": True,
        "fresh_dev_test_qrels_closed": True,
    }
    for name, threshold in thresholds.items():
        row = comparisons[name]
        checks[f"{name}_mean_threshold"] = row["mean"] >= threshold
        checks[f"{name}_positive_interval"] = row["percentile_95_ci"][0] > 0.0
        checks[f"{name}_each_seed_positive"] = all(
            seed_differences[str(seed)][name] > 0.0 for seed in seeds
        )
        checks[f"{name}_two_of_three_folds_positive"] = sum(
            fold["mean_difference"] > 0.0 for fold in row["hash_folds"]
        ) >= 2
    passed = all(checks.values())
    result = {
        "candidate_id": "c64",
        "created_at": timestamp(),
        "stage": "exposed_fit_representation_learnability_A1",
        "status": "passed" if passed else "failed_terminal",
        "decision": "authorize_same_probe_on_amazon_exposed_fit"
        if passed
        else "close_c64_without_fresh_label_access",
        "execution_lock_sha256": execution_lock_hash,
        "validation_requests": len(request_ids),
        "candidate_hash": expected_hash,
        "metrics": {name: float(values.mean()) for name, values in ndcg.items()},
        "comparisons": comparisons,
        "seed_differences": seed_differences,
        "checks": checks,
        "fit_train_labels_opened": True,
        "validation_exposed_fit_labels_opened_after_A0": True,
        "fresh_features_scores_labels_opened": False,
        "dev_test_qrels_opened": False,
    }
    target = REPO_ROOT / config["paths"]["promoted_report"]
    atomic_json(target, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/kuai_probe.yaml")
    parser.add_argument("--stage", choices=("seed", "a0", "a1"), required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "seed":
        if args.seed is None:
            parser.error("--seed is required for seed stage")
        value = run_seed(config, seed=args.seed, device=torch.device("cuda:0"))
    elif args.stage == "a0":
        value = a0(config)
    else:
        value = a1(config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
