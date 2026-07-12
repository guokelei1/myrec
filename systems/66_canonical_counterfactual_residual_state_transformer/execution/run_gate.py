"""Train and aggregate C66's exposed-fit counterfactual-state gate."""

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
from model.canonical_residual import (  # noqa: E402
    MODES,
    CanonicalCounterfactualResidualStateTransformer,
    counterfactual_training_loss,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from train.data_bridge import (  # noqa: E402
    C64Store,
    candidate_keys,
    iter_training_batches,
    iter_validation_batches,
    to_device,
)
from train.gate_metrics import compare  # noqa: E402


PRIMARY = "hidden_residual_wrong_neutral"
WRONG = "hidden_residual_wrong_history"
SCORE_NAMES = (
    "base",
    PRIMARY,
    WRONG,
    "hidden_residual_no_wrong",
    "ordinary_factual_wrong_neutral",
    "logit_difference_wrong_neutral",
    "primary_correction",
    "wrong_correction",
)
HISTORY_NAMES = (
    "history_input_ids",
    "history_attention_mask",
    "history_content_mask",
    "history_event_mask",
)
CANDIDATE_NAMES = (
    "candidate_keys",
    "candidate_input_ids",
    "candidate_attention_mask",
    "candidate_content_mask",
    "candidate_mask",
    "base_scores",
    "item_only_scores",
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
) -> CanonicalCounterfactualResidualStateTransformer:
    backbone = AutoModel.from_pretrained(
        REPO_ROOT / config["paths"]["bge_snapshot"], local_files_only=True
    )
    row = config["model"]
    return CanonicalCounterfactualResidualStateTransformer(
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
        null_reference_stop_gradient=bool(row["null_reference_stop_gradient"]),
    ).to(device)


def merge_batches(
    true_batch: Mapping[str, Any],
    wrong_batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output = to_device(true_batch, device)
    output["candidate_keys"] = torch.from_numpy(candidate_keys(true_batch)).to(device)
    for name in HISTORY_NAMES:
        output[f"wrong_{name}"] = torch.from_numpy(
            np.asarray(wrong_batch[name])
        ).to(device)
    return output


def reverse_candidates(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = dict(values)
    permutation = torch.arange(
        values["candidate_mask"].shape[1] - 1,
        -1,
        -1,
        device=values["candidate_mask"].device,
    )
    for name in CANDIDATE_NAMES:
        output[name] = values[name][:, permutation]
    output["labels"] = values["labels"][:, permutation]
    return output


def gradient_groups(
    model: CanonicalCounterfactualResidualStateTransformer,
    active_names: set[str],
) -> dict[str, bool]:
    layers = len(model.backbone_layers())
    first = layers - model.trainable_last_lm_layers
    return {
        "first_adaptive_lm_layer": any(
            name.startswith(f"inner.core.backbone.encoder.layer.{first}.")
            for name in active_names
        ),
        "last_lm_layer": any(
            name.startswith(f"inner.core.backbone.encoder.layer.{layers - 1}.")
            for name in active_names
        ),
        "joint_transformer": any(
            name.startswith("inner.core.joint_transformer.")
            for name in active_names
        ),
        "output_head": any(
            name.startswith("inner.core.output_head.") for name in active_names
        ),
        "residual_norm": (
            any(name.startswith("inner.residual_norm.") for name in active_names)
            if model.mode
            in {"hidden_residual_wrong_neutral", "hidden_residual_no_wrong"}
            else True
        ),
        "frozen_earlier_lm_has_no_gradient": not any(
            name.startswith("inner.core.backbone.encoder.layer.")
            and int(name.split(".")[5]) < first
            for name in active_names
        ),
    }


def train_model(
    model: CanonicalCounterfactualResidualStateTransformer,
    store: C64Store,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(training["backbone_learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(training["head_learning_rate"]),
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )
    losses: list[float] = []
    ranking_losses: list[float] = []
    wrong_losses: list[float] = []
    active_names: set[str] = set()
    steps = 0
    wrong_weight = (
        0.0
        if model.mode == "hidden_residual_no_wrong"
        else float(training["wrong_neutrality_weight"])
    )
    for epoch in range(int(training["epochs"])):
        model.train()
        # Gradients remain enabled, while stochastic LM behavior remains disabled.
        model.backbone.eval()
        sample_rng = np.random.default_rng(seed + epoch * 1009 + 66)
        for batch_indices in iter_training_batches(
            store.train_indices,
            seed=seed + epoch * 1009,
            batch_size=int(training["max_requests_per_batch"]),
        ):
            true_batch = store.collate(
                batch_indices,
                label_access=True,
                history_source="true",
                sampled_candidates=int(config["selection"]["sampled_candidates"]),
                rng=sample_rng,
            )
            wrong_batch = store.collate(
                batch_indices, label_access=False, history_source="wrong"
            )
            tensors = merge_batches(true_batch, wrong_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**tensors)
                loss, components = counterfactual_training_loss(
                    output,
                    tensors["labels"],
                    tensors["candidate_mask"],
                    correction_l2_weight=float(training["correction_l2_weight"]),
                    wrong_neutrality_weight=wrong_weight,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"C66 {model.mode} loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"C66 nonfinite gradient: {name}")
                    if bool(parameter.grad.ne(0).any()):
                        active_names.add(name)
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad],
                float(training["gradient_clip_norm"]),
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ranking_losses.append(float(components["ranking"].detach().cpu()))
            wrong_losses.append(float(components["wrong_neutrality"].detach().cpu()))
            steps += 1
    window = min(50, max(1, len(losses) // 2))
    groups = gradient_groups(model, active_names)
    return {
        "steps": steps,
        "wrong_neutrality_weight": wrong_weight,
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:]))
        < float(np.mean(losses[:window])),
        "ranking_loss_last": float(np.mean(ranking_losses[-window:])),
        "wrong_neutrality_last": float(np.mean(wrong_losses[-window:])),
        "finite": bool(np.isfinite(losses).all()),
        "gradient_groups": groups,
        "all_gradient_groups": all(groups.values()),
        "trainable_parameters": model.trainable_parameter_count(),
    }


def score_model(
    model: CanonicalCounterfactualResidualStateTransformer,
    store: C64Store,
    config: Mapping[str, Any],
    *,
    device: torch.device,
    include_wrong: bool,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    model.eval()
    model.backbone.eval()
    rows: dict[str, list[np.ndarray]] = {
        "true": [],
        "wrong": [],
        "base": [],
        "correction": [],
        "wrong_correction": [],
    }
    deterministic_error = 0.0
    permutation_error = 0.0
    permutation_exact = True
    nohistory_error = 0.0
    repeat_error = 0.0
    first_batch = True
    with torch.inference_mode():
        for batch_indices in iter_validation_batches(
            store,
            store.validation_indices,
            max_requests=int(config["training"]["validation_max_requests_per_batch"]),
            max_sequences=int(config["training"]["max_encoded_sequences_per_batch"]),
        ):
            true_batch = store.collate(
                batch_indices, label_access=False, history_source="true"
            )
            wrong_batch = store.collate(
                batch_indices, label_access=False, history_source="wrong"
            )
            tensors = merge_batches(true_batch, wrong_batch, device)
            # Frozen full-candidate scoring is deliberately fp32.
            output = model(**tensors)
            if first_batch:
                repeated = model(**tensors)
                reversed_tensors = reverse_candidates(tensors)
                reversed_output = model(**reversed_tensors)
                permutation = torch.arange(
                    output.scores.shape[1] - 1, -1, -1, device=device
                )
                restored = reversed_output.scores[:, permutation]
                deterministic_error = float(
                    (output.scores - repeated.scores).abs().max().cpu()
                )
                permutation_error = float(
                    (output.scores - restored).abs().max().cpu()
                )
                permutation_exact = torch.equal(output.scores, restored)
                empty_tensors = dict(tensors)
                empty_tensors["history_event_mask"] = torch.zeros_like(
                    tensors["history_event_mask"]
                )
                empty_output = model(**empty_tensors)
                nohistory_error = float(
                    (empty_output.scores - tensors["base_scores"]).abs().max().cpu()
                )
                repeat_tensors = dict(tensors)
                repeat_tensors["repeat_request"] = torch.ones_like(
                    tensors["repeat_request"]
                )
                repeat_tensors["item_only_scores"] = torch.randn_like(
                    tensors["item_only_scores"]
                ).masked_fill(~tensors["candidate_mask"], 0.0)
                repeat_output = model(**repeat_tensors)
                repeat_error = float(
                    (
                        repeat_output.scores
                        - repeat_tensors["item_only_scores"]
                    )
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
                if include_wrong:
                    rows["wrong"].append(
                        output.wrong_scores[row, :count].float().cpu().numpy()
                    )
                    rows["wrong_correction"].append(
                        output.wrong_correction[row, :count].float().cpu().numpy()
                    )
    return rows, {
        "precision": "fp32",
        "deterministic_max_abs": deterministic_error,
        "candidate_permutation_max_abs": permutation_error,
        "candidate_permutation_bit_exact": bool(permutation_exact),
        "nohistory_max_abs": nohistory_error,
        "repeat_max_abs": repeat_error,
        "validation_labels_opened": False,
    }


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.concatenate(rows).astype(np.float32, copy=False),
    )


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(
            values[int(offsets[row]) : int(offsets[row + 1])], dtype=np.float32
        ).copy()
        for row in range(len(offsets) - 1)
    ]


def ranking(
    request_id: str, item_ids: Sequence[str], values: np.ndarray
) -> list[str]:
    return [
        row.item_id
        for row in sort_candidates(
            request_id,
            [
                ScoredCandidate(str(item), float(score))
                for item, score in zip(item_ids, values)
            ],
        )
    ]


def activity(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
) -> dict[str, Any]:
    order_changes: list[bool] = []
    top10_changes: list[bool] = []
    for request_id, items, left, right in zip(
        request_ids, item_ids, first, second
    ):
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
        raise ValueError("C66 seed is not registered")
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C66 physical GPU registration differs")
    if (
        str(device) != "cuda:0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("C66 requires one registered visible GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C66 deterministic CUBLAS setting is absent")
    store = C64Store(config, REPO_ROOT)
    expected_candidate_hash = json.loads(
        (REPO_ROOT / config["paths"]["c64_split_manifest"]).read_text(
            encoding="utf-8"
        )
    )["validation_candidate_hash"]
    if store.candidate_hash(store.validation_indices) != expected_candidate_hash:
        raise RuntimeError("C66 validation candidate hash differs before training")

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
        print(f"C66 seed={seed} mode={mode} trained", flush=True)
        checkpoint = checkpoint_root / f"seed_{seed}_{mode}.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        torch.save(
            {
                "candidate_id": "c66",
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
            include_wrong=mode == PRIMARY,
        )
        score_reports[mode] = score_report
        all_rows[mode] = rows["true"]
        if mode == PRIMARY:
            all_rows["base"] = rows["base"]
            all_rows[WRONG] = rows["wrong"]
            all_rows["primary_correction"] = rows["correction"]
            all_rows["wrong_correction"] = rows["wrong_correction"]
        print(f"C66 seed={seed} mode={mode} scored", flush=True)
        del model
        torch.cuda.empty_cache()

    if store.candidate_hash(store.validation_indices) != expected_candidate_hash:
        raise RuntimeError("C66 validation candidate hash differs after scoring")
    offsets, _ = flatten(all_rows["base"])
    score_path = root / f"seed_{seed}_scores.npz"
    seed_report_path = root / f"seed_{seed}_report.json"
    if score_path.exists() or seed_report_path.exists():
        raise FileExistsError(
            score_path if score_path.exists() else seed_report_path
        )
    with score_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=np.asarray(store.validation_indices, dtype=np.int64),
            offsets=offsets,
            **{name: flatten(all_rows[name])[1] for name in SCORE_NAMES},
        )
    request_ids = [
        store.data.request_ids[index] for index in store.validation_indices
    ]
    item_ids = [
        store.data.candidate_ids(index) for index in store.validation_indices
    ]
    activities = {
        "primary_vs_base": activity(
            request_ids, item_ids, all_rows[PRIMARY], all_rows["base"]
        ),
        "true_vs_wrong": activity(
            request_ids, item_ids, all_rows[PRIMARY], all_rows[WRONG]
        ),
        **{
            f"primary_vs_{mode}": activity(
                request_ids, item_ids, all_rows[PRIMARY], all_rows[mode]
            )
            for mode in MODES
            if mode != PRIMARY
        },
    }
    evaluation = config["evaluation"]
    mechanics = {
        "all_training_finite": all(
            value["finite"] for value in training_reports.values()
        ),
        "all_loss_decreased": all(
            value["loss_decreased"] for value in training_reports.values()
        ),
        "all_gradient_groups": all(
            value["all_gradient_groups"] for value in training_reports.values()
        ),
        "deterministic": all(
            value["deterministic_max_abs"]
            <= float(evaluation["deterministic_tolerance"])
            for value in score_reports.values()
        ),
        "candidate_permutation": all(
            value["candidate_permutation_max_abs"]
            <= float(evaluation["candidate_permutation_tolerance"])
            and value["candidate_permutation_bit_exact"]
            for value in score_reports.values()
        ),
        "nohistory_exact": all(
            value["nohistory_max_abs"]
            <= float(evaluation["exact_fallback_tolerance"])
            for value in score_reports.values()
        ),
        "repeat_exact": all(
            value["repeat_max_abs"]
            <= float(evaluation["exact_fallback_tolerance"])
            for value in score_reports.values()
        ),
        "candidate_hash": True,
        "validation_labels_closed_during_scoring": True,
        "fresh_dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c66",
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
    reports: list[tuple[Path, dict[str, Any]]] = []
    for seed in config["training"]["seeds"]:
        path = root / f"seed_{int(seed)}_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        score_path = REPO_ROOT / report["scores"]["path"]
        if sha256_file(score_path) != report["scores"]["sha256"]:
            raise RuntimeError("C66 score artifact changed before A0")
        reports.append((path, report))
    evaluation = config["evaluation"]
    checks = {
        "three_registered_seed_reports": len(reports) == 3,
        "every_seed_scored": all(
            report["status"] == "scored" for _, report in reports
        ),
        "same_execution_lock": all(
            report["execution_lock_sha256"] == execution_lock_hash
            for _, report in reports
        ),
        "same_candidate_hash": len(
            {report["validation_candidate_hash"] for _, report in reports}
        )
        == 1,
        "every_seed_mechanics_pass": all(
            all(report["mechanics"].values()) for _, report in reports
        ),
        "every_seed_primary_active": all(
            report["activity"]["primary_vs_base"]["order_change_fraction"]
            >= float(evaluation["active_order_change_fraction_min"])
            and report["activity"]["primary_vs_base"]["top10_change_fraction"]
            >= float(evaluation["active_top10_change_fraction_min"])
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
            report["fresh_dev_test_qrels_opened"] is False
            for _, report in reports
        ),
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c66",
        "created_at": timestamp(),
        "stage": "A0_label_release_gate",
        "status": "passed" if passed else "failed_terminal",
        "decision": "authorize_exposed_validation_labels"
        if passed
        else "close_c66_before_validation_labels",
        "execution_lock_sha256": execution_lock_hash,
        "checks": checks,
        "seed_activity": {
            str(report["seed"]): report["activity"]
            for _, report in reports
        },
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
    for request_id, items, values, label in zip(
        request_ids, item_ids, scores, labels
    ):
        ranked = ranking(request_id, items, values)
        positive = {
            str(item) for item, value in zip(items, label) if value > 0
        }
        output.append(ndcg_at_k(ranked, positive, 10))
    return np.asarray(output, dtype=np.float64)


def mean_rows(seed_rows: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [
        np.mean(
            np.stack([rows[position] for rows in seed_rows]), axis=0
        ).astype(np.float32)
        for position in range(len(seed_rows[0]))
    ]


def a1(config: Mapping[str, Any]) -> dict[str, Any]:
    _, execution_lock_hash = verify_execution_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    gate = json.loads((root / "a0_report.json").read_text(encoding="utf-8"))
    if gate["status"] != "passed" or gate["execution_lock_sha256"] != execution_lock_hash:
        raise PermissionError("C66 validation labels are not authorized")
    store = C64Store(config, REPO_ROOT)
    expected_hash = json.loads(
        (REPO_ROOT / config["paths"]["c64_split_manifest"]).read_text(
            encoding="utf-8"
        )
    )["validation_candidate_hash"]
    if store.candidate_hash(store.validation_indices) != expected_hash:
        raise RuntimeError("C66 candidate hash differs before A1")
    seeds = [int(value) for value in config["training"]["seeds"]]
    score_sets = [
        load_scores(root / f"seed_{seed}_scores.npz") for seed in seeds
    ]
    for score in score_sets[1:]:
        if any(
            not np.array_equal(left, right)
            for left, right in zip(score_sets[0]["base"], score["base"])
        ):
            raise RuntimeError("C66 base differs across seeds")
    request_ids = [
        store.data.request_ids[index] for index in store.validation_indices
    ]
    item_ids = [
        store.data.candidate_ids(index) for index in store.validation_indices
    ]
    labels = [store.labels(index) for index in store.validation_indices]
    ensemble = {
        name: mean_rows([score[name] for score in score_sets])
        for name in SCORE_NAMES
    }
    metric_names = ("base", PRIMARY, WRONG, *[mode for mode in MODES if mode != PRIMARY])
    ndcg = {
        name: ndcg_rows(request_ids, item_ids, ensemble[name], labels)
        for name in metric_names
    }
    references = {
        "base": ndcg["base"],
        "wrong_history": ndcg[WRONG],
        "hidden_residual_no_wrong": ndcg["hidden_residual_no_wrong"],
        "ordinary_factual_wrong_neutral": ndcg[
            "ordinary_factual_wrong_neutral"
        ],
        "logit_difference_wrong_neutral": ndcg[
            "logit_difference_wrong_neutral"
        ],
    }
    evaluation = config["evaluation"]
    comparisons = compare(
        request_ids,
        ndcg[PRIMARY],
        references,
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]),
        folds=int(evaluation["hash_folds"]),
    )
    seed_differences: dict[str, dict[str, float]] = {}
    for seed, scores in zip(seeds, score_sets):
        seed_ndcg = {
            name: ndcg_rows(request_ids, item_ids, scores[name], labels)
            for name in metric_names
        }
        seed_differences[str(seed)] = {
            "base": float((seed_ndcg[PRIMARY] - seed_ndcg["base"]).mean()),
            "wrong_history": float((seed_ndcg[PRIMARY] - seed_ndcg[WRONG]).mean()),
            "hidden_residual_no_wrong": float(
                (
                    seed_ndcg[PRIMARY]
                    - seed_ndcg["hidden_residual_no_wrong"]
                ).mean()
            ),
            "ordinary_factual_wrong_neutral": float(
                (
                    seed_ndcg[PRIMARY]
                    - seed_ndcg["ordinary_factual_wrong_neutral"]
                ).mean()
            ),
            "logit_difference_wrong_neutral": float(
                (
                    seed_ndcg[PRIMARY]
                    - seed_ndcg["logit_difference_wrong_neutral"]
                ).mean()
            ),
        }
    thresholds = {
        "base": float(evaluation["primary_minus_base_min"]),
        "wrong_history": float(evaluation["true_minus_wrong_min"]),
        "hidden_residual_no_wrong": float(
            evaluation["primary_minus_each_control_min"]
        ),
        "ordinary_factual_wrong_neutral": float(
            evaluation["primary_minus_each_control_min"]
        ),
        "logit_difference_wrong_neutral": float(
            evaluation["primary_minus_each_control_min"]
        ),
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
        "candidate_id": "c66",
        "created_at": timestamp(),
        "stage": "exposed_fit_counterfactual_state_A1",
        "status": "passed" if passed else "failed_terminal",
        "decision": "authorize_same_primitive_cross_domain_probe"
        if passed
        else "close_c66_without_fresh_label_access",
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
    parser.add_argument(
        "--config", type=Path, default=SYSTEM_ROOT / "configs/train_gate.yaml"
    )
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
