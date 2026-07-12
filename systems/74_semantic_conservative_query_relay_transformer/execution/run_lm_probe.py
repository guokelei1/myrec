"""Train, label-free gate, and evaluate C74's exposed-fit LM probe."""

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

from execution.lm_locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    timestamp,
    verify_execution_lock,
)
from model.adaptive_semantic_relay import (  # noqa: E402
    MODES,
    PRIMARY,
    AdaptiveSemanticRelayLMRanker,
    listwise_loss,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from train.data_bridge import (  # noqa: E402
    C74Store,
    iter_training_batches,
    iter_validation_batches,
    to_device,
)
from train.gate_metrics import bootstrap, compare  # noqa: E402


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
    PRIMARY,
    "wrong_history",
    "coupled_value_relay",
    "pooled_semantic_relay",
    "factual_semantic_relay",
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
) -> AdaptiveSemanticRelayLMRanker:
    backbone = AutoModel.from_pretrained(
        REPO_ROOT / config["paths"]["bge_snapshot"], local_files_only=True
    )
    row = config["model"]
    return AdaptiveSemanticRelayLMRanker(
        backbone=backbone,
        mode=mode,
        trainable_last_lm_layers=int(row["trainable_last_lm_layers"]),
        input_dim=int(row["input_dim"]),
        route_rank=int(row["route_rank"]),
        max_history=int(config["selection"]["max_history"]),
        temperature=float(row["temperature"]),
        profile_scale=float(row["profile_scale"]),
        correction_scale=float(row["correction_scale"]),
        route_init_std=float(row["route_init_std"]),
    ).to(device)


def forward_kwargs(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensors[name] for name in FORWARD_NAMES}


def gradient_groups(
    model: AdaptiveSemanticRelayLMRanker, active: set[str]
) -> dict[str, bool]:
    layers = len(model._backbone_layers())
    first = layers - model.trainable_last_lm_layers
    return {
        "first_adaptive_lm_layer": any(name.startswith(f"backbone.encoder.layer.{first}.") for name in active),
        "last_lm_layer": any(name.startswith(f"backbone.encoder.layer.{layers - 1}.") for name in active),
        "history_route_down": "history_route.down.weight" in active,
        "history_route_up": "history_route.up.weight" in active,
        "candidate_route_down": "candidate_route.down.weight" in active,
        "candidate_route_up": "candidate_route.up.weight" in active,
        "chronology_bias": "chronology_bias" in active,
        "frozen_earlier_lm": not any(
            name.startswith("backbone.encoder.layer.")
            and int(name.split(".")[3]) < first
            for name in active
        ),
    }


def train_model(
    model: AdaptiveSemanticRelayLMRanker,
    store: C74Store,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    route_parameters = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": float(training["backbone_learning_rate"])},
            {"params": route_parameters, "lr": float(training["route_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    losses: list[float] = []
    active: set[str] = set()
    steps = 0
    for epoch in range(int(training["epochs"])):
        model.train()
        sample_rng = np.random.default_rng(seed + epoch * 1009 + 74)
        for indices in iter_training_batches(
            store.train_indices,
            seed=seed + epoch * 1009,
            batch_size=int(training["max_requests_per_batch"]),
        ):
            batch = store.collate(
                indices,
                label_access=True,
                history_source="true",
                sampled_candidates=int(config["selection"]["sampled_candidates"]),
                rng=sample_rng,
            )
            tensors = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**forward_kwargs(tensors))
                loss = listwise_loss(output, tensors["labels"], tensors["candidate_mask"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"C74 {model.mode} nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"C74 nonfinite gradient: {name}")
                    if bool(parameter.grad.ne(0).any()):
                        active.add(name)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(training["gradient_clip_norm"]),
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
    window = min(50, max(1, len(losses) // 2))
    groups = gradient_groups(model, active)
    return {
        "steps": steps,
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:])) < float(np.mean(losses[:window])),
        "finite": bool(np.isfinite(losses).all()),
        "gradient_groups": groups,
        "all_gradient_groups": all(groups.values()),
        "total_parameters": model.parameter_count(),
        "trainable_parameters": model.trainable_parameter_count(),
        "chronology_bias": model.chronology_bias.detach().float().cpu().tolist(),
    }


def _reverse_candidates(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = dict(tensors)
    reverse = torch.arange(
        tensors["candidate_mask"].shape[1] - 1, -1, -1,
        device=tensors["candidate_mask"].device,
    )
    for name in (
        "candidate_input_ids", "candidate_attention_mask", "candidate_content_mask",
        "candidate_mask", "base_scores", "item_only_scores", "labels",
    ):
        output[name] = tensors[name][:, reverse]
    return output


def score_model(
    model: AdaptiveSemanticRelayLMRanker,
    store: C74Store,
    config: Mapping[str, Any],
    *,
    device: torch.device,
    include_wrong: bool,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    model.eval()
    rows = {name: [] for name in (
        "true", "wrong", "base", "correction", "wrong_correction"
    )}
    deterministic_error = permutation_error = 0.0
    nohistory_error = query_error = repeat_error = 0.0
    first_batch = True
    with torch.inference_mode():
        for indices in iter_validation_batches(
            store,
            store.validation_indices,
            max_requests=int(config["training"]["validation_max_requests_per_batch"]),
            max_sequences=int(config["training"]["max_encoded_sequences_per_batch"]),
        ):
            batch = store.collate(indices, label_access=False, history_source="true")
            tensors = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**forward_kwargs(tensors))
            wrong_output = None
            if include_wrong:
                wrong_batch = store.collate(indices, label_access=False, history_source="wrong")
                wrong_tensors = to_device(wrong_batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wrong_output = model(**forward_kwargs(wrong_tensors))
            if first_batch:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    repeated = model(**forward_kwargs(tensors))
                    reversed_tensors = _reverse_candidates(tensors)
                    reversed_output = model(**forward_kwargs(reversed_tensors))
                    empty = dict(tensors); empty["history_event_mask"] = torch.zeros_like(tensors["history_event_mask"])
                    empty_output = model(**forward_kwargs(empty))
                    masked = dict(tensors); masked["query_present"] = torch.zeros_like(tensors["query_present"])
                    masked_output = model(**forward_kwargs(masked))
                    repeat = dict(tensors); repeat["repeat_request"] = torch.ones_like(tensors["repeat_request"])
                    repeat_output = model(**forward_kwargs(repeat))
                reverse = torch.arange(output.scores.shape[1] - 1, -1, -1, device=device)
                deterministic_error = float((output.scores - repeated.scores).abs().max().cpu())
                permutation_error = float((output.scores - reversed_output.scores[:, reverse]).abs().max().cpu())
                mask = tensors["candidate_mask"]
                expected_base = tensors["base_scores"].float().masked_fill(~mask, 0.0)
                expected_item = tensors["item_only_scores"].float().masked_fill(~mask, 0.0)
                nohistory_error = float((empty_output.scores - expected_base).abs().max().cpu())
                query_error = float((masked_output.scores - expected_base).abs().max().cpu())
                repeat_error = float((repeat_output.scores - expected_item).abs().max().cpu())
                first_batch = False
            for row, count in enumerate(tensors["candidate_mask"].sum(-1).tolist()):
                count = int(count)
                rows["true"].append(output.scores[row, :count].float().cpu().numpy())
                rows["base"].append(tensors["base_scores"][row, :count].float().cpu().numpy())
                rows["correction"].append(output.correction[row, :count].float().cpu().numpy())
                if wrong_output is not None:
                    rows["wrong"].append(wrong_output.scores[row, :count].float().cpu().numpy())
                    rows["wrong_correction"].append(wrong_output.correction[row, :count].float().cpu().numpy())
    return rows, {
        "deterministic_max_abs": deterministic_error,
        "candidate_permutation_max_abs": permutation_error,
        "nohistory_max_abs": nohistory_error,
        "query_mask_max_abs": query_error,
        "repeat_max_abs": repeat_error,
        "validation_labels_opened": False,
    }


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, dtype=np.int64), np.concatenate(rows).astype(np.float32, copy=False)


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[i]) : int(offsets[i + 1])], dtype=np.float32).copy()
        for i in range(len(offsets) - 1)
    ]


def ranking(request_id: str, item_ids: Sequence[str], values: np.ndarray) -> list[str]:
    return [
        row.item_id for row in sort_candidates(
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
    order = []; top10 = []
    for request_id, items, left, right in zip(request_ids, item_ids, first, second):
        first_rank = ranking(request_id, items, left)
        second_rank = ranking(request_id, items, right)
        order.append(first_rank != second_rank)
        top10.append(set(first_rank[:10]) != set(second_rank[:10]))
    return {
        "requests": len(order),
        "order_change_count": int(sum(order)),
        "order_change_fraction": float(np.mean(order)),
        "top10_change_count": int(sum(top10)),
        "top10_change_fraction": float(np.mean(top10)),
    }


def run_seed(config: Mapping[str, Any], *, seed: int, device: torch.device) -> dict[str, Any]:
    _, execution_hash = verify_execution_lock(config)
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C74 physical GPU registration differs")
    if str(device) != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C74 requires one registered visible GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C74 deterministic CUBLAS setting absent")
    store = C74Store(config, REPO_ROOT)
    split = json.loads((REPO_ROOT / config["paths"]["artifact_root"] / "split_manifest.json").read_text())
    expected_hash = split["validation_candidate_hash"]
    if store.candidate_hash(store.validation_indices) != expected_hash:
        raise RuntimeError("C74 validation candidate hash differs")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    root.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[np.ndarray]] = {}
    training_reports = {}; scoring_reports = {}; checkpoints = {}
    for mode in MODES:
        seed_all(seed)
        model = make_model(config, mode=mode, device=device)
        training_reports[mode] = train_model(model, store, config, seed=seed, device=device)
        print(f"C74 seed={seed} mode={mode} trained", flush=True)
        checkpoint = checkpoint_root / f"seed_{seed}_{mode}.pt"
        if checkpoint.exists(): raise FileExistsError(checkpoint)
        torch.save({
            "candidate_id": "c74", "seed": seed, "mode": mode,
            "execution_lock_sha256": execution_hash, "state_dict": model.state_dict(),
        }, checkpoint)
        checkpoints[mode] = {"path": str(checkpoint.relative_to(REPO_ROOT)), "sha256": sha256_file(checkpoint)}
        rows, scoring = score_model(model, store, config, device=device, include_wrong=mode == PRIMARY)
        scoring_reports[mode] = scoring; all_rows[mode] = rows["true"]
        if mode == PRIMARY:
            all_rows["base"] = rows["base"]
            all_rows["wrong_history"] = rows["wrong"]
            all_rows["primary_correction"] = rows["correction"]
            all_rows["wrong_correction"] = rows["wrong_correction"]
        print(f"C74 seed={seed} mode={mode} scored", flush=True)
        del model; torch.cuda.empty_cache()
    if store.candidate_hash(store.validation_indices) != expected_hash:
        raise RuntimeError("C74 candidate hash changed after scoring")
    offsets, _ = flatten(all_rows["base"])
    score_path = root / f"seed_{seed}_scores.npz"
    report_path = root / f"seed_{seed}_report.json"
    if score_path.exists() or report_path.exists(): raise FileExistsError(score_path)
    with score_path.open("wb") as handle:
        np.savez(handle, request_indices=np.asarray(store.validation_indices), offsets=offsets,
                 **{name: flatten(all_rows[name])[1] for name in SCORE_NAMES})
    request_ids = [store.data.request_ids[i] for i in store.validation_indices]
    item_ids = [store.data.candidate_ids(i) for i in store.validation_indices]
    activities = {
        "primary_vs_base": activity(request_ids, item_ids, all_rows[PRIMARY], all_rows["base"]),
        "true_vs_wrong": activity(request_ids, item_ids, all_rows[PRIMARY], all_rows["wrong_history"]),
        "primary_vs_coupled": activity(request_ids, item_ids, all_rows[PRIMARY], all_rows["coupled_value_relay"]),
        "primary_vs_pooled": activity(request_ids, item_ids, all_rows[PRIMARY], all_rows["pooled_semantic_relay"]),
        "primary_vs_factual": activity(request_ids, item_ids, all_rows[PRIMARY], all_rows["factual_semantic_relay"]),
    }
    correction = np.concatenate(all_rows["primary_correction"])
    wrong_correction = np.concatenate(all_rows["wrong_correction"])
    mechanics = {
        "all_training_finite": all(row["finite"] for row in training_reports.values()),
        "all_loss_decreased": all(row["loss_decreased"] for row in training_reports.values()),
        "all_gradient_groups": all(row["all_gradient_groups"] for row in training_reports.values()),
        "equal_parameters": len({(row["total_parameters"], row["trainable_parameters"]) for row in training_reports.values()}) == 1,
        "deterministic": all(row["deterministic_max_abs"] <= float(config["evaluation"]["deterministic_tolerance"]) for row in scoring_reports.values()),
        "candidate_permutation": all(row["candidate_permutation_max_abs"] <= float(config["evaluation"]["candidate_permutation_tolerance"]) for row in scoring_reports.values()),
        "nohistory_exact": all(row["nohistory_max_abs"] <= float(config["evaluation"]["exact_fallback_tolerance"]) for row in scoring_reports.values()),
        "query_mask_exact": all(row["query_mask_max_abs"] <= float(config["evaluation"]["exact_fallback_tolerance"]) for row in scoring_reports.values()),
        "repeat_exact": all(row["repeat_max_abs"] <= float(config["evaluation"]["exact_fallback_tolerance"]) for row in scoring_reports.values()),
        "candidate_hash": True,
        "validation_labels_closed_during_scoring": store._labels is None,
        "fresh_dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c74", "created_at": timestamp(),
        "stage": "exposed_fit_training_and_label_free_validation_scoring",
        "status": "scored" if all(mechanics.values()) else "failed_terminal",
        "seed": seed, "physical_gpu": physical,
        "execution_lock_sha256": execution_hash,
        "validation_candidate_hash": expected_hash,
        "training": training_reports, "scoring": scoring_reports,
        "activity": activities,
        "correction": {
            "primary_rms": float(np.sqrt(np.mean(correction ** 2))),
            "true_wrong_difference_rms": float(np.sqrt(np.mean((correction - wrong_correction) ** 2))),
        },
        "mechanics": mechanics, "checkpoints": checkpoints,
        "scores": {"path": str(score_path.relative_to(REPO_ROOT)), "sha256": sha256_file(score_path)},
        "fit_train_labels_opened": True, "validation_labels_opened": False,
        "fresh_dev_test_qrels_opened": False,
    }
    atomic_json(report_path, report)
    return report


def a0(config: Mapping[str, Any]) -> dict[str, Any]:
    _, execution_hash = verify_execution_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    reports = []
    for seed in config["training"]["seeds"]:
        path = root / f"seed_{int(seed)}_report.json"; report = json.loads(path.read_text())
        if sha256_file(REPO_ROOT / report["scores"]["path"]) != report["scores"]["sha256"]:
            raise RuntimeError("C74 score artifact changed before A0")
        reports.append((path, report))
    e = config["evaluation"]
    checks = {
        "three_registered_seed_reports": len(reports) == 3,
        "every_seed_scored": all(report["status"] == "scored" for _, report in reports),
        "same_execution_lock": all(report["execution_lock_sha256"] == execution_hash for _, report in reports),
        "same_candidate_hash": len({report["validation_candidate_hash"] for _, report in reports}) == 1,
        "every_seed_primary_active": all(
            report["activity"]["primary_vs_base"]["order_change_fraction"] >= float(e["primary_order_change_fraction_min"])
            and report["activity"]["primary_vs_base"]["top10_change_fraction"] >= float(e["primary_top10_change_fraction_min"])
            and report["correction"]["primary_rms"] >= float(e["primary_correction_rms_min"])
            for _, report in reports
        ),
        "every_seed_wrong_history_active": all(
            report["activity"]["true_vs_wrong"]["order_change_fraction"] >= float(e["wrong_order_change_fraction_min"])
            and report["activity"]["true_vs_wrong"]["top10_change_fraction"] >= float(e["wrong_top10_change_fraction_min"])
            for _, report in reports
        ),
        "every_seed_control_distinct": all(
            all(
                report["activity"][name]["order_change_fraction"] >= float(e["control_order_change_fraction_min"])
                and report["activity"][name]["top10_change_fraction"] >= float(e["control_top10_change_fraction_min"])
                for name in ("primary_vs_coupled", "primary_vs_pooled", "primary_vs_factual")
            ) for _, report in reports
        ),
        "validation_labels_closed_during_all_scoring": all(not report["validation_labels_opened"] for _, report in reports),
        "fresh_dev_test_qrels_closed": all(not report["fresh_dev_test_qrels_opened"] for _, report in reports),
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c74", "created_at": timestamp(), "stage": "A0_label_release_gate",
        "status": "passed" if passed else "failed_terminal",
        "decision": "authorize_exposed_validation_labels" if passed else "close_c74_before_validation_labels",
        "execution_lock_sha256": execution_hash, "checks": checks,
        "seed_reports": {str(report["seed"]): {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)} for path, report in reports},
        "activity": {str(report["seed"]): report["activity"] for _, report in reports},
        "correction": {str(report["seed"]): report["correction"] for _, report in reports},
        "validation_labels_opened": False, "fresh_dev_test_qrels_opened": False,
    }
    atomic_json(REPO_ROOT / config["paths"]["a0_report"], value)
    return value


def load_scores(path: Path) -> dict[str, list[np.ndarray]]:
    with np.load(path, allow_pickle=False) as values:
        offsets = np.asarray(values["offsets"], dtype=np.int64)
        return {name: unflatten(offsets, values[name]) for name in SCORE_NAMES}


def ndcg_rows(request_ids, item_ids, scores, labels) -> np.ndarray:
    output = []
    for request_id, items, values, label in zip(request_ids, item_ids, scores, labels):
        ranked = ranking(request_id, items, values)
        positive = {str(item) for item, value in zip(items, label) if value > 0}
        output.append(ndcg_at_k(ranked, positive, 10))
    return np.asarray(output, dtype=np.float64)


def mean_rows(seed_rows: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [np.mean(np.stack([rows[i] for rows in seed_rows]), axis=0).astype(np.float32) for i in range(len(seed_rows[0]))]


def a1(config: Mapping[str, Any]) -> dict[str, Any]:
    _, execution_hash = verify_execution_lock(config)
    gate = json.loads((REPO_ROOT / config["paths"]["a0_report"]).read_text())
    if gate["status"] != "passed" or gate["execution_lock_sha256"] != execution_hash:
        raise PermissionError("C74 validation labels not authorized")
    store = C74Store(config, REPO_ROOT)
    split = json.loads((REPO_ROOT / config["paths"]["artifact_root"] / "split_manifest.json").read_text())
    expected_hash = split["validation_candidate_hash"]
    if store.candidate_hash(store.validation_indices) != expected_hash:
        raise RuntimeError("C74 candidate hash differs before A1")
    seeds = [int(v) for v in config["training"]["seeds"]]
    score_sets = [load_scores(REPO_ROOT / config["paths"]["artifact_root"] / f"seed_{seed}_scores.npz") for seed in seeds]
    request_ids = [store.data.request_ids[i] for i in store.validation_indices]
    item_ids = [store.data.candidate_ids(i) for i in store.validation_indices]
    labels = [store.labels(i) for i in store.validation_indices]
    ensemble = {name: mean_rows([score[name] for score in score_sets]) for name in SCORE_NAMES}
    ndcg = {name: ndcg_rows(request_ids, item_ids, values, labels) for name, values in ensemble.items() if name not in {"primary_correction", "wrong_correction"}}
    e = config["evaluation"]
    comparisons = compare(
        request_ids, ndcg[PRIMARY],
        {"base": ndcg["base"], "coupled_value_relay": ndcg["coupled_value_relay"],
         "pooled_semantic_relay": ndcg["pooled_semantic_relay"],
         "factual_semantic_relay": ndcg["factual_semantic_relay"],
         "wrong_history": ndcg["wrong_history"]},
        samples=int(e["bootstrap_samples"]), seed=int(e["bootstrap_seed"]), folds=int(e["hash_folds"]),
    )
    seed_differences = {}
    refs = ("base", "coupled_value_relay", "pooled_semantic_relay", "factual_semantic_relay", "wrong_history")
    for seed, scores in zip(seeds, score_sets):
        seed_ndcg = {name: ndcg_rows(request_ids, item_ids, scores[name], labels) for name in (PRIMARY, *refs)}
        seed_differences[str(seed)] = {name: float((seed_ndcg[PRIMARY] - seed_ndcg[name]).mean()) for name in refs}
    thresholds = {
        "base": float(e["primary_minus_base_min"]),
        "coupled_value_relay": float(e["primary_minus_coupled_min"]),
        "pooled_semantic_relay": float(e["primary_minus_pooled_min"]),
        "factual_semantic_relay": float(e["primary_minus_factual_min"]),
        "wrong_history": float(e["true_minus_wrong_min"]),
    }
    checks = {"candidate_hash_asserted": True, "labels_opened_only_after_A0": True, "fresh_dev_test_qrels_closed": True}
    for name, threshold in thresholds.items():
        row = comparisons[name]
        checks[f"{name}_mean_threshold"] = row["mean"] >= threshold
        checks[f"{name}_positive_interval"] = row["percentile_95_ci"][0] > 0.0
        checks[f"{name}_each_seed_positive"] = all(seed_differences[str(seed)][name] > 0.0 for seed in seeds)
        checks[f"{name}_two_of_three_folds_positive"] = sum(fold["mean_difference"] > 0.0 for fold in row["hash_folds"]) >= 2
    clicked_values = []
    for correction, label in zip(ensemble["primary_correction"], labels):
        positive = np.asarray(label) > 0
        if positive.any(): clicked_values.extend(np.asarray(correction)[positive].tolist())
    clicked = bootstrap(np.asarray(clicked_values), samples=int(e["bootstrap_samples"]), seed=int(e["bootstrap_seed"]) + 17)
    passed = all(checks.values())
    result = {
        "candidate_id": "c74", "created_at": timestamp(),
        "stage": "exposed_fit_pretrained_token_lm_A1",
        "status": "passed" if passed else "failed_terminal",
        "decision": "authorize_same_graph_on_amazon_exposed_fit" if passed else "close_c74_lm_probe_without_fresh_access",
        "execution_lock_sha256": execution_hash, "validation_requests": len(request_ids),
        "candidate_hash": expected_hash,
        "metrics": {name: float(values.mean()) for name, values in ndcg.items()},
        "comparisons": comparisons, "seed_differences": seed_differences,
        "clicked_primary_correction": clicked, "checks": checks,
        "fit_train_labels_opened": True, "validation_exposed_fit_labels_opened_after_A0": True,
        "fresh_features_scores_labels_opened": False, "dev_test_qrels_opened": False,
    }
    atomic_json(REPO_ROOT / config["paths"]["promoted_report"], result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("seed", "a0", "a1"), required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(); config = load_config()
    if args.stage == "seed":
        if args.seed is None: parser.error("--seed required")
        value = run_seed(config, seed=args.seed, device=torch.device("cuda:0"))
    elif args.stage == "a0": value = a0(config)
    else: value = a1(config)
    print(json.dumps({"stage": args.stage, "status": value["status"], "decision": value.get("decision")}))


if __name__ == "__main__":
    main()
