#!/usr/bin/env python
"""Train the five frozen C02 variants and run train-internal falsifiers."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for entry in (SYSTEM_ROOT, REPO_ROOT / "src"):
    sys.path.insert(0, str(entry))

from model.chht import multi_positive_listwise_loss
from model.data import C02Split, collate_requests, frozen_train_indices, iter_request_batches
from myrec.eval.metrics import ScoredCandidate, request_metrics
from train.runtime import (
    FrozenFeatureStore,
    assert_candidate_hash,
    assert_proposal_lock,
    build_model,
    corruption_inputs,
    load_config,
    model_inputs,
    parameter_counts,
    seed_everything,
    sha256_file,
    validate_gpu,
    write_json,
)

CORRUPTIONS = ("wrong", "shuffle", "coarse", "query_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="systems/02_history_hyperadapter/configs/screen.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    validate_gpu(args.device)
    seed_everything(int(config["seed"]))
    candidate_hash = assert_candidate_hash(config)
    proposal_lock = assert_proposal_lock(config)
    feature_manifest = Path(config["paths"]["feature_root"]) / "manifest.json"
    if not feature_manifest.exists():
        raise FileNotFoundError("run prepare_features.py before training")

    data = C02Split.load(
        config["paths"]["shared_packed_data"],
        config["paths"]["feature_root"],
        "train",
    )
    train_indices, validation_indices = frozen_train_indices(
        data,
        int(config["seed"]),
        int(config["data"]["train_sample_requests"]),
        int(config["data"]["internal_validation_requests"]),
        float(config["data"]["internal_train_fraction_boundary"]),
    )
    store = FrozenFeatureStore(config, "train")
    variants = list(config["model"]["variants"])
    summaries: dict[str, Any] = {}
    total_started = time.time()
    for variant in variants:
        summaries[variant] = train_variant(
            config,
            config_path,
            data,
            store,
            train_indices,
            validation_indices,
            variant,
            args.device,
        )

    counts = {name: row["parameter_count"]["total"] for name, row in summaries.items()}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"matched-capacity variants differ: {counts}")
    steps = {name: row["optimizer_steps"] for name, row in summaries.items()}
    if len(set(steps.values())) != 1:
        raise AssertionError(f"matched-step variants differ: {steps}")

    chht = summaries["chht"]
    controls = [name for name in variants if name != "chht"]
    best_control = max(
        summaries[name]["selected_internal"]["nonrepeat"]["model_ndcg@10"]
        for name in controls
    )
    internal = chht["selected_internal"]
    corruption_pass = all(
        internal["corruptions"][name]["true_to_corrupt_core_norm_ratio"]
        >= float(config["internal_gate"]["true_to_corrupt_core_norm_ratio_min"])
        and internal["corruptions"][name]["mean_paired_core_distance"] > 0.0
        for name in CORRUPTIONS
    )
    gate_checks = {
        "finite_and_loss_decreased": bool(
            all(math.isfinite(row["train_loss"]) for row in chht["epochs"])
            and chht["epochs"][-1]["train_loss"] < chht["epochs"][0]["train_loss"]
        ),
        "nonrepeat_delta_vs_d2p": bool(
            internal["nonrepeat"]["model_minus_d2p"]
            >= float(config["internal_gate"]["nonrepeat_delta_vs_d2p_min"])
        ),
        "repeat_delta_vs_item_teacher": bool(
            internal["repeat"]["model_minus_item_teacher"]
            >= float(config["internal_gate"]["repeat_delta_vs_item_teacher_min"])
        ),
        "margin_over_best_control": bool(
            internal["nonrepeat"]["model_ndcg@10"] - best_control
            >= float(config["internal_gate"]["margin_over_best_control_min"])
        ),
        "corruption_contract": corruption_pass,
        "no_history_exact": bool(
            internal["no_history"]["max_abs_score_delta"]
            <= float(config["internal_gate"]["no_history_max_abs_score_delta"])
        ),
    }
    summary = {
        "analysis_id": config["analysis_id"],
        "candidate_id": config["candidate_id"],
        "candidate_manifest_sha256": candidate_hash,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - total_started,
        "environment": config["environment"],
        "feature_manifest_path": str(feature_manifest),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "gpu": {"physical": 1, "program_device": args.device, "name": torch.cuda.get_device_name(0)},
        "internal_gate": {
            "best_control_nonrepeat_ndcg@10": best_control,
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
        },
        "label_boundary": {
            "evaluation_labels_read": False,
            "test_data_read": False,
            "train_labels_only": True,
        },
        "optimizer_steps_by_variant": steps,
        "parameter_counts_by_variant": counts,
        "proposal_lock_path": str(
            Path(config["paths"]["candidate_source_root"]) / "notes/proposal_lock.json"
        ),
        "proposal_lock_sha256": sha256_file(
            Path(config["paths"]["candidate_source_root"]) / "notes/proposal_lock.json"
        ),
        "seed": int(config["seed"]),
        "train_request_count": len(train_indices),
        "validation_request_count": len(validation_indices),
        "variants": summaries,
    }
    output_root = Path(config["paths"]["diagnostic_root"])
    write_json(output_root / "train_summary.json", summary)
    print(json.dumps(summary["internal_gate"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def train_variant(
    config: dict[str, Any],
    config_path: Path,
    data: C02Split,
    store: FrozenFeatureStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    variant: str,
    device: str,
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    model = build_model(config, variant, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    amp = str(config["training"]["amp_dtype"])
    amp_dtype = torch.bfloat16 if amp == "bfloat16" else torch.float16
    variant_started = time.time()
    epoch_rows: list[dict[str, Any]] = []
    optimizer_steps = 0
    best_key: tuple[float, float, float, int] | None = None
    checkpoint_root = Path(config["paths"]["checkpoint_root"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / f"{variant}_selected.pt"

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        loss_sums: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for request_indices in _batches(data, train_indices, config, int(config["seed"]) + epoch, True):
            numpy_batch = collate_requests(
                data, request_indices, history_limit=int(config["data"]["history_limit"])
            )
            tensors = store.tensors(
                numpy_batch,
                device,
                include_corruptions=(variant == "chht"),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                output = model(**model_inputs(tensors), variant=variant)
                listwise = multi_positive_listwise_loss(
                    output.scores,
                    tensors["candidate_labels"],
                    tensors["candidate_mask"],
                )
                preservation = preservation_loss(
                    output.scores,
                    tensors["item_teacher_scores"],
                    tensors["candidate_mask"],
                    tensors["repeat_mask"],
                    float(config["training"]["preservation_temperature"]),
                    float(config["training"]["repeat_margin"]),
                )
                corruption = output.scores.new_zeros(())
                if variant == "chht":
                    corruption = corruption_loss(
                        model,
                        tensors,
                        output.core_norm,
                        float(config["training"]["corruption_margin"]),
                    )
                valid_core = tensors["candidate_mask"] & tensors["history_mask"].any(dim=-1)[:, None]
                core_norm = (
                    output.core_norm[valid_core].square().mean()
                    if valid_core.any()
                    else output.core_norm.sum() * 0.0
                )
                loss = (
                    float(config["training"]["listwise_weight"]) * listwise
                    + float(config["training"]["preservation_weight"]) * preservation
                    + float(config["training"]["corruption_weight"]) * corruption
                    + float(config["training"]["core_norm_weight"]) * core_norm
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {variant} loss at epoch {epoch}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite {variant} gradient")
            optimizer.step()
            optimizer_steps += 1
            batches += 1
            for name, value in (
                ("train_loss", loss),
                ("listwise_loss", listwise),
                ("preservation_loss", preservation),
                ("corruption_loss", corruption),
                ("core_norm_penalty", core_norm),
            ):
                loss_sums[name] += float(value.detach().float().cpu())

        internal = evaluate_internal(
            model,
            data,
            store,
            validation_indices,
            config,
            variant,
            device,
            include_corruptions=(variant == "chht"),
        )
        row = {
            "batches": batches,
            "epoch": epoch,
            **{name: value / batches for name, value in loss_sums.items()},
            "internal": internal,
        }
        epoch_rows.append(row)
        key = (
            float(internal["nonrepeat"]["model_minus_d2p"]),
            float(internal["repeat"]["model_minus_item_teacher"]),
            -float(row["train_loss"]),
            -epoch,
        )
        if best_key is None or key > best_key:
            best_key = key
            torch.save(
                {
                    "analysis_id": config["analysis_id"],
                    "candidate_manifest_sha256": config["integrity"]["candidate_manifest_sha256"],
                    "config_sha256": sha256_file(config_path),
                    "epoch": epoch,
                    "model_state": {
                        name: value.detach().cpu() for name, value in model.state_dict().items()
                    },
                    "seed": int(config["seed"]),
                    "variant": variant,
                },
                checkpoint_path,
            )

    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected_epoch = int(selected["epoch"])
    model.load_state_dict(selected["model_state"], strict=True)
    inference_benchmark = benchmark_forward(
        model,
        data,
        store,
        validation_indices[: min(2400, len(validation_indices))],
        config,
        variant,
        device,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "elapsed_seconds": time.time() - variant_started,
        "epochs": epoch_rows,
        "optimizer_steps": optimizer_steps,
        "parameter_count": parameter_counts(model),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "inference_benchmark": inference_benchmark,
        "selected_epoch": selected_epoch,
        "selected_internal": epoch_rows[selected_epoch - 1]["internal"],
        "variant": variant,
    }


def benchmark_forward(
    model: torch.nn.Module,
    data: C02Split,
    store: FrozenFeatureStore,
    indices: np.ndarray,
    config: dict[str, Any],
    variant: str,
    device: str,
) -> dict[str, Any]:
    """Matched true-history latency, excluding CHHT-only corruption training."""

    model.eval()
    batches = list(_batches(data, indices, config, 0, False))
    torch.cuda.synchronize()
    started = time.perf_counter()
    rows = 0
    with torch.inference_mode():
        for request_indices in batches:
            numpy_batch = collate_requests(
                data, request_indices, history_limit=int(config["data"]["history_limit"])
            )
            tensors = store.tensors(numpy_batch, device, include_corruptions=False)
            model(**model_inputs(tensors), variant=variant)
            rows += int(np.asarray(numpy_batch["candidate_mask"]).sum())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "candidate_rows": rows,
        "elapsed_seconds": elapsed,
        "requests": len(indices),
        "rows_per_second": rows / elapsed,
    }


def preservation_loss(
    scores: torch.Tensor,
    teacher: torch.Tensor,
    candidate_mask: torch.Tensor,
    repeat_mask: torch.Tensor,
    temperature: float,
    margin: float,
) -> torch.Tensor:
    repeat_candidates = repeat_mask.any(dim=-1) & candidate_mask
    repeat_requests = repeat_candidates.any(dim=-1)
    if not repeat_requests.any():
        return scores.sum() * 0.0
    valid = candidate_mask[repeat_requests]
    student = scores[repeat_requests] / temperature
    target = teacher[repeat_requests] / temperature
    student = student.masked_fill(~valid, -1e4)
    target = target.masked_fill(~valid, -1e4)
    target_log = F.log_softmax(target, dim=-1)
    target_prob = target_log.exp()
    student_log = F.log_softmax(student, dim=-1)
    kl = (target_prob * (target_log - student_log)).sum(dim=-1).mean()

    repeated = repeat_candidates[repeat_requests]
    nonrepeated = valid & ~repeated
    usable = nonrepeated.any(dim=-1)
    if usable.any():
        repeat_best = student.masked_fill(~repeated, -1e4).max(dim=-1).values
        other_best = student.masked_fill(~nonrepeated, -1e4).max(dim=-1).values
        hinge = F.relu(margin - (repeat_best[usable] - other_best[usable])).mean()
    else:
        hinge = scores.sum() * 0.0
    return kl + hinge


def corruption_loss(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    true_norm: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    valid = tensors["candidate_mask"] & tensors["history_mask"].any(dim=-1)[:, None]
    if not valid.any():
        return true_norm.sum() * 0.0
    losses = []
    for name in CORRUPTIONS:
        corrupt = model(**corruption_inputs(tensors, name), variant="chht")
        losses.append(F.relu(margin + corrupt.core_norm[valid] - true_norm[valid]).mean())
    return torch.stack(losses).sum()


def evaluate_internal(
    model: torch.nn.Module,
    data: C02Split,
    store: FrozenFeatureStore,
    indices: np.ndarray,
    config: dict[str, Any],
    variant: str,
    device: str,
    *,
    include_corruptions: bool,
) -> dict[str, Any]:
    model.eval()
    sums: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    max_no_history_delta = 0.0
    corrupt_sums: dict[str, defaultdict[str, float]] = {
        name: defaultdict(float) for name in CORRUPTIONS
    }
    with torch.inference_mode():
        for request_indices in _batches(data, indices, config, 0, False):
            numpy_batch = collate_requests(
                data, request_indices, history_limit=int(config["data"]["history_limit"])
            )
            tensors = store.tensors(
                numpy_batch, device, include_corruptions=include_corruptions
            )
            output = model(**model_inputs(tensors), variant=variant)
            scores = output.scores.float().cpu().numpy()
            bases = tensors["base_scores"].float().cpu().numpy()
            teachers = tensors["item_teacher_scores"].float().cpu().numpy()
            candidate_mask = np.asarray(numpy_batch["candidate_mask"])
            history_mask = np.asarray(numpy_batch["history_mask"])
            repeat_mask = np.asarray(numpy_batch["repeat_mask"])
            if include_corruptions:
                corrupt_outputs = {
                    name: model(**corruption_inputs(tensors, name), variant="chht")
                    for name in CORRUPTIONS
                }
            for row, raw_request_index in enumerate(request_indices):
                request_index = int(raw_request_index)
                count = int(candidate_mask[row].sum())
                labels = np.asarray(numpy_batch["candidate_labels"])[row, :count]
                item_ids = np.asarray(numpy_batch["candidate_item_ids"])[row, :count]
                subset = data.structural_subset(request_index)
                history_present = subset != "no_history"
                model_ndcg = _request_ndcg(
                    data.request_ids[request_index], item_ids, scores[row, :count], labels
                )
                base_ndcg = _request_ndcg(
                    data.request_ids[request_index], item_ids, bases[row, :count], labels
                )
                teacher_ndcg = _request_ndcg(
                    data.request_ids[request_index], item_ids, teachers[row, :count], labels
                )
                for name, value in (
                    ("model", model_ndcg),
                    ("d2p", base_ndcg),
                    ("item_teacher", teacher_ndcg),
                ):
                    sums[f"{subset}:{name}"] += value
                    sums[f"overall:{name}"] += value
                counts[subset] += 1
                counts["overall"] += 1
                if not history_present:
                    max_no_history_delta = max(
                        max_no_history_delta,
                        float(np.max(np.abs(scores[row, :count] - bases[row, :count]))),
                    )
                if include_corruptions and history_present:
                    valid_candidates = torch.from_numpy(candidate_mask[row]).to(device)
                    true_core = output.core[row, valid_candidates]
                    true_norm = float(output.core_norm[row, valid_candidates].mean().float().cpu())
                    for name, corrupt_output in corrupt_outputs.items():
                        corrupt_core = corrupt_output.core[row, valid_candidates]
                        corrupt_norm = float(
                            corrupt_output.core_norm[row, valid_candidates].mean().float().cpu()
                        )
                        distance = float(
                            torch.linalg.matrix_norm(
                                true_core - corrupt_core, ord="fro", dim=(-2, -1)
                            ).mean().float().cpu()
                        )
                        corrupt_sums[name]["true_norm"] += true_norm
                        corrupt_sums[name]["corrupt_norm"] += corrupt_norm
                        corrupt_sums[name]["distance"] += distance
                        corrupt_sums[name]["requests"] += 1
    result: dict[str, Any] = {}
    for subset in ("overall", "repeat", "nonrepeat"):
        denominator = counts[subset]
        model_value = sums[f"{subset}:model"] / denominator
        d2p_value = sums[f"{subset}:d2p"] / denominator
        teacher_value = sums[f"{subset}:item_teacher"] / denominator
        result[subset] = {
            "requests": denominator,
            "model_ndcg@10": model_value,
            "d2p_ndcg@10": d2p_value,
            "item_teacher_ndcg@10": teacher_value,
            "model_minus_d2p": model_value - d2p_value,
            "model_minus_item_teacher": model_value - teacher_value,
        }
    result["no_history"] = {
        "requests": counts["no_history"],
        "max_abs_score_delta": max_no_history_delta,
    }
    result["corruptions"] = {}
    if include_corruptions:
        for name, values in corrupt_sums.items():
            denominator = int(values["requests"])
            true_mean = values["true_norm"] / denominator
            corrupt_mean = values["corrupt_norm"] / denominator
            result["corruptions"][name] = {
                "requests": denominator,
                "mean_true_core_norm": true_mean,
                "mean_corrupt_core_norm": corrupt_mean,
                "true_to_corrupt_core_norm_ratio": (
                    true_mean / corrupt_mean if corrupt_mean > 0 else math.inf
                ),
                "mean_paired_core_distance": values["distance"] / denominator,
            }
    return result


def _request_ndcg(
    request_id: str,
    item_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    row = request_metrics(
        request_id=request_id,
        scored_candidates=[
            ScoredCandidate(item_id=str(item_id), score=float(score))
            for item_id, score in zip(item_ids, scores)
        ],
        clicked_item_ids={
            str(item_id) for item_id, label in zip(item_ids, labels) if label > 0
        },
        purchased_item_ids=set(),
    )
    return float(row["ndcg@10"])


def _batches(
    data: C02Split,
    indices: np.ndarray,
    config: dict[str, Any],
    seed: int,
    shuffle: bool,
):
    return iter_request_batches(
        data,
        indices,
        history_limit=int(config["data"]["history_limit"]),
        max_requests=int(config["data"]["max_requests_per_batch"]),
        max_padded_candidates=int(config["data"]["max_padded_candidate_rows"]),
        max_padded_history=int(config["data"]["max_padded_history_rows"]),
        seed=seed,
        shuffle=shuffle,
    )


if __name__ == "__main__":
    raise SystemExit(main())
