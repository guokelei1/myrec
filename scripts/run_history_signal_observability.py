#!/usr/bin/env python
"""Train and label-free score one HSO mode across all user-held-out folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.history_signal_observability import (  # noqa: E402
    MODES,
    CompactFoldLabels,
    FrozenSemanticStore,
    HistorySignalTransformer,
    PackedObservabilityData,
    atomic_json,
    build_fold_popularity,
    collate_requests,
    forward_kwargs,
    iter_evaluation_batches,
    iter_training_batches,
    listwise_loss,
    load_config,
    sha256_array,
    sha256_file,
    to_device,
)


SOURCE_KEYS = ("module", "prepare_script", "run_script", "summarize_script", "protocol")
INPUT_FILES = (
    "packed_manifest",
    "train_candidate_labels",
    "raw_query_embeddings",
    "raw_item_embeddings",
    "request_metadata",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def verify_lock(config: Mapping[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    paths = config["paths"]
    lock_path = ROOT / paths["execution_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source_paths = {"config": config_path}
    source_paths.update({key: ROOT / paths[key] for key in SOURCE_KEYS})
    input_paths = {key: ROOT / paths[key] for key in INPUT_FILES}
    if {key: sha256_file(value) for key, value in source_paths.items()} != lock[
        "source_sha256"
    ]:
        raise RuntimeError("HSO source changed after lock")
    if {key: sha256_file(value) for key, value in input_paths.items()} != lock[
        "input_sha256"
    ]:
        raise RuntimeError("HSO input changed after lock")
    selection_path = ROOT / paths["artifact_root"] / "selection_manifest.json"
    if sha256_file(selection_path) != lock["selection_manifest_sha256"]:
        raise RuntimeError("HSO selection changed after lock")
    label_manifest = ROOT / paths["artifact_root"] / "fold_label_manifest.json"
    if not label_manifest.exists():
        raise FileNotFoundError(label_manifest)
    return lock, sha256_file(lock_path)


def assert_device(config: Mapping[str, Any], mode: str, device_name: str) -> torch.device:
    physical = int(config["resources"]["mode_to_physical_gpu"][mode])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("HSO physical GPU registration differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("HSO runner requires exactly one visible CUDA GPU")
    return torch.device(device_name)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def make_model(config: Mapping[str, Any], mode: str, device: torch.device) -> HistorySignalTransformer:
    row = config["model"]
    return HistorySignalTransformer(
        mode=mode,
        input_dim=int(row["input_dim"]),
        width=int(row["width"]),
        heads=int(row["heads"]),
        context_layers=int(row["context_layers"]),
        candidate_layers=int(row["candidate_layers"]),
        ffn_dim=int(row["ffn_dim"]),
        dropout=float(row["dropout"]),
        id_buckets=int(row["id_buckets"]),
        id_dim=int(row["id_dim"]),
        max_history=int(config["selection"]["max_history"]),
        zero_initial_output=bool(row["zero_initial_output"]),
    ).to(device)


def gradient_is_nonzero(value: torch.Tensor | None) -> bool:
    if value is None:
        return False
    if value.is_sparse:
        return bool(value._values().ne(0).any())
    return bool(value.ne(0).any())


def train_fold(
    model: HistorySignalTransformer,
    data: PackedObservabilityData,
    features: FrozenSemanticStore,
    labels: CompactFoldLabels,
    popularity: np.ndarray,
    train_indices: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    row = config["training"]
    dense_optimizer = torch.optim.AdamW(
        model.dense_parameters(),
        lr=float(row["dense_learning_rate"]),
        weight_decay=float(row["weight_decay"]),
    )
    sparse_optimizer = torch.optim.SparseAdam(
        model.sparse_parameters(), lr=float(row["sparse_learning_rate"])
    )
    losses: list[float] = []
    active_names: set[str] = set()
    steps = 0
    for epoch in range(int(row["epochs"])):
        model.train()
        candidate_rng = np.random.default_rng(seed + epoch * 1009 + 17)
        dropout_rng = np.random.default_rng(seed + epoch * 1009 + 29)
        for batch_indices in iter_training_batches(
            train_indices,
            seed=seed + epoch * 1009,
            batch_size=int(row["requests_per_batch"]),
        ):
            batch = collate_requests(
                data,
                features,
                batch_indices,
                popularity,
                max_history=int(config["selection"]["max_history"]),
                label_access=True,
                fold_labels=labels,
                sampled_candidates=int(row["sampled_candidates"]),
                rng=candidate_rng,
            )
            tensors = to_device(batch, device)
            if model.mode != "null":
                dropped = dropout_rng.random(len(batch_indices)) < float(
                    row["history_dropout"]
                )
                if bool(dropped.any()):
                    tensors["history_mask"][
                        torch.from_numpy(dropped).to(device)
                    ] = False
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**forward_kwargs(tensors))
                loss = listwise_loss(
                    output,
                    tensors["labels"],
                    tensors["candidate_mask"],
                    residual_l2_weight=float(row["residual_l2_weight"]),
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"HSO {model.mode} nonfinite loss")
            dense_optimizer.zero_grad(set_to_none=True)
            sparse_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    values = parameter.grad._values() if parameter.grad.is_sparse else parameter.grad
                    if not bool(torch.isfinite(values).all()):
                        raise RuntimeError(f"HSO nonfinite gradient: {name}")
                    if gradient_is_nonzero(parameter.grad):
                        active_names.add(name)
            torch.nn.utils.clip_grad_norm_(
                model.dense_parameters(), float(row["gradient_clip_norm"])
            )
            dense_optimizer.step()
            sparse_optimizer.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
    window = min(100, max(1, len(losses) // 4))
    return {
        "steps": steps,
        "epochs": int(row["epochs"]),
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:]))
        < float(np.mean(losses[:window])),
        "finite": bool(np.isfinite(losses).all()),
        "active_parameter_groups": {
            "semantic_projection": any(
                name.startswith("semantic_projection.") for name in active_names
            ),
            "candidate_id": "item_id_embedding.weight" in active_names,
            "context_transformer": any(
                name.startswith("context_encoder.") for name in active_names
            ),
            "candidate_cross_attention": any(
                name.startswith("candidate_blocks.") for name in active_names
            ),
            "output_head": "output_head.weight" in active_names,
        },
    }


def reversed_candidates(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = dict(tensors)
    permutation = torch.arange(
        tensors["candidate_mask"].shape[1] - 1,
        -1,
        -1,
        device=tensors["candidate_mask"].device,
    )
    for name in (
        "candidate_semantic",
        "candidate_indices",
        "candidate_mask",
        "candidate_popularity",
        "labels",
    ):
        output[name] = tensors[name][:, permutation]
    return output


def score_scenario(
    model: HistorySignalTransformer,
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    numeric_check: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    tensors = to_device(batch, device)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first = model(**forward_kwargs(tensors))
        diagnostics = {"deterministic_max_abs": 0.0, "permutation_max_abs": 0.0}
        if numeric_check:
            second = model(**forward_kwargs(tensors))
            diagnostics["deterministic_max_abs"] = float(
                (first.scores - second.scores).abs().max().float().cpu()
            )
            reversed_tensor = reversed_candidates(tensors)
            reversed_output = model(**forward_kwargs(reversed_tensor)).scores.flip(1)
            diagnostics["permutation_max_abs"] = float(
                (first.scores - reversed_output).abs().max().float().cpu()
            )
    return (
        first.scores.float().cpu().numpy(),
        first.base_scores.float().cpu().numpy(),
        diagnostics,
    )


def save_npz(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    temporary.replace(path)


def score_fold(
    model: HistorySignalTransformer,
    data: PackedObservabilityData,
    features: FrozenSemanticStore,
    popularity: np.ndarray,
    evaluation_indices: np.ndarray,
    wrong: Mapping[int, int],
    config: Mapping[str, Any],
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    if data.labels_opened:
        raise PermissionError("HSO scoring process opened source labels")
    flat: dict[str, list[np.ndarray]] = {
        name: [] for name in ("true", "wrong", "shuffle", "null", "base")
    }
    offsets = [0]
    request_order: list[int] = []
    numeric = {"deterministic_max_abs": 0.0, "permutation_max_abs": 0.0}
    checked = False
    evaluation = config["evaluation"]
    for batch_indices in iter_evaluation_batches(
        data,
        evaluation_indices,
        max_requests=int(evaluation["max_requests_per_batch"]),
        max_candidates=int(evaluation["max_candidates_per_batch"]),
    ):
        scenarios = {
            "true": {},
            "wrong": {"history_sources": wrong},
            "shuffle": {"reverse_history": True},
            "null": {"empty_history": True},
        }
        scenario_scores: dict[str, np.ndarray] = {}
        base_scores: np.ndarray | None = None
        for name, options in scenarios.items():
            if model.mode == "null" and name != "null":
                continue
            batch = collate_requests(
                data,
                features,
                batch_indices,
                popularity,
                max_history=int(config["selection"]["max_history"]),
                label_access=False,
                **options,
            )
            scores, base, diagnostics = score_scenario(
                model, batch, device, numeric_check=not checked and name in {"true", "null"}
            )
            scenario_scores[name] = scores
            base_scores = base
            numeric = {key: max(numeric[key], value) for key, value in diagnostics.items()}
            if not checked and name in {"true", "null"}:
                checked = True
        if model.mode == "null":
            for name in ("true", "wrong", "shuffle"):
                scenario_scores[name] = scenario_scores["null"]
        if base_scores is None:
            raise RuntimeError("HSO base scores missing")
        for row, index_value in enumerate(batch_indices):
            count = len(data.candidates(int(index_value)))
            for name in ("true", "wrong", "shuffle", "null"):
                flat[name].append(scenario_scores[name][row, :count].astype(np.float32))
            flat["base"].append(base_scores[row, :count].astype(np.float32))
            offsets.append(offsets[-1] + count)
            request_order.append(int(index_value))
    values = {
        name: np.concatenate(rows).astype(np.float32, copy=False)
        for name, rows in flat.items()
    }
    save_npz(
        output_path,
        request_indices=np.asarray(request_order, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        **values,
    )
    return {
        "requests": len(request_order),
        "candidate_rows": offsets[-1],
        "candidate_hash": data.candidate_hash(request_order),
        "score_path": str(output_path.relative_to(ROOT)),
        "score_sha256": sha256_file(output_path),
        "score_array_sha256": {name: sha256_array(value) for name, value in values.items()},
        **numeric,
        "heldout_labels_opened": False,
    }


def run_mode(config: Mapping[str, Any], mode: str, device: torch.device, lock_hash: str) -> None:
    paths = config["paths"]
    artifact_root = ROOT / paths["artifact_root"]
    checkpoint_root = ROOT / paths["checkpoint_root"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    data = PackedObservabilityData(ROOT / paths["packed_train_root"])
    features = FrozenSemanticStore(
        data,
        ROOT / paths["raw_query_embeddings"],
        ROOT / paths["raw_item_embeddings"],
    )
    selected = np.load(artifact_root / "strict_indices.npy", mmap_mode="r")
    assignments = np.load(artifact_root / "fold_assignments.npy", mmap_mode="r")
    wrong_values = np.load(artifact_root / "wrong_request_indices.npy", mmap_mode="r")
    wrong = {int(index): int(donor) for index, donor in zip(selected, wrong_values)}
    reports = []
    for fold, seed in enumerate(config["training"]["seeds_by_fold"]):
        report_path = artifact_root / f"{mode}_fold{fold}_report.json"
        score_path = artifact_root / f"{mode}_fold{fold}_scores.npz"
        checkpoint_path = checkpoint_root / f"{mode}_fold{fold}.pt"
        for path in (report_path, score_path, checkpoint_path):
            if path.exists():
                raise FileExistsError(path)
        seed_all(int(seed))
        train_indices = np.asarray(
            selected[np.asarray(assignments) != fold], dtype=np.int64
        )
        evaluation_indices = np.asarray(
            selected[np.asarray(assignments) == fold], dtype=np.int64
        )
        labels = CompactFoldLabels.load(artifact_root, fold)
        if set(labels.request_indices.tolist()) != set(train_indices.tolist()):
            raise RuntimeError("HSO compact fit labels differ from fold")
        popularity = build_fold_popularity(
            data, labels, train_indices, len(features.items)
        )
        if data.labels_opened:
            raise PermissionError("HSO runner opened global train labels")
        model = make_model(config, mode, device)
        training = train_fold(
            model,
            data,
            features,
            labels,
            popularity,
            train_indices,
            config,
            seed=int(seed),
            device=device,
        )
        torch.save(
            {
                "analysis_id": config["analysis_id"],
                "mode": mode,
                "fold": fold,
                "seed": int(seed),
                "model_state": model.state_dict(),
            },
            checkpoint_path,
        )
        scoring = score_fold(
            model,
            data,
            features,
            popularity,
            evaluation_indices,
            wrong,
            config,
            device,
            score_path,
        )
        checks = {
            "loss_decreased": bool(training["loss_decreased"]),
            "finite": bool(training["finite"]),
            "candidate_hash": scoring["candidate_hash"]
            == data.candidate_hash(evaluation_indices),
            "deterministic": scoring["deterministic_max_abs"]
            <= float(config["evaluation"]["deterministic_tolerance"]),
            "candidate_permutation": scoring["permutation_max_abs"]
            <= float(config["evaluation"]["candidate_permutation_tolerance"]),
            "global_train_labels_closed": not data.labels_opened,
            "heldout_labels_closed": not scoring["heldout_labels_opened"],
            "dev_test_qrels_closed": True,
        }
        report = {
            "analysis_id": config["analysis_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "fit_fold_training_and_label_free_scoring",
            "mode": mode,
            "fold": fold,
            "seed": int(seed),
            "execution_lock_sha256": lock_hash,
            "train_requests": len(train_indices),
            "evaluation_requests": len(evaluation_indices),
            "parameters": {
                "total": model.parameter_count(),
                "trainable": model.trainable_parameter_count(),
            },
            "popularity_sha256": sha256_array(popularity),
            "training": training,
            "scoring": scoring,
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": sha256_file(checkpoint_path),
            },
            "checks": checks,
            "passed_mechanics": all(checks.values()),
        }
        atomic_json(report_path, report)
        reports.append(report)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "fold": fold,
                    "loss_first": training["loss_first"],
                    "loss_last": training["loss_last"],
                    "mechanics": report["passed_mechanics"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
    if len({row["parameters"]["total"] for row in reports}) != 1:
        raise RuntimeError("HSO parameter count differs across folds")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    _, lock_hash = verify_lock(config, config_path)
    device = assert_device(config, args.mode, args.device)
    run_mode(config, args.mode, device, lock_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
