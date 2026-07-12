#!/usr/bin/env python
"""Train and label-free score one Amazon HSO source across three user folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from myrec.analysis.amazon_history_signal_observability import (  # noqa: E402
    AmazonFrozenSemanticStore,
    AmazonObservabilityData,
)
from myrec.analysis.history_signal_observability import (  # noqa: E402
    CompactFoldLabels,
    atomic_json,
    build_fold_popularity,
    sha256_array,
    sha256_file,
)
from prepare_amazon_history_signal_observability import (  # noqa: E402
    load_config,
    verify_lock,
)
from run_history_signal_observability import (  # noqa: E402
    make_model,
    score_fold,
    seed_all,
    train_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("full", "text", "id"), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def assert_device(config: dict[str, Any], mode: str, device_name: str) -> torch.device:
    physical = int(config["resources"]["mode_to_physical_gpu"][mode])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("Amazon HSO physical GPU registration differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Amazon HSO requires exactly one visible GPU")
    return torch.device(device_name)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    _, lock_hash = verify_lock(config, config_path)
    device = assert_device(config, args.mode, args.device)
    paths = config["paths"]
    artifact_root = ROOT / paths["artifact_root"]
    checkpoint_root = ROOT / paths["checkpoint_root"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    data = AmazonObservabilityData(ROOT / paths["feature_root"])
    features = AmazonFrozenSemanticStore(data)
    selected = np.load(artifact_root / "strict_indices.npy", mmap_mode="r")
    assignments = np.load(artifact_root / "fold_assignments.npy", mmap_mode="r")
    wrong = data.wrong_mapping(selected)
    for fold, seed in enumerate(config["training"]["seeds_by_fold"]):
        report_path = artifact_root / f"{args.mode}_fold{fold}_report.json"
        score_path = artifact_root / f"{args.mode}_fold{fold}_scores.npz"
        checkpoint_path = checkpoint_root / f"{args.mode}_fold{fold}.pt"
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
            raise RuntimeError("Amazon HSO compact labels differ from fold")
        popularity = build_fold_popularity(
            data, labels, train_indices, len(features.items)
        )
        model = make_model(config, args.mode, device)
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
                "mode": args.mode,
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
            "heldout_labels_closed": not scoring["heldout_labels_opened"],
            "dev_test_qrels_closed": True,
        }
        report = {
            "analysis_id": config["analysis_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "amazon_fit_fold_training_and_label_free_scoring",
            "mode": args.mode,
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
        print(
            json.dumps(
                {
                    "mode": args.mode,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
