#!/usr/bin/env python
"""Write the one label-free C02 dev score file and functional diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for entry in (SYSTEM_ROOT, REPO_ROOT / "src"):
    sys.path.insert(0, str(entry))

from model.data import C02Split, collate_requests, iter_request_batches
from train.runtime import (
    FrozenFeatureStore,
    assert_candidate_hash,
    assert_proposal_lock,
    build_model,
    corruption_inputs,
    load_config,
    model_inputs,
    read_json,
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
    train_summary_path = Path(config["paths"]["diagnostic_root"]) / "train_summary.json"
    train_summary = read_json(train_summary_path)
    checkpoint_path = Path(
        train_summary["variants"]["chht"]["checkpoint_path"]
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["candidate_manifest_sha256"] != candidate_hash:
        raise ValueError("checkpoint candidate hash mismatch")

    data = C02Split.load(
        config["paths"]["shared_packed_data"],
        config["paths"]["feature_root"],
        "dev",
    )
    if data.base_scores is None:
        raise ValueError("frozen D2p dev scores are missing")
    store = FrozenFeatureStore(config, "dev")
    model = build_model(config, "chht", args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    subset_indices: dict[str, list[int]] = {name: [] for name in ("repeat", "nonrepeat", "no_history")}
    for index in range(len(data)):
        subset_indices[data.structural_subset(index)].append(index)
    expected = {
        "repeat": int(config["integrity"]["expected_repeat_present"]),
        "nonrepeat": int(config["integrity"]["expected_nonrepeat_present"]),
        "no_history": int(config["integrity"]["expected_no_history"]),
    }
    observed = {name: len(values) for name, values in subset_indices.items()}
    if observed != expected:
        raise AssertionError(f"frozen structural subset mismatch: {observed} != {expected}")

    diagnostic_root = Path(config["paths"]["diagnostic_root"])
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    subset_paths = {}
    for name, indices in subset_indices.items():
        path = diagnostic_root / f"{name}_request_ids.txt"
        _write_lines(path, [data.request_ids[index] for index in indices])
        subset_paths[name] = str(path)

    run_id = str(config["run_id"])
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scores_path = run_dir / "scores.jsonl"
    temporary_scores = scores_path.with_suffix(".jsonl.tmp")
    scoring_started = time.time()
    rows = 0
    no_history_score_mismatches = 0
    no_history_rank_mismatches = 0
    corruption_totals: dict[str, defaultdict[str, float]] = {
        name: defaultdict(float) for name in CORRUPTIONS
    }
    first_pass: dict[int, np.ndarray] = {}
    all_indices = np.arange(len(data), dtype=np.int64)
    with temporary_scores.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for request_indices in _batches(data, all_indices, config):
            numpy_batch = collate_requests(
                data, request_indices, history_limit=int(config["data"]["history_limit"])
            )
            tensors = store.dev_tensors(numpy_batch, args.device, include_corruptions=True)
            output = model(**model_inputs(tensors), variant="chht")
            corrupt_outputs = {
                name: model(**corruption_inputs(tensors, name), variant="chht")
                for name in CORRUPTIONS
            }
            scores = output.scores.cpu().numpy()
            bases = tensors["base_scores"].cpu().numpy()
            candidate_mask = np.asarray(numpy_batch["candidate_mask"])
            for row, raw_index in enumerate(request_indices):
                index = int(raw_index)
                count = int(candidate_mask[row].sum())
                values = np.asarray(scores[row, :count], dtype=np.float64)
                base_values = np.asarray(bases[row, :count], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"non-finite dev score for {data.request_ids[index]}")
                if index < int(config["dev_gate"]["deterministic_requests"]):
                    first_pass[index] = values.copy()
                subset = data.structural_subset(index)
                if subset == "no_history":
                    no_history_score_mismatches += int(not np.array_equal(values, base_values))
                    rank = np.argsort(-values, kind="stable")
                    base_rank = np.argsort(-base_values, kind="stable")
                    no_history_rank_mismatches += int(not np.array_equal(rank, base_rank))
                else:
                    valid_candidates = torch.from_numpy(candidate_mask[row]).to(args.device)
                    true_core = output.core[row, valid_candidates]
                    true_norm = float(
                        output.core_norm[row, valid_candidates].mean().float().cpu()
                    )
                    for name, corrupt in corrupt_outputs.items():
                        corrupt_core = corrupt.core[row, valid_candidates]
                        corrupt_norm = float(
                            corrupt.core_norm[row, valid_candidates].mean().float().cpu()
                        )
                        max_abs = float((true_core - corrupt_core).abs().max().float().cpu())
                        distance = float(
                            torch.linalg.matrix_norm(
                                true_core - corrupt_core, ord="fro", dim=(-2, -1)
                            ).mean().float().cpu()
                        )
                        corruption_totals[name]["requests"] += 1
                        corruption_totals[name]["true_norm"] += true_norm
                        corruption_totals[name]["corrupt_norm"] += corrupt_norm
                        corruption_totals[name]["distance"] += distance
                        corruption_totals[name]["changed"] += int(max_abs > 0.0)
                request_id = data.request_ids[index]
                item_ids = np.asarray(numpy_batch["candidate_item_ids"])[row, :count]
                for item_id, value in zip(item_ids, values):
                    handle.write(
                        json.dumps(
                            {
                                "candidate_item_id": str(int(item_id)),
                                "method_id": "c02_chht",
                                "request_id": request_id,
                                "score": float(value),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    rows += 1
    temporary_scores.replace(scores_path)
    if rows != int(config["integrity"]["expected_score_rows"]):
        raise AssertionError(f"score-row mismatch: {rows}")

    deterministic = _deterministic_rescore(
        model, data, store, config, args.device, first_pass
    )
    corruptions = {}
    for name, values in corruption_totals.items():
        requests = int(values["requests"])
        true_mean = values["true_norm"] / requests
        corrupt_mean = values["corrupt_norm"] / requests
        corruptions[name] = {
            "affected_requests": requests,
            "changed_request_fraction": values["changed"] / requests,
            "mean_paired_core_distance": values["distance"] / requests,
            "mean_true_core_norm": true_mean,
            "mean_corrupt_core_norm": corrupt_mean,
            "corrupt_to_true_core_norm_ratio": (
                corrupt_mean / true_mean if true_mean > 0 else math.inf
            ),
        }

    diagnostics = {
        "candidate_manifest_sha256": candidate_hash,
        "corruptions": corruptions,
        "deterministic_rescore": deterministic,
        "no_history": {
            "requests": observed["no_history"],
            "rank_mismatch_requests": no_history_rank_mismatches,
            "score_mismatch_requests": no_history_score_mismatches,
        },
        "structural_subsets": observed,
        "subset_paths": subset_paths,
    }
    write_json(diagnostic_root / "dev_functional_diagnostics.json", diagnostics)
    pre_eval_log_rows = _count_run_log_rows(Path("reports/dev_eval_log.jsonl"), run_id)
    if pre_eval_log_rows != 0:
        raise AssertionError(f"C02 run already has {pre_eval_log_rows} dev-eval rows")
    metadata = {
        "analysis_id": config["analysis_id"],
        "candidate_manifest_path": str(config["paths"]["candidate_manifest"]),
        "candidate_manifest_sha256": candidate_hash,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_id": "kuaisearch",
        "dataset_version": "v0_lite",
        "dev_eval_log_rows_before_evaluation": pre_eval_log_rows,
        "elapsed_seconds": time.time() - scoring_started,
        "environment": config["environment"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu": {"physical": 1, "program_device": args.device, "name": torch.cuda.get_device_name(0)},
        "input_fields_used": [
            "label-free query state",
            "candidate title state",
            "strictly-prior history item/category/event/recency",
            "train-only popularity coordinate",
        ],
        "label_boundary": {
            "evaluation_labels_read": False,
            "test_data_read": False,
        },
        "method_id": "c02_chht",
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "proposal_lock_sha256": sha256_file(
            Path(config["paths"]["candidate_source_root"]) / "notes/proposal_lock.json"
        ),
        "request_count": len(data),
        "run_id": run_id,
        "score_rows": rows,
        "scores_sha256": sha256_file(scores_path),
        "seed": int(config["seed"]),
        "split": "dev",
    }
    write_json(run_dir / "metadata.json", metadata)
    shutil.copyfile(config_path, run_dir / "config_snapshot.yaml")
    write_json(diagnostic_root / "score_summary.json", metadata)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _deterministic_rescore(
    model: torch.nn.Module,
    data: C02Split,
    store: FrozenFeatureStore,
    config: dict[str, Any],
    device: str,
    first_pass: dict[int, np.ndarray],
) -> dict[str, Any]:
    count = int(config["dev_gate"]["deterministic_requests"])
    indices = np.arange(count, dtype=np.int64)
    missing = 0
    max_abs = 0.0
    rows = 0
    with torch.inference_mode():
        for request_indices in _batches(data, indices, config):
            numpy_batch = collate_requests(
                data, request_indices, history_limit=int(config["data"]["history_limit"])
            )
            tensors = store.dev_tensors(numpy_batch, device, include_corruptions=False)
            output = model(**model_inputs(tensors), variant="chht")
            scores = output.scores.cpu().numpy()
            mask = np.asarray(numpy_batch["candidate_mask"])
            for row, raw_index in enumerate(request_indices):
                index = int(raw_index)
                valid = int(mask[row].sum())
                values = np.asarray(scores[row, :valid], dtype=np.float64)
                expected = first_pass.get(index)
                if expected is None or expected.shape != values.shape:
                    missing += 1
                    continue
                if values.size:
                    max_abs = max(max_abs, float(np.max(np.abs(values - expected))))
                rows += values.size
    return {
        "max_abs_score_delta": max_abs,
        "missing_requests": missing,
        "requests": count,
        "score_rows": rows,
    }


def _batches(data: C02Split, indices: np.ndarray, config: dict[str, Any]):
    return iter_request_batches(
        data,
        indices,
        history_limit=int(config["data"]["history_limit"]),
        max_requests=int(config["data"]["max_requests_per_batch"]),
        max_padded_candidates=int(config["data"]["max_padded_candidate_rows"]),
        max_padded_history=int(config["data"]["max_padded_history_rows"]),
        seed=0,
        shuffle=False,
    )


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(value + "\n")
    temporary.replace(path)


def _count_run_log_rows(path: Path, run_id: str) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and str(json.loads(line).get("run_id")) == run_id:
                count += 1
    return count


if __name__ == "__main__":
    raise SystemExit(main())
