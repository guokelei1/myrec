#!/usr/bin/env python
"""Summarize the frozen full-token attention-edge attribution audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_amazon_token_edge_attribution import load_config, verify_lock  # noqa: E402
from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402
from myrec.analysis.token_history_observability import TokenHistoryData  # noqa: E402
from prepare_amazon_token_history_observability import load_config as load_upstream_config  # noqa: E402
from summarize_amazon_token_history_observability import load_reserve_labels  # noqa: E402
from summarize_history_signal_observability import cluster_bootstrap, derived_seed, request_ndcg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def comparison(left: np.ndarray, right: np.ndarray, users: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    return cluster_bootstrap(left - right, users, samples=samples, seed=seed)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    _, lock_hash = verify_lock(config, config_path)
    paths = config["paths"]
    output_path = ROOT / paths["report"]
    if output_path.exists():
        raise FileExistsError(output_path)
    token_root = ROOT / paths["token_root"]
    output_root = ROOT / paths["artifact_root"]
    data = TokenHistoryData(token_root)
    upstream = load_upstream_config(ROOT / paths["upstream_config"])
    seeds = [int(value) for value in config["seeds"]]
    masked: dict[int, Any] = {}
    original: dict[int, Any] = {}
    checks: dict[str, bool] = {}
    for seed in seeds:
        seed_report = json.loads((output_root / f"seed_{seed}_report.json").read_text(encoding="utf-8"))
        masked_path = output_root / f"seed_{seed}_scores.npz"
        checks[str(seed)] = bool(
            seed_report["passed_mechanics"]
            and seed_report["execution_lock_sha256"] == lock_hash
            and seed_report["scores"]["sha256"] == sha256_file(masked_path)
        )
        masked[seed] = np.load(masked_path, allow_pickle=False)
        original[seed] = np.load(token_root / f"seed_{seed}_scores.npz", allow_pickle=False)
    checks["all_seed_mechanics"] = all(checks.values())
    if not all(checks.values()):
        raise RuntimeError("token-edge mechanics failed")
    reference = masked[seeds[0]]
    for seed in seeds:
        if not (
            np.array_equal(masked[seed]["request_indices"], data.reserve_indices)
            and np.array_equal(masked[seed]["offsets"], reference["offsets"])
            and np.array_equal(original[seed]["offsets"], reference["offsets"])
        ):
            raise RuntimeError("token-edge candidate surface differs")

    labels = load_reserve_labels(ROOT / paths["records_train"], data)
    users_by_index = {}
    with (token_root / "requests.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            users_by_index[int(row["position"])] = str(row["user_id"])
    users = np.asarray([users_by_index[int(index)] for index in data.reserve_indices])
    score_names = ["full_true", "full_null", "full_wrong"]
    for mode in config["modes"]:
        score_names.append(f"{mode}_true")
        if mode != "history_isolated":
            score_names.append(f"{mode}_wrong")
    ensemble_flat: dict[str, np.ndarray] = {}
    for name in score_names:
        if name.startswith("full_"):
            key = name.removeprefix("full_")
            ensemble_flat[name] = np.mean(np.stack([np.asarray(original[seed][key]) for seed in seeds]), axis=0)
        else:
            ensemble_flat[name] = np.mean(np.stack([np.asarray(masked[seed][name]) for seed in seeds]), axis=0)
    per_request = {name: np.empty(len(data.reserve_indices), dtype=np.float64) for name in score_names}
    per_seed = {
        seed: {name: np.empty(len(data.reserve_indices), dtype=np.float64) for name in score_names}
        for seed in seeds
    }
    for row, index_value in enumerate(data.reserve_indices):
        index = int(index_value)
        start, stop = int(reference["offsets"][row]), int(reference["offsets"][row + 1])
        item_ids = np.asarray(data.candidate_ids(index), dtype=object)
        for name in score_names:
            per_request[name][row] = request_ndcg(data.request_ids[index], item_ids, ensemble_flat[name][start:stop], labels[index])
            for seed in seeds:
                if name.startswith("full_"):
                    values = original[seed][name.removeprefix("full_")]
                else:
                    values = masked[seed][name]
                per_seed[seed][name][row] = request_ndcg(data.request_ids[index], item_ids, values[start:stop], labels[index])

    samples = int(config["evaluation"]["bootstrap_samples"])
    base_seed = int(config["evaluation"]["bootstrap_seed"])
    full_effect = float((per_request["full_true"] - per_request["full_null"]).mean())
    metrics: dict[str, Any] = {
        "full": {
            "true": float(per_request["full_true"].mean()),
            "null": float(per_request["full_null"].mean()),
            "wrong": float(per_request["full_wrong"].mean()),
            "true_minus_null": full_effect,
            "true_minus_wrong": float((per_request["full_true"] - per_request["full_wrong"]).mean()),
        },
        "modes": {},
    }
    classifications = {}
    retained_threshold = float(config["evaluation"]["retained_fraction"])
    destroyed_threshold = float(config["evaluation"]["destroyed_fraction"])
    for mode in config["modes"]:
        true_name = f"{mode}_true"
        true_null = comparison(per_request[true_name], per_request["full_null"], users, samples=samples, seed=derived_seed(base_seed, f"{mode}_tn"))
        full_minus = comparison(per_request["full_true"], per_request[true_name], users, samples=samples, seed=derived_seed(base_seed, f"{mode}_fm"))
        row: dict[str, Any] = {
            "true": float(per_request[true_name].mean()),
            "true_minus_null": true_null,
            "full_true_minus_masked_true": full_minus,
            "true_minus_null_seed_means": {
                str(seed): float((per_seed[seed][true_name] - per_seed[seed]["full_null"]).mean())
                for seed in seeds
            },
        }
        specificity_positive = False
        if mode != "history_isolated":
            wrong_name = f"{mode}_wrong"
            true_wrong = comparison(per_request[true_name], per_request[wrong_name], users, samples=samples, seed=derived_seed(base_seed, f"{mode}_tw"))
            row["wrong"] = float(per_request[wrong_name].mean())
            row["true_minus_wrong"] = true_wrong
            row["true_minus_wrong_seed_means"] = {
                str(seed): float((per_seed[seed][true_name] - per_seed[seed][wrong_name]).mean())
                for seed in seeds
            }
            specificity_positive = float(true_wrong["user_cluster_95_ci"][0]) > 0
        retention = float(true_null["mean"] / full_effect) if full_effect != 0 else float("nan")
        row["effect_retention"] = retention
        if mode == "history_isolated":
            classification = "mechanical_null"
        elif retention >= retained_threshold and specificity_positive:
            classification = "retained"
        elif retention <= destroyed_threshold or not specificity_positive:
            classification = "destroyed"
        else:
            classification = "partial"
        row["classification"] = classification
        classifications[mode] = classification
        metrics["modes"][mode] = row

    query_path = classifications["no_candidate_history"]
    candidate_path = classifications["no_query_history"]
    if query_path == "retained" and candidate_path != "retained":
        direction = "query_mediated_raw_token_path_sufficient"
    elif candidate_path == "retained" and query_path != "retained":
        direction = "candidate_mediated_raw_token_path_sufficient"
    elif query_path == "retained" and candidate_path == "retained":
        direction = "redundant_raw_token_paths_preserve_both_in_c76"
    else:
        direction = "joint_query_history_and_candidate_history_edges_required"
    metric_path = output_root / "per_request_metrics.npz"
    with metric_path.open("wb") as handle:
        np.savez(handle, request_indices=np.asarray(data.reserve_indices, dtype=np.int64), **{name: value.astype(np.float32) for name, value in per_request.items()})
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "post_outcome_attention_edge_attribution",
        "decision": direction,
        "execution_lock_sha256": lock_hash,
        "requests": len(data.reserve_indices),
        "users": len(np.unique(users)),
        "metrics": metrics,
        "classifications": classifications,
        "mechanics": checks,
        "boundary": {
            "same_already_open_reserve": True,
            "weights_changed": False,
            "checkpoint_selected": False,
            "fresh_generalization_claim": False,
            "dev_test_qrels_opened": False,
        },
        "per_request_metrics": {"path": str(metric_path.relative_to(ROOT)), "sha256": sha256_file(metric_path)},
    }
    atomic_json(output_path, report)
    print(json.dumps({"decision": direction, "classifications": classifications, "metrics": metrics, "report": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
