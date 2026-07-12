#!/usr/bin/env python
"""Open the token-HSO reserve labels after all seed scores pass mechanics."""

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

from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402
from myrec.analysis.token_history_observability import TokenHistoryData  # noqa: E402
from prepare_amazon_token_history_observability import load_config, verify_lock  # noqa: E402
from summarize_history_signal_observability import (  # noqa: E402
    cluster_bootstrap,
    derived_seed,
    request_ndcg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_reserve_labels(
    records_path: Path, data: TokenHistoryData
) -> dict[int, np.ndarray]:
    wanted = {
        int(data.original_indices[index]): int(index) for index in data.reserve_indices
    }
    output: dict[int, np.ndarray] = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for original_index, line in enumerate(handle):
            local = wanted.get(original_index)
            if local is None:
                continue
            row = json.loads(line)
            if str(row["request_id"]) != data.request_ids[local]:
                raise ValueError("token HSO reserve request differs")
            if [str(value["item_id"]) for value in row["candidates"]] != data.candidate_ids(local):
                raise ValueError("token HSO reserve candidate order differs")
            labels = np.asarray(
                [float(value.get("clicked", 0) or 0) for value in row["candidates"]],
                dtype=np.float32,
            )
            if int((labels > 0).sum()) != 1:
                raise ValueError("token HSO reserve expects one positive")
            output[local] = labels
    if set(output) != set(data.reserve_indices.tolist()):
        raise ValueError("token HSO reserve label coverage differs")
    return output


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    _, lock_hash = verify_lock(config, config_path)
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    report_path = ROOT / paths["report"]
    if report_path.exists():
        raise FileExistsError(report_path)
    data = TokenHistoryData(root)
    seeds = [int(value) for value in config["training"]["seeds"]]
    score_files: dict[int, Any] = {}
    checks: dict[str, bool] = {}
    parameters = set()
    for seed in seeds:
        report = json.loads(
            (root / f"seed_{seed}_report.json").read_text(encoding="utf-8")
        )
        score_path = root / f"seed_{seed}_scores.npz"
        scores = np.load(score_path, allow_pickle=False)
        checks[str(seed)] = bool(
            report["passed_mechanics"]
            and report["execution_lock_sha256"] == lock_hash
            and sha256_file(score_path) == report["scoring"]["sha256"]
            and np.array_equal(scores["request_indices"], data.reserve_indices)
            and not report["checks"]["reserve_labels_opened"]
        )
        parameters.add(int(report["parameters"]))
        score_files[seed] = scores
    checks["all_three_scores"] = len(score_files) == 3
    checks["equal_parameters"] = len(parameters) == 1
    if not all(checks.values()):
        raise RuntimeError("token HSO pre-label mechanics failed")
    reference = score_files[seeds[0]]
    for seed in seeds[1:]:
        current = score_files[seed]
        if not (
            np.array_equal(reference["request_indices"], current["request_indices"])
            and np.array_equal(reference["offsets"], current["offsets"])
        ):
            raise RuntimeError("token HSO seed candidate surface differs")

    labels = load_reserve_labels(ROOT / paths["records_train"], data)
    users_by_index = {}
    with (root / "requests.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            users_by_index[int(row["position"])] = str(row["user_id"])
    users = np.asarray([users_by_index[int(index)] for index in data.reserve_indices])
    scenarios = ("true", "null", "wrong", "shuffle")
    per_seed: dict[int, dict[str, np.ndarray]] = {
        seed: {name: np.empty(len(data.reserve_indices), dtype=np.float64) for name in scenarios}
        for seed in seeds
    }
    ensemble_scores = {
        name: np.mean(
            np.stack([np.asarray(score_files[seed][name]) for seed in seeds], axis=0), axis=0
        )
        for name in scenarios
    }
    ensemble = {name: np.empty(len(data.reserve_indices), dtype=np.float64) for name in scenarios}
    for row, index_value in enumerate(data.reserve_indices):
        index = int(index_value)
        start, stop = int(reference["offsets"][row]), int(reference["offsets"][row + 1])
        item_ids = np.asarray(data.candidate_ids(index), dtype=object)
        for seed in seeds:
            for scenario in scenarios:
                per_seed[seed][scenario][row] = request_ndcg(
                    data.request_ids[index],
                    item_ids,
                    score_files[seed][scenario][start:stop],
                    labels[index],
                )
        for scenario in scenarios:
            ensemble[scenario][row] = request_ndcg(
                data.request_ids[index],
                item_ids,
                ensemble_scores[scenario][start:stop],
                labels[index],
            )
    metrics = {
        "ensemble": {name: float(values.mean()) for name, values in ensemble.items()},
        "seeds": {
            str(seed): {
                name: float(values.mean()) for name, values in per_seed[seed].items()
            }
            for seed in seeds
        },
    }
    samples = int(config["evaluation"]["bootstrap_samples"])
    base_seed = int(config["evaluation"]["bootstrap_seed"])
    comparisons = {}
    for name, right in (("true_minus_null", "null"), ("true_minus_wrong", "wrong"), ("true_minus_shuffle", "shuffle")):
        difference = ensemble["true"] - ensemble[right]
        row = cluster_bootstrap(
            difference,
            users,
            samples=samples,
            seed=derived_seed(base_seed, name),
        )
        row["seed_means"] = {
            str(seed): float((per_seed[seed]["true"] - per_seed[seed][right]).mean())
            for seed in seeds
        }
        row["all_seeds_positive"] = all(value > 0 for value in row["seed_means"].values())
        comparisons[name] = row
    minimum = float(config["evaluation"]["observable_min_ndcg"])
    gate = {
        "true_null_minimum_effect": comparisons["true_minus_null"]["mean"] >= minimum,
        "true_null_ci_positive": comparisons["true_minus_null"]["user_cluster_95_ci"][0] > 0,
        "true_null_all_seeds_positive": comparisons["true_minus_null"]["all_seeds_positive"],
        "true_wrong_ci_positive": comparisons["true_minus_wrong"]["user_cluster_95_ci"][0] > 0,
        "true_wrong_all_seeds_positive": comparisons["true_minus_wrong"]["all_seeds_positive"],
        "all_mechanics": all(checks.values()),
    }
    passed = all(gate.values())
    decision = (
        "token_semantic_history_observable_authorize_architecture_primitive"
        if passed
        else "token_semantic_history_not_observable_change_data_contract_or_narrow_claim"
    )
    metric_path = root / "per_request_metrics.npz"
    with metric_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=np.asarray(data.reserve_indices, dtype=np.int64),
            **{f"ensemble_{name}": value.astype(np.float32) for name, value in ensemble.items()},
            **{
                f"seed_{seed}_{name}": value.astype(np.float32)
                for seed in seeds
                for name, value in per_seed[seed].items()
            },
        )
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "full_token_unopened_reserve_outcome",
        "decision": decision,
        "passed": passed,
        "execution_lock_sha256": lock_hash,
        "reserve_requests": len(data.reserve_indices),
        "reserve_users": len(np.unique(users)),
        "metrics": metrics,
        "comparisons": comparisons,
        "gate": gate,
        "pre_label_checks": checks,
        "parameters": next(iter(parameters)),
        "label_boundary": {
            "reserve_labels_opened_only_after_all_scores": True,
            "dev_test_qrels_opened": False,
        },
        "per_request_metrics": {
            "path": str(metric_path.relative_to(ROOT)),
            "sha256": sha256_file(metric_path),
        },
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "decision": decision,
                "passed": passed,
                "metrics": metrics,
                "comparisons": comparisons,
                "gate": gate,
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
