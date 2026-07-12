#!/usr/bin/env python
"""Freeze and execute the token-HSO closed-label report-flag recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402
from myrec.analysis.token_history_observability import TokenHistoryData  # noqa: E402
from summarize_amazon_token_history_observability import load_reserve_labels  # noqa: E402
from summarize_history_signal_observability import (  # noqa: E402
    cluster_bootstrap,
    derived_seed,
    request_ndcg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("token HSO recovery config must be a mapping")
    return value


def source_paths(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    output = {
        "recovery_config": config_path,
        "recovery_protocol": ROOT / paths["protocol"],
        "recovery_script": ROOT / paths["script"],
        "base_summarizer": ROOT / paths["base_summarizer"],
        "source_config": ROOT / paths["source_config"],
        "source_execution_lock": ROOT / paths["source_execution_lock"],
        "token_manifest": root / "token_manifest.json",
        "request_original_indices": root / "request_original_indices.npy",
        "request_roles": root / "request_roles.npy",
        "candidate_offsets": root / "candidate_offsets.npy",
        "candidate_positions": root / "candidate_item_positions.npy",
        "requests": root / "requests.jsonl",
        "items": root / "items.jsonl",
    }
    for seed in config["evaluation"]["seeds"]:
        output[f"seed_{seed}_report"] = root / f"seed_{seed}_report.json"
        output[f"seed_{seed}_scores"] = root / f"seed_{seed}_scores.npz"
    return output


def audit_reports(config: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / config["paths"]["artifact_root"]
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    rows = {}
    for seed in seeds:
        report = json.loads((root / f"seed_{seed}_report.json").read_text(encoding="utf-8"))
        substantive = {
            key: value
            for key, value in report["checks"].items()
            if key != "reserve_labels_opened"
        }
        rows[str(seed)] = {
            "all_substantive_checks": all(substantive.values()),
            "reserve_labels_closed": not bool(report["checks"]["reserve_labels_opened"]),
            "aggregate_false_only_from_closed_flag": not bool(report["passed_mechanics"])
            and all(substantive.values())
            and not bool(report["checks"]["reserve_labels_opened"]),
            "score_hash": sha256_file(root / f"seed_{seed}_scores.npz")
            == report["scoring"]["sha256"],
        }
    return {
        "seeds": rows,
        "all_valid_under_corrected_semantics": all(
            all(row.values()) for row in rows.values()
        ),
        "report_absent": not (ROOT / config["paths"]["report"]).exists(),
    }


def freeze(config: dict[str, Any], config_path: Path) -> None:
    lock_path = ROOT / config["paths"]["recovery_lock"]
    if lock_path.exists():
        raise FileExistsError(lock_path)
    audit = audit_reports(config)
    if not audit["all_valid_under_corrected_semantics"] or not audit["report_absent"]:
        raise RuntimeError("token HSO recovery pre-outcome audit failed")
    lock = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "authorize_existing_score_readout_with_closed_flag_semantics_corrected",
        "source_sha256": {
            key: sha256_file(path) for key, path in source_paths(config, config_path).items()
        },
        "preoutcome_audit": audit,
        "outcome_boundary": {
            "training": False,
            "rescoring": False,
            "checkpoint_selection": False,
            "reserve_labels_before_recovery_lock": False,
            "dev_test_qrels": False,
        },
    }
    atomic_json(lock_path, lock)
    print(json.dumps({"path": str(lock_path), "sha256": sha256_file(lock_path)}, sort_keys=True))


def verify(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    lock_path = ROOT / config["paths"]["recovery_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if {key: sha256_file(path) for key, path in source_paths(config, config_path).items()} != lock[
        "source_sha256"
    ]:
        raise RuntimeError("token HSO recovery source changed")
    if not lock["preoutcome_audit"]["all_valid_under_corrected_semantics"]:
        raise RuntimeError("token HSO recovery lock is not valid")
    return lock, sha256_file(lock_path)


def evaluate(config: dict[str, Any], config_path: Path) -> None:
    _, recovery_hash = verify(config, config_path)
    paths = config["paths"]
    source = load_yaml(ROOT / paths["source_config"])
    root = ROOT / paths["artifact_root"]
    report_path = ROOT / paths["report"]
    if report_path.exists():
        raise FileExistsError(report_path)
    data = TokenHistoryData(root)
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    score_files = {
        seed: np.load(root / f"seed_{seed}_scores.npz", allow_pickle=False)
        for seed in seeds
    }
    reference = score_files[seeds[0]]
    for seed in seeds[1:]:
        current = score_files[seed]
        if not (
            np.array_equal(reference["request_indices"], current["request_indices"])
            and np.array_equal(reference["offsets"], current["offsets"])
        ):
            raise RuntimeError("token HSO recovery seed surface differs")
    labels = load_reserve_labels(ROOT / paths["records_train"], data)
    users_by_index = {}
    with (root / "requests.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            users_by_index[int(row["position"])] = str(row["user_id"])
    users = np.asarray([users_by_index[int(index)] for index in data.reserve_indices])
    scenarios = ("true", "null", "wrong", "shuffle")
    per_seed = {
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
                    data.request_ids[index], item_ids, score_files[seed][scenario][start:stop], labels[index]
                )
        for scenario in scenarios:
            ensemble[scenario][row] = request_ndcg(
                data.request_ids[index], item_ids, ensemble_scores[scenario][start:stop], labels[index]
            )
    metrics = {
        "ensemble": {name: float(value.mean()) for name, value in ensemble.items()},
        "seeds": {
            str(seed): {name: float(value.mean()) for name, value in per_seed[seed].items()}
            for seed in seeds
        },
    }
    comparisons = {}
    samples = int(config["evaluation"]["bootstrap_samples"])
    base_seed = int(config["evaluation"]["bootstrap_seed"])
    for name, right in (
        ("true_minus_null", "null"),
        ("true_minus_wrong", "wrong"),
        ("true_minus_shuffle", "shuffle"),
    ):
        result = cluster_bootstrap(
            ensemble["true"] - ensemble[right],
            users,
            samples=samples,
            seed=derived_seed(base_seed, name),
        )
        result["seed_means"] = {
            str(seed): float((per_seed[seed]["true"] - per_seed[seed][right]).mean())
            for seed in seeds
        }
        result["all_seeds_positive"] = all(value > 0 for value in result["seed_means"].values())
        comparisons[name] = result
    minimum = float(config["evaluation"]["observable_min_ndcg"])
    gate = {
        "true_null_minimum_effect": comparisons["true_minus_null"]["mean"] >= minimum,
        "true_null_ci_positive": comparisons["true_minus_null"]["user_cluster_95_ci"][0] > 0,
        "true_null_all_seeds_positive": comparisons["true_minus_null"]["all_seeds_positive"],
        "true_wrong_ci_positive": comparisons["true_minus_wrong"]["user_cluster_95_ci"][0] > 0,
        "true_wrong_all_seeds_positive": comparisons["true_minus_wrong"]["all_seeds_positive"],
        "all_mechanics": True,
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
        "stage": "full_token_unopened_reserve_recovery_outcome",
        "decision": decision,
        "passed": passed,
        "source_execution_lock_sha256": sha256_file(ROOT / paths["source_execution_lock"]),
        "recovery_lock_sha256": recovery_hash,
        "reserve_requests": len(data.reserve_indices),
        "reserve_users": len(np.unique(users)),
        "metrics": metrics,
        "comparisons": comparisons,
        "gate": gate,
        "mechanical_recovery": {
            "correct_semantics": "all substantive checks true and reserve_labels_opened false",
            "training_or_rescoring": False,
        },
        "label_boundary": {
            "reserve_labels_opened_only_after_recovery_lock": True,
            "dev_test_qrels_opened": False,
        },
        "parameters": int(
            json.loads((root / f"seed_{seeds[0]}_report.json").read_text(encoding="utf-8"))[
                "parameters"
            ]
        ),
        "per_request_metrics": {
            "path": str(metric_path.relative_to(ROOT)),
            "sha256": sha256_file(metric_path),
        },
    }
    atomic_json(report_path, report)
    print(json.dumps({"decision": decision, "passed": passed, "metrics": metrics, "comparisons": comparisons, "gate": gate, "report": str(report_path)}, sort_keys=True))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    if args.stage == "freeze":
        freeze(config, config_path)
    else:
        evaluate(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
