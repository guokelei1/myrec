#!/usr/bin/env python
"""Open C80 fresh labels after all scores pass and adjudicate the terminal gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_lock import verify_lock  # noqa: E402
from model.pitt import MODES  # noqa: E402
from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402
from myrec.analysis.token_history_observability import TokenHistoryData  # noqa: E402
from prepare_real_gate import load_config  # noqa: E402
from summarize_history_signal_observability import (  # noqa: E402
    cluster_bootstrap,
    derived_seed,
    request_ndcg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_fresh_labels(records_path: Path, data: TokenHistoryData) -> dict[int, np.ndarray]:
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
                raise ValueError("C80 fresh request differs at label opening")
            candidate_ids = [str(value["item_id"]) for value in row["candidates"]]
            if candidate_ids != data.candidate_ids(local):
                raise ValueError("C80 fresh candidate order differs at label opening")
            labels = np.asarray(
                [float(value.get("clicked", 0) or 0) for value in row["candidates"]],
                dtype=np.float32,
            )
            if int((labels > 0).sum()) != 1:
                raise ValueError("C80 expects exactly one clicked candidate")
            output[local] = labels
    if set(output) != set(int(value) for value in data.reserve_indices):
        raise ValueError("C80 fresh label coverage differs")
    return output


def user_ids(path: Path, indices: np.ndarray) -> np.ndarray:
    wanted = set(int(value) for value in indices)
    mapping = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            position = int(row["position"])
            if position in wanted:
                mapping[position] = str(row["user_id"])
    output = np.asarray([mapping[int(value)] for value in indices], dtype=object)
    if len(np.unique(output)) != len(output):
        raise ValueError("C80 fresh role unexpectedly shares users")
    return output


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    _, lock_hash = verify_lock(config, config_path)
    paths = config["paths"]
    root = ROOT / paths["fresh_root"]
    report_path = ROOT / paths["report"]
    if report_path.exists():
        raise FileExistsError(report_path)
    data = TokenHistoryData(root)
    seeds = [int(value) for value in config["training"]["seeds"]]
    score_files: dict[int, Any] = {}
    pre_label_checks: dict[str, bool] = {}
    parameter_surfaces = set()
    for seed in seeds:
        seed_report = json.loads(
            (root / f"seed_{seed}_report.json").read_text(encoding="utf-8")
        )
        score_path = root / f"seed_{seed}_scores.npz"
        scores = np.load(score_path, allow_pickle=False)
        pre_label_checks[str(seed)] = bool(
            seed_report["passed_mechanics"]
            and seed_report["execution_lock_sha256"] == lock_hash
            and sha256_file(score_path) == seed_report["scores"]["sha256"]
            and np.array_equal(scores["request_indices"], data.reserve_indices)
            and not seed_report["checks"]["fresh_labels_opened"]
        )
        parameter_surfaces.add(tuple(sorted(seed_report["trainable_parameters"].items())))
        score_files[seed] = scores
    pre_label_checks["all_three_seed_scores"] = len(score_files) == 3
    pre_label_checks["same_parameter_surface_across_seeds"] = len(parameter_surfaces) == 1
    if not all(pre_label_checks.values()):
        raise RuntimeError("C80 pre-label mechanics failed")
    reference = score_files[seeds[0]]
    for seed in seeds[1:]:
        current = score_files[seed]
        if not (
            np.array_equal(reference["request_indices"], current["request_indices"])
            and np.array_equal(reference["offsets"], current["offsets"])
        ):
            raise RuntimeError("C80 seed candidate surfaces differ")

    labels = load_fresh_labels(ROOT / paths["records_train"], data)
    users = user_ids(root / "requests.jsonl", data.reserve_indices)
    method_keys = ["base"]
    method_keys.extend(f"external_{scenario}" for scenario in ("true", "wrong", "shuffle"))
    method_keys.extend(
        f"{mode}_{scenario}"
        for mode in MODES
        for scenario in ("true", "wrong", "shuffle", "null")
    )
    per_seed = {
        seed: {key: np.empty(len(data.reserve_indices), dtype=np.float64) for key in method_keys}
        for seed in seeds
    }
    ensemble_raw = {
        key: np.mean(
            np.stack([np.asarray(score_files[seed][key]) for seed in seeds]), axis=0
        )
        for key in method_keys
    }
    ensemble = {
        key: np.empty(len(data.reserve_indices), dtype=np.float64) for key in method_keys
    }
    for row, index_value in enumerate(data.reserve_indices):
        index = int(index_value)
        start, stop = int(reference["offsets"][row]), int(reference["offsets"][row + 1])
        item_ids = np.asarray(data.candidate_ids(index), dtype=object)
        for seed in seeds:
            for key in method_keys:
                per_seed[seed][key][row] = request_ndcg(
                    data.request_ids[index],
                    item_ids,
                    score_files[seed][key][start:stop],
                    labels[index],
                )
        for key in method_keys:
            ensemble[key][row] = request_ndcg(
                data.request_ids[index],
                item_ids,
                ensemble_raw[key][start:stop],
                labels[index],
            )

    metrics = {
        "ensemble": {key: float(values.mean()) for key, values in ensemble.items()},
        "seeds": {
            str(seed): {
                key: float(values.mean()) for key, values in per_seed[seed].items()
            }
            for seed in seeds
        },
    }
    primary = "triadic_set_true"
    comparisons_to = {
        "primary_minus_base": "base",
        "primary_true_minus_wrong": "triadic_set_wrong",
        "primary_minus_query_filtered_set": "query_filtered_set_true",
        "primary_minus_pairwise_set": "pairwise_set_true",
        "primary_minus_triadic_positional": "triadic_positional_true",
        "primary_minus_ungated_full": "ungated_full_true",
        "primary_minus_external_full_token": "external_true",
    }
    comparisons = {}
    samples = int(config["evaluation"]["bootstrap_samples"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    for name, right in comparisons_to.items():
        difference = ensemble[primary] - ensemble[right]
        row = cluster_bootstrap(
            difference,
            users,
            samples=samples,
            seed=derived_seed(bootstrap_seed, name),
        )
        row["seed_means"] = {
            str(seed): float((per_seed[seed][primary] - per_seed[seed][right]).mean())
            for seed in seeds
        }
        row["all_seeds_positive"] = all(value > 0 for value in row["seed_means"].values())
        comparisons[name] = row
    shuffle_difference = ensemble[primary] - ensemble["triadic_set_shuffle"]
    shuffle_row = cluster_bootstrap(
        shuffle_difference,
        users,
        samples=samples,
        seed=derived_seed(bootstrap_seed, "primary_true_minus_shuffle"),
    )
    shuffle_row["absolute_mean"] = abs(float(shuffle_row["mean"]))
    shuffle_row["seed_means"] = {
        str(seed): float(
            (per_seed[seed][primary] - per_seed[seed]["triadic_set_shuffle"]).mean()
        )
        for seed in seeds
    }
    comparisons["primary_true_minus_shuffle"] = shuffle_row

    positive_names = list(comparisons_to)
    gate = {
        "base_practical_effect": comparisons["primary_minus_base"]["mean"]
        >= float(config["evaluation"]["practical_effect_min"]),
        "all_registered_differences_ci_positive": all(
            comparisons[name]["user_cluster_95_ci"][0] > 0 for name in positive_names
        ),
        "all_registered_differences_all_seeds_positive": all(
            comparisons[name]["all_seeds_positive"] for name in positive_names
        ),
        "shuffle_absolute_effect_bounded": shuffle_row["absolute_mean"]
        <= float(config["evaluation"]["shuffle_abs_effect_max"]),
        "all_pre_label_mechanics": all(pre_label_checks.values()),
        "terminal_candidate_no_successor": True,
    }
    passed = all(gate.values())
    decision = (
        "c80_passes_fresh_amazon_gate_authorize_cross_domain_boundary_only"
        if passed
        else "close_c80_and_architecture_search_begin_c01_c80_retrospective"
    )
    metric_path = root / "per_request_metrics.npz"
    with metric_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=np.asarray(data.reserve_indices, dtype=np.int64),
            users=users.astype(str),
            **{f"ensemble_{key}": value.astype(np.float32) for key, value in ensemble.items()},
            **{
                f"seed_{seed}_{key}": value.astype(np.float32)
                for seed in seeds
                for key, value in per_seed[seed].items()
            },
        )
    report = {
        "candidate_id": "c80",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "fresh_amazon_terminal_real_gate_outcome",
        "decision": decision,
        "passed": passed,
        "execution_lock_sha256": lock_hash,
        "fresh_requests": len(data.reserve_indices),
        "fresh_users": len(np.unique(users)),
        "metrics": metrics,
        "comparisons": comparisons,
        "gate": gate,
        "pre_label_checks": pre_label_checks,
        "label_boundary": {
            "fresh_labels_opened_only_after_all_scores": True,
            "dev_test_qrels_opened": False,
        },
        "terminal_search_boundary": {
            "c80_is_final_architecture": True,
            "c81_authorized": False,
            "c01_c80_retrospective_required": True,
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
                "primary_ndcg": metrics["ensemble"][primary],
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
