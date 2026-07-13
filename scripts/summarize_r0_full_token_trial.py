#!/usr/bin/env python
"""Summarize a shared-evaluator R0 full-token trial with paired surface inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from summarize_history_signal_observability import cluster_bootstrap, derived_seed  # noqa: E402


SCENARIOS = ("true", "null", "wrong", "shuffle")
COMPARISONS = {
    "true_minus_null": ("true", "null"),
    "true_minus_wrong": ("true", "wrong"),
    "true_minus_shuffle": ("true", "shuffle"),
    "wrong_minus_null": ("wrong", "null"),
}


def request_surfaces(record: dict[str, Any], *, wrong_matched: bool) -> set[str]:
    candidates = {str(value["item_id"]) for value in record["candidates"]}
    history = {str(value["item_id"]) for value in record["history"]}
    surfaces = {"all"}
    if not history:
        surfaces.add("no_history")
    else:
        surfaces.add("history_present")
        surfaces.add("repeat_present" if candidates & history else "strict_nonrepeat")
        if wrong_matched:
            surfaces.add("wrong_matched")
    return surfaces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--records-dev", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20267100)
    return parser.parse_args()


def trial_slug(trial_id: str) -> str:
    return trial_id.split("-")[-1].lower()


def load_per_request(run_id: str) -> dict[str, float]:
    path = ROOT / "runs" / run_id / "per_request_metrics.jsonl"
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[str(row["request_id"])] = float(row["ndcg@10"])
    return output


def load_records(path: Path) -> list[dict[str, Any]]:
    if "qrels" in path.name.lower() or "test" in path.name.lower():
        raise PermissionError(f"unauthorized summary input: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if any("clicked" in value or "purchased" in value for value in row["candidates"]):
                raise PermissionError("R0 full-token summary requires blind dev records")
            rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    slug = trial_slug(args.trial_id)
    run_ids = {
        scenario: f"20260713_kuaisearch_r0ft_{slug}_{scenario}_dev"
        for scenario in SCENARIOS
    }
    metrics = {
        scenario: json.loads(
            (ROOT / "runs" / run_id / "metrics.json").read_text(encoding="utf-8")
        )
        for scenario, run_id in run_ids.items()
    }
    candidate_hashes = {row["candidate_manifest_sha256"] for row in metrics.values()}
    qrels_hashes = {row["qrels_sha256"] for row in metrics.values()}
    if len(candidate_hashes) != 1 or len(qrels_hashes) != 1:
        raise RuntimeError("R0 full-token evaluator surfaces differ")
    values_by_scenario = {scenario: load_per_request(run_id) for scenario, run_id in run_ids.items()}
    request_sets = {frozenset(values) for values in values_by_scenario.values()}
    if len(request_sets) != 1:
        raise RuntimeError("R0 full-token per-request surfaces differ")

    records = load_records(ROOT / args.records_dev)
    record_by_id = {str(row["request_id"]): row for row in records}
    request_ids = sorted(next(iter(request_sets)))
    if set(record_by_id) != set(request_ids):
        raise RuntimeError("R0 full-token evaluator requests differ from blind dev")
    data = TokenHistoryData(ROOT / args.artifact_root)
    matched_wrong = {
        data.request_ids[int(index)]
        for index in data.reserve_indices
        if len(data.history(int(index), "wrong", 50)) > 0
    }
    users = np.asarray([str(record_by_id[request_id]["user_id"]) for request_id in request_ids])
    arrays = {
        scenario: np.asarray([values_by_scenario[scenario][request_id] for request_id in request_ids])
        for scenario in SCENARIOS
    }
    surface_names = ("all", "history_present", "repeat_present", "strict_nonrepeat", "no_history", "wrong_matched")
    masks = {
        surface: np.asarray(
            [
                surface in request_surfaces(
                    record_by_id[request_id], wrong_matched=request_id in matched_wrong
                )
                for request_id in request_ids
            ],
            dtype=bool,
        )
        for surface in surface_names
    }
    surface_metrics = {
        surface: {
            "requests": int(mask.sum()),
            "users": int(len(np.unique(users[mask]))),
            **{
                scenario: float(arrays[scenario][mask].mean()) if bool(mask.any()) else None
                for scenario in SCENARIOS
            },
        }
        for surface, mask in masks.items()
    }
    comparisons = {}
    for name, (left, right) in COMPARISONS.items():
        comparisons[name] = {}
        for surface, mask in masks.items():
            difference = arrays[left][mask] - arrays[right][mask]
            seed = derived_seed(args.bootstrap_seed, f"{args.trial_id}:{name}:{surface}")
            user_row = cluster_bootstrap(
                difference, users[mask], samples=args.bootstrap_samples, seed=seed
            )
            request_row = cluster_bootstrap(
                difference,
                np.asarray(request_ids)[mask],
                samples=args.bootstrap_samples,
                seed=seed + 1,
            )
            user_row["request_paired_95_ci"] = request_row["user_cluster_95_ci"]
            comparisons[name][surface] = user_row

    mechanics_report = json.loads(
        (ROOT / args.artifact_root / "trials" / f"{slug}_report.json").read_text(
            encoding="utf-8"
        )
    )
    pooled = json.loads(
        (ROOT / "reports/pps_history_signal_observability_r1.json").read_text(encoding="utf-8")
    )
    amazon = json.loads(
        (ROOT / "reports/pps_amazon_token_history_observability_v1.json").read_text(
            encoding="utf-8"
        )
    )
    true_null = comparisons["true_minus_null"]["all"]
    true_wrong = comparisons["true_minus_wrong"]["wrong_matched"]
    report = {
        "report_id": f"pps_r0_full_token_{slug}_summary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_phase": "R0-B" if args.trial_id.startswith("R0B") else "R0-C",
        "trial_id": args.trial_id,
        "run_ids": run_ids,
        "candidate_manifest_sha256": next(iter(candidate_hashes)),
        "shared_evaluator_qrels_sha256": next(iter(qrels_hashes)),
        "metrics": {scenario: row["ndcg@10"] for scenario, row in metrics.items()},
        "surface_metrics": surface_metrics,
        "comparisons": comparisons,
        "mechanics": mechanics_report["mechanics"],
        "gpu_hours": mechanics_report["gpu_hours"],
        "observability_gate": {
            "minimum_effect": 0.002,
            "true_null_effect": true_null["mean"],
            "true_null_user_ci_positive": true_null["user_cluster_95_ci"][0] > 0,
            "true_wrong_matched_effect": true_wrong["mean"],
            "true_wrong_matched_user_ci_positive": true_wrong["user_cluster_95_ci"][0] > 0,
            "passed": (
                true_null["mean"] >= 0.002
                and true_null["user_cluster_95_ci"][0] > 0
                and true_wrong["user_cluster_95_ci"][0] > 0
            ),
        },
        "parity_context": {
            "historical_kuai_pooled_true_minus_null": pooled["comparisons"][
                "full_true_minus_own_null"
            ],
            "historical_amazon_full_token_true_minus_null": amazon["comparisons"][
                "true_minus_null"
            ],
            "target_attention_control": {
                "source": "experiments/pps_results.md D1a",
                "boundary": "historical frozen-embedding target-attention residual, not same-checkpoint parity",
                "three_seed_mean_ndcg": 0.3148,
            },
        },
        "strong_baseline_status": {
            "current_true_ndcg": metrics["true"]["ndcg@10"],
            "current_static_item_only_waterline": 0.3454,
            "normally_tuned": False,
            "next_registered_trial": {
                "R0B-T001": "R0C-T002",
                "R0C-T002": "R0C-T003",
                "R0C-T003": "R0C-T004",
                "R0C-T004": None,
            }.get(args.trial_id),
            "reason": "full-token source is observable, but the current recipe remains below the static waterline and registered tuning classes remain",
        },
        "label_boundary": {
            "summary_read_qrels_directly": False,
            "shared_evaluator_dev_outputs_used": True,
            "test_opened": False,
            "c80_fresh_labels_opened": False,
        },
    }
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "observability_passed": report["observability_gate"]["passed"],
                "true_minus_null": true_null,
                "true_minus_wrong_matched": true_wrong,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
