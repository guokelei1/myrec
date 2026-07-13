#!/usr/bin/env python
"""Aggregate the frozen three-seed R0 full-token stability replication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("true", "null", "wrong", "shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs=3, required=True)
    parser.add_argument("--stability-lock", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def comparison_value(report: dict[str, Any], comparison: str, surface: str) -> float:
    return float(report["comparisons"][comparison][surface]["mean"])


def main() -> int:
    args = parse_args()
    lock_path = ROOT / args.stability_lock
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    reports = [json.loads((ROOT / path).read_text(encoding="utf-8")) for path in args.summaries]
    expected = {str(row["trial_id"]) for row in lock["stability_seeds"]}
    actual = {str(row["trial_id"]) for row in reports}
    if actual != expected or len(reports) != len(actual):
        raise RuntimeError(f"stability report set differs: expected={expected} actual={actual}")
    reports.sort(key=lambda row: str(row["trial_id"]))
    if not all(bool(row["observability_gate"]["passed"]) for row in reports):
        observability_all_seeds = False
    else:
        observability_all_seeds = True
    candidate_hashes = {str(row["candidate_manifest_sha256"]) for row in reports}
    qrels_hashes = {str(row["shared_evaluator_qrels_sha256"]) for row in reports}
    if len(candidate_hashes) != 1 or len(qrels_hashes) != 1:
        raise RuntimeError("stability reports do not share evaluator inputs")

    scenario_stats = {
        scenario: stats([float(row["metrics"][scenario]) for row in reports])
        for scenario in SCENARIOS
    }
    effect_stats = {
        "true_minus_null_all": stats(
            [comparison_value(row, "true_minus_null", "all") for row in reports]
        ),
        "true_minus_wrong_matched": stats(
            [comparison_value(row, "true_minus_wrong", "wrong_matched") for row in reports]
        ),
        "true_minus_null_strict_nonrepeat": stats(
            [comparison_value(row, "true_minus_null", "strict_nonrepeat") for row in reports]
        ),
        "true_minus_shuffle_all": stats(
            [comparison_value(row, "true_minus_shuffle", "all") for row in reports]
        ),
    }
    true_mean = float(scenario_stats["true"]["mean"])
    waterlines = {
        "B7_BGE": 0.3305,
        "D2s_three_seed_mean": 0.34162898451860496,
        "item_only_three_seed_mean": 0.3453755427,
    }
    report = {
        "report_id": "pps_r0_full_token_stability",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_phase": "R0-C",
        "selection_lock": str(lock_path.relative_to(ROOT)),
        "selection_rule": lock["selection_rule"],
        "trial_ids": [str(row["trial_id"]) for row in reports],
        "candidate_manifest_sha256": next(iter(candidate_hashes)),
        "shared_evaluator_qrels_sha256": next(iter(qrels_hashes)),
        "scenario_ndcg_at_10": scenario_stats,
        "effects": effect_stats,
        "observability_passed_all_seeds": observability_all_seeds,
        "waterlines": {
            name: {"ndcg_at_10": value, "selected_true_mean_minus_waterline": true_mean - value}
            for name, value in waterlines.items()
        },
        "strong_baseline_status": {
            "normally_tuned": True,
            "configuration_frozen": True,
            "seeds": len(reports),
            "best_seed_reporting_forbidden": True,
            "ready_for_failure_atlas": observability_all_seeds,
        },
        "label_boundary": {
            "direct_qrels_read": False,
            "shared_evaluator_outputs_used": True,
            "test_opened": False,
            "c80_fresh_labels_opened": False,
        },
    }
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "true": scenario_stats["true"], "effects": effect_stats}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
