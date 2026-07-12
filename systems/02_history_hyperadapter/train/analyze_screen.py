#!/usr/bin/env python
"""Audit the frozen C02 screen after its single shared-evaluator call."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for entry in (SYSTEM_ROOT, REPO_ROOT / "src"):
    sys.path.insert(0, str(entry))

from myrec.eval.compare import compare_per_request_metrics
from train.runtime import (
    assert_candidate_hash,
    assert_proposal_lock,
    load_config,
    read_json,
    sha256_file,
    write_json,
)

CORRUPTIONS = ("wrong", "shuffle", "coarse", "query_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="systems/02_history_hyperadapter/configs/screen.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    candidate_hash = assert_candidate_hash(config)
    proposal_lock = assert_proposal_lock(config)
    run_id = str(config["run_id"])
    run_dir = Path("runs") / run_id
    diagnostic_root = Path(config["paths"]["diagnostic_root"])
    metrics_path = run_dir / "metrics.json"
    per_request_path = run_dir / "per_request_metrics.jsonl"
    if not metrics_path.exists() or not per_request_path.exists():
        raise FileNotFoundError("the single shared evaluation must run before analysis")
    metrics = read_json(metrics_path)
    metadata = read_json(run_dir / "metadata.json")
    train_summary = read_json(diagnostic_root / "train_summary.json")
    score_summary = read_json(diagnostic_root / "score_summary.json")
    diagnostics = read_json(diagnostic_root / "dev_functional_diagnostics.json")

    nonrepeat_comparison = compare_per_request_metrics(
        run_a_path=per_request_path,
        run_b_path=Path("runs")
        / config["paths"]["d2p_dev_run"]
        / "per_request_metrics.jsonl",
        output_path=diagnostic_root / "nonrepeat_vs_d2p.json",
        metric="ndcg@10",
        samples=10000,
        seed=int(config["seed"]),
        request_ids_path=diagnostic_root / "nonrepeat_request_ids.txt",
    )
    repeat_comparison = compare_per_request_metrics(
        run_a_path=per_request_path,
        run_b_path=Path("runs")
        / config["paths"]["item_only_dev_run"]
        / "per_request_metrics.jsonl",
        output_path=diagnostic_root / "repeat_vs_item_only.json",
        metric="ndcg@10",
        samples=10000,
        seed=int(config["seed"]),
        request_ids_path=diagnostic_root / "repeat_request_ids.txt",
    )

    c02_log_rows, exact_log_rows = _dev_log_counts(
        Path("reports/dev_eval_log.jsonl"), run_id
    )
    internal = train_summary["internal_gate"]
    selected = train_summary["variants"]["chht"]["selected_internal"]
    benchmark = {
        name: row["inference_benchmark"]["elapsed_seconds"]
        for name, row in train_summary["variants"].items()
    }
    chht_time = float(benchmark["chht"])
    latency_relative_deltas = {
        name: abs(float(value) - chht_time) / chht_time
        for name, value in benchmark.items()
        if name != "chht"
    }
    max_latency_delta = max(latency_relative_deltas.values())

    feature_manifest = read_json(Path(config["paths"]["feature_root"]) / "manifest.json")
    elapsed_seconds = (
        float(feature_manifest["elapsed_seconds"])
        + float(train_summary["elapsed_seconds"])
        + float(score_summary["elapsed_seconds"])
    )
    gpu_hours = elapsed_seconds / 3600.0

    integrity_checks = {
        "candidate_hash": bool(
            candidate_hash == config["integrity"]["candidate_manifest_sha256"]
            == metrics["candidate_manifest_sha256"]
            == metadata["candidate_manifest_sha256"]
        ),
        "proposal_source_hashes": True,
        "environment_gpu": bool(
            metadata["environment"] == "myrec-c02"
            and metadata["gpu"]["physical"] == 1
            and metadata["gpu"]["program_device"] == "cuda:0"
        ),
        "score_coverage": bool(
            int(metadata["request_count"]) == int(config["integrity"]["expected_requests"])
            and int(metadata["score_rows"]) == int(config["integrity"]["expected_score_rows"])
        ),
        "structural_subset_counts": bool(
            diagnostics["structural_subsets"]
            == {
                "repeat": int(config["integrity"]["expected_repeat_present"]),
                "nonrepeat": int(config["integrity"]["expected_nonrepeat_present"]),
                "no_history": int(config["integrity"]["expected_no_history"]),
            }
        ),
        "single_dev_evaluator_row": bool(c02_log_rows == 1 and exact_log_rows == 1),
        "label_and_test_boundary": bool(
            feature_manifest["label_boundary"]["separated_evaluation_labels_read"] is False
            and feature_manifest["label_boundary"]["held_out_test_data_read"] is False
            and train_summary["label_boundary"]["evaluation_labels_read"] is False
            and score_summary["label_boundary"]["evaluation_labels_read"] is False
            and score_summary["label_boundary"]["test_data_read"] is False
        ),
        "gpu_budget": bool(gpu_hours <= float(config["training"]["max_gpu_hours"])),
        "matched_parameter_count": bool(
            len(set(train_summary["parameter_counts_by_variant"].values())) == 1
        ),
        "matched_optimizer_steps": bool(
            len(set(train_summary["optimizer_steps_by_variant"].values())) == 1
        ),
        "matched_inference_wall_time": bool(
            max_latency_delta <= float(config["controls"]["wall_time_relative_tolerance"])
        ),
    }

    corrupt_checks = {
        name: bool(
            diagnostics["corruptions"][name]["corrupt_to_true_core_norm_ratio"]
            <= float(config["dev_gate"]["corrupt_to_true_core_norm_ratio_max"])
            and diagnostics["corruptions"][name]["changed_request_fraction"]
            >= float(config["dev_gate"]["corrupt_changed_request_fraction_min"])
        )
        for name in CORRUPTIONS
    }
    scientific_checks = {
        "train_internal_gate": bool(internal["passed"]),
        "overall_not_below_d2p": bool(
            float(metrics["ndcg@10"]) >= float(config["dev_gate"]["overall_d2p_floor"])
        ),
        "nonrepeat_gain_and_ci": bool(
            float(nonrepeat_comparison["delta"])
            >= float(config["dev_gate"]["nonrepeat_delta_vs_d2p_min"])
            and float(nonrepeat_comparison["ci95"][0]) > 0.0
        ),
        "repeat_noninferiority": bool(
            float(repeat_comparison["delta"])
            >= float(config["dev_gate"]["repeat_delta_vs_item_min"])
            and float(repeat_comparison["ci95"][0])
            >= float(config["dev_gate"]["repeat_ci_low_min"])
        ),
        "no_history_exact_score_and_rank": bool(
            diagnostics["no_history"]["score_mismatch_requests"] == 0
            and diagnostics["no_history"]["rank_mismatch_requests"] == 0
        ),
        "all_corruption_contracts": all(corrupt_checks.values()),
        "history_only_does_not_match": bool(
            selected["nonrepeat"]["model_ndcg@10"]
            - train_summary["variants"]["history_only"]["selected_internal"]["nonrepeat"]["model_ndcg@10"]
            >= float(config["internal_gate"]["margin_over_best_control_min"])
        ),
        "deterministic_rescore": bool(
            diagnostics["deterministic_rescore"]["requests"]
            == int(config["dev_gate"]["deterministic_requests"])
            and diagnostics["deterministic_rescore"]["missing_requests"] == 0
            and diagnostics["deterministic_rescore"]["max_abs_score_delta"]
            <= float(config["dev_gate"]["deterministic_max_abs_delta"])
        ),
    }
    integrity_passed = all(integrity_checks.values())
    scientific_passed = all(scientific_checks.values())
    if not integrity_passed:
        decision = "stop"
    elif scientific_passed:
        decision = "advance-to-full-gate"
    else:
        decision = "pivot-before-more-dev"

    result: dict[str, Any] = {
        "analysis_id": config["analysis_id"],
        "candidate_id": config["candidate_id"],
        "candidate_manifest_sha256": candidate_hash,
        "comparisons": {
            "nonrepeat_vs_d2p": nonrepeat_comparison,
            "repeat_vs_item_only": repeat_comparison,
        },
        "corruption_checks": corrupt_checks,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "dev_evaluator": {
            "all_c02_log_rows": c02_log_rows,
            "exact_run_log_rows": exact_log_rows,
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "ndcg@10": metrics["ndcg@10"],
            "per_request_path": str(per_request_path),
            "per_request_sha256": sha256_file(per_request_path),
        },
        "gpu_budget": {
            "elapsed_seconds": elapsed_seconds,
            "gpu_hours": gpu_hours,
            "limit_hours": float(config["training"]["max_gpu_hours"]),
        },
        "integrity_checks": integrity_checks,
        "integrity_passed": integrity_passed,
        "matched_capacity": {
            "inference_elapsed_seconds": benchmark,
            "latency_relative_delta_vs_chht": latency_relative_deltas,
            "max_latency_relative_delta": max_latency_delta,
            "parameter_counts": train_summary["parameter_counts_by_variant"],
            "optimizer_steps": train_summary["optimizer_steps_by_variant"],
        },
        "proposal_lock_sha256": sha256_file(
            Path(config["paths"]["candidate_source_root"]) / "notes/proposal_lock.json"
        ),
        "run_id": run_id,
        "scientific_checks": scientific_checks,
        "scientific_passed": scientific_passed,
        "seed": int(config["seed"]),
    }
    write_json(diagnostic_root / "gate_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _dev_log_counts(path: Path, run_id: str) -> tuple[int, int]:
    all_c02 = 0
    exact = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row.get("run_id", ""))
            all_c02 += int(identifier.startswith("20260710_kuaisearch_c02_"))
            exact += int(identifier == run_id)
    return all_c02, exact


if __name__ == "__main__":
    raise SystemExit(main())
