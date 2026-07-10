#!/usr/bin/env python
"""Summarize the locked C3-R history-identity controls."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.compare import compare_per_request_metrics
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/analysis/c3_history_identity_controls.yaml",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/analysis/c3_history_identity_controls/materialization_manifest.json",
    )
    parser.add_argument(
        "--output-json",
        default="reports/pps_c3r_history_identity_control.json",
    )
    parser.add_argument(
        "--output-md",
        default="reports/pps_c3r_history_identity_control.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("locked config changed after control materialization")

    runs_dir = Path("runs")
    reports_dir = Path("reports")
    seeds = [int(value) for value in config["seeds"]]
    true_b7 = config["frozen_runs"]["true_static_mixture"]
    true_b0b = config["frozen_runs"]["true_history"]
    query_run = config["frozen_runs"]["query_only_reference"]
    history_present_path = Path(
        manifest["subset_paths"]["history_present"]["path"]
    )
    history_absent_path = Path(manifest["subset_paths"]["history_absent"]["path"])
    same_query_path = Path(
        manifest["subset_paths"]["same_query_all_seeds"]["path"]
    )

    comparisons: dict[str, dict[str, dict[str, Any]]] = {
        "b7_true_vs_wrong_history_present": {},
        "b7_true_vs_wrong_same_query": {},
        "b0b_true_vs_wrong_history_present": {},
    }
    for seed in seeds:
        wrong_b7 = manifest["mixture_run_ids"][str(seed)]
        wrong_b0b = manifest["history_run_ids"][str(seed)]
        comparisons["b7_true_vs_wrong_history_present"][str(seed)] = _compare(
            true_b7,
            wrong_b7,
            history_present_path,
            reports_dir
            / f"compare_c3r_b7_true_vs_wrong_history_present_s{seed}.json",
        )
        comparisons["b7_true_vs_wrong_same_query"][str(seed)] = _compare(
            true_b7,
            wrong_b7,
            same_query_path,
            reports_dir / f"compare_c3r_b7_true_vs_wrong_same_query_s{seed}.json",
        )
        comparisons["b0b_true_vs_wrong_history_present"][str(seed)] = _compare(
            true_b0b,
            wrong_b0b,
            history_present_path,
            reports_dir
            / f"compare_c3r_b0b_true_vs_wrong_history_present_s{seed}.json",
        )

    existing_comparisons = {
        "b7_vs_b0b": _read_json(reports_dir / "compare_b7_bge_vs_b0b.json"),
        "b7_vs_b2z": _read_json(reports_dir / "compare_b7_bge_vs_b2z.json"),
    }
    no_history_equivalence = _metric_equivalence(
        runs_dir / true_b7 / "per_request_metrics.jsonl",
        runs_dir / query_run / "per_request_metrics.jsonl",
        history_absent_path,
    )
    run_metrics = {
        "true_b0b": _metric_summary(runs_dir / true_b0b / "metrics.json"),
        "true_b7": _metric_summary(runs_dir / true_b7 / "metrics.json"),
        "query_b2z": _metric_summary(runs_dir / query_run / "metrics.json"),
        "wrong_b0b": {
            str(seed): _metric_summary(
                runs_dir / manifest["history_run_ids"][str(seed)] / "metrics.json"
            )
            for seed in seeds
        },
        "wrong_b7": {
            str(seed): _metric_summary(
                runs_dir / manifest["mixture_run_ids"][str(seed)] / "metrics.json"
            )
            for seed in seeds
        },
    }
    structure = _history_structure(Path("artifacts/m4/m4_features_dev.parquet"))
    control_run_ids = [
        manifest[family][str(seed)]
        for family in ["history_run_ids", "mixture_run_ids"]
        for seed in seeds
    ]
    run_integrity = _run_integrity(
        control_run_ids,
        expected_candidate_sha256=manifest["candidate_manifest_sha256"],
        expected_config_sha256=manifest["config_sha256"],
    )
    dev_eval_reconciliation = _dev_eval_reconciliation(
        Path("reports/dev_eval_log.jsonl"), control_run_ids
    )

    three_seed_summary = {
        "wrong_b0b_ndcg10": _mean_std(
            [value["ndcg@10"] for value in run_metrics["wrong_b0b"].values()]
        ),
        "wrong_b7_ndcg10": _mean_std(
            [value["ndcg@10"] for value in run_metrics["wrong_b7"].values()]
        ),
        "true_minus_wrong_b0b_history_present": _comparison_summary(
            comparisons["b0b_true_vs_wrong_history_present"].values()
        ),
        "true_minus_wrong_b7_history_present": _comparison_summary(
            comparisons["b7_true_vs_wrong_history_present"].values()
        ),
        "true_minus_wrong_b7_same_query": _comparison_summary(
            comparisons["b7_true_vs_wrong_same_query"].values()
        ),
    }

    history_checks = comparisons["b7_true_vs_wrong_history_present"].values()
    same_query_checks = comparisons["b7_true_vs_wrong_same_query"].values()
    same_query_deltas = [float(value["delta"]) for value in same_query_checks]
    gate_checks = {
        "aggregate_b7_vs_b0b": existing_comparisons["b7_vs_b0b"]["ci95"][0] > 0,
        "aggregate_b7_vs_b2z": existing_comparisons["b7_vs_b2z"]["ci95"][0] > 0,
        "all_history_present_ci_lower_gt_zero": all(
            float(value["ci95"][0]) > 0 for value in history_checks
        ),
        "history_absent_b7_equals_query_only": no_history_equivalence[
            "all_metrics_identical"
        ],
        "same_query_count_at_least_minimum": manifest["subset_counts"][
            "same_query_all_seeds"
        ]
        >= int(config["gate"]["min_same_query_requests"]),
        "same_query_mean_delta_gt_zero": statistics.mean(same_query_deltas) > 0,
        "same_query_significant_seed_count_at_least_minimum": sum(
            float(value["ci95"][0]) > 0 for value in same_query_checks
        )
        >= int(config["gate"]["same_query_min_significant_seeds"]),
        "control_run_integrity": run_integrity["passed"],
        "dev_eval_log_reconciled": dev_eval_reconciliation["passed"],
    }
    passed = all(gate_checks.values())
    result = {
        "analysis_id": config["analysis_id"],
        "analysis_type": "locked matched wrong-user history control",
        "comparisons": comparisons,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "decision": (
            "Correct-user history is identity-specific and query/history evidence is "
            "complementary in aggregate. This supports interaction-aware fusion, not "
            "per-request oracle routing."
            if passed
            else "The repaired personalization motivation did not pass its locked control."
        ),
        "existing_aggregate_comparisons": existing_comparisons,
        "dev_eval_reconciliation": dev_eval_reconciliation,
        "gate_checks": gate_checks,
        "history_structure": structure,
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "no_history_equivalence": no_history_equivalence,
        "qrels_read": False,
        "qrels_derived_per_request_metrics_read": True,
        "run_integrity": run_integrity,
        "run_metrics": run_metrics,
        "same_query_mean_delta": statistics.mean(same_query_deltas),
        "status": "passed" if passed else "failed",
        "subset_counts": manifest["subset_counts"],
        "three_seed_summary": three_seed_summary,
    }
    write_json(args.output_json, result)
    Path(args.output_md).write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 2


def _compare(
    run_a: str,
    run_b: str,
    request_ids_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    left_path = Path("runs") / run_a / "per_request_metrics.jsonl"
    right_path = Path("runs") / run_b / "per_request_metrics.jsonl"
    provenance = {
        "request_ids_sha256": sha256_file(request_ids_path),
        "run_a": run_a,
        "run_a_per_request_sha256": sha256_file(left_path),
        "run_b": run_b,
        "run_b_per_request_sha256": sha256_file(right_path),
    }
    if output_path.exists():
        existing = _read_json(output_path)
        core_matches = (
            existing.get("request_ids_path") == str(request_ids_path)
            and existing.get("samples") == 10000
            and existing.get("seed") == 20260708
        )
        recorded = existing.get("provenance")
        if core_matches and (recorded is None or recorded == provenance):
            existing["provenance"] = provenance
            write_json(output_path, existing)
            return existing
    result = compare_per_request_metrics(
        run_a_path=left_path,
        run_b_path=right_path,
        output_path=output_path,
        metric="ndcg@10",
        samples=10000,
        seed=20260708,
        request_ids_path=request_ids_path,
    )
    result["provenance"] = provenance
    write_json(output_path, result)
    return result


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def _comparison_summary(values: Any) -> dict[str, float]:
    rows = list(values)
    deltas = [float(row["delta"]) for row in rows]
    return {
        "conservative_ci_high": max(float(row["ci95"][1]) for row in rows),
        "conservative_ci_low": min(float(row["ci95"][0]) for row in rows),
        "mean_delta": statistics.mean(deltas),
        "sample_std": statistics.stdev(deltas),
    }


def _metric_equivalence(
    left_path: Path,
    right_path: Path,
    request_ids_path: Path,
) -> dict[str, Any]:
    request_ids = _read_request_ids(request_ids_path)
    fields = ["ndcg@10", "mrr", "recall@10", "purchase_ndcg@10"]
    left = {str(row["request_id"]): row for row in iter_jsonl(left_path)}
    right = {str(row["request_id"]): row for row in iter_jsonl(right_path)}
    mismatches = []
    for request_id in sorted(request_ids):
        if request_id not in left or request_id not in right:
            raise ValueError(f"missing no-history request {request_id}")
        if any(
            _comparable_metric(left[request_id][field])
            != _comparable_metric(right[request_id][field])
            for field in fields
        ):
            mismatches.append(request_id)
    return {
        "all_metrics_identical": not mismatches,
        "fields": fields,
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:10],
        "num_requests": len(request_ids),
        "request_ids_path": str(request_ids_path),
    }


def _comparable_metric(value: Any) -> float | None:
    return None if value is None else float(value)


def _history_structure(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    history = frame.loc[frame["history_length"] > 0]
    overlap = history["history_candidate_cat_overlap"].astype(float)
    return {
        "all_requests": int(len(frame)),
        "history_absent": int((frame["history_length"] == 0).sum()),
        "history_absent_rate": float((frame["history_length"] == 0).mean()),
        "history_present": int(len(history)),
        "history_present_rate": float(len(history) / len(frame)),
        "history_length_median": float(history["history_length"].median()),
        "history_candidate_deepest_category_jaccard_median": float(overlap.median()),
        "history_candidate_deepest_category_zero_rate": float((overlap == 0).mean()),
        "source": str(path),
        "source_sha256": sha256_file(path),
    }


def _run_integrity(
    run_ids: list[str],
    expected_candidate_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    checks = {}
    for run_id in run_ids:
        run_dir = Path("runs") / run_id
        metrics = _read_json(run_dir / "metrics.json")
        metadata = _read_json(run_dir / "metadata.json")
        checks[run_id] = {
            "candidate_manifest_matches": metrics["candidate_manifest_sha256"]
            == expected_candidate_sha256
            == metadata["candidate_manifest_sha256"],
            "config_matches": metadata["config_sha256"] == expected_config_sha256,
            "generated_by_shared_evaluator": metrics["generated_by"]
            == "myrec.eval.evaluator",
            "metadata_qrels_read_false": metadata["qrels_read"] is False,
            "scores_hash_matches": metrics["scores_sha256"]
            == sha256_file(run_dir / "scores.jsonl"),
        }
    return {
        "passed": all(all(values.values()) for values in checks.values()),
        "runs": checks,
    }


def _dev_eval_reconciliation(path: Path, run_ids: list[str]) -> dict[str, Any]:
    expected = set(run_ids)
    counts = {run_id: 0 for run_id in run_ids}
    for row in iter_jsonl(path):
        run_id = str(row["run_id"])
        if run_id in expected:
            counts[run_id] += 1
    return {
        "entries": sum(counts.values()),
        "expected_entries": len(run_ids),
        "per_run_counts": counts,
        "passed": all(value == 1 for value in counts.values()),
        "role": "fixed claim controls, not tuning",
    }


def _metric_summary(path: Path) -> dict[str, Any]:
    row = _read_json(path)
    return {
        "ndcg@10": row["ndcg@10"],
        "run_id": row["run_id"],
        "scores_sha256": row["scores_sha256"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_request_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# C3-R History Identity Control",
        "",
        f"Status: **{result['status']}**",
        "",
        "This is the locked replacement for the invalid M3/M4 positive claim. It",
        "supports aggregate complementarity and identity-specific history only; it",
        "does not validate per-request oracle routing.",
        "",
        "## Aggregate Complementarity",
        "",
        "| Comparison | Delta NDCG@10 | 95% CI |",
        "|---|---:|---:|",
    ]
    for label, key in [("B7 vs B0b", "b7_vs_b0b"), ("B7 vs B2z", "b7_vs_b2z")]:
        value = result["existing_aggregate_comparisons"][key]
        lines.append(
            f"| {label} | {value['delta']:+.4f} | "
            f"[{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Matched Wrong-User Control",
            "",
            "| Seed | Wrong B7 full NDCG@10 | True-minus-wrong, history present | 95% CI | True-minus-wrong, same query | 95% CI |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed, full in result["run_metrics"]["wrong_b7"].items():
        hp = result["comparisons"]["b7_true_vs_wrong_history_present"][seed]
        sq = result["comparisons"]["b7_true_vs_wrong_same_query"][seed]
        lines.append(
            f"| {seed} | {full['ndcg@10']:.4f} | {hp['delta']:+.4f} | "
            f"[{hp['ci95'][0]:+.4f}, {hp['ci95'][1]:+.4f}] | "
            f"{sq['delta']:+.4f} | [{sq['ci95'][0]:+.4f}, {sq['ci95'][1]:+.4f}] |"
        )
    structure = result["history_structure"]
    lines.extend(
        [
            "",
            "## Evidence Structure",
            "",
            f"- History is absent for {structure['history_absent']:,}/{structure['all_requests']:,} requests "
            f"({structure['history_absent_rate']:.1%}).",
            f"- Among history-present requests, median history length is {structure['history_length_median']:.0f}.",
            "- Deepest-category history/candidate Jaccard is zero for "
            f"{structure['history_candidate_deepest_category_zero_rate']:.1%} and has median "
            f"{structure['history_candidate_deepest_category_jaccard_median']:.3f}.",
            f"- Same-query wrong-history subset: {result['subset_counts']['same_query_all_seeds']:,} requests.",
            "- B7 and B2z are identical on all history-absent requests: "
            f"{result['no_history_equivalence']['all_metrics_identical']}.",
            "",
            "## Decision",
            "",
            result["decision"],
            "",
            "The permitted architecture hypothesis is a query-anchored personalized",
            "residual with exact masking for absent history. Whether it can identify",
            "irrelevant events remains a system-level falsification, not a result of",
            "this control. The failed M3/M4 oracle evidence is not restored.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
