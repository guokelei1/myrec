#!/usr/bin/env python
"""Summarize frozen V1.1 history-response bundles without changing V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.controlled_composition import (  # noqa: E402
    cluster_bootstrap_group_mean_contrast,
    cluster_bootstrap_mean_ci,
)
from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import iter_jsonl, write_json  # noqa: E402


SURFACES = (
    "all",
    "target_repeat",
    "target_nonrepeat_other_candidate_overlap",
    "target_nonrepeat_no_candidate_overlap",
    "target_nonrepeat_no_history",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="experiments/motivation_v1_1/protocol.md")
    parser.add_argument(
        "--data-admission",
        default="reports/pps_motivation_v11_kuaisearch_data_admission.json",
    )
    parser.add_argument(
        "--main-standardized-dir",
        default="data/standardized/kuaisearch/full_confirm_preceding40k_v11",
    )
    parser.add_argument(
        "--main-analysis",
        action="append",
        required=True,
        metavar="MODEL=ANALYSIS_RUN_ID",
    )
    parser.add_argument(
        "--second-analysis",
        default="20260716_jdsearch_v11_existing_full_history_response",
    )
    parser.add_argument(
        "--second-standardized-dir",
        default="data/standardized/jdsearch/hash_scout10k_v3",
    )
    parser.add_argument("--output", default="reports/pps_motivation_v11_robustness.json")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    args = parser.parse_args()
    if len(args.main_analysis) < 2:
        raise SystemExit("V1.1 requires at least two pre-declared main analyses")
    return args


def main() -> int:
    args = parse_args()
    protocol = Path(args.protocol)
    admission = _read_json(Path(args.data_admission))
    main_results = {}
    for item in args.main_analysis:
        if "=" not in item:
            raise SystemExit(f"invalid --main-analysis: {item}")
        model, analysis_id = item.split("=", 1)
        main_results[model] = _summarize_analysis(
            analysis_id,
            Path(args.main_standardized_dir),
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    second = _summarize_analysis(
        args.second_analysis,
        Path(args.second_standardized_dir),
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 1000,
    )

    report = {
        "schema_version": 1,
        "report_id": "pps_motivation_v11_robustness",
        "evidence_mode": "frozen_v1_1_separate_report",
        "generated_from": {
            "protocol": {"path": str(protocol), "sha256": sha256_file(protocol)},
            "data_admission": {
                "path": str(Path(args.data_admission)),
                "sha256": sha256_file(args.data_admission),
            },
            "v1_entry_unchanged": True,
            "test_opened": False,
        },
        "main_population": {
            "dataset": admission["dataset_version"],
            "admission_passed": admission["admission_passed"],
            "confirmation_lock": admission["confirmation_lock"],
            "analyses": main_results,
        },
        "second_population": {
            "dataset": "jdsearch/hash_scout10k_v3",
            "role": "functional_replication_anonymized_query_boundary",
            "semantic_claim_authorized": False,
            "analysis": second,
        },
        "interpretation": _interpret(main_results, second),
        "v1_not_modified": True,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _summarize_analysis(
    analysis_id: str, standardized_dir: Path, *, samples: int, seed: int
) -> dict[str, Any]:
    analysis_dir = Path("runs") / analysis_id
    metrics = _read_json(analysis_dir / "metrics.json")
    rows = {str(row["request_id"]): row for row in iter_jsonl(analysis_dir / "per_request_history_response.jsonl")}
    records_path = standardized_dir / f"records_{metrics['split']}.jsonl"
    records = {str(row["request_id"]): row for row in iter_jsonl(records_path)}
    if set(rows) != set(records):
        raise ValueError(f"analysis/records coverage mismatch: {analysis_id}")
    clusters = {
        request_id: "".join(str(record.get("query", "")).casefold().split())
        for request_id, record in records.items()
    }
    surface_dir = analysis_dir / "target_aware_surfaces"
    surface_values: dict[str, Any] = {}
    for offset, name in enumerate(SURFACES):
        ids = _read_ids(surface_dir / f"{name}.txt")
        selected = [rows[request_id] for request_id in sorted(ids)]
        if not selected:
            surface_values[name] = {"num_requests": 0, "mean": None, "ci95": None}
            continue
        ci = cluster_bootstrap_mean_ci(
            selected,
            clusters,
            ("true_minus_null_ndcg@10",),
            samples=samples,
            seed=seed + offset,
        )["true_minus_null_ndcg@10"]
        surface_values[name] = {
            "num_requests": len(selected),
            "mean": sum(float(row["true_minus_null_ndcg@10"]) for row in selected)
            / len(selected),
            "ci95_query_cluster": ci,
        }
    contrast_rows = []
    for surface in ("target_repeat", "target_nonrepeat_no_candidate_overlap"):
        for request_id in sorted(_read_ids(surface_dir / f"{surface}.txt")):
            contrast_rows.append(
                {
                    "request_id": request_id,
                    "surface": surface,
                    "recovery": float(rows[request_id]["true_minus_null_ndcg@10"]),
                }
            )
    contrast = cluster_bootstrap_group_mean_contrast(
        contrast_rows,
        clusters,
        group_field="surface",
        value_field="recovery",
        left_group="target_repeat",
        right_group="target_nonrepeat_no_candidate_overlap",
        samples=samples,
        seed=seed + 100,
    )
    return {
        "analysis_run_id": analysis_id,
        "candidate_manifest_sha256": metrics["candidate_manifest_sha256"],
        "split": metrics["split"],
        "label_mode": metrics["label_mode"],
        "num_requests": metrics["num_requests"],
        "task_capability": {
            "true_ndcg_at_10_all": metrics["mean_true_ndcg@10"],
            "true_ndcg_at_10_positive": metrics.get("mean_true_ndcg@10_positive"),
        },
        "overall_history_gain": {
            "mean_true_minus_null_ndcg_at_10": metrics["mean_true_minus_null_ndcg@10"],
            "active_response_rate": metrics["active_response_rate"],
        },
        "surfaces": surface_values,
        "repeat_minus_no_overlap": contrast,
        "shared_evaluator_target_surface_counts": metrics.get(
            "target_aware_surface_counts", {}
        ),
        "true_run_id": metrics["true_run_id"],
        "null_run_id": metrics["null_run_id"],
        "wrong_run_id": metrics.get("wrong_run_id"),
    }


def _interpret(main_results: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    main_repeat_positive = all(
        result["surfaces"]["target_repeat"]["mean"] > 0 for result in main_results.values()
    )
    main_no_overlap_not_reliable = all(
        result["surfaces"]["target_nonrepeat_no_candidate_overlap"]["ci95_query_cluster"][0]
        <= 0
        for result in main_results.values()
    )
    second_repeat_positive = second["surfaces"]["target_repeat"]["mean"] > 0
    second_no_overlap_positive = second["surfaces"]["target_nonrepeat_no_candidate_overlap"]["mean"] > 0
    return {
        "main_all_seeds_repeat_positive": main_repeat_positive,
        "main_all_seeds_no_overlap_not_reliably_positive": main_no_overlap_not_reliable,
        "second_population_repeat_positive": second_repeat_positive,
        "second_population_no_overlap_point_estimate_positive": second_no_overlap_positive,
        "family_prevalence_upgrade": False,
        "reason": (
            "JDsearch is functional/anonymized and uses a different ordinary ranker; "
            "this V1.1 report therefore does not upgrade the bounded V1 observation "
            "to a universal or family-level claim."
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
