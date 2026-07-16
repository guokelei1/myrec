#!/usr/bin/env python
"""Build a concise report for one staged Motivation V1.1 axis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from build_motivation_v11_report import _summarize_analysis  # noqa: E402
from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-id", required=True)
    parser.add_argument("--axis-description", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--analysis", action="append", required=True, metavar="MODEL=ANALYSIS_RUN_ID")
    parser.add_argument("--output", required=True)
    parser.add_argument("--next-action")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: dict[str, Any] = {}
    for item in args.analysis:
        if "=" not in item:
            raise SystemExit(f"invalid --analysis: {item}")
        model, analysis_id = item.split("=", 1)
        results[model] = _summarize_analysis(
            analysis_id,
            Path(args.standardized_dir),
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + len(results),
        )

    instructrec = {
        key: value for key, value in results.items() if key.startswith("instructrec")
    }
    tem = {key: value for key, value in results.items() if key.startswith("tem")}
    interpretation = _interpret(instructrec, tem)
    report = {
        "schema_version": 1,
        "report_id": f"pps_motivation_v11_{args.axis_id}",
        "evidence_mode": "staged_axis_confirmation_only",
        "axis_id": args.axis_id,
        "axis_description": args.axis_description,
        "generated_from": {
            "protocol": {"path": args.protocol, "sha256": sha256_file(args.protocol)},
            "standardized_dir": args.standardized_dir,
            "v1_entry_unchanged": True,
            "test_opened": False,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "analyses": results,
        "interpretation": interpretation,
        "next_action": args.next_action or _default_next_action(args.axis_id),
        "v1_not_modified": True,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _interpret(instructrec: dict[str, dict[str, Any]], tem: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not instructrec:
        raise SystemExit("Axis report requires an instructrec analysis")
    repeat = [result["surfaces"]["target_repeat"] for result in instructrec.values()]
    no_overlap = [
        result["surfaces"]["target_nonrepeat_no_candidate_overlap"]
        for result in instructrec.values()
    ]
    contrast = [result["repeat_minus_no_overlap"] for result in instructrec.values()]
    tem_repeat = [result["surfaces"]["target_repeat"] for result in tem.values()]
    tem_contrast = [result["repeat_minus_no_overlap"] for result in tem.values()]
    return {
        "instructrec_repeat_positive_all_seeds": all(item["mean"] > 0 for item in repeat),
        "instructrec_no_overlap_point_negative_all_seeds": all(
            item["mean"] < 0 for item in no_overlap
        ),
        "instructrec_no_overlap_not_reliably_positive": (
            all(item["ci95_query_cluster"][0] <= 0 <= item["ci95_query_cluster"][1] for item in no_overlap)
        ),
        "instructrec_repeat_minus_no_overlap_positive": (
            all(item["point_estimate"] > 0 and item["bootstrap_ci95"][0] > 0 for item in contrast)
        ),
        "tem_available": bool(tem),
        "tem_replicates_instructrec_repeat_pattern": bool(
            tem_repeat
            and all(item["mean"] > 0 for item in tem_repeat)
            and all(item["ci95_query_cluster"][0] > 0 for item in tem_repeat)
            and tem_contrast
            and all(item["point_estimate"] > 0 for item in tem_contrast)
            and all(item["bootstrap_ci95"][0] > 0 for item in tem_contrast)
        ),
        "family_prevalence_upgrade": False,
        "claim_boundary": (
            "On fixed-population KuaiSearch Axis A, the ordinary InstructRec model "
            "shows a replicated positive target-repeat response and a positive "
            "repeat-versus-nonrepeat-no-overlap contrast. The non-overlap point "
            "estimate is negative but not individually reliable, so this does not "
            "license a universal negative-history-effect claim. TEM does not provide "
            "the same stable decomposition, and no family-level upgrade is made."
        ),
    }


def _default_next_action(axis_id: str) -> str:
    if "axis_b" in axis_id.lower() or "population" in axis_id.lower():
        return (
            "Audit the completed KuaiSearch Axis B result, then only after its "
            "decision run the pre-registered second-population replication."
        )
    return (
        "Freeze Axis A interpretation, then run Axis B with the Axis A epoch rule "
        "and a strictly larger preceding KuaiSearch training population."
    )


if __name__ == "__main__":
    raise SystemExit(main())
