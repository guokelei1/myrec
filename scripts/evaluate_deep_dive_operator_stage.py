#!/usr/bin/env python3
"""Evaluate N15/N16 operator bundles after qrels-blind integrity checks."""

from __future__ import annotations

import argparse
import sys

from myrec.mechanism.operator_stage_evaluator import evaluate_operator_bundles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--q2-bundle", required=True)
    parser.add_argument("--q3-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument(
        "--kind",
        choices=(
            "n15_residual_composition",
            "n16_rmsnorm",
            "n17_head_norm",
            "n18_gqa_grouping",
            "n25_swiglu_formation",
        ),
        required=True,
    )
    args = parser.parse_args()
    result = evaluate_operator_bundles(
        args.standardized_dir,
        {
            "q2_recranker_generalqwen": args.q2_bundle,
            "q3_tallrec_generalqwen": args.q3_bundle,
        },
        args.output_dir,
        args.analysis_run_id,
        kind=args.kind,
        command=sys.argv,
    )
    print(result["analysis_type"], result["status"])


if __name__ == "__main__":
    main()
