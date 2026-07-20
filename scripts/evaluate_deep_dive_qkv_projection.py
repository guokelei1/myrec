#!/usr/bin/env python3
"""Evaluate N13 Q/K/V projection bundles after qrels-blind integrity checks."""

from __future__ import annotations

import argparse

from myrec.mechanism.qkv_projection_evaluator import evaluate_qkv_projection_bundles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--q2-b13", required=True)
    parser.add_argument("--q2-b20", required=True)
    parser.add_argument("--q2-b27", required=True)
    parser.add_argument("--q3-b13", required=True)
    parser.add_argument("--q3-b20", required=True)
    parser.add_argument("--q3-b27", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    evaluate_qkv_projection_bundles(
        args.standardized_dir,
        {
            "q2_recranker_generalqwen": {
                13: args.q2_b13,
                20: args.q2_b20,
                27: args.q2_b27,
            },
            "q3_tallrec_generalqwen": {
                13: args.q3_b13,
                20: args.q3_b20,
                27: args.q3_b27,
            },
        },
        args.output_dir,
        args.analysis_run_id,
    )


if __name__ == "__main__":
    main()
