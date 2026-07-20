#!/usr/bin/env python3
"""Evaluate N14 embedding-stage bundles after qrels-blind integrity checks."""

from __future__ import annotations

import argparse

from myrec.mechanism.embedding_stage_evaluator import evaluate_embedding_stage_bundles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--q2-bundle", required=True)
    parser.add_argument("--q3-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    evaluate_embedding_stage_bundles(
        args.standardized_dir,
        {
            "q2_recranker_generalqwen": args.q2_bundle,
            "q3_tallrec_generalqwen": args.q3_bundle,
        },
        args.output_dir,
        args.analysis_run_id,
    )


if __name__ == "__main__":
    main()
