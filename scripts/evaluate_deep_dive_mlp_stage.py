#!/usr/bin/env python
"""CLI for the shared qrels-gated N12 MLP stage evaluator."""

from __future__ import annotations

import argparse

from myrec.mechanism.mlp_stage_evaluator import evaluate_mlp_stage_bundles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-dir", required=True)
    for method in ("q2", "q3"):
        for block in (13, 20, 27):
            parser.add_argument(f"--{method}-b{block}", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    evaluate_mlp_stage_bundles(
        args.standardized_dir,
        {
            "q2_recranker_generalqwen": {13: args.q2_b13, 20: args.q2_b20, 27: args.q2_b27},
            "q3_tallrec_generalqwen": {13: args.q3_b13, 20: args.q3_b20, 27: args.q3_b27},
        },
        args.output_dir,
        args.analysis_run_id,
    )


if __name__ == "__main__":
    main()

