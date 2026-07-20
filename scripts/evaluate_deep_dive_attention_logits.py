#!/usr/bin/env python
"""CLI for the shared qrels-gated N11 attention-logit evaluator."""

from __future__ import annotations

import argparse

from myrec.mechanism.attention_edge_runtime import FIXED_BLOCKS, SUPPORTED_METHODS
from myrec.mechanism.attention_logit_evaluator import evaluate_attention_logit_bundles


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
    evaluate_attention_logit_bundles(
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

