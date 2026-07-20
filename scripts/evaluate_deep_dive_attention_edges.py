#!/usr/bin/env python3
"""Evaluate all six registered Q2/Q3 D3 attention-edge bundles."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.attention_edge_evaluator import evaluate_attention_edge_bundles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    for method in ("q2", "q3"):
        for block in (13, 20, 27):
            parser.add_argument(f"--{method}-b{block}", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    metrics = evaluate_attention_edge_bundles(
        args.standardized_dir,
        {
            "q2_recranker_generalqwen": {
                block: getattr(args, f"q2_b{block}") for block in (13, 20, 27)
            },
            "q3_tallrec_generalqwen": {
                block: getattr(args, f"q3_b{block}") for block in (13, 20, 27)
            },
        },
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in (
                    "analysis_run_id",
                    "eligible_requests",
                    "strict_transfer_requests",
                    "strict_transfer_eligible_requests",
                    "status",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
