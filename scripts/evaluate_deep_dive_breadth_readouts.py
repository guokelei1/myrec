#!/usr/bin/env python3
"""Evaluate Q0/Q1 native final-RMSNorm/readout causal bundles."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.breadth_readout_evaluator import (
    Q0_METHOD_ID,
    Q1_METHOD_ID,
    evaluate_breadth_readouts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--q0-bundle", required=True)
    parser.add_argument("--q1-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    result = evaluate_breadth_readouts(
        args.standardized_dir,
        {Q0_METHOD_ID: args.q0_bundle, Q1_METHOD_ID: args.q1_bundle},
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                "analysis_run_id": result["analysis_run_id"],
                "status": result["status"],
                "strict_transfer_requests": result["strict_transfer_requests"],
                "rows": len(result["rows"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
