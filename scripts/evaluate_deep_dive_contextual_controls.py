#!/usr/bin/env python3
"""Evaluate Q2/Q3 D5 fixed-length contextual-control bundles."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.contextual_control_evaluator import (
    evaluate_contextual_control_bundles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--q2-bundle", required=True)
    parser.add_argument("--q3-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    result = evaluate_contextual_control_bundles(
        args.standardized_dir,
        {
            "q2_recranker_generalqwen": args.q2_bundle,
            "q3_tallrec_generalqwen": args.q3_bundle,
        },
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "analysis_run_id",
                    "eligible_requests",
                    "strict_transfer_requests",
                    "status",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
