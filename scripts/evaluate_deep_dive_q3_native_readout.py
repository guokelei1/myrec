#!/usr/bin/env python3
"""Evaluate a complete registered D6 Q3 native-readout bundle."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.q3_native_readout_evaluator import evaluate_q3_native_readout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    result = evaluate_q3_native_readout(
        args.standardized_dir,
        args.bundle,
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(
        json.dumps(
            {
                "analysis_run_id": result["analysis_run_id"],
                "family_size": result["multiple_testing"]["family_size"],
                "strict_transfer_requests": result["strict_transfer_requests"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
