#!/usr/bin/env python3
"""Evaluate the registered D6 Q2 native-readout bundle."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.native_readout_evaluator import evaluate_q2_native_readout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    result = evaluate_q2_native_readout(
        args.standardized_dir,
        args.bundle,
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
        command=sys.argv,
    )
    print(json.dumps({"analysis_run_id": result["analysis_run_id"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
