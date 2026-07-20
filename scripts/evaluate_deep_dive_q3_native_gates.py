#!/usr/bin/env python3
"""Evaluate Q3 block-13/27 all-native scientific gates."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.deep_dive_native_evaluator import evaluate_q3_native_gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--block", action="append", nargs=2, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    blocks = {int(block): path for block, path in args.block}
    result = evaluate_q3_native_gates(
        args.standardized_dir,
        blocks,
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                "analysis_run_id": result["analysis_run_id"],
                "q3_sweep_admitted": result["q3_sweep_admitted"],
                "status": result["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

