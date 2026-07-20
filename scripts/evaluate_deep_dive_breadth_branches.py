#!/usr/bin/env python3
"""Evaluate the registered 96-cell Q0/Q1 branch extension."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.breadth_branch_evaluator import evaluate_breadth_branches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--q0", action="append", nargs=2, metavar=("BLOCK", "DIR"), required=True)
    parser.add_argument("--q1", action="append", nargs=2, metavar=("BLOCK", "DIR"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    result = evaluate_breadth_branches(
        args.standardized_dir,
        {
            "q0_qwen3_reranker_06b": {int(block): path for block, path in args.q0},
            "q1_instructrec_generalqwen": {int(block): path for block, path in args.q1},
        },
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(json.dumps({"analysis_type": result["analysis_type"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
