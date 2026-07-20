#!/usr/bin/env python3
"""Aggregate all fixed D4 SwiGLU groups and residual geometry."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.mlp_group_evaluator import evaluate_mlp_group_bundles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2", action="append", nargs=2, metavar=("BLOCK", "DIR"), required=True)
    parser.add_argument("--q3", action="append", nargs=2, metavar=("BLOCK", "DIR"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    result = evaluate_mlp_group_bundles(
        {
            "q2_recranker_generalqwen": {int(block): path for block, path in args.q2},
            "q3_tallrec_generalqwen": {int(block): path for block, path in args.q3},
        },
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(json.dumps({"analysis_type": result["analysis_type"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
