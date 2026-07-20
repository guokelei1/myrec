#!/usr/bin/env python3
"""Evaluate one D2 selected-block branch bundle on frozen fold 1."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.selected_branch_evaluator import evaluate_selected_branches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--qrels-split-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    result = evaluate_selected_branches(
        args.standardized_dir,
        args.qrels_split_dir,
        args.bundle,
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "method_id",
                    "selected_block",
                    "evidence_role",
                    "strict_transfer_requests",
                    "status",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
