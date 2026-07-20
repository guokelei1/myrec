#!/usr/bin/env python3
"""Aggregate all fixed Q2/Q3 GQA-group causal localization bundles."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.attention_group_evaluator import (
    evaluate_attention_group_bundles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for method in ("q2", "q3"):
        for block in (13, 20, 27):
            parser.add_argument(f"--{method}-b{block}", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    bundles = {
        "q2_recranker_generalqwen": {
            block: getattr(args, f"q2_b{block}") for block in (13, 20, 27)
        },
        "q3_tallrec_generalqwen": {
            block: getattr(args, f"q3_b{block}") for block in (13, 20, 27)
        },
    }
    result = evaluate_attention_group_bundles(
        bundles,
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
                "gqa_groups": result["gqa_groups"],
                "sample_rows_per_bundle": result["sample_rows_per_bundle"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
