#!/usr/bin/env python3
"""Evaluate the six registered N9 history-path bundles."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.history_path_evaluator import evaluate_history_path_bundles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--qrels-split-dir", required=True)
    parser.add_argument("--q2-b13", required=True)
    parser.add_argument("--q2-b20", required=True)
    parser.add_argument("--q2-b27", required=True)
    parser.add_argument("--q3-b13", required=True)
    parser.add_argument("--q3-b20", required=True)
    parser.add_argument("--q3-b27", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    result = evaluate_history_path_bundles(
        args.standardized_dir,
        args.qrels_split_dir,
        {
            "q2_recranker_generalqwen": {13: args.q2_b13, 20: args.q2_b20, 27: args.q2_b27},
            "q3_tallrec_generalqwen": {13: args.q3_b13, 20: args.q3_b20, 27: args.q3_b27},
        },
        args.output_dir,
        args.analysis_run_id,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                "analysis_run_id": result["analysis_run_id"],
                "status": result["status"],
                "family_rows": len(result["family_rows"]),
                "strict_transfer_eligible_requests": result["strict_transfer_eligible_requests"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

