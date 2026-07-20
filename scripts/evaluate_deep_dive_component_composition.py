#!/usr/bin/env python3
"""Evaluate the frozen N8 joint attention/MLP composition family."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.component_composition_evaluator import (
    evaluate_component_composition,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--qrels-split-dir", required=True)
    parser.add_argument("--q2-bundle", required=True)
    parser.add_argument("--q3-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    result = evaluate_component_composition(
        args.standardized_dir,
        args.qrels_split_dir,
        {
            "q2_recranker_generalqwen": args.q2_bundle,
            "q3_tallrec_generalqwen": args.q3_bundle,
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
                "strict_transfer_requests": result["strict_transfer_requests"],
                "family_rows": len(result["family_rows"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
