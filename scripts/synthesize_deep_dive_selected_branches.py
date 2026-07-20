#!/usr/bin/env python3
"""Synthesize Q2/Q3 D2 selected-branch families with fixed missing cells."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.selected_branch_synthesis import (
    MODELS,
    synthesize_selected_branches,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", action="append", nargs=2, metavar=("METHOD", "PATH")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    metrics = dict(args.metrics or [])
    if set(metrics) - set(MODELS):
        parser.error("--metrics METHOD must be Q2 or Q3")
    result = synthesize_selected_branches(
        metrics, args.output_dir, args.analysis_run_id, command=sys.argv
    )
    print(
        json.dumps(
            {
                "registered_support_rows": result["registered_support_rows"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
