#!/usr/bin/env python3
"""Evaluate N10 candidate-gap geometry without opening qrels."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.candidate_gap_evaluator import evaluate_candidate_gap_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    metrics = evaluate_candidate_gap_bundle(
        args.standardized_dir,
        args.bundle,
        args.output_dir,
        args.analysis_run_id,
        command=list(sys.argv),
    )
    print(json.dumps({"status": metrics["status"], "output_dir": args.output_dir}, sort_keys=True))


if __name__ == "__main__":
    main()
