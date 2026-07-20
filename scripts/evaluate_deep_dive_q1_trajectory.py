#!/usr/bin/env python3
"""Evaluate a complete Q1 KV-cache trajectory bundle."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.q1_trajectory_evaluator import evaluate_q1_kv_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    result = evaluate_q1_kv_trajectory(
        args.standardized_dir,
        args.bundle,
        args.output_dir,
        args.analysis_run_id,
    )
    print(json.dumps({key: result[key] for key in ("analysis_run_id", "request_count", "candidate_count", "response_token_count", "status")}, sort_keys=True))


if __name__ == "__main__":
    main()
