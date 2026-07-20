#!/usr/bin/env python3
"""Evaluate a complete qrels-blind N20 Q1 cache-phase bundle."""

from __future__ import annotations

import argparse

from myrec.mechanism.q1_cache_phase_evaluator import evaluate_q1_cache_phase_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log-path", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    evaluate_q1_cache_phase_bundle(
        args.standardized_dir,
        args.bundle,
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log_path,
    )


if __name__ == "__main__":
    main()

