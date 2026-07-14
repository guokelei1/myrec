#!/usr/bin/env python
"""Evaluate a frozen true/null/matched-wrong score bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.history_response_evaluator import evaluate_history_response_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--true-run-id", required=True)
    parser.add_argument("--null-run-id", required=True)
    parser.add_argument("--wrong-run-id", default=None)
    parser.add_argument("--split", default="dev", choices=("dev", "confirmation"))
    parser.add_argument("--label-mode", default="click", choices=("click", "purchase", "graded"))
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--activity-epsilon", required=True, type=float)
    parser.add_argument("--utility-epsilon", required=True, type=float)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_history_response_runs(
        analysis_run_id=args.analysis_run_id,
        true_run_id=args.true_run_id,
        null_run_id=args.null_run_id,
        wrong_run_id=args.wrong_run_id,
        split=args.split,
        label_mode=args.label_mode,
        candidate_manifest_path=args.candidate_manifest,
        standardized_dir=args.standardized_dir,
        activity_epsilon=args.activity_epsilon,
        utility_epsilon=args.utility_epsilon,
        runs_dir=args.runs_dir,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
