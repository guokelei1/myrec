#!/usr/bin/env python3
"""Admit and shared-evaluate the four Q2 matched-control score bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.matched_control_evaluator import (  # noqa: E402
    evaluate_q2_matched_control,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--original-full-run-id", required=True)
    parser.add_argument("--original-null-run-id", required=True)
    parser.add_argument("--balanced-full-run-id", required=True)
    parser.add_argument("--balanced-null-run-id", required=True)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--candidate-manifest")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_q2_matched_control(
        analysis_run_id=args.analysis_run_id,
        original_full_run_id=args.original_full_run_id,
        original_null_run_id=args.original_null_run_id,
        balanced_full_run_id=args.balanced_full_run_id,
        balanced_null_run_id=args.balanced_null_run_id,
        standardized_dir=args.standardized_dir,
        candidate_manifest_path=args.candidate_manifest,
        runs_dir=args.runs_dir,
        dev_eval_log_path=args.dev_eval_log,
        command=sys.argv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
