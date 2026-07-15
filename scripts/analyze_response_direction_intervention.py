#!/usr/bin/env python
"""Evaluate fixed-budget label-aligned history-response interventions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.response_direction_intervention_evaluator import (
    evaluate_response_direction_intervention_runs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--true-run-id", required=True)
    parser.add_argument("--null-run-id", required=True)
    parser.add_argument("--split", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--label-mode", choices=("click", "purchase", "graded"), default="click")
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--random-permutations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    result = evaluate_response_direction_intervention_runs(
        analysis_run_id=args.analysis_run_id,
        true_run_id=args.true_run_id,
        null_run_id=args.null_run_id,
        split=args.split,
        label_mode=args.label_mode,
        candidate_manifest_path=args.candidate_manifest,
        standardized_dir=args.standardized_dir,
        random_permutations=args.random_permutations,
        seed=args.seed,
        runs_dir=args.runs_dir,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
