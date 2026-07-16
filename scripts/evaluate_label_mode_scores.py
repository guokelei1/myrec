#!/usr/bin/env python
"""Evaluate one frozen score run under click, purchase, or graded gains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.label_mode_evaluator import evaluate_label_mode_score_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--score-run-id", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--label-mode", choices=("click", "purchase", "graded"), required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    result = evaluate_label_mode_score_run(
        args.analysis_run_id,
        args.score_run_id,
        args.split,
        args.candidate_manifest,
        args.standardized_dir,
        label_mode=args.label_mode,
        runs_dir=args.runs_dir,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

