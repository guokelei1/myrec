#!/usr/bin/env python
"""Evaluate one paired Motivation mechanism probe on internal dev."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.evaluator import evaluate_mechanism_probe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--treatment-run-id", required=True)
    parser.add_argument("--control-run-id", required=True)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--candidate-manifest", default=None)
    parser.add_argument(
        "--split",
        default="dev",
        choices=("dev", "confirmation", "test"),
        help="Only dev is authorized; locked values are accepted only to fail explicitly.",
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_mechanism_probe(
        analysis_run_id=args.analysis_run_id,
        treatment_run_id=args.treatment_run_id,
        control_run_id=args.control_run_id,
        standardized_dir=args.standardized_dir,
        candidate_manifest_path=args.candidate_manifest,
        split=args.split,
        runs_dir=args.runs_dir,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
