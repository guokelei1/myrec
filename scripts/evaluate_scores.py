#!/usr/bin/env python
"""Evaluate a run's scores with the shared PPS evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.evaluator import evaluate_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--candidate-manifest",
        default="data/standardized/kuaisearch/v0_lite/candidate_manifest.json",
    )
    parser.add_argument("--standardized-dir", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = evaluate_run(
        run_id=args.run_id,
        split=args.split,
        candidate_manifest_path=args.candidate_manifest,
        standardized_dir=args.standardized_dir,
        runs_dir=args.runs_dir,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
