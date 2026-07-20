#!/usr/bin/env python3
"""Run the evaluator-side Motivation data/signal/power audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.data_power_audit import run_data_power_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    parser.add_argument(
        "--analysis",
        action="append",
        required=True,
        metavar="METHOD_ID=RUN_DIR",
        help="Frozen internal-dev analysis directory; repeat for Q0--Q3.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analyses: dict[str, str] = {}
    for value in args.analysis:
        if "=" not in value:
            raise ValueError("--analysis must use METHOD_ID=RUN_DIR")
        method_id, path = value.split("=", 1)
        if not method_id or not path or method_id in analyses:
            raise ValueError("invalid or duplicate --analysis")
        analyses[method_id] = path
    result = run_data_power_audit(
        args.standardized_dir,
        args.feature_store,
        args.run_id,
        frozen_analysis_dirs=analyses,
        runs_dir=args.runs_dir,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
