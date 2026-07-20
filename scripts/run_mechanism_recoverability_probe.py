#!/usr/bin/env python3
"""Fit or score the Motivation train-only recoverability probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.recoverability_probe import (  # noqa: E402
    fit_recoverability_probe,
    score_recoverability_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--standardized-dir", required=True)
    fit.add_argument("--feature-store", required=True)
    fit.add_argument("--output-model-dir", required=True)
    fit.add_argument("--run-id", required=True)
    fit.add_argument("--runs-dir", default="runs")
    fit.add_argument("--label-shuffle", action="store_true")

    score = sub.add_parser("score")
    score.add_argument("--standardized-dir", required=True)
    score.add_argument("--feature-store", required=True)
    score.add_argument("--model-dir", required=True)
    score.add_argument(
        "--condition",
        required=True,
        choices=("full", "null", "history_shuffle", "routing_query_shuffle"),
    )
    score.add_argument("--run-id", required=True)
    score.add_argument("--runs-dir", default="runs")
    score.add_argument("--wrong-history-assignments")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "fit":
        result = fit_recoverability_probe(
            args.standardized_dir,
            args.feature_store,
            args.output_model_dir,
            args.run_id,
            runs_dir=args.runs_dir,
            label_shuffle=args.label_shuffle,
            command=sys.argv,
        )
    else:
        result = score_recoverability_probe(
            args.standardized_dir,
            args.feature_store,
            args.model_dir,
            args.condition,
            args.run_id,
            runs_dir=args.runs_dir,
            wrong_history_assignments=args.wrong_history_assignments,
            command=sys.argv,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
