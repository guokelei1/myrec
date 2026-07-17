#!/usr/bin/env python3
"""Score one label-free full/null/wrong condition with a frozen Q0--Q3 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.motivation_v12_ranker import (  # noqa: E402
    write_motivation_v12_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--history-assignments", required=True)
    parser.add_argument("--history-condition", choices=("true", "null", "wrong"), required=True)
    parser.add_argument("--split", choices=("dev", "confirmation"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True, help="Explicit device, e.g. cuda:0")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--max-score-requests",
        type=int,
        help="Smoke-test cap; any capped score bundle is permanently marked non-result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_motivation_v12_scores(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.history_assignments,
        args.history_condition,
        args.split,
        args.run_id,
        device=args.device,
        runs_dir=args.runs_dir,
        command=sys.argv,
        max_score_requests=args.max_score_requests,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
