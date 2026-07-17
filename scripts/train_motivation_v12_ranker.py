#!/usr/bin/env python3
"""Train or resume one frozen Motivation V1.2 Q0--Q3 recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.motivation_v12_ranker import (  # noqa: E402
    train_motivation_v12_ranker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--device", required=True, help="Explicit device, e.g. cuda:0")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument(
        "--max-train-groups",
        type=int,
        help="Smoke-test cap; any capped run is permanently marked non-result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_motivation_v12_ranker(
        args.standardized_dir,
        args.config,
        args.run_id,
        args.output_model_dir,
        device=args.device,
        runs_dir=args.runs_dir,
        resume=args.resume,
        command=sys.argv,
        max_wall_seconds=args.max_wall_seconds,
        max_train_groups=args.max_train_groups,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
