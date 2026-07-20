#!/usr/bin/env python3
"""Score full/null internal dev for a formal Q2 matched-control checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.matched_control_scorer import (  # noqa: E402
    HISTORY_CONDITIONS,
    write_q2_matched_control_scores,
)
from myrec.mechanism.matched_training_control import (  # noqa: E402
    CONDITIONS,
    MAX_WALL_SECONDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--sampling-condition", required=True, choices=CONDITIONS)
    parser.add_argument("--history-condition", required=True, choices=HISTORY_CONDITIONS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--probe-manifest",
        default="experiments/motivation/probe_manifest.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-wall-seconds", type=float, default=MAX_WALL_SECONDS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_q2_matched_control_scores(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.sampling_condition,
        args.history_condition,
        args.run_id,
        device=args.device,
        runs_dir=args.runs_dir,
        probe_manifest_path=args.probe_manifest,
        command=sys.argv,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
