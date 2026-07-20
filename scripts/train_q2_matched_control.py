#!/usr/bin/env python3
"""Train or CPU-mock one registered M3 Q2 matched-control condition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.matched_training_control import (  # noqa: E402
    CONDITIONS,
    MAX_WALL_SECONDS,
    run_q2_matched_training_control,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-model-dir", required=True)
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
    parser.add_argument(
        "--mock-cpu",
        action="store_true",
        help="Execute the exact schedule without Qwen; permanently non-result.",
    )
    parser.add_argument(
        "--mock-stop-after-optimizer-steps",
        type=int,
        help="Pause a CPU mock after this many additional steps to test resume.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_q2_matched_training_control(
        args.standardized_dir,
        args.config,
        args.condition,
        args.run_id,
        args.output_model_dir,
        device=args.device,
        runs_dir=args.runs_dir,
        probe_manifest_path=args.probe_manifest,
        command=sys.argv,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        mock_cpu=args.mock_cpu,
        mock_stop_after_optimizer_steps=args.mock_stop_after_optimizer_steps,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
