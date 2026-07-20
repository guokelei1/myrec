#!/usr/bin/env python3
"""Run one train-only M3 Q2/Q3 gradient diagnostic state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.gradient_diagnostic import (  # noqa: E402
    MAX_WALL_SECONDS,
    PROBE_MANIFEST_PATH,
    SUPPORTED_STATES,
    run_gradient_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--state", required=True, choices=SUPPORTED_STATES)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True, help="Explicit device, e.g. cuda:0")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--probe-manifest",
        default=str(PROBE_MANIFEST_PATH),
        help="Exact frozen mechanism probe manifest.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=MAX_WALL_SECONDS,
        help=f"Resumable segment wall limit; must be <= {int(MAX_WALL_SECONDS)}.",
    )
    parser.add_argument(
        "--max-requests-per-surface",
        type=int,
        help="CPU/engineering smoke cap in [1,95]; permanently non-result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_gradient_diagnostic(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.state,
        args.run_id,
        device=args.device,
        runs_dir=args.runs_dir,
        probe_manifest_path=args.probe_manifest,
        command=sys.argv,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        max_requests_per_surface=args.max_requests_per_surface,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
