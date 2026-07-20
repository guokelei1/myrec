#!/usr/bin/env python3
"""Score one audited Motivation history intervention on frozen internal-dev."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.scorer import (  # noqa: E402
    MAX_WALL_SECONDS,
    MECHANISM_PROBE_MANIFEST_PATH,
    write_mechanism_intervention_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--assignment-manifest", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--reference-run-id", required=True)
    parser.add_argument(
        "--probe-manifest",
        default=str(MECHANISM_PROBE_MANIFEST_PATH),
        help="Exact frozen mechanism-stage probe manifest (path and hash locked).",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("dev",),
        help="Mechanism scoring is locked to frozen v11 internal-dev.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True, help="Explicit device, e.g. cuda:0")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after validating every completed request block and hash.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=MAX_WALL_SECONDS,
        help=f"Resumable segment wall limit; must be <= {int(MAX_WALL_SECONDS)}.",
    )
    parser.add_argument(
        "--max-score-requests",
        type=int,
        help="Smoke cap; a capped bundle is permanently marked non-result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_mechanism_intervention_scores(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.assignment,
        args.assignment_manifest,
        args.condition_id,
        args.reference_run_id,
        args.split,
        args.run_id,
        device=args.device,
        runs_dir=args.runs_dir,
        command=sys.argv,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        max_score_requests=args.max_score_requests,
        probe_manifest_path=args.probe_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
