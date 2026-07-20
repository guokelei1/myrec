#!/usr/bin/env python3
"""Run one Q2 full-parameter RankNet/ListNet gradient-conflict state."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.objective_conflict_runtime import run_q2_objective_conflict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument(
        "--state",
        choices=("base_initialization", "frozen_final_checkpoint"),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-requests-per-surface", type=int)
    args = parser.parse_args()
    result = run_q2_objective_conflict(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.state,
        args.run_id,
        device=args.device,
        runs_dir=args.runs_dir,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        max_requests_per_surface=args.max_requests_per_surface,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "run_id",
                    "state",
                    "evidence_mode",
                    "status",
                    "completed_request_diagnostics",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
