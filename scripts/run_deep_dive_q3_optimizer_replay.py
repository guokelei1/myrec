#!/usr/bin/env python3
"""Run the registered Q3 step-501 LoRA optimizer replay."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.q3_optimizer_replay_runtime import run_q3_step501_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    result = run_q3_step501_replay(
        args.standardized_dir,
        args.config,
        args.run_id,
        device=args.device,
        runs_dir=args.runs_dir,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        max_tasks=args.max_tasks,
    )
    print(json.dumps({key: result.get(key) for key in ("run_id", "status", "completed_replay_blocks", "exact_effective_updates_computed", "elapsed_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
