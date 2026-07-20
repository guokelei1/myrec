#!/usr/bin/env python3
"""Score one fixed Q0 branch-aggregate block over internal dev."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.q0_branch_runtime import write_q0_branch_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block", type=int, choices=(13, 20, 27), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    result = write_q0_branch_bundle(
        args.standardized_dir, args.config, args.checkpoint_root, args.run_id,
        block=args.block, device=args.device, runs_dir=args.runs_dir,
        resume=args.resume, max_wall_seconds=args.max_wall_seconds,
        max_requests=args.max_requests,
    )
    print(json.dumps({key: result.get(key) for key in ("run_id", "block_zero_based", "status", "request_count", "maximum_identity_delta")}, sort_keys=True))


if __name__ == "__main__":
    main()
