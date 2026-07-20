#!/usr/bin/env python3
"""Score one Q2/Q3 D5 fixed-length contextual-control bundle."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.contextual_control_runtime import (
    write_contextual_control_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    result = write_contextual_control_bundle(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        device=args.device,
        runs_dir=args.runs_dir,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        max_requests=args.max_requests,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "run_id",
                    "method_id",
                    "evidence_mode",
                    "status",
                    "request_count",
                    "score_rows",
                    "identity_passed",
                    "maximum_identity_delta",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
