#!/usr/bin/env python3
"""Score all registered nodes at one frozen D2 selected block."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.selected_branch_runtime import write_selected_branch_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--branch-contract", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--request-shard-index", type=int, default=0)
    parser.add_argument("--request-shard-count", type=int, default=1)
    args = parser.parse_args()
    result = write_selected_branch_bundle(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        device=args.device,
        branch_contract_path=args.branch_contract,
        runs_dir=args.runs_dir,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        max_requests=args.max_requests,
        request_shard_index=args.request_shard_index,
        request_shard_count=args.request_shard_count,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "run_id",
                    "method_id",
                    "selected_block",
                    "status",
                    "request_count",
                    "score_rows",
                    "maximum_identity_delta",
                    "elapsed_seconds",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
