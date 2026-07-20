#!/usr/bin/env python3
"""Score the frozen reverse component-state removal extension."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.component_necessity_runtime import (
    write_component_necessity_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--branch-contract", required=True)
    parser.add_argument("--parent-selected-branch", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    result = write_component_necessity_bundle(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        device=args.device,
        branch_contract_path=args.branch_contract,
        parent_selected_branch_dir=args.parent_selected_branch,
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
