#!/usr/bin/env python3
"""Score qrels-blind N15 residual-composition or N16 RMSNorm bundles."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.operator_stage_runtime import (
    write_rmsnorm_bundle,
    write_residual_composition_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--kind", choices=("residual", "rmsnorm"), required=True)
    parser.add_argument("--block", type=int)
    parser.add_argument("--branch", choices=("attention", "mlp"))
    parser.add_argument("--scope", choices=("input", "post_attention", "final"))
    parser.add_argument("--max-wall-seconds", type=float, default=13500.0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    common = dict(
        standardized_dir=args.standardized_dir,
        config_path=args.config,
        checkpoint_root=args.checkpoint_root,
        run_id=args.run_id,
        device=args.device,
        max_wall_seconds=args.max_wall_seconds,
        max_requests=args.max_requests,
        resume=args.resume,
        command=sys.argv,
    )
    if args.kind == "residual":
        if args.block is None or args.branch is None:
            parser.error("residual kind requires --block and --branch")
        result = write_residual_composition_bundle(
            **common, block=args.block, branch=args.branch
        )
    else:
        if args.scope is None or (args.scope != "final" and args.block is None):
            parser.error("rmsnorm requires --scope and a block except for final scope")
        result = write_rmsnorm_bundle(**common, scope=args.scope, block=args.block)
    print(json.dumps({key: result.get(key) for key in (
        "run_id", "method_id", "analysis_stage", "status", "result_eligible",
        "maximum_identity_delta", "maximum_frozen_baseline_delta",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
