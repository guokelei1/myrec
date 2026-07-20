#!/usr/bin/env python3
"""Score qrels-blind N17/N18 routing or N19 Q3 LoRA bundles."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.routing_boundary_runtime import (
    write_gqa_grouping_bundle,
    write_q3_lora_branch_bundle,
    write_qk_head_norm_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--kind", choices=("head_norm", "gqa", "lora_branch"), required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--component", choices=("q", "k", "v"))
    parser.add_argument("--max-wall-seconds", type=float, default=13500.0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    common = dict(
        standardized_dir=args.standardized_dir,
        config_path=args.config,
        checkpoint_root=args.checkpoint_root,
        run_id=args.run_id,
        block=args.block,
        device=args.device,
        max_wall_seconds=args.max_wall_seconds,
        max_requests=args.max_requests,
        resume=args.resume,
        command=sys.argv,
    )
    if args.kind == "head_norm":
        if args.component is None:
            parser.error("head_norm requires --component q or k")
        result = write_qk_head_norm_bundle(**common, component=args.component)
    elif args.kind == "gqa":
        if args.component is not None:
            parser.error("gqa does not take --component")
        result = write_gqa_grouping_bundle(**common)
    else:
        if args.component not in {"q", "v"}:
            parser.error("lora_branch requires --component q or v")
        result = write_q3_lora_branch_bundle(**common, component=args.component)
    print(json.dumps({key: result.get(key) for key in (
        "run_id", "method_id", "analysis_stage", "status", "result_eligible",
        "maximum_identity_delta", "maximum_frozen_baseline_delta",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
