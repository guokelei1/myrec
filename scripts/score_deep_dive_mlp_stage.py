#!/usr/bin/env python
"""CLI for the qrels-blind N12 SwiGLU stage scorer."""

from __future__ import annotations

import argparse

from myrec.mechanism.mlp_stage_runtime import write_mlp_stage_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    write_mlp_stage_bundle(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        block=args.block,
        device=args.device,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        max_requests=args.max_requests,
    )


if __name__ == "__main__":
    main()

