#!/usr/bin/env python3
"""CLI for the qrels-blind N10 Q3 LoRA rank-path scorer."""

from __future__ import annotations

import argparse
import sys

from myrec.mechanism.q3_lora_rank_runtime import write_q3_lora_rank_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--manifest", default="experiments/motivation/transformer_n10_q3_lora_rank_manifest_v1.yaml")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    write_q3_lora_rank_bundle(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        device=args.device,
        manifest_path=args.manifest,
        max_wall_seconds=args.max_wall_seconds,
        max_requests=args.max_requests,
        resume=args.resume,
        command=sys.argv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

