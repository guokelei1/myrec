#!/usr/bin/env python3
"""Score registered N14 history-embedding operators without qrels."""

from __future__ import annotations

import argparse

from myrec.mechanism.embedding_stage_runtime import write_embedding_stage_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=13500.0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = write_embedding_stage_bundle(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        device=args.device,
        max_wall_seconds=args.max_wall_seconds,
        max_requests=args.max_requests,
        resume=args.resume,
    )
    print(result)


if __name__ == "__main__":
    main()
