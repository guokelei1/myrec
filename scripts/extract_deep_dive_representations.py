#!/usr/bin/env python3
"""Extract resumable all-29-state Transformer deep-dive bundles."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.deep_dive_representation_runtime import (
    extract_deep_dive_representations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--role", required=True, choices=("train_probe", "dev_representation")
    )
    parser.add_argument("--condition", required=True, choices=("full", "null"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--manifest",
        default="experiments/motivation/transformer_deep_dive_manifest.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--requests-per-shard", type=int, default=32)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    result = extract_deep_dive_representations(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        role=args.role,
        condition_id=args.condition,
        device=args.device,
        runs_dir=args.runs_dir,
        manifest_path=args.manifest,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        requests_per_shard=args.requests_per_shard,
        max_requests=args.max_requests,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "run_id",
                    "method_id",
                    "condition_id",
                    "evidence_mode",
                    "status",
                    "request_count",
                    "candidate_count",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

