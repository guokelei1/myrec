#!/usr/bin/env python3
"""Extract resumable train/dev M2 activation bundles without qrels access."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.representation_runtime import extract_m2_activations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--role", required=True, choices=("train_probe", "dev_representation")
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=("full", "null", "relevant_6", "irrelevant_6"),
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--assignment")
    parser.add_argument("--assignment-manifest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--requests-per-shard", type=int, default=32)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    result = extract_m2_activations(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        role=args.role,
        condition_id=args.condition,
        device=args.device,
        runs_dir=args.runs_dir,
        assignment_path=args.assignment,
        assignment_manifest_path=args.assignment_manifest,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        requests_per_shard=args.requests_per_shard,
        max_requests=args.max_requests,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
