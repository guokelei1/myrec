#!/usr/bin/env python3
"""Score Q3 all-native and first-position-only block-state gate cells."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.deep_dive_native_patch import write_q3_native_patch_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block", type=int, required=True, choices=(13, 27))
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--manifest",
        default="experiments/motivation/transformer_deep_dive_manifest.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    result = write_q3_native_patch_gate(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        block=args.block,
        device=args.device,
        runs_dir=args.runs_dir,
        manifest_path=args.manifest,
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
                    "block_zero_based",
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

