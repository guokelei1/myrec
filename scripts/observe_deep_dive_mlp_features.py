#!/usr/bin/env python3
"""Observe gate/SiLU/up/product formation on the frozen 512-row grid."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.mlp_feature_runtime import (
    write_mlp_feature_observation_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block", type=int, choices=(13, 20, 27), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    result = write_mlp_feature_observation_bundle(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        block=args.block,
        device=args.device,
        runs_dir=args.runs_dir,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        max_rows=args.max_rows,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "run_id",
                    "method_id",
                    "block_zero_based",
                    "status",
                    "observation_rows",
                    "maximum_score_identity_delta",
                    "maximum_product_recomposition_low_precision_ratio",
                    "maximum_delta_recomposition_abs_error",
                    "maximum_actual_product_quantization_abs_error",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
