#!/usr/bin/env python3
"""Analyze Q3 LoRA factor/function paths at base, step-500, and final."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.lora_path_runtime import analyze_q3_lora_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    result = analyze_q3_lora_path(
        args.config,
        args.checkpoint_root,
        args.run_id,
        device=args.device,
        runs_dir=args.runs_dir,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "run_id",
                    "method_id",
                    "status",
                    "parameter_paths",
                    "orthogonal_gauge_identity_passed",
                    "base_b_exact_zero",
                    "elapsed_seconds",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
