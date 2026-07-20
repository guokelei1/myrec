#!/usr/bin/env python3
"""Run one qrels-blind Transformer deep-dive identity smoke."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.deep_dive_smoke import run_deep_dive_identity_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--identity-gate", action="store_true")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--manifest",
        default="experiments/motivation/transformer_deep_dive_manifest.yaml",
    )
    args = parser.parse_args()
    result = run_deep_dive_identity_smoke(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        device=args.device,
        max_requests=args.max_requests,
        identity_gate=args.identity_gate,
        runs_dir=args.runs_dir,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "run_id",
                    "method_id",
                    "status",
                    "evidence_mode",
                    "request_count",
                    "maximum_identity_error",
                    "algebra_recomposition_passed",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
