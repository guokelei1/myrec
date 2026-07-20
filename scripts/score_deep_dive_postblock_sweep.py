#!/usr/bin/env python3
"""Score one registered Q2/Q3 D2 post-block sweep bundle."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.postblock_sweep_runtime import write_postblock_sweep_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--q3-gate-metrics")
    parser.add_argument("--fold0-selection")
    parser.add_argument("--q2-reuse-identity-dir")
    parser.add_argument("--q2-reuse-same-dir")
    parser.add_argument("--q2-reuse-cross-dir")
    parser.add_argument("--q2-reuse-audit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    result = write_postblock_sweep_bundle(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        block=args.block,
        fold=args.fold,
        device=args.device,
        runs_dir=args.runs_dir,
        q3_gate_metrics_path=args.q3_gate_metrics,
        fold0_selection_path=args.fold0_selection,
        q2_reuse_identity_dir=args.q2_reuse_identity_dir,
        q2_reuse_same_dir=args.q2_reuse_same_dir,
        q2_reuse_cross_dir=args.q2_reuse_cross_dir,
        q2_reuse_audit_path=args.q2_reuse_audit,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        max_requests=args.max_requests,
    )
    print(json.dumps({key: result.get(key) for key in ("run_id", "method_id", "block_zero_based", "normalized_query_fold", "status", "request_count", "score_rows", "maximum_identity_delta", "elapsed_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
