#!/usr/bin/env python3
"""Score one resumable preregistered M2 activation patch bundle."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.patch_scorer import PATCH_KINDS, write_m2_patch_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--full-activation-bundle", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--patch-kind", choices=PATCH_KINDS, required=True)
    parser.add_argument("--block", type=int, choices=(13, 27), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    result = write_m2_patch_scores(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.full_activation_bundle,
        args.run_id,
        patch_kind=args.patch_kind,
        block=args.block,
        device=args.device,
        runs_dir=args.runs_dir,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
        max_requests=args.max_requests,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
