#!/usr/bin/env python3
"""Evaluate all six M2 patch bundles after score integrity checks."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.patch_evaluator import evaluate_m2_patches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--full-baseline", required=True)
    parser.add_argument("--null-baseline", required=True)
    parser.add_argument(
        "--patch",
        action="append",
        nargs=3,
        metavar=("KIND", "BLOCK", "PATH"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    patches: dict[str, dict[int, str]] = {}
    for kind, raw_block, path in args.patch:
        block = int(raw_block)
        if block in patches.setdefault(kind, {}):
            parser.error(f"duplicate --patch {kind} {block}")
        patches[kind][block] = path
    result = evaluate_m2_patches(
        args.standardized_dir,
        {"full": args.full_baseline, "null": args.null_baseline},
        patches,
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
        command=sys.argv,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
