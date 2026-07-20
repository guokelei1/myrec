#!/usr/bin/env python3
"""Confirm one immutable D2 selected transition on fold 1."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.postblock_sweep_evaluator import (
    POSTBLOCK_BLOCKS,
    confirm_postblock_fold1,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--qrels-split-dir", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--bundle", action="append", nargs=2, metavar=("BLOCK", "DIR"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    bundles = {int(block): path for block, path in args.bundle}
    if set(bundles) != set(POSTBLOCK_BLOCKS):
        parser.error("exactly one --bundle is required for each block 13..27")
    result = confirm_postblock_fold1(
        args.standardized_dir, args.qrels_split_dir, args.selection, bundles,
        args.output_dir, args.analysis_run_id, dev_eval_log_path=args.dev_eval_log,
    )
    print(json.dumps(result["fixed_transition_confirmation"], sort_keys=True))


if __name__ == "__main__":
    main()
