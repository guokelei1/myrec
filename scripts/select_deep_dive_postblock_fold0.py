#!/usr/bin/env python3
"""Select and freeze one model's D2 block from all 15 fold-0 bundles."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.postblock_sweep_evaluator import (
    POSTBLOCK_BLOCKS,
    select_postblock_fold0,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--qrels-split-dir", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--bundle", action="append", nargs=2, metavar=("BLOCK", "DIR"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    bundles = {int(block): path for block, path in args.bundle}
    if set(bundles) != set(POSTBLOCK_BLOCKS):
        parser.error("exactly one --bundle is required for each block 13..27")
    result = select_postblock_fold0(
        args.standardized_dir, args.qrels_split_dir, args.method_id, bundles,
        args.output_dir, args.analysis_run_id, dev_eval_log_path=args.dev_eval_log,
    )
    print(json.dumps({key: result[key] for key in ("method_id", "selected_block", "minimum_adjacent_margin_step", "status")}, sort_keys=True))


if __name__ == "__main__":
    main()
