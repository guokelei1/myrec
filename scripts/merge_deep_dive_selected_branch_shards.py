#!/usr/bin/env python3
"""Merge deterministic selected-branch request shards without reading qrels."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.selected_branch_shard_merge import (
    merge_selected_branch_request_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    result = merge_selected_branch_request_shards(
        args.standardized_dir,
        args.shard,
        args.output_dir,
        args.analysis_run_id,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "run_id",
                    "method_id",
                    "selected_block",
                    "status",
                    "request_count",
                    "score_rows",
                    "scores_sha256",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
