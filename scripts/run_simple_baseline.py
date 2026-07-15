#!/usr/bin/env python
"""Score one exploratory standardized split with a simple shared baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.core import (
    write_bm25_scores,
    write_popularity_scores,
    write_recent_behavior_scores,
    write_source_order_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("source_order", "popularity", "recent_behavior", "bm25_request"),
    )
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--split", default="dev", choices=("train", "dev"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--artifacts-dir", default="artifacts/baselines")
    parser.add_argument("--config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common = {
        "standardized_dir": args.standardized_dir,
        "split": args.split,
        "run_id": args.run_id,
        "runs_dir": args.runs_dir,
        "config_path": args.config,
    }
    if args.method == "source_order":
        metadata = write_source_order_scores(**common)
    elif args.method == "popularity":
        metadata = write_popularity_scores(
            **common,
            artifacts_dir=args.artifacts_dir,
        )
    elif args.method == "recent_behavior":
        metadata = write_recent_behavior_scores(**common)
    else:
        metadata = write_bm25_scores(
            **common,
            artifacts_dir=args.artifacts_dir,
            idf_scope="request",
            tokenizer_mode="cjk_2_3gram",
        )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
