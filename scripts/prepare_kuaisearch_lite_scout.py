#!/usr/bin/env python
"""Build a label-isolated KuaiSearch Lite or Full-source exploratory scout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.kuaisearch_scout import build_kuaisearch_lite_scout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/kuaisearch")
    parser.add_argument(
        "--output-dir",
        default="data/standardized/kuaisearch/lite_scout10k_v1",
    )
    parser.add_argument(
        "--report-path",
        default="reports/history_response_gap_kuaisearch_lite_scout.json",
    )
    parser.add_argument("--dataset-version", default="lite_scout10k_v1")
    parser.add_argument("--max-requests", type=int, default=10_000)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--min-candidate-count", type=int, default=2)
    parser.add_argument("--max-candidate-count", type=int, default=100)
    parser.add_argument("--max-history-len", type=int, default=20)
    parser.add_argument(
        "--include-history-query",
        action="store_true",
        help="Retain the prior query attached to each clicked/purchased history item.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_kuaisearch_lite_scout(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        report_path=args.report_path,
        dataset_version=args.dataset_version,
        max_requests=args.max_requests,
        dev_fraction=args.dev_fraction,
        min_candidate_count=args.min_candidate_count,
        max_candidate_count=args.max_candidate_count,
        max_history_len=args.max_history_len,
        include_history_query=args.include_history_query,
    )
    preview = {
        "dataset_version": manifest["dataset_version"],
        "selection": manifest["selection"],
        "item_join": manifest["item_join"],
        "counts": manifest["outputs"]["counts"],
        "report_path": args.report_path,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
