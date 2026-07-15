#!/usr/bin/env python
"""Run the exploratory, outcome-free KuaiSearch Lite or Full source audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.kuaisearch_source_audit import audit_kuaisearch_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/kuaisearch")
    parser.add_argument(
        "--report-path",
        default="reports/history_response_gap_kuaisearch_lite_source_audit.json",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--collision-query-limit", type=int, default=500)
    parser.add_argument("--collision-requests-per-query", type=int, default=50)
    parser.add_argument("--rank-history-sample-size", type=int, default=1000)
    parser.add_argument("--max-recall-rows", type=int)
    parser.add_argument("--max-rank-scan-rows", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_kuaisearch_source(
        raw_dir=args.raw_dir,
        report_path=args.report_path,
        seed=args.seed,
        collision_query_limit=args.collision_query_limit,
        collision_requests_per_query=args.collision_requests_per_query,
        rank_history_sample_size=args.rank_history_sample_size,
        max_recall_rows=args.max_recall_rows,
        max_rank_scan_rows=args.max_rank_scan_rows,
        included_source_splits=("train",),
    )
    preview = {
        "dataset_id": report["dataset_id"],
        "direct_observations": report["direct_observations"],
        "report_path": args.report_path,
        "scope_warning": report["scope_warning"],
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
