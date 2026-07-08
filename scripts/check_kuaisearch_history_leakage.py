#!/usr/bin/env python
"""Run the KuaiSearch C0 recent-history leakage cross-check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.kuaisearch_leakage import (
    check_history_leakage,
    merge_history_leakage_into_c0,
)
from myrec.utils.jsonl import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/kuaisearch")
    parser.add_argument("--report-path", default="reports/pps_c0_history_leakage_check.json")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--merge-c0-report", default="reports/pps_c0_data_audit.json")
    parser.add_argument("--no-merge", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = check_history_leakage(
        raw_dir=args.raw_dir,
        seed=args.seed,
        sample_size=args.sample_size,
        max_examples=args.max_examples,
    )
    write_json(args.report_path, report)
    if not args.no_merge:
        merge_history_leakage_into_c0(
            c0_report_path=args.merge_c0_report,
            leakage_report_path=args.report_path,
        )
    print(
        {
            "status": report["status"],
            "checked_requests": report["classification"]["checked_requests"],
            "total_history_items": report["classification"]["total_history_items"],
            "rates": report["classification"]["rates"],
            "manual_review_required": report["classification"]["manual_review_required"],
        }
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
