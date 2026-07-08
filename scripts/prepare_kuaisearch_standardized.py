#!/usr/bin/env python
"""Prepare KuaiSearch standardized JSONL records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.kuaisearch_standardize import build_standardized_kuaisearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/kuaisearch")
    parser.add_argument(
        "--window-requests",
        default="data/interim/kuaisearch/v0_lite/time_window_seed20260708/requests.jsonl",
    )
    parser.add_argument("--output-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--c0-report", default="reports/pps_c0_data_audit.json")
    parser.add_argument("--leakage-report", default="reports/pps_c0_history_leakage_check.json")
    parser.add_argument("--max-history-len", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_standardized_kuaisearch(
        raw_dir=args.raw_dir,
        window_requests_path=args.window_requests,
        output_dir=args.output_dir,
        c0_report_path=args.c0_report,
        leakage_report_path=args.leakage_report,
        max_history_len=args.max_history_len,
    )
    print(
        {
            "output_dir": manifest["output_dir"],
            "counts_by_split": manifest["outputs"]["counts_by_split"],
            "history": manifest["outputs"]["history"],
            "item_join": manifest["item_join"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
