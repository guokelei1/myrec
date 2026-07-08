#!/usr/bin/env python
"""Run KuaiSearch Phase 0/C0 data audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.kuaisearch_audit import audit_kuaisearch_c0, dump_report_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/kuaisearch")
    parser.add_argument("--report-path", default="reports/pps_c0_data_audit.json")
    parser.add_argument(
        "--sample-dir",
        default="data/interim/kuaisearch/v0_lite/time_window_seed20260708",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--recall-sample-size", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=250_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_kuaisearch_c0(
        raw_dir=args.raw_dir,
        report_path=args.report_path,
        sample_dir=args.sample_dir,
        seed=args.seed,
        recall_sample_size=args.recall_sample_size,
        window_size=args.window_size,
    )
    print(dump_report_preview(report))
    return 0 if report["overall_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
