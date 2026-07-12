#!/usr/bin/env python
"""Write the Amazon-C4 C1-style standardized-data protocol report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.amazon_c4_protocol_audit import audit_amazon_c4_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standardized-dir",
        default="data/standardized/amazon_c4/v0_history_bm25_100",
    )
    parser.add_argument("--c0-report", default="reports/pps_c0_amazon_c4_data_audit.json")
    parser.add_argument("--report", default="reports/pps_c1_amazon_c4_protocol.json")
    parser.add_argument("--skip-unit-tests", action="store_true")
    args = parser.parse_args()
    report = audit_amazon_c4_protocol(
        standardized_dir=args.standardized_dir,
        c0_report_path=args.c0_report,
        report_path=args.report,
        run_unit_tests=not args.skip_unit_tests,
    )
    print({"status": report["overall_status"], "checks": report["checks"]})
    return 0 if report["overall_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
