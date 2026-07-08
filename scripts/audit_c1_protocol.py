#!/usr/bin/env python
"""Write the C1 standardized-data and evaluator protocol report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.protocol_audit import DEFAULT_CANARY_RUN_IDS, audit_c1_protocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--report-path", default="reports/pps_c1_protocol.json")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--tmp-dir", default="tmp/c1_protocol_audit")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--random-run-id", default=DEFAULT_CANARY_RUN_IDS["random"])
    parser.add_argument("--label-shuffle-run-id", default=DEFAULT_CANARY_RUN_IDS["label_shuffle"])
    parser.add_argument("--positive-title-leak-run-id", default=DEFAULT_CANARY_RUN_IDS["positive_title_leak"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_c1_protocol(
        standardized_dir=args.standardized_dir,
        report_path=args.report_path,
        runs_dir=args.runs_dir,
        tmp_dir=args.tmp_dir,
        canary_run_ids={
            "random": args.random_run_id,
            "label_shuffle": args.label_shuffle_run_id,
            "positive_title_leak": args.positive_title_leak_run_id,
        },
        run_unit_tests=not args.skip_unit_tests,
    )
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "report_path": args.report_path,
                "checks": {name: check["status"] for name, check in report["checks"].items()},
            },
            sort_keys=True,
        )
    )
    return 0 if report["overall_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
