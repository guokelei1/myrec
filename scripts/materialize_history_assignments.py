#!/usr/bin/env python
"""Build label-free true/null/matched-wrong history assignment files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.history_assignments import materialize_history_assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument(
        "--donor-records",
        help="Optional earlier-split donor records; target records are also included.",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--global-donor-shortlist-size",
        type=int,
        default=512,
        help="Deterministic global donor window; exact-query donors are never shortened.",
    )
    parser.add_argument(
        "--motivation-v12-release-lock",
        help=(
            "Required for the registered Motivation V1.2 holdout; validates and "
            "binds the frozen assignment generator and recipe."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = materialize_history_assignments(
        args.records,
        args.output_dir,
        args.report_path,
        donor_records_path=args.donor_records,
        seed=args.seed,
        global_donor_shortlist_size=args.global_donor_shortlist_size,
        motivation_v12_release_lock_path=args.motivation_v12_release_lock,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
