#!/usr/bin/env python3
"""Synthesize a complete registered mechanism family without reopening data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.statistical_synthesis import (  # noqa: E402
    PROBE_MANIFEST_PATH,
    synthesize_mechanism_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        required=True,
        choices=("m0_recoverability", "m1_input_interventions"),
        help="Frozen probe-manifest family key.",
    )
    parser.add_argument(
        "--analysis-dir",
        action="append",
        required=True,
        help="Completed paired mechanism analysis directory; repeat for the family.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New machine-readable JSON path; existing files are never overwritten.",
    )
    parser.add_argument(
        "--probe-manifest",
        default=str(PROBE_MANIFEST_PATH),
        help="Exact frozen mechanism probe manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = synthesize_mechanism_statistics(
        family=args.family,
        analysis_dirs=args.analysis_dir,
        output_path=args.output,
        probe_manifest_path=args.probe_manifest,
        command=sys.argv,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
