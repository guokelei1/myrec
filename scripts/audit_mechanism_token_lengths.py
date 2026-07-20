#!/usr/bin/env python
"""Audit frozen Q0--Q3 prompt lengths for all M1 internal-dev conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.token_length_audit import run_token_length_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standardized-dir",
        required=True,
        help="Frozen full_confirm_preceding40k_v11 standardized directory.",
    )
    parser.add_argument(
        "--probe-manifest",
        required=True,
        help="Frozen mechanism probe_manifest.yaml (exact SHA is enforced).",
    )
    parser.add_argument(
        "--assignment-manifest",
        required=True,
        help="Frozen M1 assignment manifest (exact SHA is enforced).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New empty run-local output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = run_token_length_audit(
        args.standardized_dir,
        args.probe_manifest,
        args.assignment_manifest,
        args.output_dir,
        command=sys.argv,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
