#!/usr/bin/env python
"""Export the shared train-only interaction artifact for Batch 2b."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.batch2b_interactions import export_batch2b_train_interactions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--output", default="artifacts/batch2b/interactions_train.jsonl")
    parser.add_argument(
        "--report",
        default="reports/pps_batch2b_interactions_train_manifest.json",
        help="Small tracked JSON manifest with counts and artifact hash.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_batch2b_train_interactions(
        standardized_dir=args.standardized_dir,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
