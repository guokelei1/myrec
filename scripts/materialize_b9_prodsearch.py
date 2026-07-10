#!/usr/bin/env python
"""Materialize KuaiSearch train/dev records for official ProdSearch B9."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.prodsearch_adapter import materialize_prodsearch_format  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--valid-examples", type=int, default=5000)
    parser.add_argument("--valid-candidate-size", type=int, default=500)
    parser.add_argument("--test-candidate-size", type=int, default=1500)
    parser.add_argument("--max-train-requests", type=int, default=None)
    parser.add_argument("--max-dev-requests", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize_prodsearch_format(
        args.standardized_dir,
        args.output_root,
        seed=args.seed,
        valid_examples=args.valid_examples,
        valid_candidate_size=args.valid_candidate_size,
        test_candidate_size=args.test_candidate_size,
        max_train_requests=args.max_train_requests,
        max_dev_requests=args.max_dev_requests,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

