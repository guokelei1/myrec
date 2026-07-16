#!/usr/bin/env python
"""Materialize unified PPS records for the official ZAM/TEM baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.prodsearch_adapter import materialize_prodsearch_format


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--valid-examples", type=int, default=5000)
    parser.add_argument("--valid-candidate-size", type=int, default=100)
    parser.add_argument("--test-candidate-size", type=int, default=100)
    parser.add_argument("--max-train-requests", type=int)
    parser.add_argument("--max-dev-requests", type=int)
    parser.add_argument("--dev-history-condition", choices=("true", "null"), default="true")
    args = parser.parse_args()
    result = materialize_prodsearch_format(
        args.standardized_dir,
        args.output_root,
        seed=args.seed,
        valid_examples=args.valid_examples,
        valid_candidate_size=args.valid_candidate_size,
        test_candidate_size=args.test_candidate_size,
        max_train_requests=args.max_train_requests,
        max_dev_requests=args.max_dev_requests,
        dev_history_condition=args.dev_history_condition,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
