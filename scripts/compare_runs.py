#!/usr/bin/env python
"""Compare two runs using paired bootstrap over per-request metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.compare import compare_per_request_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--request-ids", default=None)
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_per_request_metrics(
        run_a_path=Path(args.runs_dir) / args.run_a / "per_request_metrics.jsonl",
        run_b_path=Path(args.runs_dir) / args.run_b / "per_request_metrics.jsonl",
        output_path=args.output,
        metric=args.metric,
        samples=args.samples,
        seed=args.seed,
        request_ids_path=args.request_ids,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
