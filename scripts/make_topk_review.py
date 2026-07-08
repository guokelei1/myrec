#!/usr/bin/env python
"""Create a sampled top-k review sheet for a score run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.topk_review import write_topk_review  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = write_topk_review(
        records_path=Path(args.standardized_dir) / f"records_{args.split}.jsonl",
        scores_path=Path(args.runs_dir) / args.run_id / "scores.jsonl",
        output_path=args.output,
        run_id=args.run_id,
        sample_size=args.sample_size,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
