#!/usr/bin/env python
"""Aggregate per-request metrics over a fixed request-id subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.metrics import aggregate_request_metrics  # noqa: E402
from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import iter_jsonl, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.request_ids).open("r", encoding="utf-8") as handle:
        request_ids = {line.strip() for line in handle if line.strip()}
    rows = [
        row
        for row in iter_jsonl(Path(args.runs_dir) / args.run_id / "per_request_metrics.jsonl")
        if str(row["request_id"]) in request_ids
    ]
    if len(rows) != len(request_ids):
        raise ValueError(f"subset coverage mismatch: metrics={len(rows)} request_ids={len(request_ids)}")
    metrics = aggregate_request_metrics(rows)
    metrics.update(
        {
            "request_ids_path": args.request_ids,
            "request_ids_sha256": sha256_file(args.request_ids),
            "run_id": args.run_id,
        }
    )
    write_json(args.output, metrics)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
