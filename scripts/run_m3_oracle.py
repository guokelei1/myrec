#!/usr/bin/env python
"""Run M3 per-request oracle headroom analysis."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.oracle import run_per_request_oracle  # noqa: E402


DEFAULT_METHODS = {
    "query_b2z": "20260708_kuaisearch_b2z_bge_small_zh_dev",
    "history_b0b": "20260708_kuaisearch_b0b_recent_behavior_dev",
    "static_b7_bge": "20260708_kuaisearch_b7_bge_dev_a02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-run-id", default="20260708_kuaisearch_m3_oracle_dev")
    parser.add_argument("--report-copy", default="reports/pps_m3_headroom_summary.json")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--metric", default="ndcg@10")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = runs_dir / args.output_run_id
    paths = {
        name: runs_dir / run_id / "per_request_metrics.jsonl"
        for name, run_id in DEFAULT_METHODS.items()
    }
    summary = run_per_request_oracle(
        method_metric_paths=paths,
        output_dir=output_dir,
        metric=args.metric,
        bootstrap_samples=args.samples,
        seed=args.seed,
    )
    summary["input_run_ids"] = DEFAULT_METHODS
    summary["run_id"] = args.output_run_id
    with (output_dir / "headroom_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    report_copy = Path(args.report_copy)
    report_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_dir / "headroom_summary.json", report_copy)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["gate_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
