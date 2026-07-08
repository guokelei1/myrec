#!/usr/bin/env python
"""Generate and evaluate C1 instrumentation canary runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.canaries import (  # noqa: E402
    generate_positive_title_leak_scores,
    generate_random_scores,
    generate_shuffled_label_scores,
)
from myrec.eval.evaluator import evaluate_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    standardized_dir = Path(args.standardized_dir)
    candidate_manifest = standardized_dir / "candidate_manifest.json"
    run_specs = [
        ("20260708_kuaisearch_random_c1", generate_random_scores),
        ("20260708_kuaisearch_label_shuffle_c1", generate_shuffled_label_scores),
        ("20260708_kuaisearch_positive_title_leak_c1", generate_positive_title_leak_scores),
    ]
    results = []
    for run_id, generator in run_specs:
        metadata = generator(
            standardized_dir=standardized_dir,
            split=args.split,
            run_id=run_id,
            runs_dir=args.runs_dir,
            seed=args.seed,
        )
        metrics = evaluate_run(
            run_id=run_id,
            split=args.split,
            candidate_manifest_path=candidate_manifest,
            standardized_dir=standardized_dir,
            runs_dir=args.runs_dir,
            dev_eval_log_path=args.dev_eval_log,
        )
        results.append({"metadata": metadata, "metrics": metrics})
        print(json.dumps({"run_id": run_id, "metrics": metrics}, ensure_ascii=False, sort_keys=True))
    print(json.dumps({"completed_runs": [run_id for run_id, _ in run_specs]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
