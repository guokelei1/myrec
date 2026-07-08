#!/usr/bin/env python
"""Run a static-mixture alpha grid over two upstream score runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.core import write_static_mixture_scores  # noqa: E402
from myrec.eval.evaluator import evaluate_run  # noqa: E402
from myrec.utils.jsonl import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--query-run-id", default="20260708_kuaisearch_b1_bm25_dev")
    parser.add_argument("--history-run-id", default="20260708_kuaisearch_b0b_recent_behavior_dev")
    parser.add_argument("--run-prefix", default="20260708_kuaisearch_b7_bm25_dev")
    parser.add_argument("--method-id", default="b7_bm25")
    parser.add_argument("--config", default="configs/baselines/b7_bm25.yaml")
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    standardized_dir = Path(args.standardized_dir)
    runs_dir = Path(args.runs_dir)
    candidate_manifest = standardized_dir / "candidate_manifest.json"
    query_scores = runs_dir / args.query_run_id / "scores.jsonl"
    history_scores = runs_dir / args.history_run_id / "scores.jsonl"
    grid_dir = runs_dir / f"{args.run_prefix}_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best = None
    for step in range(11):
        alpha = step / 10.0
        run_id = f"{args.run_prefix}_a{step:02d}"
        metadata = write_static_mixture_scores(
            query_scores_path=query_scores,
            history_scores_path=history_scores,
            query_run_id=args.query_run_id,
            history_run_id=args.history_run_id,
            run_id=run_id,
            method_id=args.method_id,
            alpha=alpha,
            candidate_manifest_path=candidate_manifest,
            runs_dir=runs_dir,
            config_path=args.config,
        )
        metrics = evaluate_run(
            run_id=run_id,
            split=args.split,
            candidate_manifest_path=candidate_manifest,
            standardized_dir=standardized_dir,
            runs_dir=runs_dir,
            dev_eval_log_path=args.dev_eval_log,
        )
        row = {"alpha": alpha, "run_id": run_id, "metadata": metadata, "metrics": metrics}
        rows.append(row)
        if best is None or metrics["ndcg@10"] > best["metrics"]["ndcg@10"]:
            best = row
        print(json.dumps({"alpha": alpha, "run_id": run_id, "ndcg@10": metrics["ndcg@10"]}, sort_keys=True))

    result = {
        "best_alpha": best["alpha"],
        "best_run_id": best["run_id"],
        "best_metrics": best["metrics"],
        "grid": [
            {
                "alpha": row["alpha"],
                "run_id": row["run_id"],
                "ndcg@10": row["metrics"]["ndcg@10"],
                "mrr": row["metrics"]["mrr"],
                "recall@10": row["metrics"]["recall@10"],
                "purchase_ndcg@10": row["metrics"]["purchase_ndcg@10"],
            }
            for row in rows
        ],
        "history_run_id": args.history_run_id,
        "method_id": args.method_id,
        "query_run_id": args.query_run_id,
    }
    write_json(grid_dir / "alpha_curve.json", result)
    write_json(runs_dir / best["run_id"] / "alpha_curve.json", result)
    print(json.dumps({"alpha_curve": str(grid_dir / "alpha_curve.json"), "best": result["best_run_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
