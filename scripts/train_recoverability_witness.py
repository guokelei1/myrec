#!/usr/bin/env python
"""Train the strict-nonrepeat semantic recoverability witness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.recoverability_witness import train_recoverability_witness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--embedding-model-name", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--embedding-device", default="cuda:2")
    parser.add_argument("--history-budget", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--config")
    args = parser.parse_args()
    result = train_recoverability_witness(
        args.standardized_dir,
        args.output_model_dir,
        args.run_id,
        runs_dir=args.runs_dir,
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_model_name=args.embedding_model_name,
        embedding_device=args.embedding_device,
        history_budget=args.history_budget,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        seed=args.seed,
        config_path=args.config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
