#!/usr/bin/env python
"""Train and score one official-code ZAM/TEM baseline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.prodsearch_adapter import run_official_prodsearch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--model", choices=("zam", "tem"), required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument(
        "--history-condition", choices=("true", "null", "wrong"), default="true"
    )
    parser.add_argument("--split", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--baseline-dir", default="baselines/pps_classic/prodsearch_tem")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-train-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--valid-batch-size", type=int, default=24)
    parser.add_argument("--valid-candidate-size", type=int, default=100)
    parser.add_argument("--test-candidate-size", type=int, default=100)
    parser.add_argument("--candidate-batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    result = run_official_prodsearch(
        baseline_dir=args.baseline_dir,
        materialized_root=args.materialized_root,
        run_dir=args.run_dir,
        model=args.model,
        seed=args.seed,
        embedding_size=args.embedding_size,
        learning_rate=args.learning_rate,
        max_train_epoch=args.max_train_epoch,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        valid_candidate_size=args.valid_candidate_size,
        test_candidate_size=args.test_candidate_size,
        candidate_batch_size=args.candidate_batch_size,
        num_workers=args.num_workers,
        candidate_manifest_path=args.candidate_manifest,
        method_id=args.method_id,
        history_condition=args.history_condition,
        split=args.split,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
