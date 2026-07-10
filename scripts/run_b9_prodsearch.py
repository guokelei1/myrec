#!/usr/bin/env python
"""Run official ProdSearch ZAM/TEM and export scores without evaluating."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.prodsearch_adapter import run_official_prodsearch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines/b9_prodsearch.yaml")
    parser.add_argument("--model", required=True, choices=["zam", "tem"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--materialized-root", default=None)
    parser.add_argument("--candidate-manifest", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--max-train-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--valid-batch-size", type=int, default=None)
    parser.add_argument("--valid-candidate-size", type=int, default=None)
    parser.add_argument("--test-candidate-size", type=int, default=None)
    parser.add_argument("--candidate-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config["models"][args.model]
    common = config["common"]
    adapter = config["adapter"]
    run_dir = Path(args.runs_dir) / args.run_id
    metadata = run_official_prodsearch(
        baseline_dir=config["upstream"]["local_dir"],
        materialized_root=args.materialized_root or config["materialized_root"],
        run_dir=run_dir,
        model=args.model,
        seed=args.seed,
        embedding_size=int(model["embedding_size"]),
        learning_rate=float(model["learning_rate"]),
        max_train_epoch=args.max_train_epoch or int(common["max_train_epoch"]),
        batch_size=args.batch_size or int(model["batch_size"]),
        valid_batch_size=args.valid_batch_size or int(common["valid_batch_size"]),
        valid_candidate_size=args.valid_candidate_size or int(adapter["valid_candidate_size"]),
        test_candidate_size=args.test_candidate_size or int(adapter["dev_candidate_size"]),
        candidate_batch_size=args.candidate_batch_size or int(common["candidate_batch_size"]),
        num_workers=common["num_workers"] if args.num_workers is None else args.num_workers,
        candidate_manifest_path=args.candidate_manifest or config["candidate_manifest"],
        method_id=model["method_id"],
    )
    shutil.copyfile(config_path, run_dir / "config_snapshot.yaml")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
