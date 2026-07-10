#!/usr/bin/env python
"""Rescore an existing official B9 checkpoint for determinism checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.prodsearch_adapter import rescore_official_prodsearch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines/b9_prodsearch.yaml")
    parser.add_argument("--model", required=True, choices=["zam", "tem"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--materialized-root", required=True)
    parser.add_argument("--checkpoint-official-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--valid-candidate-size", type=int, default=None)
    parser.add_argument("--test-candidate-size", type=int, default=None)
    parser.add_argument("--candidate-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rankfname", default="determinism.ranklist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config["models"][args.model]
    common = config["common"]
    adapter = config["adapter"]
    result = rescore_official_prodsearch(
        baseline_dir=config["upstream"]["local_dir"],
        materialized_root=args.materialized_root,
        checkpoint_official_dir=args.checkpoint_official_dir,
        output_dir=args.output_dir,
        model=args.model,
        seed=args.seed,
        embedding_size=int(model["embedding_size"]),
        learning_rate=float(model["learning_rate"]),
        batch_size=int(model["batch_size"]),
        valid_batch_size=int(common["valid_batch_size"]),
        valid_candidate_size=args.valid_candidate_size or int(adapter["valid_candidate_size"]),
        test_candidate_size=args.test_candidate_size or int(adapter["dev_candidate_size"]),
        candidate_batch_size=args.candidate_batch_size or int(common["candidate_batch_size"]),
        num_workers=args.num_workers,
        candidate_manifest_path=args.candidate_manifest,
        method_id=model["method_id"],
        rankfname=args.rankfname,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
