#!/usr/bin/env python
"""Generate and evaluate the B3 cross-encoder zero-shot baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.cross_encoder import write_cross_encoder_scores  # noqa: E402
from myrec.eval.evaluator import evaluate_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--run-id", default="20260708_kuaisearch_b3_bge_reranker_base_zs_dev")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config", default="configs/baselines/b3_cross_encoder.yaml")
    parser.add_argument("--model-name", default="BAAI/bge-reranker-base")
    parser.add_argument("--cache-folder", default="models/huggingface/cross_encoders")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--pair-chunk-size", type=int, default=8192)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    standardized_dir = Path(args.standardized_dir)
    metadata = write_cross_encoder_scores(
        standardized_dir=standardized_dir,
        split=args.split,
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        config_path=args.config,
        model_name=args.model_name,
        cache_folder=args.cache_folder,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pair_chunk_size=args.pair_chunk_size,
    )
    result = {"metadata": metadata}
    if not args.no_eval:
        result["metrics"] = evaluate_run(
            run_id=args.run_id,
            split=args.split,
            candidate_manifest_path=standardized_dir / "candidate_manifest.json",
            standardized_dir=standardized_dir,
            runs_dir=args.runs_dir,
            dev_eval_log_path=args.dev_eval_log,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
