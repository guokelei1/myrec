#!/usr/bin/env python
"""Generate and evaluate a self-implemented core baseline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.core import (  # noqa: E402
    write_bm25_scores,
    write_popularity_scores,
    write_recent_behavior_scores,
)
from myrec.eval.evaluator import evaluate_run  # noqa: E402


DEFAULTS = {
    "b0a": {
        "run_id": "20260708_kuaisearch_b0a_popularity_dev",
        "config": "configs/baselines/b0a_popularity.yaml",
    },
    "b0b": {
        "run_id": "20260708_kuaisearch_b0b_recent_behavior_dev",
        "config": "configs/baselines/b0b_recent_behavior.yaml",
    },
    "b1": {
        "run_id": "20260708_kuaisearch_b1_bm25_dev",
        "config": "configs/baselines/b1_bm25.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--artifacts-dir", default="artifacts/baselines")
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    parser.add_argument("--bm25-tokenizer-mode", default="cjk_2_3gram")
    parser.add_argument("--bm25-exact-match-boost", type=float, default=2.0)
    parser.add_argument("--bm25-idf-scope", choices=["global", "request"], default="global")
    parser.add_argument("--bm25-char-coverage-boost", type=float, default=0.0)
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or DEFAULTS[args.method]["run_id"]
    config_path = args.config or DEFAULTS[args.method]["config"]
    standardized_dir = Path(args.standardized_dir)
    if args.method == "b0a":
        metadata = write_popularity_scores(
            standardized_dir=standardized_dir,
            split=args.split,
            run_id=run_id,
            runs_dir=args.runs_dir,
            artifacts_dir=args.artifacts_dir,
            config_path=config_path,
        )
    elif args.method == "b0b":
        metadata = write_recent_behavior_scores(
            standardized_dir=standardized_dir,
            split=args.split,
            run_id=run_id,
            runs_dir=args.runs_dir,
            config_path=config_path,
        )
    elif args.method == "b1":
        metadata = write_bm25_scores(
            standardized_dir=standardized_dir,
            split=args.split,
            run_id=run_id,
            runs_dir=args.runs_dir,
            artifacts_dir=args.artifacts_dir,
            config_path=config_path,
            tokenizer_mode=args.bm25_tokenizer_mode,
            exact_match_boost=args.bm25_exact_match_boost,
            idf_scope=args.bm25_idf_scope,
            char_coverage_boost=args.bm25_char_coverage_boost,
        )
    else:
        raise AssertionError(args.method)

    result = {"metadata": metadata}
    if not args.no_eval:
        result["metrics"] = evaluate_run(
            run_id=run_id,
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
