#!/usr/bin/env python
"""Generate and evaluate local trainable Batch 2 hashed-feature baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.batch2_hashed import write_hashed_batch2_scores  # noqa: E402
from myrec.eval.evaluator import evaluate_run  # noqa: E402


DEFAULTS = {
    "b4": {
        "config": "configs/baselines/b4_sasrec_style_hashed.yaml",
        "run_prefix": "20260708_kuaisearch_b4_sasrec_style_hashed_prior_dev",
    },
    "b5": {
        "config": "configs/baselines/b5_kuaisearch_dcn_din_style_hashed.yaml",
        "run_prefix": "20260708_kuaisearch_b5_dcn_din_style_hashed_dev",
    },
    "b6": {
        "config": "configs/baselines/b6_pps_classic_style_hashed.yaml",
        "run_prefix": "20260708_kuaisearch_b6_pps_classic_style_hashed_dev",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--artifacts-dir", default="artifacts/baselines")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-features", type=int, default=1 << 20)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--max-history-len", type=int, default=20)
    parser.add_argument("--max-query-tokens", type=int, default=12)
    parser.add_argument("--max-doc-tokens", type=int, default=24)
    parser.add_argument("--tokenizer-mode", default="cjk_2_3gram")
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = DEFAULTS[args.method]
    run_id = args.run_id or f"{defaults['run_prefix']}_s{args.seed}"
    config_path = args.config or defaults["config"]
    standardized_dir = Path(args.standardized_dir)
    metadata = write_hashed_batch2_scores(
        standardized_dir=standardized_dir,
        split=args.split,
        run_id=run_id,
        method=args.method,
        seed=args.seed,
        runs_dir=args.runs_dir,
        artifacts_dir=args.artifacts_dir,
        config_path=config_path,
        n_features=args.n_features,
        chunk_size=args.chunk_size,
        epochs=args.epochs,
        negatives_per_positive=args.negatives_per_positive,
        max_history_len=args.max_history_len,
        max_query_tokens=args.max_query_tokens,
        max_doc_tokens=args.max_doc_tokens,
        tokenizer_mode=args.tokenizer_mode,
        alpha=args.alpha,
    )
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
