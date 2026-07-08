#!/usr/bin/env python
"""Generate and evaluate B8 LLM rerank baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.llm_rerank import ensure_subset_request_ids, write_llm_rerank_scores  # noqa: E402
from myrec.eval.evaluator import evaluate_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["b8a", "b8b"])
    parser.add_argument("--history-len", required=True, type=int, choices=[5, 20, 50])
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-run-id", default="20260708_kuaisearch_b7_bge_dev_a02")
    parser.add_argument("--subset-request-ids", default="reports/b8_dev_subset_request_ids_seed20260708.txt")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache-dir", default="models/huggingface/llm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    standardized_dir = Path(args.standardized_dir)
    subset = ensure_subset_request_ids(
        standardized_dir=standardized_dir,
        split=args.split,
        output_path=args.subset_request_ids,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    run_id = args.run_id or f"20260708_kuaisearch_{args.variant}_qwen25_7b_h{args.history_len}_dev"
    config = args.config or f"configs/baselines/{args.variant}_llm_rerank.yaml"
    metadata = write_llm_rerank_scores(
        standardized_dir=standardized_dir,
        split=args.split,
        run_id=run_id,
        variant=args.variant,
        history_len=args.history_len,
        base_run_id=args.base_run_id,
        subset_request_ids_path=args.subset_request_ids,
        runs_dir=args.runs_dir,
        config_path=config,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.generation_batch_size,
    )
    result = {"metadata": metadata, "subset": subset}
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
