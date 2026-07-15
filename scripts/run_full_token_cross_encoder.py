#!/usr/bin/env python
"""Run one ordinary full-token cross-encoder history condition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.full_token_cross_encoder import (
    write_full_token_cross_encoder_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--split", default="dev", choices=("dev",))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--history-condition", required=True, choices=("true", "null", "wrong"))
    parser.add_argument("--history-assignments", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config")
    parser.add_argument("--model-name", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--method-id", default="e_full_zero_shot")
    parser.add_argument("--tuning-class", default="zero_shot_instrumentation")
    parser.add_argument("--cache-folder", default="models/huggingface/cross_encoders")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--history-budget", type=int, default=10)
    parser.add_argument(
        "--truncation-strategy",
        choices=("longest_first", "only_second"),
        default="longest_first",
    )
    parser.add_argument(
        "--serialization-version",
        default="query_history_event_text_v1",
        choices=("query_history_event_text_v1", "query_only_text_v1"),
    )
    parser.add_argument("--pair-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network access and require a complete local model cache.",
    )
    parser.add_argument(
        "--request-aligned-batches",
        action="store_true",
        help="Flush inference at request boundaries for counterfactual stability.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = write_full_token_cross_encoder_scores(
        standardized_dir=args.standardized_dir,
        split=args.split,
        run_id=args.run_id,
        history_condition=args.history_condition,
        history_assignments_path=args.history_assignments,
        runs_dir=args.runs_dir,
        config_path=args.config,
        model_name=args.model_name,
        checkpoint_id=args.checkpoint_id,
        cache_folder=args.cache_folder,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_length=args.max_length,
        history_budget=args.history_budget,
        serialization_version=args.serialization_version,
        pair_chunk_size=args.pair_chunk_size,
        local_files_only=args.local_files_only,
        request_aligned_batches=args.request_aligned_batches,
        method_id=args.method_id,
        tuning_class=args.tuning_class,
        truncation_strategy=args.truncation_strategy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
