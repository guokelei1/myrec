#!/usr/bin/env python
"""Train one ordinary Lite QC or full-token cross-encoder control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.full_token_training import train_pairwise_cross_encoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--input-mode", required=True, choices=("qc", "full"))
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config")
    parser.add_argument("--base-model-name", default="BAAI/bge-reranker-base")
    parser.add_argument("--cache-folder", default="models/huggingface/cross_encoders")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--history-budget", type=int, default=10)
    parser.add_argument(
        "--truncation-strategy",
        choices=("longest_first", "only_second"),
        default="longest_first",
    )
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--history-dropout-probability",
        type=float,
        default=0.0,
        help="Deterministic request-level train-only history dropout for ordinary FULL.",
    )
    parser.add_argument(
        "--objective",
        default="pairwise_logistic_softplus",
        choices=(
            "pairwise_logistic_softplus",
            "pointwise_binary_cross_entropy",
        ),
    )
    args = parser.parse_args()
    result = train_pairwise_cross_encoder(
        args.standardized_dir,
        args.run_id,
        args.output_model_dir,
        input_mode=args.input_mode,
        runs_dir=args.runs_dir,
        config_path=args.config,
        base_model_name=args.base_model_name,
        cache_folder=args.cache_folder,
        local_files_only=not args.allow_network,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        history_budget=args.history_budget,
        negatives_per_positive=args.negatives_per_positive,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        objective=args.objective,
        truncation_strategy=args.truncation_strategy,
        history_dropout_probability=args.history_dropout_probability,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
