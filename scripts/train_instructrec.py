#!/usr/bin/env python
"""Train an InstructRec T3 candidate-likelihood model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.instructrec import train_instructrec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--input-mode", choices=("qc", "full"), required=True)
    parser.add_argument("--base-model-name", default="google/flan-t5-xl")
    parser.add_argument("--cache-folder", default="models/huggingface/llm")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--history-budget", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--internal-dev-fraction", type=float, default=0.0)
    args = parser.parse_args()
    result = train_instructrec(
        args.standardized_dir,
        args.run_id,
        args.output_model_dir,
        input_mode=args.input_mode,
        base_model_name=args.base_model_name,
        cache_folder=args.cache_folder,
        runs_dir=args.runs_dir,
        config_path=args.config,
        local_files_only=not args.allow_network,
        device=args.device,
        dtype=args.dtype,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        history_budget=args.history_budget,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        max_train_examples=args.max_train_examples,
        internal_dev_fraction=args.internal_dev_fraction,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
