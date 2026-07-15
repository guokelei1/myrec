#!/usr/bin/env python
"""Train the independent PPS-adapted LLM-SRec baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.llm_srec_training import train_llm_srec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--teacher-store", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--backbone", default="models/huggingface/Qwen3-Reranker-0.6B")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-budget", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-train-requests", type=int)
    args = parser.parse_args()
    result = train_llm_srec(
        args.standardized_dir,
        args.teacher_store,
        args.run_id,
        args.output_model_dir,
        backbone=args.backbone,
        runs_dir=args.runs_dir,
        config_path=args.config,
        device=args.device,
        history_budget=args.history_budget,
        max_length=args.max_length,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        negatives=args.negatives,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_train_requests=args.max_train_requests,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

