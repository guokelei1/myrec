#!/usr/bin/env python
"""Score one InstructRec T3 history condition on a frozen candidate slate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.instructrec import write_instructrec_scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--split", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--history-condition", choices=("true", "null", "wrong"), required=True)
    parser.add_argument("--history-assignments", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config")
    parser.add_argument("--cache-folder", default="models/huggingface/llm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--history-budget", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    result = write_instructrec_scores(
        args.standardized_dir,
        args.split,
        args.run_id,
        args.history_condition,
        args.history_assignments,
        args.model_dir,
        runs_dir=args.runs_dir,
        config_path=args.config,
        cache_folder=args.cache_folder,
        device=args.device,
        dtype=args.dtype,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        history_budget=args.history_budget,
        batch_size=args.batch_size,
        local_files_only=not args.allow_network,
        method_id=args.method_id,
        max_requests=args.max_requests,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
