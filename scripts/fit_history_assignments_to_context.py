#!/usr/bin/env python
"""Fit label-free assigned histories into an only-second context budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.full_token_coverage import fit_history_assignments_to_context_budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--cache-folder", default="models/huggingface/cross_encoders")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--history-budget", type=int, required=True)
    parser.add_argument("--min-candidate-tokens", type=int, default=1)
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = fit_history_assignments_to_context_budget(
        args.records,
        args.assignments,
        args.output,
        args.report,
        model_name=args.model_name,
        cache_folder=args.cache_folder,
        max_length=args.max_length,
        history_budget=args.history_budget,
        min_candidate_tokens=args.min_candidate_tokens,
        local_files_only=not args.allow_network,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
