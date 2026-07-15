#!/usr/bin/env python
"""Audit label-free token-budget coverage for full-token inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.full_token_coverage import (
    audit_cross_encoder_preprocess_coverage,
    audit_full_token_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-name", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--cache-folder", default="models/huggingface/cross_encoders")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--history-budget", type=int, default=10)
    parser.add_argument(
        "--truncation-strategy",
        choices=("longest_first", "only_second"),
        default="longest_first",
    )
    parser.add_argument(
        "--max-candidates-per-request",
        type=int,
        help="Optional label-free prefix sample; context preservation is request-level.",
    )
    parser.add_argument("--history-assignments")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("tokenizer_pair", "cross_encoder_preprocess"),
        default="tokenizer_pair",
    )
    parser.add_argument("--audit-max-length", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    args = parser.parse_args()
    if args.mode == "tokenizer_pair":
        result = audit_full_token_coverage(
            args.records,
            args.report,
            model_name=args.model_name,
            cache_folder=args.cache_folder,
            max_length=args.max_length,
            history_budget=args.history_budget,
            truncation_strategy=args.truncation_strategy,
            max_candidates_per_request=args.max_candidates_per_request,
            history_assignments_path=args.history_assignments,
            local_files_only=not args.allow_network,
        )
    else:
        result = audit_cross_encoder_preprocess_coverage(
            args.records,
            args.report,
            model_name=args.model_name,
            max_length=args.max_length,
            history_budget=args.history_budget,
            audit_max_length=args.audit_max_length,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            local_files_only=not args.allow_network,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
