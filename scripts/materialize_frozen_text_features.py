#!/usr/bin/env python
"""Materialize frozen multilingual text features for representative baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.frozen_text_features import materialize_frozen_text_features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model", default="BAAI/bge-reranker-v2-m3"
    )
    parser.add_argument(
        "--cache-folder", default="models/huggingface/cross_encoders"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    result = materialize_frozen_text_features(
        args.records,
        args.output_dir,
        model_name_or_path=args.model,
        cache_folder=args.cache_folder,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        local_files_only=not args.allow_network,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
