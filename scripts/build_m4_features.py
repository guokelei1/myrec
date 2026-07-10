#!/usr/bin/env python
"""Build M4 request-level features without reading dev qrels or metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.m4_features import (  # noqa: E402
    build_feature_frame,
    build_train_stats,
    sample_train_subset,
)
from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--output-dir", default="artifacts/m4")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--train-subset-size", type=int, default=20000)
    parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--cache-folder", default="models/huggingface/sentence_transformers")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    standardized_dir = Path(args.standardized_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = standardized_dir / "records_train.jsonl"
    dev_path = standardized_dir / "records_dev.jsonl"

    stats = build_train_stats(train_path)
    subset = sample_train_subset(
        records_train_path=train_path,
        sample_size=args.train_subset_size,
        seed=args.seed,
    )
    dev_frame, dev_meta = build_feature_frame(
        records_path=dev_path,
        stats=stats,
        split="dev",
        semantic_model_name=args.model_name,
        cache_folder=args.cache_folder,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    train_frame, train_meta = build_feature_frame(
        records_path=train_path,
        stats=stats,
        split="train_sub",
        request_ids=set(subset.request_ids),
        semantic_model_name=args.model_name,
        cache_folder=args.cache_folder,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )

    dev_path_out = output_dir / "m4_features_dev.parquet"
    train_path_out = output_dir / "m4_features_train_sub.parquet"
    subset_path = output_dir / "m4_train_subset_request_ids.txt"
    metadata_path = output_dir / "m4_feature_manifest.json"
    dev_frame.to_parquet(dev_path_out, index=False)
    train_frame.to_parquet(train_path_out, index=False)
    with subset_path.open("w", encoding="utf-8") as handle:
        for request_id in subset.request_ids:
            handle.write(request_id + "\n")

    metadata = {
        "artifacts": {
            "dev_features": str(dev_path_out),
            "dev_features_sha256": sha256_file(dev_path_out),
            "train_sub_features": str(train_path_out),
            "train_sub_features_sha256": sha256_file(train_path_out),
            "train_subset_request_ids": str(subset_path),
            "train_subset_request_ids_sha256": sha256_file(subset_path),
        },
        "feature_columns": [col for col in dev_frame.columns if col not in {"request_id", "split"}],
        "mechanical_checks": {
            "imports_evaluator": False,
            "reads_qrels": False,
            "reads_per_request_metrics": False,
            "label_derived_columns": [],
        },
        "seed": args.seed,
        "semantic_model": {
            "model_name": args.model_name,
            "cache_folder": args.cache_folder,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
        },
        "source_files": {
            "records_dev": str(dev_path),
            "records_dev_sha256": sha256_file(dev_path),
            "records_train": str(train_path),
            "records_train_sha256": sha256_file(train_path),
        },
        "train_stats": {
            "train_request_count": stats.train_request_count,
            "unique_queries": len(stats.query_freq),
            "queries_with_click_entropy": len(stats.query_click_entropy),
            "query_term_df_terms": len(stats.query_term_df),
        },
        "train_subset": {
            "eligible_count": subset.eligible_count,
            "filters": subset.filters,
            "requested_sample_size": args.train_subset_size,
            "sample_size": subset.sample_size,
            "seed": subset.seed,
        },
        "outputs": {
            "dev": dev_meta,
            "train_sub": train_meta,
        },
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
