#!/usr/bin/env python
"""Build a fresh label-isolated Amazon-C4 history stress-test dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.amazon_c4_standardize import build_standardized_amazon_c4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c4-csv", default="data/raw/amazon_c4/source/test.csv")
    parser.add_argument("--history-root", default="data/raw/amazon_c4/history")
    parser.add_argument(
        "--sampled-metadata",
        default="data/raw/amazon_c4/source/sampled_item_metadata_1M.jsonl",
    )
    parser.add_argument(
        "--reviews-metadata-dir", default="data/raw/amazon_reviews_2023/meta"
    )
    parser.add_argument(
        "--fts-index",
        default="artifacts/history_response_gap/amazon_c4/sampled_metadata_1m_fts.sqlite",
    )
    parser.add_argument(
        "--output-dir", default="data/standardized/amazon_c4/history_bm25_100_v1"
    )
    parser.add_argument(
        "--report", default="reports/history_response_gap_amazon_c4_data_audit.json"
    )
    parser.add_argument("--dataset-version", default="history_bm25_100_v1")
    parser.add_argument("--max-history-len", type=int, default=50)
    parser.add_argument("--bm25-top-k", type=int, default=100)
    parser.add_argument("--retrieval-workers", type=int, default=4)
    parser.add_argument("--metadata-workers", type=int, default=6)
    parser.add_argument("--candidate-cache")
    parser.add_argument("--candidate-cache-report")
    args = parser.parse_args()

    manifest = build_standardized_amazon_c4(
        c4_csv_path=args.c4_csv,
        history_root=args.history_root,
        sampled_metadata_path=args.sampled_metadata,
        reviews_metadata_dir=args.reviews_metadata_dir,
        fts_index_path=args.fts_index,
        output_dir=args.output_dir,
        report_path=args.report,
        dataset_version=args.dataset_version,
        max_history_len=args.max_history_len,
        bm25_top_k=args.bm25_top_k,
        retrieval_workers=args.retrieval_workers,
        metadata_workers=args.metadata_workers,
        candidate_cache_path=args.candidate_cache,
        candidate_cache_report_path=args.candidate_cache_report,
    )
    print(json.dumps({
        "status": manifest["overall_status"],
        "output_dir": manifest["output_dir"],
        "counts": manifest["outputs"]["counts_by_split"],
        "history_text_coverage": manifest["item_join"][
            "source_history_text_coverage"
        ],
    }, indent=2, sort_keys=True))
    return 0 if manifest["overall_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
