#!/usr/bin/env python
"""Build the stable-hash, label-isolated JDsearch exploratory scout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.jdsearch_standardize import build_jdsearch_scout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/jdsearch/kaggle_mirror")
    parser.add_argument("--official-repository-dir", default="data/raw/jdsearch/repository")
    parser.add_argument("--output-dir", default="data/standardized/jdsearch/hash_scout10k_v3")
    parser.add_argument("--report-path", default="reports/history_response_gap_jdsearch_source_admission.json")
    parser.add_argument("--dataset-version", default="hash_scout10k_v3")
    parser.add_argument("--max-requests", type=int, default=10_000)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--max-history-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    result = build_jdsearch_scout(
        args.raw_dir,
        args.output_dir,
        args.report_path,
        official_repository_dir=args.official_repository_dir,
        dataset_version=args.dataset_version,
        max_requests=args.max_requests,
        dev_fraction=args.dev_fraction,
        max_history_len=args.max_history_len,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "admission_passed": result["admission_passed"],
                "admission_checks": result["admission_checks"],
                "selection": result["selection"],
                "metadata": result["metadata"],
                "counts": result["outputs"]["counts"],
                "report_path": args.report_path,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
