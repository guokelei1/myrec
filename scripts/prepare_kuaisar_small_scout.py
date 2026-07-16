#!/usr/bin/env python
"""Build a label-isolated KuaiSAR user-input search scout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.kuaisar_scout import build_kuaisar_small_scout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/kuaisar_small")
    parser.add_argument(
        "--output-dir",
        default="data/standardized/kuaisar_small/small_user_input_scout10k_v1",
    )
    parser.add_argument(
        "--report-path",
        default="reports/history_response_gap_kuaisar_small_source_admission.json",
    )
    parser.add_argument("--dataset-version", default="small_user_input_scout10k_v1")
    parser.add_argument("--dataset-id", default="kuaisar_small")
    parser.add_argument("--release-name", default="KuaiSAR Small")
    parser.add_argument(
        "--archive-md5-expected", default="daea8cbf605db6bd5841740f0e4a12d9"
    )
    parser.add_argument("--archive-path")
    parser.add_argument("--max-requests", type=int, default=10_000)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--min-candidate-count", type=int, default=2)
    parser.add_argument("--max-candidate-count", type=int, default=100)
    parser.add_argument("--max-history-len", type=int, default=20)
    args = parser.parse_args()
    result = build_kuaisar_small_scout(
        args.raw_dir,
        args.output_dir,
        args.report_path,
        dataset_version=args.dataset_version,
        max_requests=args.max_requests,
        dev_fraction=args.dev_fraction,
        min_candidate_count=args.min_candidate_count,
        max_candidate_count=args.max_candidate_count,
        max_history_len=args.max_history_len,
        dataset_id=args.dataset_id,
        release_name=args.release_name,
        archive_md5_expected=args.archive_md5_expected,
        archive_path=args.archive_path,
    )
    print(
        json.dumps(
            {
                "dataset_version": result["dataset_version"],
                "admission_passed": result["admission_passed"],
                "admission_checks": result["admission_checks"],
                "selection": result["selection"],
                "item_join": result["item_join"],
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
