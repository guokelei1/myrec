#!/usr/bin/env python3
"""Run B1 BM25 diagnostics for the KuaiSearch C2 failure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.b1_diagnostics import run_b1_diagnostics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--raw-dir", default="data/raw/kuaisearch")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--artifacts-dir", default="artifacts/baselines")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--random-catalog-size", type=int, default=100_000)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--tokenizer-mode", default="cjk_2_3gram")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--exact-match-boost", type=float, default=10.0)
    parser.add_argument("--char-coverage-boost", type=float, default=0.0)
    parser.add_argument("--b1-run-id", default="20260708_kuaisearch_b1_bm25_globalidf_exact10_dev")
    parser.add_argument("--b0a-run-id", default="20260708_kuaisearch_b0a_popularity_dev")
    parser.add_argument("--output-json", default="reports/pps_c2_b1_diagnostics.json")
    parser.add_argument("--output-md", default="reports/pps_c2_b1_diagnostics.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_b1_diagnostics(
        standardized_dir=args.standardized_dir,
        raw_dir=args.raw_dir,
        runs_dir=args.runs_dir,
        artifacts_dir=args.artifacts_dir,
        output_json_path=args.output_json,
        output_markdown_path=args.output_md,
        split=args.split,
        seed=args.seed,
        max_requests=args.max_requests or None,
        random_catalog_size=args.random_catalog_size,
        bootstrap_samples=args.bootstrap_samples,
        tokenizer_mode=args.tokenizer_mode,
        k1=args.k1,
        b=args.b,
        exact_match_boost=args.exact_match_boost,
        char_coverage_boost=args.char_coverage_boost,
        b1_run_id=args.b1_run_id,
        b0a_run_id=args.b0a_run_id,
    )
    print(report["interpretation"]["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
