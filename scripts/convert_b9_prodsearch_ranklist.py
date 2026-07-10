#!/usr/bin/env python
"""Convert a ProdSearch ranklist to exact frozen-candidate PPS scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.prodsearch_adapter import convert_prodsearch_ranklist  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranklist", required=True)
    parser.add_argument("--request-map", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument(
        "--candidate-manifest",
        default="data/standardized/kuaisearch/v0_lite/candidate_manifest.json",
    )
    parser.add_argument("--split", default="dev")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = convert_prodsearch_ranklist(
        args.ranklist,
        args.request_map,
        args.output_scores,
        method_id=args.method_id,
        candidate_manifest_path=args.candidate_manifest,
        split=args.split,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

