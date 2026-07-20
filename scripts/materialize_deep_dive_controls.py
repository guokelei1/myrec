#!/usr/bin/env python
"""Materialize qrels-free Transformer deep-dive control assignments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.deep_dive_assignments import (  # noqa: E402
    materialize_content_neutral_eligibility,
    materialize_fixed_candidate_sample,
    materialize_wrong_user_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-records", required=True)
    parser.add_argument("--donor-records")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q2-config")
    parser.add_argument("--q3-config")
    parser.add_argument(
        "--control",
        choices=("wrong-user", "content-neutral", "fixed-sample"),
        default="wrong-user",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.control == "wrong-user":
        if not args.donor_records:
            raise ValueError("wrong-user control requires --donor-records")
        report = materialize_wrong_user_mapping(
            args.target_records,
            args.donor_records,
            args.tokenizer_path,
            args.output_dir,
            expected_tokenizer_sha256=args.expected_tokenizer_sha256,
        )
    elif args.control == "content-neutral":
        if not args.q2_config or not args.q3_config:
            raise ValueError("content-neutral control requires --q2-config/--q3-config")
        report = materialize_content_neutral_eligibility(
            args.target_records,
            {
                "q2_recranker_generalqwen": args.q2_config,
                "q3_tallrec_generalqwen": args.q3_config,
            },
            args.tokenizer_path,
            args.output_dir,
            expected_tokenizer_sha256=args.expected_tokenizer_sha256,
        )
    else:
        report = materialize_fixed_candidate_sample(
            args.target_records,
            args.output_dir,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
