#!/usr/bin/env python3
"""Audit new/old Q2 post-block equivalence before formal score reuse."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.postblock_reuse_audit import (
    audit_q2_postblock_reuse_equivalence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--smoke-bundle", required=True)
    parser.add_argument("--identity-dir", required=True)
    parser.add_argument("--same-dir", required=True)
    parser.add_argument("--cross-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--block", type=int, choices=(13, 27), required=True)
    args = parser.parse_args()
    result = audit_q2_postblock_reuse_equivalence(
        args.standardized_dir,
        args.smoke_bundle,
        args.identity_dir,
        args.same_dir,
        args.cross_dir,
        args.output_dir,
        args.analysis_run_id,
        block=args.block,
        command=sys.argv,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
