#!/usr/bin/env python3
"""Freeze fold-specific dev qrels before adaptive D2 layer selection."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.fold_qrels import materialize_fold_qrels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = materialize_fold_qrels(args.standardized_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
