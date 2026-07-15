#!/usr/bin/env python
"""Audit HSTU/LLM-SRec label-free sequence inputs on standardized records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.representative_sequence_adapter import (
    audit_standardized_sequence_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-budget", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    args = parser.parse_args()
    result = audit_standardized_sequence_inputs(
        args.standardized_dir,
        args.output,
        history_budget=args.history_budget,
        splits=args.splits,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
