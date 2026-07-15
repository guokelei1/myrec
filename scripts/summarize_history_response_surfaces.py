#!/usr/bin/env python
"""Summarize one evaluated history-response bundle by request surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.history_response_surfaces import summarize_history_response_surfaces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-request", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--surfaces-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--utility-epsilon", required=True, type=float)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    result = summarize_history_response_surfaces(
        args.per_request,
        args.records,
        args.surfaces_dir,
        args.output,
        utility_epsilon=args.utility_epsilon,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
