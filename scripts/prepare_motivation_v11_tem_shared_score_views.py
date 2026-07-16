#!/usr/bin/env python
"""Build TEM null/wrong score views sharing the trained true catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.prodsearch_adapter import prepare_prodsearch_shared_history_view


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--history-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--history-condition", choices=("null", "wrong"), required=True)
    args = parser.parse_args()
    result = prepare_prodsearch_shared_history_view(
        catalog_root=args.catalog_root,
        history_root=args.history_root,
        output_root=args.output_root,
        history_condition=args.history_condition,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
