#!/usr/bin/env python
"""Validate method-visible standardized records without opening qrels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.contracts import audit_standardized_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "dev", "confirmation", "test"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(audit_standardized_file(args.records, args.split), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
