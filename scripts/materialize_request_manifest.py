#!/usr/bin/env python
"""Create a request identity manifest from label-free standardized records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.request_manifest import materialize_request_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()
    root = Path(args.standardized_dir)
    result = materialize_request_manifest(
        [(split, root / f"records_{split}.jsonl") for split in args.splits],
        root / "request_manifest.json",
        dataset_version=args.dataset_version,
    )
    print(json.dumps({"requests": len(result["entries"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
