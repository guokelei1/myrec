#!/usr/bin/env python3
"""Print the qrels-blind D2 scientific-bundle progress census."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.deep_dive_progress import audit_deep_dive_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = audit_deep_dive_progress(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
