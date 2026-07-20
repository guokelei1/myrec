#!/usr/bin/env python3
"""Print outcome-independent Transformer deep-dive breadth and depth status."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.deep_dive_overview import build_deep_dive_overview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = build_deep_dive_overview(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
