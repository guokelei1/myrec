#!/usr/bin/env python3
"""Read-only audit of Transformer deep-dive completion and data boundaries."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.deep_dive_closeout_audit import audit_deep_dive_closeout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = audit_deep_dive_closeout(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] == "failed":
        raise SystemExit(1)
    if args.strict and result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
