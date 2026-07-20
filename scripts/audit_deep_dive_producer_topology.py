#!/usr/bin/env python3
"""Audit producer coverage for all formal and supplemental evidence."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.deep_dive_producer_topology import (
    audit_deep_dive_producer_topology,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = audit_deep_dive_producer_topology(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
