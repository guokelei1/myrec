#!/usr/bin/env python3
"""Print the requirement-by-requirement Transformer completion audit."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.comprehensive_completion_audit import (
    audit_comprehensive_completion,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = audit_comprehensive_completion(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] not in {"pending", "completed"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

