#!/usr/bin/env python3
"""Report live physical-GPU ownership for project deep-dive workers."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.gpu_ownership_audit import audit_gpu_ownership


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--expected-gpu-count", type=int, default=4)
    args = parser.parse_args()
    result = audit_gpu_ownership(
        proc_root=args.proc_root,
        expected_gpu_count=args.expected_gpu_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

