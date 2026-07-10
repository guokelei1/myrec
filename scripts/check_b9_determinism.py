#!/usr/bin/env python
"""Check exact B9 score reproducibility on the first N dev requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import iter_jsonl, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--repeat", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=1000)
    return parser.parse_args()


def _first_request_ids(path: Path, count: int) -> list[str]:
    request_ids: list[str] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id not in seen:
            seen.add(request_id)
            request_ids.append(request_id)
            if len(request_ids) == count:
                break
    return request_ids


def _load_subset(path: Path, request_ids: set[str]) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id not in request_ids:
            continue
        key = (request_id, str(row["candidate_item_id"]))
        if key in values:
            raise ValueError(f"duplicate score key in {path}: {key}")
        values[key] = float(row["score"])
    return values


def main() -> int:
    args = parse_args()
    original_path = Path(args.original)
    repeat_path = Path(args.repeat)
    ordered_requests = _first_request_ids(original_path, args.requests)
    request_ids = set(ordered_requests)
    original = _load_subset(original_path, request_ids)
    repeat = _load_subset(repeat_path, request_ids)
    if set(original) != set(repeat):
        raise ValueError(
            f"determinism key mismatch: original={len(original)} repeat={len(repeat)}"
        )
    unequal = 0
    max_abs_diff = 0.0
    examples = []
    for key, score in original.items():
        difference = abs(score - repeat[key])
        max_abs_diff = max(max_abs_diff, difference)
        if difference != 0.0:
            unequal += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "request_id": key[0],
                        "candidate_item_id": key[1],
                        "original": score,
                        "repeat": repeat[key],
                        "abs_diff": difference,
                    }
                )
    result = {
        "status": "passed" if unequal == 0 else "failed",
        "request_count": len(ordered_requests),
        "score_rows": len(original),
        "unequal_rows": unequal,
        "max_abs_diff": max_abs_diff,
        "tolerance": 0.0,
        "examples": examples,
        "original_path": str(original_path),
        "original_sha256": sha256_file(original_path),
        "repeat_path": str(repeat_path),
        "repeat_sha256": sha256_file(repeat_path),
    }
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if unequal == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

