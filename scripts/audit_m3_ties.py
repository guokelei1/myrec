#!/usr/bin/env python
"""Audit M3 oracle ties without reading labels or rerunning evaluation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import iter_jsonl, write_json  # noqa: E402


METHOD_ORDER = ("query_b2z", "history_b0b", "static_b7_bge")
ABS_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-choices",
        default="runs/20260708_kuaisearch_m3_oracle_dev/oracle_choices.jsonl",
    )
    parser.add_argument("--output", default="reports/pps_m3_tie_aware_audit.json")
    return parser.parse_args()


def audit_ties(path: str | Path) -> dict:
    path = Path(path)
    patterns: Counter[str] = Counter()
    pairwise_strict: Counter[str] = Counter()
    total = 0

    for row in iter_jsonl(path):
        values = {method: float(row["values"][method]) for method in METHOD_ORDER}
        best = max(values.values())
        winners = tuple(
            method
            for method in METHOD_ORDER
            if math.isclose(values[method], best, rel_tol=0.0, abs_tol=ABS_TOLERANCE)
        )
        expected_choice = winners[0]
        if row["chosen_method"] != expected_choice:
            raise ValueError(
                f"M3 tie-rule mismatch for {row['request_id']}: "
                f"expected={expected_choice} observed={row['chosen_method']}"
            )
        prefix = "unique" if len(winners) == 1 else "tie"
        patterns[f"{prefix}__{'__'.join(winners)}"] += 1

        for left in METHOD_ORDER:
            for right in METHOD_ORDER:
                if left == right:
                    continue
                if values[left] < values[right] - ABS_TOLERANCE:
                    pairwise_strict[f"{left}_below_{right}"] += 1
        if any(
            values["static_b7_bge"] < values[other] - ABS_TOLERANCE
            for other in ("query_b2z", "history_b0b")
        ):
            pairwise_strict["static_b7_bge_below_either_alternative"] += 1
        total += 1

    if total == 0:
        raise ValueError(f"empty oracle choices: {path}")

    tie_count = sum(count for pattern, count in patterns.items() if pattern.startswith("tie__"))
    return {
        "report": "pps_m3_tie_aware_audit",
        "status": "passed",
        "analysis_type": "read-only post-hoc wording audit; no new evaluation",
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "qrels_read": False,
        "records_test_read": False,
        "method_order_and_tie_rule": list(METHOD_ORDER),
        "absolute_tolerance": ABS_TOLERANCE,
        "total_requests": total,
        "tie_requests": tie_count,
        "tie_rate": tie_count / total,
        "winner_patterns": {
            pattern: {"count": count, "rate": count / total}
            for pattern, count in sorted(patterns.items())
        },
        "pairwise_strict": {
            relation: {"count": count, "rate": count / total}
            for relation, count in sorted(pairwise_strict.items())
        },
        "interpretation": (
            "The M3 choice distribution is tie-broken assignment, not strict preference. "
            "Unique-winner and pairwise-strict rates are exact descriptive facts, but the "
            "Random-channel construct audit must pass before they support exploitable-"
            "heterogeneity claims."
        ),
    }


def main() -> int:
    args = parse_args()
    result = audit_ties(args.oracle_choices)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
