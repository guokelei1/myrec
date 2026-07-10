#!/usr/bin/env python
"""Build a label-free 20-request B9 top-5 review sheet."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.metrics import ScoredCandidate, sort_candidates  # noqa: E402
from myrec.utils.jsonl import iter_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="data/standardized/kuaisearch/v0_lite/records_dev.jsonl")
    parser.add_argument("--zam-scores", required=True)
    parser.add_argument("--tem-scores", required=True)
    parser.add_argument("--output", default="reports/pps_b9_top5_review.md")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--requests", type=int, default=20)
    return parser.parse_args()


def _load_selected_scores(path: Path, selected: set[str]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {request_id: {} for request_id in selected}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in selected:
            scores[request_id][str(row["candidate_item_id"])] = float(row["score"])
    return scores


def main() -> int:
    args = parse_args()
    records = list(iter_jsonl(args.records))
    if args.requests > len(records):
        raise ValueError("requested more review cases than dev records")
    rng = random.Random(args.seed)
    selected_records = rng.sample(records, args.requests)
    selected = {str(row["request_id"]) for row in selected_records}
    method_scores = {
        "ZAM": _load_selected_scores(Path(args.zam_scores), selected),
        "TEM": _load_selected_scores(Path(args.tem_scores), selected),
    }
    lines = [
        "# B9 Top-5 Review",
        "",
        f"Seed: `{args.seed}`; requests: `{args.requests}`.",
        "",
        "Scope: label-free qualitative integrity review. This script reads records_dev and score files only; it does not read qrels.",
        "",
        "Review status: pending manual confirmation.",
        "",
    ]
    for index, record in enumerate(selected_records, start=1):
        request_id = str(record["request_id"])
        candidates = {str(item["item_id"]): item for item in record["candidates"]}
        history = record.get("history") or []
        lines.extend(
            [
                f"## {index}. {request_id}",
                "",
                f"Query: `{record.get('query', '')}`",
                "",
                "Recent history: "
                + (" | ".join(str(item.get("title") or item["item_id"]) for item in history[-3:]) or "(empty)"),
                "",
            ]
        )
        for method, scores_by_request in method_scores.items():
            scores = scores_by_request[request_id]
            if set(scores) != set(candidates):
                raise ValueError(f"{method} score coverage mismatch for {request_id}")
            ranked = sort_candidates(
                request_id,
                [ScoredCandidate(item_id, score) for item_id, score in scores.items()],
            )[:5]
            lines.append(f"{method} top-5:")
            lines.append("")
            for rank, scored in enumerate(ranked, start=1):
                item = candidates[scored.item_id]
                lines.append(
                    f"{rank}. `{scored.item_id}` score={scored.score:.6f} | "
                    f"{item.get('title', '')} | {item.get('brand', '')} | {' / '.join(item.get('cat') or [])}"
                )
            lines.append("")
    Path(args.output).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "requests": args.requests, "qrels_read": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
