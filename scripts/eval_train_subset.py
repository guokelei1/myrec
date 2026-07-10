#!/usr/bin/env python
"""Evaluate a train-request subset using the shared metric functions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.metrics import ScoredCandidate, aggregate_request_metrics, request_metrics  # noqa: E402
from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import iter_jsonl, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--standardized-dir", default="data/standardized/kuaisearch/v0_lite")
    parser.add_argument("--subset-request-ids", required=True)
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args()


def evaluate_train_subset(
    run_id: str,
    standardized_dir: str | Path,
    subset_request_ids_path: str | Path,
    runs_dir: str | Path = "runs",
) -> dict[str, Any]:
    standardized_dir = Path(standardized_dir)
    run_dir = Path(runs_dir) / run_id
    subset_ids = _load_subset_ids(subset_request_ids_path)
    records = _load_train_records(standardized_dir / "records_train.jsonl", subset_ids)
    scores, method_id = _load_scores(run_dir / "scores.jsonl")
    if set(records) != set(scores):
        raise ValueError(f"record/score request mismatch: records={len(records)} scores={len(scores)}")
    rows = []
    for request_id in sorted(records):
        record = records[request_id]
        candidate_ids = [str(candidate["item_id"]) for candidate in record["candidates"]]
        if set(candidate_ids) != set(scores[request_id]):
            raise ValueError(f"candidate mismatch for train subset request {request_id}")
        clicked = {
            str(candidate["item_id"])
            for candidate in record["candidates"]
            if int(candidate.get("clicked", 0) or 0) > 0
        }
        purchased = {
            str(candidate["item_id"])
            for candidate in record["candidates"]
            if int(candidate.get("purchased", 0) or 0) > 0
        }
        rows.append(
            request_metrics(
                request_id=request_id,
                scored_candidates=[
                    ScoredCandidate(item_id=item_id, score=scores[request_id][item_id])
                    for item_id in candidate_ids
                ],
                clicked_item_ids=clicked,
                purchased_item_ids=purchased,
            )
        )
    metrics = aggregate_request_metrics(rows)
    metrics.update(
        {
            "run_id": run_id,
            "method_id": method_id,
            "split": "train_subset",
            "generated_by": "scripts.eval_train_subset",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "records_train_sha256": sha256_file(standardized_dir / "records_train.jsonl"),
            "scores_sha256": sha256_file(run_dir / "scores.jsonl"),
            "subset_request_ids_path": str(subset_request_ids_path),
            "subset_request_ids_sha256": sha256_file(subset_request_ids_path),
            "qrels_read": False,
        }
    )
    with (run_dir / "per_request_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def _load_subset_ids(path: str | Path) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _load_train_records(path: Path, subset_ids: set[str]) -> dict[str, dict[str, Any]]:
    records = {}
    for record in iter_jsonl(path):
        request_id = str(record["request_id"])
        if request_id in subset_ids:
            records[request_id] = record
    if set(records) != subset_ids:
        raise ValueError(f"subset ids missing from train records: {len(subset_ids - set(records))}")
    return records


def _load_scores(path: Path) -> tuple[dict[str, dict[str, float]], str]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    method_ids = set()
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        item_id = str(row["candidate_item_id"])
        scores[request_id][item_id] = float(row["score"])
        method_ids.add(str(row.get("method_id", "unknown")))
    method_id = method_ids.pop() if len(method_ids) == 1 else "mixed"
    return dict(scores), method_id


def main() -> int:
    args = parse_args()
    result = evaluate_train_subset(
        run_id=args.run_id,
        standardized_dir=args.standardized_dir,
        subset_request_ids_path=args.subset_request_ids,
        runs_dir=args.runs_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
