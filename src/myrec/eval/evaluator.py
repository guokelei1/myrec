"""Shared score-file evaluator."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.eval.metrics import ScoredCandidate, aggregate_request_metrics, request_metrics
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


def evaluate_run(
    run_id: str,
    split: str,
    candidate_manifest_path: str | Path,
    standardized_dir: str | Path | None = None,
    runs_dir: str | Path = "runs",
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
) -> dict[str, Any]:
    run_dir = Path(runs_dir) / run_id
    standardized_dir = Path(standardized_dir) if standardized_dir else Path(candidate_manifest_path).parent
    candidate_manifest_path = Path(candidate_manifest_path)
    qrels_path = standardized_dir / f"qrels_{split}.jsonl"
    scores_path = run_dir / "scores.jsonl"
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"
    per_request_path = run_dir / "per_request_metrics.jsonl"

    candidate_manifest_sha256 = sha256_file(candidate_manifest_path)
    _check_metadata(metadata_path, candidate_manifest_sha256)
    candidates = _load_candidate_manifest(candidate_manifest_path, split)
    qrels = _load_qrels(qrels_path)
    scores, method_id = _load_scores(scores_path)
    _assert_score_coverage(candidates, scores)

    per_request_rows = []
    for request_id in sorted(qrels):
        clicked, purchased = qrels[request_id]
        per_request_rows.append(
            request_metrics(
                request_id=request_id,
                scored_candidates=[
                    ScoredCandidate(item_id=item_id, score=scores[request_id][item_id])
                    for item_id in candidates[request_id]
                ],
                clicked_item_ids=clicked,
                purchased_item_ids=purchased,
            )
        )
    metrics = aggregate_request_metrics(per_request_rows)
    metrics.update(
        {
            "run_id": run_id,
            "method_id": method_id,
            "split": split,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "qrels_sha256": sha256_file(qrels_path),
            "scores_sha256": sha256_file(scores_path),
            "generated_by": "myrec.eval.evaluator",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_per_request(per_request_path, per_request_rows)
    write_json(metrics_path, metrics)
    if split == "dev":
        _append_dev_eval_log(dev_eval_log_path, metrics)
    return metrics


def _load_candidate_manifest(path: Path, split: str) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    result = {}
    for entry in manifest["entries"]:
        if entry["split"] != split:
            continue
        result[str(entry["request_id"])] = [str(item_id) for item_id in entry["candidate_item_ids"]]
    if not result:
        raise ValueError(f"no candidate manifest entries for split={split}")
    return result


def _load_qrels(path: Path) -> dict[str, tuple[set[str], set[str]]]:
    qrels = {}
    for row in iter_jsonl(path):
        qrels[str(row["request_id"])] = (
            set(str(item_id) for item_id in row.get("clicked", [])),
            set(str(item_id) for item_id in row.get("purchased", [])),
        )
    if not qrels:
        raise ValueError(f"empty qrels file: {path}")
    return qrels


def _load_scores(path: Path) -> tuple[dict[str, dict[str, float]], str]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    method_ids = set()
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        item_id = str(row["candidate_item_id"])
        if item_id in scores[request_id]:
            raise ValueError(f"duplicate score for request_id={request_id} item_id={item_id}")
        scores[request_id][item_id] = float(row["score"])
        method_ids.add(str(row.get("method_id", "unknown")))
    if not scores:
        raise ValueError(f"empty score file: {path}")
    method_id = method_ids.pop() if len(method_ids) == 1 else "mixed"
    return dict(scores), method_id


def _assert_score_coverage(
    candidates: dict[str, list[str]],
    scores: dict[str, dict[str, float]],
) -> None:
    candidate_request_ids = set(candidates)
    score_request_ids = set(scores)
    unknown_requests = score_request_ids - candidate_request_ids
    missing_requests = candidate_request_ids - score_request_ids
    if unknown_requests:
        raise ValueError(f"scores contain unknown request_ids: {sorted(unknown_requests)[:5]}")
    if missing_requests:
        raise ValueError(f"scores missing request_ids: {sorted(missing_requests)[:5]}")
    for request_id, candidate_ids in candidates.items():
        candidate_set = set(candidate_ids)
        score_set = set(scores[request_id])
        if candidate_set != score_set:
            raise ValueError(
                f"candidate mismatch for request_id={request_id}: "
                f"missing={sorted(candidate_set - score_set)[:5]} "
                f"extra={sorted(score_set - candidate_set)[:5]}"
            )


def _check_metadata(path: Path, candidate_manifest_sha256: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"run metadata missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    recorded = metadata.get("candidate_manifest_sha256")
    if recorded != candidate_manifest_sha256:
        raise ValueError(
            f"metadata candidate_manifest_sha256 mismatch: {recorded} != {candidate_manifest_sha256}"
        )


def _write_per_request(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_dev_eval_log(path: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": metrics["run_id"],
        "method_id": metrics["method_id"],
        "split": metrics["split"],
        "ndcg@10": metrics["ndcg@10"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
