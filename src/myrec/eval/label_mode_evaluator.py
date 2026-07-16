"""Shared evaluator for one frozen score run under a registered gain endpoint."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.eval.history_response import gain_ndcg_at_k
from myrec.eval.history_response_evaluator import (
    _assert_score_coverage,
    _load_candidates,
    _load_gains,
    _load_scores,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json, write_jsonl


def evaluate_label_mode_score_run(
    analysis_run_id: str,
    score_run_id: str,
    split: str,
    candidate_manifest_path: str | Path,
    standardized_dir: str | Path,
    *,
    label_mode: str,
    runs_dir: str | Path = "runs",
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
) -> dict[str, Any]:
    """Evaluate one score artifact without modifying its immutable run directory."""

    if split == "test":
        raise ValueError("test label-mode evaluation is locked")
    if split not in {"dev", "internal", "confirmation"}:
        raise ValueError(
            "label-mode evaluation supports dev, internal, or confirmation only"
        )
    if label_mode not in {"click", "purchase", "graded"}:
        raise ValueError(f"unsupported label_mode: {label_mode}")

    runs_dir = Path(runs_dir)
    score_dir = runs_dir / score_run_id
    analysis_dir = runs_dir / analysis_run_id
    if analysis_dir.exists():
        raise FileExistsError(f"analysis run already exists: {analysis_dir}")
    metadata_path = score_dir / "metadata.json"
    scores_path = score_dir / "scores.jsonl"
    if not metadata_path.exists() or not scores_path.exists():
        raise FileNotFoundError(f"incomplete score run: {score_dir}")

    standardized_dir = Path(standardized_dir)
    candidate_manifest_path = Path(candidate_manifest_path)
    candidate_sha256 = sha256_file(candidate_manifest_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        score_metadata = json.load(handle)
    if score_metadata.get("candidate_manifest_sha256") != candidate_sha256:
        raise ValueError("score run candidate manifest hash mismatch")
    if score_metadata.get("split") != split:
        raise ValueError(
            f"score run split={score_metadata.get('split')} expected={split}"
        )

    candidates = _load_candidates(candidate_manifest_path, split)
    scores = _load_scores(scores_path)
    _assert_score_coverage(candidates, scores)
    qrels_path = standardized_dir / f"qrels_{split}.jsonl"
    gains = _load_gains(qrels_path, label_mode)
    if set(gains) != set(candidates):
        raise ValueError("qrels request coverage differs from candidate manifest")

    rows = []
    for request_id in sorted(candidates):
        item_ids = candidates[request_id]
        unknown_gains = set(gains[request_id]) - set(item_ids)
        if unknown_gains:
            raise ValueError(
                f"qrels contain non-candidate items for request_id={request_id}: "
                f"{sorted(unknown_gains)[:5]}"
            )
        candidate_gains = [float(gains[request_id].get(item_id, 0.0)) for item_id in item_ids]
        positive_count = sum(gain > 0 for gain in candidate_gains)
        rows.append(
            {
                "request_id": request_id,
                "ndcg@10": gain_ndcg_at_k(
                    request_id,
                    item_ids,
                    [scores[request_id][item_id] for item_id in item_ids],
                    candidate_gains,
                    10,
                ),
                "positive_candidates": positive_count,
                "positive_eligible": positive_count > 0,
            }
        )

    positive_rows = [row for row in rows if row["positive_eligible"]]
    all_mean = _mean(float(row["ndcg@10"]) for row in rows)
    positive_mean = _mean(float(row["ndcg@10"]) for row in positive_rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "analysis_run_id": analysis_run_id,
        "analysis_type": "label_mode_score_evaluation",
        "candidate_manifest_sha256": candidate_sha256,
        "generated_at": generated_at,
        "label_mode": label_mode,
        "ndcg@10": all_mean,
        "ndcg@10_all_requests": all_mean,
        "ndcg@10_positive_requests": positive_mean,
        "no_positive_request_convention": "ndcg@10 equals zero",
        "num_positive_eligible_requests": len(positive_rows),
        "num_requests": len(rows),
        "positive_eligible_rate": len(positive_rows) / len(rows),
        "qrels_sha256": sha256_file(qrels_path),
        "score_run_id": score_run_id,
        "scores_sha256": sha256_file(scores_path),
        "split": split,
    }
    analysis_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(analysis_dir / "per_request_metrics.jsonl", rows)
    write_json(analysis_dir / "metrics.json", metrics)
    write_json(
        analysis_dir / "metadata.json",
        {
            "analysis_type": "label_mode_score_evaluation",
            "candidate_manifest_path": str(candidate_manifest_path),
            "candidate_manifest_sha256": candidate_sha256,
            "generated_at": generated_at,
            "label_mode": label_mode,
            "qrels_path": str(qrels_path),
            "qrels_read": True,
            "score_metadata_path": str(metadata_path),
            "score_metadata_sha256": sha256_file(metadata_path),
            "score_run_id": score_run_id,
            "scores_path": str(scores_path),
            "scores_sha256": sha256_file(scores_path),
            "split": split,
        },
    )
    if split in {"dev", "confirmation"}:
        _append_dev_log(dev_eval_log_path, metrics)
    return metrics


def _append_dev_log(path: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "analysis_type": metrics["analysis_type"],
        "label_mode": metrics["label_mode"],
        "method_id": "shared_label_mode_score_evaluator",
        "ndcg@10": metrics["ndcg@10"],
        "run_id": metrics["analysis_run_id"],
        "score_run_id": metrics["score_run_id"],
        "split": metrics["split"],
        "timestamp": metrics["generated_at"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
