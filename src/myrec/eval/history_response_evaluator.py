"""Shared evaluator for true/null/matched-wrong score bundles."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.eval.history_response import (
    ResponseCandidate,
    aggregate_history_response,
    request_history_response,
)
from myrec.eval.target_aware_surfaces import materialize_target_aware_surfaces
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json, write_jsonl

COUNTERFACTUAL_IDENTITY_KEYS = (
    "candidate_manifest_sha256",
    "checkpoint_id",
    "dataset_id",
    "dataset_version",
    "request_manifest_sha256",
    "scoring_signature",
    "split",
)


def evaluate_history_response_runs(
    analysis_run_id: str,
    true_run_id: str,
    null_run_id: str,
    split: str,
    candidate_manifest_path: str | Path,
    standardized_dir: str | Path,
    activity_epsilon: float,
    utility_epsilon: float,
    label_mode: str = "click",
    wrong_run_id: str | None = None,
    runs_dir: str | Path = "runs",
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
) -> dict[str, Any]:
    """Evaluate one counterfactual bundle through the qrels-only boundary."""

    if split == "test":
        raise ValueError("test history-response evaluation is locked")
    if split not in {"dev", "confirmation"}:
        raise ValueError("history-response evaluation supports dev or confirmation only")
    if label_mode not in {"click", "purchase", "graded"}:
        raise ValueError(f"unsupported label_mode: {label_mode}")

    runs_dir = Path(runs_dir)
    analysis_dir = runs_dir / analysis_run_id
    if analysis_dir.exists():
        raise FileExistsError(f"analysis run already exists: {analysis_dir}")
    standardized_dir = Path(standardized_dir)
    candidate_manifest_path = Path(candidate_manifest_path)
    candidate_sha256 = sha256_file(candidate_manifest_path)
    request_manifest_path = standardized_dir / "request_manifest.json"
    request_sha256 = sha256_file(request_manifest_path)
    candidates = _load_candidates(candidate_manifest_path, split)

    bundle = {
        "true": _load_run(
            runs_dir, true_run_id, "true", candidate_sha256, request_sha256
        ),
        "null": _load_run(
            runs_dir, null_run_id, "null", candidate_sha256, request_sha256
        ),
    }
    if wrong_run_id is not None:
        bundle["wrong"] = _load_run(
            runs_dir, wrong_run_id, "wrong", candidate_sha256, request_sha256
        )
    _assert_counterfactual_identity(bundle)
    if bundle["true"]["metadata"]["split"] != split:
        raise ValueError(
            f"score-bundle split={bundle['true']['metadata']['split']} expected={split}"
        )
    for condition in bundle.values():
        _assert_score_coverage(candidates, condition["scores"])

    qrels_path = standardized_dir / f"qrels_{split}.jsonl"
    gains = _load_gains(qrels_path, label_mode)
    if set(gains) != set(candidates):
        raise ValueError(
            "qrels request coverage differs from candidate manifest: "
            f"missing={sorted(set(candidates) - set(gains))[:5]} "
            f"extra={sorted(set(gains) - set(candidates))[:5]}"
        )

    per_request = []
    for request_id in sorted(candidates):
        item_ids = candidates[request_id]
        unknown_labels = set(gains[request_id]) - set(item_ids)
        if unknown_labels:
            raise ValueError(
                f"qrels contain non-candidate items for request_id={request_id}: "
                f"{sorted(unknown_labels)[:5]}"
            )
        per_request.append(
            request_history_response(
                request_id,
                [
                    ResponseCandidate(
                        item_id=item_id,
                        true_score=bundle["true"]["scores"][request_id][item_id],
                        null_score=bundle["null"]["scores"][request_id][item_id],
                        wrong_score=(
                            bundle["wrong"]["scores"][request_id][item_id]
                            if "wrong" in bundle
                            else None
                        ),
                        gain=float(gains[request_id].get(item_id, 0.0)),
                    )
                    for item_id in item_ids
                ],
                activity_epsilon=activity_epsilon,
            )
        )

    metrics = aggregate_history_response(per_request, utility_epsilon=utility_epsilon)
    generated_at = datetime.now(timezone.utc).isoformat()
    metrics.update(
        {
            "activity_epsilon": activity_epsilon,
            "analysis_run_id": analysis_run_id,
            "candidate_manifest_sha256": candidate_sha256,
            "generated_at": generated_at,
            "label_mode": label_mode,
            "null_run_id": null_run_id,
            "qrels_sha256": sha256_file(qrels_path),
            "request_manifest_sha256": request_sha256,
            "split": split,
            "true_run_id": true_run_id,
            "utility_epsilon": utility_epsilon,
            "wrong_run_id": wrong_run_id,
        }
    )

    analysis_dir.mkdir(parents=True, exist_ok=False)
    records_path = standardized_dir / f"records_{split}.jsonl"
    target_surfaces = materialize_target_aware_surfaces(
        records_path,
        candidates,
        gains,
        analysis_dir / "target_aware_surfaces",
        label_mode=label_mode,
        candidate_manifest_path=candidate_manifest_path,
        qrels_path=qrels_path,
    )
    metrics["target_aware_surface_counts"] = {
        name: values["requests"] for name, values in target_surfaces["files"].items()
    }
    write_jsonl(analysis_dir / "per_request_history_response.jsonl", per_request)
    write_json(analysis_dir / "metrics.json", metrics)
    metadata = {
        "analysis_type": "history_response_direction_gap",
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": candidate_sha256,
        "counterfactual_identity": {
            key: bundle["true"]["metadata"][key] for key in COUNTERFACTUAL_IDENTITY_KEYS
        },
        "generated_at": generated_at,
        "input_runs": {
            condition: values["run_id"] for condition, values in bundle.items()
        },
        "history_assignment_sha256": {
            condition: values["metadata"]["history_assignment_sha256"]
            for condition, values in bundle.items()
        },
        "label_mode": label_mode,
        "qrels_path": str(qrels_path),
        "qrels_read": True,
        "request_manifest_path": str(request_manifest_path),
        "request_manifest_sha256": request_sha256,
        "target_aware_surfaces_manifest": str(
            analysis_dir / "target_aware_surfaces" / "manifest.json"
        ),
        "target_aware_surfaces_manifest_sha256": sha256_file(
            analysis_dir / "target_aware_surfaces" / "manifest.json"
        ),
        "split": split,
    }
    write_json(analysis_dir / "metadata.json", metadata)
    if split in {"dev", "confirmation"}:
        _append_dev_log(dev_eval_log_path, metrics)
    return metrics


def _load_run(
    runs_dir: Path,
    run_id: str,
    expected_condition: str,
    candidate_sha256: str,
    request_sha256: str,
) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    metadata_path = run_dir / "metadata.json"
    scores_path = run_dir / "scores.jsonl"
    if not metadata_path.exists() or not scores_path.exists():
        raise FileNotFoundError(f"incomplete score run: {run_dir}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    missing = [
        key
        for key in (
            *COUNTERFACTUAL_IDENTITY_KEYS,
            "history_assignment_sha256",
            "history_condition",
        )
        if key not in metadata
    ]
    if missing:
        raise ValueError(f"run {run_id} is missing counterfactual metadata: {missing}")
    if metadata["history_condition"] != expected_condition:
        raise ValueError(
            f"run {run_id} history_condition={metadata['history_condition']} "
            f"expected={expected_condition}"
        )
    if metadata["candidate_manifest_sha256"] != candidate_sha256:
        raise ValueError(f"run {run_id} candidate manifest hash mismatch")
    if metadata["request_manifest_sha256"] != request_sha256:
        raise ValueError(f"run {run_id} request manifest hash mismatch")
    return {
        "metadata": metadata,
        "run_id": run_id,
        "scores": _load_scores(scores_path),
        "scores_sha256": sha256_file(scores_path),
    }


def _assert_counterfactual_identity(bundle: dict[str, dict[str, Any]]) -> None:
    reference = bundle["true"]["metadata"]
    for condition, values in bundle.items():
        metadata = values["metadata"]
        for key in COUNTERFACTUAL_IDENTITY_KEYS:
            if _canonical(metadata[key]) != _canonical(reference[key]):
                raise ValueError(
                    f"counterfactual identity mismatch for {condition}.{key}: "
                    f"{metadata[key]!r} != {reference[key]!r}"
                )


def _load_candidates(path: Path, split: str) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    result: dict[str, list[str]] = {}
    for entry in manifest.get("entries", []):
        if entry.get("split") != split:
            continue
        request_id = str(entry["request_id"])
        item_ids = [str(item_id) for item_id in entry["candidate_item_ids"]]
        if request_id in result:
            raise ValueError(f"duplicate candidate manifest request_id={request_id}")
        if len(item_ids) < 2 or len(set(item_ids)) != len(item_ids):
            raise ValueError(f"invalid candidate slate for request_id={request_id}")
        result[request_id] = item_ids
    if not result:
        raise ValueError(f"no candidate entries for split={split}")
    return result


def _load_scores(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        item_id = str(row["candidate_item_id"])
        if item_id in result[request_id]:
            raise ValueError(f"duplicate score for request_id={request_id} item_id={item_id}")
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite score for request_id={request_id} item_id={item_id}")
        result[request_id][item_id] = score
    if not result:
        raise ValueError(f"empty score file: {path}")
    return dict(result)


def _assert_score_coverage(
    candidates: dict[str, list[str]], scores: dict[str, dict[str, float]]
) -> None:
    if set(candidates) != set(scores):
        raise ValueError("score request coverage differs from candidate manifest")
    for request_id, item_ids in candidates.items():
        if set(item_ids) != set(scores[request_id]):
            raise ValueError(f"score candidate coverage mismatch for request_id={request_id}")


def _load_gains(path: Path, label_mode: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate qrels request_id={request_id}")
        if label_mode == "click":
            result[request_id] = {str(item_id): 1.0 for item_id in row.get("clicked", [])}
        elif label_mode == "purchase":
            result[request_id] = {str(item_id): 1.0 for item_id in row.get("purchased", [])}
        else:
            relevance = row.get("relevance", {})
            if not isinstance(relevance, dict):
                raise ValueError("graded qrels relevance must be an item-to-gain object")
            result[request_id] = {str(item_id): float(gain) for item_id, gain in relevance.items()}
    if not result:
        raise ValueError(f"empty qrels file: {path}")
    return result


def _append_dev_log(path: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "analysis_type": "history_response_direction_gap",
        "activity_epsilon": metrics["activity_epsilon"],
        "label_mode": metrics["label_mode"],
        "method_id": "shared_history_response_evaluator",
        "ndcg@10": metrics["mean_true_ndcg@10"],
        "null_run_id": metrics["null_run_id"],
        "run_id": metrics["analysis_run_id"],
        "split": metrics["split"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "true_run_id": metrics["true_run_id"],
        "utility_epsilon": metrics["utility_epsilon"],
        "wrong_run_id": metrics["wrong_run_id"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
