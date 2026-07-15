"""Shared evaluator for label-oracle history-response interventions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.eval.history_response_evaluator import (
    COUNTERFACTUAL_IDENTITY_KEYS,
    _assert_counterfactual_identity,
    _assert_score_coverage,
    _load_candidates,
    _load_gains,
    _load_run,
)
from myrec.eval.response_direction_intervention import (
    DirectionInterventionCandidate,
    aggregate_direction_interventions,
    request_direction_intervention,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json, write_jsonl


def evaluate_response_direction_intervention_runs(
    analysis_run_id: str,
    true_run_id: str,
    null_run_id: str,
    split: str,
    candidate_manifest_path: str | Path,
    standardized_dir: str | Path,
    *,
    label_mode: str = "click",
    random_permutations: int = 128,
    seed: int = 20260714,
    runs_dir: str | Path = "runs",
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
) -> dict[str, Any]:
    if split == "test":
        raise ValueError("test direction-intervention evaluation is locked")
    if split not in {"dev", "confirmation"}:
        raise ValueError("direction intervention supports dev or confirmation only")
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
    _assert_counterfactual_identity(bundle)
    for condition in bundle.values():
        _assert_score_coverage(candidates, condition["scores"])

    qrels_path = standardized_dir / f"qrels_{split}.jsonl"
    gains = _load_gains(qrels_path, label_mode)
    if set(gains) != set(candidates):
        raise ValueError("qrels request coverage differs from candidate manifest")
    per_request = []
    for request_id in sorted(candidates):
        item_ids = candidates[request_id]
        per_request.append(
            request_direction_intervention(
                request_id,
                [
                    DirectionInterventionCandidate(
                        item_id=item_id,
                        true_score=bundle["true"]["scores"][request_id][item_id],
                        null_score=bundle["null"]["scores"][request_id][item_id],
                        gain=float(gains[request_id].get(item_id, 0.0)),
                    )
                    for item_id in item_ids
                ],
                random_permutations=random_permutations,
                seed=seed,
            )
        )

    metrics = aggregate_direction_interventions(per_request)
    generated_at = datetime.now(timezone.utc).isoformat()
    metrics.update(
        {
            "analysis_run_id": analysis_run_id,
            "candidate_manifest_sha256": candidate_sha256,
            "generated_at": generated_at,
            "label_mode": label_mode,
            "label_oracle_diagnostic": True,
            "null_run_id": null_run_id,
            "qrels_sha256": sha256_file(qrels_path),
            "random_permutations": random_permutations,
            "request_manifest_sha256": request_sha256,
            "seed": seed,
            "split": split,
            "true_run_id": true_run_id,
        }
    )
    analysis_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(analysis_dir / "per_request_direction_intervention.jsonl", per_request)
    write_json(analysis_dir / "metrics.json", metrics)
    write_json(
        analysis_dir / "metadata.json",
        {
            "analysis_type": "response_direction_intervention",
            "candidate_manifest_path": str(candidate_manifest_path),
            "candidate_manifest_sha256": candidate_sha256,
            "counterfactual_identity": {
                key: bundle["true"]["metadata"][key]
                for key in COUNTERFACTUAL_IDENTITY_KEYS
            },
            "generated_at": generated_at,
            "input_runs": {"true": true_run_id, "null": null_run_id},
            "intervention": (
                "Preserve each request's exact true-minus-null score-delta multiset; "
                "reassign it monotonically by observed gain or by random permutation."
            ),
            "label_mode": label_mode,
            "label_oracle_diagnostic": True,
            "qrels_path": str(qrels_path),
            "qrels_read": True,
            "request_manifest_path": str(request_manifest_path),
            "request_manifest_sha256": request_sha256,
            "split": split,
        },
    )
    if split == "dev":
        _append_dev_log(dev_eval_log_path, metrics)
    return metrics


def _append_dev_log(path: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "analysis_type": "response_direction_intervention",
        "label_mode": metrics["label_mode"],
        "label_oracle_diagnostic": True,
        "method_id": "shared_response_direction_intervention_evaluator",
        "ndcg@10": metrics["mean_actual_ndcg@10"],
        "null_run_id": metrics["null_run_id"],
        "run_id": metrics["analysis_run_id"],
        "split": metrics["split"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "true_run_id": metrics["true_run_id"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
