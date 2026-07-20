"""Shared qrels-gated evaluator for the N8 joint composition bundle."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_evaluator import (
    _append_jsonl,
    _load_bundle_content_control_eligibility,
)
from myrec.mechanism.component_composition_scoring import composition_conditions
from myrec.mechanism.deep_dive_native_evaluator import (
    benjamini_hochberg,
    cluster_mean_inference,
)
from myrec.mechanism.fold_qrels import audit_fold_qrels
from myrec.mechanism.postblock_sweep_evaluator import (
    _load_fold_qrels,
    _ndcg,
    _strict_transfer_mask,
    _target_margins,
)
from myrec.mechanism.representation_evaluator import _audit_candidate_and_request_manifests
from myrec.mechanism.representation_probe import normalize_query, normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


METHODS = ("q2_recranker_generalqwen", "q3_tallrec_generalqwen")
ENDPOINTS = ("target_margin", "ndcg@10")
JOINT = "joint_attention_mlp_neutral_removal"
ATTENTION = "attention_neutral_removal"
MLP = "mlp_neutral_removal"
FULL = "baseline_full"
CONDITIONS = composition_conditions()


def evaluate_component_composition(
    standardized_dir: str | Path,
    qrels_split_dir: str | Path,
    model_bundles: Mapping[str, str | Path],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate exactly one completed Q2 and one completed Q3 bundle."""

    if set(model_bundles) != set(METHODS):
        raise ValueError("component-composition evaluator requires explicit Q2 and Q3")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"component-composition output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = list(iter_jsonl(standardized_dir / "records_dev.jsonl"))
    all_records = [sanitize_record_for_model(row) for row in raw_records]
    if len(all_records) != 8000:
        raise ValueError("component-composition evaluator requires 8000 requests")
    all_candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        all_records,
        raw_records,
    )
    records = [record for record in all_records if normalized_query_fold(record.query) == 1]
    candidates = {record.request_id: all_candidates[record.request_id] for record in records}
    qrels_path, qrels_manifest = audit_fold_qrels(standardized_dir, qrels_split_dir, 1)
    gains = _load_fold_qrels(qrels_path, candidates)
    strict = _strict_transfer_mask(records, candidates, gains)
    request_ids = [record.request_id for record in records]
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    results: dict[str, Any] = {}
    family_rows: list[dict[str, Any]] = []
    per_request: dict[str, np.ndarray] = {}
    input_identities: dict[str, Any] = {}
    for method_id in METHODS:
        root = Path(model_bundles[method_id])
        metadata = _read_json(root / "metadata.json")
        expected = {
            "analysis_stage": "transformer_component_composition_next_wave",
            "status": "completed",
            "result_eligible": True,
            "identity_passed": True,
            "complete_finite_score_coverage": True,
            "qrels_read": False,
            "source_test_opened": False,
            "method_id": method_id,
            "normalized_query_fold": 1,
            "score_conditions": list(CONDITIONS),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"component-composition metadata mismatch: {method_id}:{key}")
        for key in ("maximum_identity_delta", "maximum_full_baseline_delta", "maximum_null_baseline_delta"):
            if float(metadata.get(key, math.inf)) > 1.0e-5:
                raise ValueError(f"component-composition identity gate failed: {method_id}:{key}")
        scores_path = root / "scores.jsonl"
        if metadata.get("scores_sha256") != sha256_file(scores_path):
            raise ValueError(f"component-composition score hash mismatch: {method_id}")
        input_identities[method_id] = {
            "status": "completed_bundle",
            "path": str(root),
            "metadata_sha256": sha256_file(root / "metadata.json"),
            "scores_sha256": sha256_file(scores_path),
        }
        observed = audit_scalar_partial(scores_path, records, CONDITIONS)
        if observed["completed_requests"] != len(records):
            raise ValueError(f"component-composition request coverage mismatch: {method_id}")
        eligibility = _load_bundle_content_control_eligibility(
            metadata,
            records,
            label=f"component-composition {method_id}",
            identity_key="content_neutral_control",
        )
        condition_scores = {name: {} for name in CONDITIONS}
        for block_row in iter_jsonl(scores_path):
            request_id = str(block_row["request_id"])
            for name in CONDITIONS:
                condition_scores[name][request_id] = {
                    str(row["candidate_item_id"]): float(row["conditions"][name])
                    for row in block_row["rows"]
                }
        margins = {
            name: _target_margins(request_ids, candidates, gains, condition_scores[name])
            for name in CONDITIONS
        }
        ndcgs = {
            name: _ndcg(request_ids, candidates, gains, condition_scores[name])
            for name in CONDITIONS
        }
        interaction = {
            "target_margin": margins[JOINT] - margins[ATTENTION] - margins[MLP] + margins[FULL],
            "ndcg@10": ndcgs[JOINT] - ndcgs[ATTENTION] - ndcgs[MLP] + ndcgs[FULL],
        }
        results[method_id] = {
            "selected_block": metadata["selected_block"],
            "eligible_requests": int(sum(eligibility.values())),
            "interaction": {},
            "single_removal_effects": {},
        }
        for endpoint, values in interaction.items():
            mask = strict & np.asarray([eligibility[request_id] for request_id in request_ids]) & np.isfinite(values)
            inference = {
                **cluster_mean_inference(values[mask], clusters[mask]),
                "contrast": "joint - attention_single - mlp_single + full",
                "endpoint": endpoint,
                "surface": "strict_transfer",
                "eligibility": "frozen_content_neutral_eligible",
                "status": "completed",
            }
            results[method_id]["interaction"][endpoint] = inference
            family_rows.append(
                {
                    "method_id": method_id,
                    "endpoint": endpoint,
                    "contrast": "joint_minus_additive_single_removals",
                    "two_sided_p": float(inference["two_sided_p"]),
                }
            )
            per_request[f"{method_id}__interaction__{endpoint}"] = values
        for name in (ATTENTION, MLP, JOINT):
            results[method_id]["single_removal_effects"][name] = {}
            for endpoint, values in (
                ("target_margin", margins[name] - margins[FULL]),
                ("ndcg@10", ndcgs[name] - ndcgs[FULL]),
            ):
                mask = strict & np.asarray([eligibility[request_id] for request_id in request_ids]) & np.isfinite(values)
                results[method_id]["single_removal_effects"][name][endpoint] = {
                    **cluster_mean_inference(values[mask], clusters[mask]),
                    "surface": "strict_transfer",
                    "eligibility": "frozen_content_neutral_eligible",
                }
    q_values = benjamini_hochberg([row["two_sided_p"] for row in family_rows])
    for row, q_value in zip(family_rows, q_values):
        row["bh_q"] = q_value
        results[row["method_id"]]["interaction"][row["endpoint"]]["bh_q"] = q_value
    per_request_path = output_dir / "per_request_contrasts.npz"
    np.savez(
        per_request_path,
        **per_request,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        normalized_queries=clusters,
        strict_mask=strict,
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_component_composition_next_wave",
        "analysis_run_id": analysis_run_id,
        "methods": list(METHODS),
        "conditions": list(CONDITIONS),
        "normalized_query_fold": 1,
        "strict_transfer_requests": int(strict.sum()),
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "family_policy": {"units": len(family_rows), "multiple_testing": "benjamini_hochberg"},
        "family_rows": family_rows,
        "results": results,
        "input_bundles": input_identities,
        "qrels_read": True,
        "qrels_fold_opened": 1,
        "qrels_fold_sha256": sha256_file(qrels_path),
        "qrels_source_sha256": qrels_manifest["source_qrels_sha256"],
        "per_request_contrasts_path": str(per_request_path),
        "per_request_contrasts_sha256": sha256_file(per_request_path),
        "source_test_opened": False,
        "status": "completed",
        "claim_boundary": {
            "highest_authorized_claim": "joint_attention_mlp_state_interaction_candidate",
            "operator_necessity_authorized": False,
            "exclusive_origin_authorized": False,
        },
        "command": list(command or []),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_jsonl(
        Path(dev_eval_log_path),
        {
            "analysis_type": metrics["analysis_type"],
            "run_id": analysis_run_id,
            "method_ids": list(METHODS),
            "split": "dev_fold1",
            "qrels_sha256": metrics["qrels_fold_sha256"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
