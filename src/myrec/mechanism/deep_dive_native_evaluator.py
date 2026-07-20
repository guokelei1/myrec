"""Independent qrels-gated evaluator for Q3 all-native position gates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.eval.history_response import gain_ndcg_at_k
from myrec.eval.target_aware_surfaces import build_target_aware_surface_memberships
from myrec.mechanism.deep_dive_native_patch import (
    GATE_BLOCKS,
    NATIVE_TERMS,
    SCORE_CONDITIONS,
)
from myrec.mechanism.patch_evaluator import _target_margins
from myrec.mechanism.representation_evaluator import (
    STRICT_TRANSFER_SURFACE,
    _audit_candidate_and_request_manifests,
    _load_dev_qrels,
)
from myrec.mechanism.representation_probe import (
    load_m2_probe_manifest,
    normalize_query,
    normalized_query_fold,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20_260_715


@dataclass(frozen=True)
class NativeBundle:
    root: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, dict[str, Any]]]


def evaluate_q3_native_gates(
    standardized_dir: str | Path,
    block_dirs: Mapping[int, str | Path],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    if set(map(int, block_dirs)) != set(GATE_BLOCKS):
        raise ValueError("Q3 native evaluator requires blocks 13 and 27")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Q3 native evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_dev.jsonl"
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("Q3 native evaluator requires all 8000 dev requests")
    candidates = _audit_candidate_and_request_manifests(
        candidate_manifest_path,
        request_manifest_path,
        records,
        raw_records,
    )
    bundles = {
        block: _audit_native_bundle(block_dirs[block], records, block)
        for block in GATE_BLOCKS
    }
    reference = bundles[13].metadata
    invariants = (
        "method_id",
        "checkpoint_id",
        "config_sha256",
        "records_sha256",
        "candidate_manifest_sha256",
        "request_manifest_sha256",
        "dataset_manifest_sha256",
        "deep_dive_manifest_sha256",
    )
    for block, bundle in bundles.items():
        for key in invariants:
            if bundle.metadata.get(key) != reference.get(key):
                raise ValueError(f"Q3 native bundle invariant differs at block {block}: {key}")
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d2_q3_native_gate_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "both_registered_blocks_present": True,
            "all_requests_and_candidates_complete_finite": True,
            "all_four_logprob_terms_present": True,
            "full_and_null_identity_at_most_1e-5": True,
            "shared_prompt_yes_no_path_exact": True,
            "candidate_and_request_manifests_reconstructed": True,
        },
        "invariants": {key: reference.get(key) for key in invariants},
        "bundles": {
            str(block): {
                "path": str(bundle.root),
                "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
                "scores_sha256": sha256_file(bundle.root / "scores.jsonl"),
                "maximum_identity_delta": bundle.metadata["maximum_identity_delta"],
                "shared_prompt_path_max_abs_delta": bundle.metadata[
                    "shared_prompt_path_max_abs_delta"
                ],
            }
            for block, bundle in bundles.items()
        },
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    frozen = load_m2_probe_manifest()["frozen_inputs"]
    qrels_sha256 = sha256_file(qrels_path)
    if qrels_sha256 != frozen["qrels_dev_sha256"]:
        raise ValueError("Q3 native evaluator qrels hash mismatch")
    gains = _load_dev_qrels(qrels_path, candidates)
    memberships = build_target_aware_surface_memberships(records_path, candidates, gains)
    request_ids = [row.request_id for row in records]
    clusters = np.asarray([normalize_query(row.query) for row in records], dtype=np.str_)
    folds = np.asarray([normalized_query_fold(row.query) for row in records], dtype=np.int8)
    strict = np.asarray(
        [request_id in memberships[STRICT_TRANSFER_SURFACE] for request_id in request_ids],
        dtype=bool,
    )

    per_request: dict[str, np.ndarray] = {}
    block_results: dict[str, Any] = {}
    gate_p_values: list[float] = []
    scope_p_values: list[float] = []
    for block in GATE_BLOCKS:
        condition_scores = {
            condition: _condition_scores(bundles[block], condition)
            for condition in SCORE_CONDITIONS
        }
        margins = {
            condition: _target_margins(
                request_ids,
                candidates,
                gains,
                condition_scores[condition],
            )
            for condition in SCORE_CONDITIONS
        }
        ndcg = {
            condition: _ndcg_values(
                request_ids,
                candidates,
                gains,
                condition_scores[condition],
            )
            for condition in SCORE_CONDITIONS
        }
        contrasts = {
            "all_native_minus_null_margin": (
                margins["same_all_native_positions"] - margins["baseline_null"]
            ),
            "first_only_minus_null_margin": (
                margins["same_first_position_only"] - margins["baseline_null"]
            ),
            "all_native_minus_first_only_margin": (
                margins["same_all_native_positions"]
                - margins["same_first_position_only"]
            ),
            "all_native_minus_null_ndcg@10": (
                ndcg["same_all_native_positions"] - ndcg["baseline_null"]
            ),
            "first_only_minus_null_ndcg@10": (
                ndcg["same_first_position_only"] - ndcg["baseline_null"]
            ),
            "all_native_minus_first_only_ndcg@10": (
                ndcg["same_all_native_positions"]
                - ndcg["same_first_position_only"]
            ),
        }
        summaries: dict[str, Any] = {}
        for name, values in contrasts.items():
            rows = []
            for fold_name, fold_mask in (
                ("all", np.ones(len(records), dtype=bool)),
                ("0", folds == 0),
                ("1", folds == 1),
            ):
                mask = strict & fold_mask & np.isfinite(values)
                rows.append(
                    {
                        "surface": STRICT_TRANSFER_SURFACE,
                        "normalized_query_fold": fold_name,
                        **cluster_mean_inference(values[mask], clusters[mask]),
                    }
                )
            summaries[name] = rows
        gate_row = next(
            row
            for row in summaries["all_native_minus_null_margin"]
            if row["normalized_query_fold"] == "all"
        )
        scope_row = next(
            row
            for row in summaries["all_native_minus_first_only_margin"]
            if row["normalized_query_fold"] == "all"
        )
        gate_p_values.append(float(gate_row["two_sided_p"]))
        scope_p_values.append(float(scope_row["two_sided_p"]))
        for name, values in contrasts.items():
            per_request[f"block_{block}__{name}"] = values
        block_results[str(block)] = {
            "block_zero_based": block,
            "contrasts": summaries,
        }
    gate_q = benjamini_hochberg(gate_p_values)
    scope_q = benjamini_hochberg(scope_p_values)
    for index, block in enumerate(GATE_BLOCKS):
        result = block_results[str(block)]
        gate_rows = result["contrasts"]["all_native_minus_null_margin"]
        scope_rows = result["contrasts"]["all_native_minus_first_only_margin"]
        next(row for row in gate_rows if row["normalized_query_fold"] == "all")[
            "bh_q"
        ] = gate_q[index]
        next(row for row in scope_rows if row["normalized_query_fold"] == "all")[
            "bh_q"
        ] = scope_q[index]
        expected_sign = 1 if block == 13 else -1
        all_row = next(row for row in gate_rows if row["normalized_query_fold"] == "all")
        fold_rows = [
            next(row for row in gate_rows if row["normalized_query_fold"] == value)
            for value in ("0", "1")
        ]
        result["scientific_gate"] = {
            "expected_sign": "positive" if expected_sign > 0 else "negative",
            "point_expected_sign": expected_sign * all_row["mean"] > 0,
            "both_folds_expected_sign": all(
                expected_sign * row["mean"] > 0 for row in fold_rows
            ),
            "bh_q_below_0.05": gate_q[index] < 0.05,
        }
        result["scientific_gate"]["passed"] = all(
            result["scientific_gate"][key]
            for key in (
                "point_expected_sign",
                "both_folds_expected_sign",
                "bh_q_below_0.05",
            )
        )
    per_request_path = output_dir / "per_request_contrasts.npz"
    np.savez(
        per_request_path,
        **per_request,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        normalized_queries=clusters,
        folds=folds,
        strict_mask=strict,
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d2_q3_native_position_gate",
        "analysis_run_id": analysis_run_id,
        "method_id": reference["method_id"],
        "checkpoint_id": reference["checkpoint_id"],
        "primary_endpoint": "strict_transfer_target_margin_patch_minus_null",
        "native_terms": list(NATIVE_TERMS),
        "bootstrap": {
            "cluster": "normalized_query",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "two_sided_p": "min(1,2*min((1+#draw<=0)/(B+1),(1+#draw>=0)/(B+1)))",
        },
        "multiple_testing": {
            "gate_family": "blocks_13_27_all_native_minus_null_margin",
            "gate_family_size": 2,
            "scope_family": "blocks_13_27_all_native_minus_first_only_margin",
            "scope_family_size": 2,
            "method": "benjamini_hochberg",
        },
        "strict_transfer_requests": int(strict.sum()),
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": True,
        "qrels_opened_only_after_score_integrity": True,
        "qrels_dev_sha256": qrels_sha256,
        "block_results": block_results,
        "q3_sweep_admitted": all(
            result["scientific_gate"]["passed"]
            for result in block_results.values()
        ),
        "per_request_contrasts_path": str(per_request_path),
        "per_request_contrasts_sha256": sha256_file(per_request_path),
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    _append_jsonl(
        Path(dev_eval_log_path),
        {
            "analysis_type": metrics["analysis_type"],
            "run_id": analysis_run_id,
            "method_id": metrics["method_id"],
            "checkpoint_id": metrics["checkpoint_id"],
            "split": "dev",
            "qrels_sha256": qrels_sha256,
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def cluster_mean_inference(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=np.str_)
    if values.shape != clusters.shape or not values.size:
        raise ValueError("cluster mean inference requires aligned nonempty arrays")
    if not np.isfinite(values).all():
        raise ValueError("cluster mean inference values are non-finite")
    unique, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(unique), size=len(unique))
        draws[index] = sums[selected].sum() / counts[selected].sum()
    lower, upper = np.percentile(draws, [2.5, 97.5])
    lower_tail = (1 + int(np.count_nonzero(draws <= 0))) / (samples + 1)
    upper_tail = (1 + int(np.count_nonzero(draws >= 0))) / (samples + 1)
    return {
        "requests": int(values.size),
        "normalized_query_clusters": int(len(unique)),
        "mean": float(values.mean()),
        "ci95": [float(lower), float(upper)],
        "two_sided_p": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "bootstrap_samples": samples,
    }


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not values.size or np.any((values < 0) | (values > 1)):
        raise ValueError("BH p-values are invalid")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return [float(value) for value in result]


def _audit_native_bundle(
    bundle_dir: str | Path, records: Sequence[Any], block: int
) -> NativeBundle:
    root = Path(bundle_dir)
    metadata = _read_json(root / "metadata.json")
    expected = {
        "analysis_stage": "transformer_deep_dive_d2_q3_native_position_gate",
        "block_zero_based": block,
        "status": "completed",
        "result_eligible": True,
        "identity_passed": True,
        "complete_finite_score_coverage": True,
        "qrels_read": False,
        "source_test_opened": False,
        "native_terms": list(NATIVE_TERMS),
        "score_conditions": list(SCORE_CONDITIONS),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Q3 native bundle metadata mismatch: {key}")
    if float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("Q3 native bundle identity exceeds tolerance")
    if float(metadata.get("shared_prompt_path_max_abs_delta", math.inf)) != 0.0:
        raise ValueError("Q3 native shared prompt differs across Yes/No paths")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("Q3 native scores changed after metadata")
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for row in iter_jsonl(scores_path):
        request_id = str(row.get("request_id") or "")
        item_id = str(row.get("candidate_item_id") or "")
        request = scores.setdefault(request_id, {})
        if not request_id or not item_id or item_id in request:
            raise ValueError("Q3 native score identity is empty/duplicate")
        conditions = row.get("conditions")
        if len(conditions) != len(SCORE_CONDITIONS) or set(conditions) != set(
            SCORE_CONDITIONS
        ):
            raise ValueError("Q3 native score condition set drifted")
        for name in SCORE_CONDITIONS:
            value = conditions[name]
            if not math.isfinite(float(value["score"])):
                raise ValueError("Q3 native score is non-finite")
            if len(value["terms"]) != 4 or not all(
                math.isfinite(float(term)) for term in value["terms"]
            ):
                raise ValueError("Q3 native term is non-finite")
        request[item_id] = conditions
    if list(scores) != [row.request_id for row in records]:
        raise ValueError("Q3 native request identity/order coverage mismatch")
    for record in records:
        if list(scores[record.request_id]) != [
            str(candidate["item_id"]) for candidate in record.candidates
        ]:
            raise ValueError("Q3 native candidate identity/order coverage mismatch")
    return NativeBundle(root=root, metadata=metadata, scores=scores)


def _condition_scores(
    bundle: NativeBundle, condition: str
) -> dict[str, dict[str, float]]:
    return {
        request_id: {
            item_id: float(values[condition]["score"])
            for item_id, values in request.items()
        }
        for request_id, request in bundle.scores.items()
    }


def _ndcg_values(
    request_ids: Sequence[str],
    candidates: Mapping[str, Sequence[str]],
    gains: Mapping[str, Mapping[str, float]],
    scores: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    result = []
    for request_id in request_ids:
        item_ids = list(candidates[request_id])
        result.append(
            gain_ndcg_at_k(
                request_id,
                item_ids,
                [scores[request_id][item_id] for item_id in item_ids],
                [float(gains[request_id].get(item_id, 0.0)) for item_id in item_ids],
                10,
            )
        )
    return np.asarray(result, dtype=np.float64)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
