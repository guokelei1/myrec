"""Independent qrels-gated evaluator for M2 activation-patch score bundles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord, sanitize_record_for_model
from myrec.eval.target_aware_surfaces import build_target_aware_surface_memberships
from myrec.mechanism.patch_scorer import (
    M2_PATCH_BLOCKS,
    PATCH_KINDS,
    _cross_request_mapping,
)
from myrec.mechanism.representation_probe import (
    load_m2_probe_manifest,
    normalize_query,
    normalized_query_fold,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


BOOTSTRAP_SEED = 20_260_715
BOOTSTRAP_SAMPLES = 5000
STRICT_TRANSFER_SURFACE = "target_nonrepeat_no_candidate_overlap"
IDENTITY_MAX_ABS_SCORE_DELTA_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class AuditedScoreBundle:
    root: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, float]]
    scores_sha256: str


def evaluate_m2_patches(
    standardized_dir: str | Path,
    baseline_dirs: Mapping[str, str | Path],
    patch_dirs: Mapping[str, Mapping[int, str | Path]],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit full/null and all six patch bundles before opening qrels_dev."""

    if set(baseline_dirs) != {"full", "null"}:
        raise ValueError("M2 patch evaluator requires full and null baselines")
    if set(patch_dirs) != set(PATCH_KINDS):
        raise ValueError("M2 patch evaluator requires all three registered patch kinds")
    if any(set(values) != set(M2_PATCH_BLOCKS) for values in patch_dirs.values()):
        raise ValueError("every M2 patch kind requires blocks 13 and 27")
    if not analysis_run_id or "/" in analysis_run_id or "\\" in analysis_run_id:
        raise ValueError("invalid M2 patch analysis_run_id")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"M2 patch analysis output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_manifest = load_m2_probe_manifest()
    implementation_identity = patch_evaluator_implementation_identity()
    frozen = probe_manifest["frozen_inputs"]
    records_path = standardized_dir / "records_dev.jsonl"
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    dataset_manifest_path = standardized_dir / "manifest.json"
    for path in (
        records_path,
        qrels_path,
        candidate_manifest_path,
        request_manifest_path,
        dataset_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    # Do not hash qrels here: the pre-qrels report must exist first.
    hashes = {
        "records_dev_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
    }
    for key, value in hashes.items():
        if value != frozen[key]:
            raise ValueError(f"frozen M2 patch evaluator input mismatch: {key}")
    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("M2 patch evaluator requires all 8000 dev requests")
    candidates = {
        row.request_id: [str(value["item_id"]) for value in row.candidates]
        for row in records
    }
    _audit_external_manifests(
        candidate_manifest_path,
        request_manifest_path,
        records,
        raw_records,
    )

    full = _audit_score_bundle(
        baseline_dirs["full"], records, expected_condition="full", patch=None
    )
    null = _audit_score_bundle(
        baseline_dirs["null"], records, expected_condition="null", patch=None
    )
    patches: dict[tuple[str, int], AuditedScoreBundle] = {}
    for kind in PATCH_KINDS:
        for block in M2_PATCH_BLOCKS:
            patches[(kind, block)] = _audit_score_bundle(
                patch_dirs[kind][block],
                records,
                expected_condition=None,
                patch=(kind, block),
            )
    if full.metadata.get("candidate_manifest_sha256") != hashes[
        "candidate_manifest_sha256"
    ]:
        raise ValueError("full baseline candidate manifest hash mismatch")
    if full.metadata.get("request_manifest_sha256") != hashes[
        "request_manifest_sha256"
    ]:
        raise ValueError("full baseline request manifest hash mismatch")
    frozen_model = frozen["models"].get(full.metadata.get("method_id"))
    if not isinstance(frozen_model, dict):
        raise ValueError("M2 patch baseline method is absent from frozen manifest")
    if full.metadata.get("config_sha256") != frozen_model.get("config_sha256"):
        raise ValueError("M2 patch baseline config differs from frozen manifest")
    if full.metadata.get("checkpoint_id") != frozen_model.get("checkpoint_id"):
        raise ValueError("M2 patch baseline checkpoint differs from frozen manifest")
    invariant_keys = (
        "method_id",
        "checkpoint_id",
        "config_sha256",
        "dataset_id",
        "dataset_version",
        "split",
        "candidate_manifest_sha256",
        "request_manifest_sha256",
    )
    for name, bundle in {
        "null": null,
        **{f"{kind}:{block}": value for (kind, block), value in patches.items()},
    }.items():
        for key in invariant_keys:
            if bundle.metadata.get(key) != full.metadata.get(key):
                raise ValueError(f"M2 patch score invariant differs for {name}: {key}")
    for (kind, block), bundle in patches.items():
        identity = bundle.metadata.get("mechanism_probe_manifest", {})
        if identity.get("sha256") != probe_manifest["sha256"]:
            raise ValueError(f"M2 patch bundle manifest identity mismatch: {kind}/{block}")
        if bundle.metadata.get("records_sha256") != hashes["records_dev_sha256"]:
            raise ValueError(f"M2 patch records hash mismatch: {kind}/{block}")
        expected_mapping_sha256 = sha256_text(
            json.dumps(
                _cross_request_mapping(records),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if bundle.metadata.get("cross_request_mapping_sha256") != expected_mapping_sha256:
            raise ValueError(f"M2 cross-request mapping hash mismatch: {kind}/{block}")
    identity_control_audit: dict[str, Any] = {}
    for block in M2_PATCH_BLOCKS:
        identity = patches[("full_to_full_identity", block)]
        deltas = [
            abs(
                identity.scores[request_id][item_id]
                - full.scores[request_id][item_id]
            )
            for request_id, item_ids in candidates.items()
            for item_id in item_ids
        ]
        maximum = max(deltas, default=math.inf)
        row = {
            "patch_block_zero_based": block,
            "score_rows": len(deltas),
            "mean_abs_score_delta": float(np.mean(deltas)),
            "max_abs_score_delta": float(maximum),
            "max_abs_score_delta_tolerance": IDENTITY_MAX_ABS_SCORE_DELTA_TOLERANCE,
            "passed": maximum <= IDENTITY_MAX_ABS_SCORE_DELTA_TOLERANCE,
            "qrels_read": False,
        }
        identity_control_audit[f"block_{block}"] = row
        if not row["passed"]:
            raise ValueError(
                "M2 identity patch differs from frozen full scores beyond numerical "
                f"tolerance at block={block}: {maximum}"
            )

    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "m2_patch_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "qrels_read": False,
        "status": "passed",
        "checks": {
            "full_and_null_baselines_complete": True,
            "all_six_registered_patch_bundles_complete": True,
            "all_candidate_scores_finite_and_unique": True,
            "candidate_and_request_hashes_exact": True,
            "method_checkpoint_config_dataset_invariants_equal": True,
            "patch_manifest_provenance_exact": True,
            "identity_patch_within_numerical_tolerance": True,
        },
        "invariants": {key: full.metadata.get(key) for key in invariant_keys},
        "input_hashes": hashes,
        "baseline_bundles": {
            "full": _score_identity(full),
            "null": _score_identity(null),
        },
        "patch_bundles": {
            f"{kind}:block_{block}": _score_identity(bundle)
            for (kind, block), bundle in patches.items()
        },
        "identity_control_audit": identity_control_audit,
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json_atomic(pre_qrels_path, pre_qrels)

    # First qrels access occurs after all score integrity checks are durable.
    qrels_sha256 = sha256_file(qrels_path)
    if qrels_sha256 != frozen["qrels_dev_sha256"]:
        raise ValueError("frozen M2 qrels_dev hash mismatch")
    gains = _load_dev_qrels(qrels_path, candidates)
    memberships = build_target_aware_surface_memberships(
        records_path, candidates, gains
    )
    request_ids = [row.request_id for row in records]
    clusters = np.asarray([normalize_query(row.query) for row in records], dtype=np.str_)
    folds = np.asarray([normalized_query_fold(row.query) for row in records], dtype=np.int8)
    strict = np.asarray(
        [request_id in memberships[STRICT_TRANSFER_SURFACE] for request_id in request_ids],
        dtype=bool,
    )
    full_margin = _target_margins(request_ids, candidates, gains, full.scores)
    null_margin = _target_margins(request_ids, candidates, gains, null.scores)
    denominator = full_margin - null_margin
    results: dict[str, Any] = {}
    for (kind, block), bundle in patches.items():
        patch_margin = _target_margins(request_ids, candidates, gains, bundle.scores)
        numerator = patch_margin - null_margin
        rows = []
        for surface, surface_mask in (
            ("observed_positive", np.isfinite(denominator)),
            (STRICT_TRANSFER_SURFACE, strict),
        ):
            for fold_name, fold_mask in (
                ("all", np.ones(len(records), dtype=bool)),
                ("0", folds == 0),
                ("1", folds == 1),
            ):
                mask = (
                    surface_mask
                    & fold_mask
                    & np.isfinite(numerator)
                    & np.isfinite(denominator)
                )
                rows.append(
                    {
                        "surface": surface,
                        "normalized_query_fold": fold_name,
                        **mediated_fraction_summary(
                            numerator[mask], denominator[mask], clusters[mask]
                        ),
                    }
                )
        results[f"{kind}:block_{block}"] = {
            "patch_kind": kind,
            "patch_block_zero_based": block,
            "negative_control": kind
            in {"full_to_full_identity", "cross_request_same_layer"},
            "rows": rows,
        }
    metrics = {
        "schema_version": 1,
        "analysis_type": "m2_activation_patch_mediation",
        "analysis_run_id": analysis_run_id,
        "method_id": full.metadata["method_id"],
        "checkpoint_id": full.metadata["checkpoint_id"],
        "mechanism_probe_manifest": {
            key: probe_manifest[key]
            for key in ("path", "sha256", "expected_sha256", "verified", "manifest_id")
        },
        "implementation_identity": implementation_identity,
        "primary_endpoint": "strict_transfer_target_margin_mediated_fraction",
        "mediated_fraction_definition": (
            "mean(patch_margin-null_margin) / mean(full_margin-null_margin); "
            "target is first maximum-gain candidate in frozen order and competitor "
            "is each bundle's best-scoring strictly lower-gain candidate"
        ),
        "bootstrap": {
            "cluster": "normalized_query",
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
        },
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": True,
        "qrels_opened_only_after_score_integrity": True,
        "qrels_dev_sha256": qrels_sha256,
        "strict_transfer_requests": int(strict.sum()),
        "identity_control_audit": identity_control_audit,
        "patch_results": results,
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    _write_json_atomic(metrics_path, metrics)
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
            "mechanism_probe_manifest_sha256": probe_manifest["sha256"],
        },
    )
    return metrics


def patch_evaluator_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "src/myrec/mechanism/patch_scorer.py",
        root / "src/myrec/mechanism/patch_evaluator.py",
        root / "scripts/evaluate_m2_activation_patches.py",
    )
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return {
        "files": files,
        "digest": sha256_text(
            json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
    }


def mediated_fraction_summary(
    numerator: np.ndarray,
    denominator: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Ratio-of-means mediation with normalized-query cluster bootstrap."""

    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=np.str_)
    if numerator.shape != denominator.shape or numerator.shape != clusters.shape:
        raise ValueError("mediated-fraction arrays are misaligned")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if numerator.size == 0:
        return {
            "requests": 0,
            "normalized_query_clusters": 0,
            "mean_patch_minus_null_margin": None,
            "mean_full_minus_null_margin": None,
            "mediated_fraction": None,
            "ci95": [None, None],
            "bootstrap_valid_samples": 0,
        }
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise ValueError("mediated-fraction input is non-finite")
    mean_num = float(numerator.mean())
    mean_den = float(denominator.mean())
    point = mean_num / mean_den if abs(mean_den) > 1.0e-12 else None
    unique, inverse = np.unique(clusters, return_inverse=True)
    cluster_num = np.bincount(inverse, weights=numerator)
    cluster_den = np.bincount(inverse, weights=denominator)
    cluster_count = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        selected = rng.integers(0, unique.size, size=unique.size)
        selected_den = float(cluster_den[selected].sum())
        selected_count = int(cluster_count[selected].sum())
        if selected_count and abs(selected_den) > 1.0e-12:
            values.append(float(cluster_num[selected].sum() / selected_den))
    if values:
        lower, upper = np.percentile(np.asarray(values), [2.5, 97.5]).tolist()
        ci = [float(lower), float(upper)]
    else:
        ci = [None, None]
    return {
        "requests": int(numerator.size),
        "normalized_query_clusters": int(unique.size),
        "mean_patch_minus_null_margin": mean_num,
        "mean_full_minus_null_margin": mean_den,
        "mediated_fraction": point,
        "ci95": ci,
        "bootstrap_valid_samples": len(values),
    }


def _audit_score_bundle(
    root: str | Path,
    records: Sequence[ModelRecord],
    *,
    expected_condition: str | None,
    patch: tuple[str, int] | None,
) -> AuditedScoreBundle:
    root = Path(root)
    metadata = _read_json(root / "metadata.json")
    scores_path = root / "scores.jsonl"
    if metadata.get("qrels_read") is not False:
        raise ValueError(f"score bundle crossed qrels boundary: {root}")
    if patch is None:
        # Immutable first-round score metadata predates the status/result_eligible
        # fields.  Admit only that exact completed evidence role; coverage and
        # byte hashes are reconstructed below rather than inferred from absence.
        if metadata.get("evidence_mode") != "first_round_pilot":
            raise ValueError(f"baseline is not a frozen first-round score: {root}")
        if metadata.get("result_eligible") is False:
            raise ValueError(f"baseline score is a smoke non-result: {root}")
    else:
        if metadata.get("status") != "completed":
            raise ValueError(f"patch score bundle is incomplete: {root}")
        if metadata.get("result_eligible") is not True:
            raise ValueError(f"patch score bundle is a smoke non-result: {root}")
    observed_hash = sha256_file(scores_path)
    if metadata.get("scores_sha256") != observed_hash:
        raise ValueError(f"score bytes changed after metadata: {root}")
    if expected_condition is not None:
        condition = metadata.get("condition_id")
        if condition is None:
            condition = {
                "true": "full",
                "full": "full",
                "null": "null",
            }.get(metadata.get("history_condition"))
        if condition != expected_condition:
            raise ValueError(f"baseline score condition mismatch: {root}")
    if patch is not None:
        kind, block = patch
        if metadata.get("analysis_stage") != "m2_mediation_patch":
            raise ValueError("patch score bundle has wrong analysis stage")
        if metadata.get("patch_kind") != kind:
            raise ValueError("patch score kind mismatch")
        if int(metadata.get("patch_block_zero_based", -1)) != block:
            raise ValueError("patch score block mismatch")
        if metadata.get("complete_finite_score_coverage") is not True:
            raise ValueError("patch score bundle lacks completion attestation")
    scores: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(scores_path):
        request_id = str(row.get("request_id") or "")
        item_id = str(row.get("candidate_item_id") or "")
        request_scores = scores.setdefault(request_id, {})
        if not request_id or not item_id or item_id in request_scores:
            raise ValueError("score bundle has empty/duplicate identity")
        value = float(row.get("score"))
        if not math.isfinite(value):
            raise ValueError("score bundle contains non-finite value")
        request_scores[item_id] = value
    expected_ids = [row.request_id for row in records]
    if list(scores) != expected_ids:
        raise ValueError("score bundle request identity/order coverage mismatch")
    for record in records:
        if list(scores[record.request_id]) != [
            str(value["item_id"]) for value in record.candidates
        ]:
            raise ValueError("score bundle candidate identity/order coverage mismatch")
    expected_rows = sum(len(row.candidates) for row in records)
    if int(metadata.get("request_count", -1)) != len(records):
        raise ValueError("score bundle metadata request count mismatch")
    if int(metadata.get("score_rows", -1)) != expected_rows:
        raise ValueError("score bundle metadata candidate count mismatch")
    return AuditedScoreBundle(root, metadata, scores, observed_hash)


def _target_margins(
    request_ids: Sequence[str],
    candidates: Mapping[str, Sequence[str]],
    gains: Mapping[str, Mapping[str, float]],
    scores: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    result = np.full(len(request_ids), np.nan, dtype=np.float64)
    for ordinal, request_id in enumerate(request_ids):
        item_ids = list(candidates[request_id])
        gain_values = [float(gains[request_id].get(item_id, 0.0)) for item_id in item_ids]
        maximum = max(gain_values, default=0.0)
        if maximum <= 0:
            continue
        target_index = next(index for index, value in enumerate(gain_values) if value == maximum)
        lower = [index for index, value in enumerate(gain_values) if value < maximum]
        if not lower:
            continue
        best_competitor = max(lower, key=lambda index: scores[request_id][item_ids[index]])
        result[ordinal] = (
            scores[request_id][item_ids[target_index]]
            - scores[request_id][item_ids[best_competitor]]
        )
    return result


def _audit_external_manifests(
    candidate_path: Path,
    request_path: Path,
    records: Sequence[ModelRecord],
    raw_records: Sequence[Mapping[str, Any]],
) -> None:
    if len(raw_records) != len(records):
        raise ValueError("raw and sanitized dev record coverage mismatch")
    raw_by_request: dict[str, Mapping[str, Any]] = {}
    for record, raw in zip(records, raw_records):
        raw_request_id = str(raw.get("request_id") or "")
        if raw_request_id != record.request_id or raw_request_id in raw_by_request:
            raise ValueError("raw and sanitized dev record identity/order mismatch")
        raw_by_request[raw_request_id] = raw
    candidates = _read_json(candidate_path)
    candidate_rows = [
        value for value in candidates.get("entries", []) if value.get("split") == "dev"
    ]
    if [str(value.get("request_id")) for value in candidate_rows] != [
        row.request_id for row in records
    ]:
        raise ValueError("candidate manifest dev request order mismatch")
    requests = _read_json(request_path)
    request_rows = [
        value for value in requests.get("entries", []) if value.get("split") == "dev"
    ]
    if [str(value.get("request_id")) for value in request_rows] != [
        row.request_id for row in records
    ]:
        raise ValueError("request manifest dev request order mismatch")
    for record, candidate_row, request_row in zip(records, candidate_rows, request_rows):
        item_ids = [str(value["item_id"]) for value in record.candidates]
        if [str(value) for value in candidate_row.get("candidate_item_ids", [])] != item_ids:
            raise ValueError("candidate manifest differs from dev records")
        from myrec.utils.hashing import sha256_text

        if request_row.get("candidate_item_ids_sha256") != sha256_text(
            json.dumps(item_ids, separators=(",", ":"))
        ):
            raise ValueError("request manifest candidate identity hash mismatch")
        # request_manifest.json seals the raw standardized query, whereas the
        # ModelRecord query is prompt-sanitized with surrounding whitespace
        # removed.  Audit the same raw representation used by the manifest.
        raw_query = str(raw_by_request[record.request_id].get("query", ""))
        if request_row.get("query_sha256") != sha256_text(raw_query):
            raise ValueError("request manifest query hash mismatch")


def _load_dev_qrels(
    path: Path, candidates: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError("qrels_dev has empty/duplicate request")
        relevance = row.get("relevance") or {}
        if not isinstance(relevance, dict):
            raise ValueError("qrels_dev relevance must be an object")
        values: dict[str, float] = {}
        for item_id, raw in relevance.items():
            gain = float(raw)
            if not math.isfinite(gain) or gain < 0:
                raise ValueError("qrels_dev gain is invalid")
            if gain > 0:
                values[str(item_id)] = gain
        result[request_id] = values
    if set(result) != set(candidates):
        raise ValueError("qrels_dev request coverage mismatch")
    if any(set(result[key]) - set(candidates[key]) for key in result):
        raise ValueError("qrels_dev contains an out-of-slate item")
    return result


def _score_identity(bundle: AuditedScoreBundle) -> dict[str, Any]:
    return {
        "path": str(bundle.root),
        "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
        "scores_sha256": bundle.scores_sha256,
        "request_count": len(bundle.scores),
        "score_rows": sum(len(value) for value in bundle.scores.values()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
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
