"""Independent M2 representation evaluator with a hard dev-qrels boundary."""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord, sanitize_record_for_model
from myrec.eval.target_aware_surfaces import build_target_aware_surface_memberships
from myrec.mechanism.representation_probe import (
    CANDIDATE_POSITIONS,
    M2_CONDITIONS,
    M2_HIDDEN_STATE_INDICES,
    REQUEST_POSITIONS,
    AuditedActivationBundle,
    audit_activation_bundle,
    load_fitted_readout,
    load_m2_probe_manifest,
    load_request_activations,
    normalize_query,
    normalized_query_fold,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


STRICT_TRANSFER_SURFACE = "target_nonrepeat_no_candidate_overlap"
PAIR_ORDER = (
    ("full", "null"),
    ("full", "relevant_6"),
    ("full", "irrelevant_6"),
    ("null", "relevant_6"),
    ("null", "irrelevant_6"),
    ("relevant_6", "irrelevant_6"),
)


def evaluate_m2_representations(
    standardized_dir: str | Path,
    bundle_dirs: Mapping[str, str | Path],
    probe_model_dir: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit all four bundles, then and only then open internal-dev qrels."""

    if tuple(sorted(bundle_dirs)) != tuple(sorted(M2_CONDITIONS)):
        raise ValueError("M2 representation evaluation requires exactly four conditions")
    if not analysis_run_id or any(value in analysis_run_id for value in ("/", "\\")):
        raise ValueError("invalid M2 analysis_run_id")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"M2 analysis output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_manifest = load_m2_probe_manifest()
    implementation_identity = representation_evaluator_implementation_identity()
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
    # Hashing qrels before the boundary would itself read qrels.  Deliberately
    # omit qrels_path here; it is checked after pre_qrels_audit.json is durable.
    pre_qrels_hashes = {
        "records_dev_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
    }
    for key, observed in pre_qrels_hashes.items():
        if observed != frozen[key]:
            raise ValueError(f"frozen M2 dev input hash mismatch: {key}")

    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("M2 evaluator requires all 8000 internal-dev requests")
    candidates = _audit_candidate_and_request_manifests(
        candidate_manifest_path,
        request_manifest_path,
        records,
        raw_records,
    )
    bundles: dict[str, AuditedActivationBundle] = {}
    for condition in M2_CONDITIONS:
        bundles[condition] = audit_activation_bundle(
            bundle_dirs[condition],
            expected_records=records,
            expected_role="dev_representation",
            expected_condition=condition,
            require_result_eligible=True,
        )
    invariant_keys = (
        "method_id",
        "config_sha256",
        "checkpoint_id",
        "records_sha256",
        "candidate_manifest_sha256",
        "request_manifest_sha256",
        "dataset_manifest_sha256",
        "dataset_id",
        "dataset_version",
        "split",
    )
    reference = bundles["full"].metadata
    expected_reference_hashes = {
        "records_sha256": pre_qrels_hashes["records_dev_sha256"],
        "candidate_manifest_sha256": pre_qrels_hashes["candidate_manifest_sha256"],
        "request_manifest_sha256": pre_qrels_hashes["request_manifest_sha256"],
        "dataset_manifest_sha256": pre_qrels_hashes["dataset_manifest_sha256"],
    }
    for key, value in expected_reference_hashes.items():
        if reference.get(key) != value:
            raise ValueError(f"M2 activation metadata/external input mismatch: {key}")
    frozen_model = frozen["models"].get(reference.get("method_id"))
    if not isinstance(frozen_model, dict):
        raise ValueError("M2 activation method is absent from the frozen manifest")
    if reference.get("config_sha256") != frozen_model.get("config_sha256"):
        raise ValueError("M2 activation config differs from the frozen manifest")
    if reference.get("checkpoint_id") != frozen_model.get("checkpoint_id"):
        raise ValueError("M2 activation checkpoint differs from the frozen manifest")
    for condition, bundle in bundles.items():
        for key in invariant_keys:
            if bundle.metadata.get(key) != reference.get(key):
                raise ValueError(f"M2 bundle invariant differs for {condition}: {key}")
        if bundle.metadata.get("qrels_read") is not False:
            raise ValueError(f"M2 activation bundle crossed qrels boundary: {condition}")
        if bundle.metadata.get("activation_passes") != reference.get(
            "activation_passes"
        ):
            raise ValueError(f"M2 activation pass contract differs for {condition}")

    probe_model_dir = Path(probe_model_dir)
    probe_metadata = _read_json(probe_model_dir / "metadata.json")
    weights_path = probe_model_dir / "probe_weights.npz"
    if probe_metadata.get("weights_sha256") != sha256_file(weights_path):
        raise ValueError("M2 probe weights changed after train-only fitting")
    if probe_metadata.get("method_id") != reference.get("method_id"):
        raise ValueError("M2 probe/activation method mismatch")
    if probe_metadata.get("dev_qrels_read") is not False:
        raise ValueError("M2 fitted probe metadata claims dev qrels access")
    probe_identity = probe_metadata.get("mechanism_probe_manifest", {})
    if probe_identity.get("sha256") != probe_manifest["sha256"]:
        raise ValueError("M2 fitted probe manifest identity mismatch")

    pre_qrels_report = {
        "schema_version": 1,
        "analysis_type": "m2_representation_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "qrels_read": False,
        "status": "passed",
        "checks": {
            "all_four_conditions_present": True,
            "all_8000_requests_complete": True,
            "all_candidate_readouts_complete_finite": True,
            "candidate_and_request_manifests_reconstructed": True,
            "checkpoint_config_dataset_and_request_invariants_equal": True,
            "probe_train_only_boundary_attested": True,
            "mechanism_probe_manifest_exact": True,
        },
        "invariants": {key: reference.get(key) for key in invariant_keys},
        "input_hashes": pre_qrels_hashes,
        "probe_model": {
            "path": str(probe_model_dir),
            "metadata_sha256": sha256_file(probe_model_dir / "metadata.json"),
            "weights_sha256": sha256_file(weights_path),
            "checkpoint_id": probe_metadata.get("probe_checkpoint_id"),
        },
        "bundles": {
            condition: {
                "path": str(bundle.root),
                "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
                "index_sha256": sha256_file(bundle.root / "index.json"),
                "request_count": len(bundle.request_ids),
                "candidate_count": bundle.candidate_count,
                "hidden_size": bundle.hidden_size,
                "assignment": bundle.metadata.get("assignment"),
            }
            for condition, bundle in bundles.items()
        },
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json_atomic(pre_qrels_path, pre_qrels_report)

    # Hard boundary: the first qrels read occurs after the audit is durable.
    qrels_sha256 = sha256_file(qrels_path)
    if qrels_sha256 != frozen["qrels_dev_sha256"]:
        raise ValueError("frozen M2 qrels_dev hash mismatch")
    gains = _load_dev_qrels(qrels_path, candidates)
    memberships = build_target_aware_surface_memberships(
        records_path, candidates, gains
    )
    labels = _dev_preference_labels(records, gains, probe_metadata)
    folds = np.asarray([normalized_query_fold(row.query) for row in records], dtype=np.int8)
    strict_mask = np.asarray(
        [row.request_id in memberships[STRICT_TRANSFER_SURFACE] for row in records],
        dtype=bool,
    )

    probe_metrics = _evaluate_probe_accuracy(
        bundles,
        probe_model_dir,
        labels,
        folds,
        strict_mask,
    )
    shift_metrics = _evaluate_layer_shifts(
        bundles,
        folds,
        strict_mask,
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": "m2_representation_and_preference_readout",
        "analysis_run_id": analysis_run_id,
        "method_id": reference["method_id"],
        "checkpoint_id": reference["checkpoint_id"],
        "mechanism_probe_manifest": {
            key: probe_manifest[key]
            for key in ("path", "sha256", "expected_sha256", "verified", "manifest_id")
        },
        "implementation_identity": implementation_identity,
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": True,
        "qrels_opened_only_after_bundle_integrity": True,
        "qrels_dev_path": str(qrels_path),
        "qrels_dev_sha256": qrels_sha256,
        "request_count": len(records),
        "candidate_count": sum(len(value) for value in candidates.values()),
        "strict_transfer_requests": int(strict_mask.sum()),
        "two_fold_rule": "sha256(normalized_query) mod 2",
        "fold_request_counts": {
            "0": int((folds == 0).sum()),
            "1": int((folds == 1).sum()),
        },
        "preference_classifier_feature_position": "history_summary_end",
        "candidate_text_visible_to_preference_classifier": False,
        "activation_pass_contract": reference["activation_passes"],
        "request_and_candidate_positions_share_same_forward": False,
        "probe_accuracy": probe_metrics,
        "layerwise_shift": shift_metrics,
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    _write_json_atomic(metrics_path, metrics)
    ledger_row = {
        "analysis_type": metrics["analysis_type"],
        "run_id": analysis_run_id,
        "method_id": metrics["method_id"],
        "checkpoint_id": metrics["checkpoint_id"],
        "split": "dev",
        "qrels_sha256": qrels_sha256,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "mechanism_probe_manifest_sha256": probe_manifest["sha256"],
        "strict_transfer_requests": int(strict_mask.sum()),
    }
    _append_jsonl(Path(dev_eval_log_path), ledger_row)
    return metrics


def representation_evaluator_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "src/myrec/mechanism/representation_probe.py",
        root / "src/myrec/mechanism/representation_evaluator.py",
        root / "scripts/evaluate_m2_representations.py",
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


def _evaluate_probe_accuracy(
    bundles: Mapping[str, AuditedActivationBundle],
    model_dir: Path,
    labels: Mapping[str, np.ndarray],
    folds: np.ndarray,
    strict_mask: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in M2_CONDITIONS:
        activations = load_request_activations(
            bundles[condition], position="history_summary_end"
        )
        condition_rows: list[dict[str, Any]] = []
        for task in ("brand", "category"):
            target = labels[task]
            label_present = target != ""
            for label_control in ("real_labels", "random_labels"):
                for state in M2_HIDDEN_STATE_INDICES:
                    readout, _metadata = load_fitted_readout(
                        model_dir,
                        task=task,
                        hidden_state_index=state,
                        label_control=label_control,
                    )
                    prediction = readout.predict(activations[state])
                    for surface, surface_mask in (
                        ("all", np.ones(target.size, dtype=bool)),
                        (STRICT_TRANSFER_SURFACE, strict_mask),
                    ):
                        for fold_name, fold_mask in (
                            ("all", np.ones(target.size, dtype=bool)),
                            ("0", folds == 0),
                            ("1", folds == 1),
                        ):
                            mask = label_present & surface_mask & fold_mask
                            condition_rows.append(
                                {
                                    "task": task,
                                    "hidden_state_index": state,
                                    "embedding_state_negative_control": state == 0,
                                    "label_control": label_control,
                                    "surface": surface,
                                    "normalized_query_fold": fold_name,
                                    **_classification_summary(target[mask], prediction[mask]),
                                }
                            )
        result[condition] = condition_rows
    return result


def _evaluate_layer_shifts(
    bundles: Mapping[str, AuditedActivationBundle],
    folds: np.ndarray,
    strict_mask: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left, right in PAIR_ORDER:
        request_l2: list[np.ndarray] = []
        request_cosine: list[np.ndarray] = []
        candidate_l2: list[np.ndarray] = []
        candidate_cosine: list[np.ndarray] = []
        left_shards = bundles[left].index["shards"]
        right_shards = bundles[right].index["shards"]
        if len(left_shards) != len(right_shards):
            raise ValueError(f"M2 condition shard partitions differ: {left}/{right}")
        for left_index, right_index in zip(left_shards, right_shards):
            with np.load(
                bundles[left].root / "shards" / left_index["path"],
                allow_pickle=False,
            ) as left_payload, np.load(
                bundles[right].root / "shards" / right_index["path"],
                allow_pickle=False,
            ) as right_payload:
                _assert_shard_alignment(left_payload, right_payload, left, right)
                left_request = np.asarray(
                    left_payload["request_activations"], dtype=np.float32
                )
                right_request = np.asarray(
                    right_payload["request_activations"], dtype=np.float32
                )
                request_l2.append(_normalized_l2(left_request, right_request))
                request_cosine.append(_cosine_distance(left_request, right_request))
                left_candidate = np.asarray(
                    left_payload["candidate_activations"], dtype=np.float32
                )
                right_candidate = np.asarray(
                    right_payload["candidate_activations"], dtype=np.float32
                )
                row_l2 = _normalized_l2(left_candidate, right_candidate)
                row_cosine = _cosine_distance(left_candidate, right_candidate)
                offsets = np.asarray(left_payload["candidate_offsets"], dtype=np.int64)
                candidate_l2.append(_request_mean(row_l2, offsets))
                candidate_cosine.append(_request_mean(row_cosine, offsets))
        # request arrays: [request, request_position, state]
        l2 = np.concatenate(request_l2, axis=0)
        cosine = np.concatenate(request_cosine, axis=0)
        # candidate arrays are already request-weighted: [request, state]
        candidate_l2_values = np.concatenate(candidate_l2, axis=0)
        candidate_cosine_values = np.concatenate(candidate_cosine, axis=0)
        if l2.shape[0] != folds.size or candidate_l2_values.shape[0] != folds.size:
            raise ValueError("M2 layer-shift request coverage mismatch")
        rows: list[dict[str, Any]] = []
        for state_ordinal, state in enumerate(M2_HIDDEN_STATE_INDICES):
            for position_ordinal, position in enumerate(REQUEST_POSITIONS):
                rows.extend(
                    _shift_summaries(
                        l2[:, position_ordinal, state_ordinal],
                        cosine[:, position_ordinal, state_ordinal],
                        folds,
                        strict_mask,
                        state=state,
                        position=position,
                    )
                )
            rows.extend(
                _shift_summaries(
                    candidate_l2_values[:, state_ordinal],
                    candidate_cosine_values[:, state_ordinal],
                    folds,
                    strict_mask,
                    state=state,
                    position=CANDIDATE_POSITIONS[0],
                )
            )
        result[f"{left}_vs_{right}"] = {
            "left_condition": left,
            "right_condition": right,
            "all_preregistered_pair_reported": True,
            "candidate_shift_weighting": "mean_within_request_then_mean_across_requests",
            "rows": rows,
        }
    return result


def _shift_summaries(
    l2: np.ndarray,
    cosine: np.ndarray,
    folds: np.ndarray,
    strict_mask: np.ndarray,
    *,
    state: int,
    position: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for surface, surface_mask in (
        ("all", np.ones(folds.size, dtype=bool)),
        (STRICT_TRANSFER_SURFACE, strict_mask),
    ):
        for fold_name, fold_mask in (
            ("all", np.ones(folds.size, dtype=bool)),
            ("0", folds == 0),
            ("1", folds == 1),
        ):
            mask = surface_mask & fold_mask
            rows.append(
                {
                    "hidden_state_index": state,
                    "embedding_state_negative_control": state == 0,
                    "position": position,
                    "surface": surface,
                    "normalized_query_fold": fold_name,
                    "requests": int(mask.sum()),
                    "mean_l2_per_sqrt_hidden": _mean_or_none(l2[mask]),
                    "mean_cosine_distance": _mean_or_none(cosine[mask]),
                    "sd_l2_per_sqrt_hidden": _sd_or_none(l2[mask]),
                    "sd_cosine_distance": _sd_or_none(cosine[mask]),
                }
            )
    return rows


def _audit_candidate_and_request_manifests(
    candidate_path: Path,
    request_path: Path,
    records: Sequence[ModelRecord],
    raw_records: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    if len(raw_records) != len(records):
        raise ValueError("raw and sanitized dev record coverage mismatch")
    raw_by_request: dict[str, Mapping[str, Any]] = {}
    for record, raw in zip(records, raw_records):
        raw_request_id = str(raw.get("request_id") or "")
        if raw_request_id != record.request_id or raw_request_id in raw_by_request:
            raise ValueError("raw and sanitized dev record identity/order mismatch")
        raw_by_request[raw_request_id] = raw
    candidate_manifest = _read_json(candidate_path)
    result: dict[str, list[str]] = {}
    for entry in candidate_manifest.get("entries", []):
        if entry.get("split") != "dev":
            continue
        request_id = str(entry.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError("candidate manifest has empty/duplicate dev request")
        item_ids = [str(value) for value in entry.get("candidate_item_ids", [])]
        if len(item_ids) < 2 or len(item_ids) != len(set(item_ids)):
            raise ValueError("candidate manifest dev slate is invalid")
        result[request_id] = item_ids
    expected_ids = [row.request_id for row in records]
    if list(result) != expected_ids:
        raise ValueError("candidate manifest dev request identity/order mismatch")
    for record in records:
        if result[record.request_id] != [str(row["item_id"]) for row in record.candidates]:
            raise ValueError("candidate manifest differs from label-free record")
    request_manifest = _read_json(request_path)
    observed: dict[str, dict[str, Any]] = {}
    for entry in request_manifest.get("entries", []):
        if entry.get("split") != "dev":
            continue
        request_id = str(entry.get("request_id") or "")
        if not request_id or request_id in observed:
            raise ValueError("request manifest has empty/duplicate dev request")
        observed[request_id] = entry
    if list(observed) != expected_ids:
        raise ValueError("request manifest dev identity/order mismatch")
    for record in records:
        item_ids = result[record.request_id]
        expected_candidate_hash = sha256_text(json.dumps(item_ids, separators=(",", ":")))
        if observed[record.request_id].get("candidate_item_ids_sha256") != expected_candidate_hash:
            raise ValueError("request manifest candidate hash mismatch")
        # The frozen request manifest hashes the raw standardized query.  Prompt
        # sanitization intentionally strips surrounding whitespace, so hashing
        # ``record.query`` here would compare two different contracts.
        raw_query = str(raw_by_request[record.request_id].get("query", ""))
        if observed[record.request_id].get("query_sha256") != sha256_text(raw_query):
            raise ValueError("request manifest query hash mismatch")
    return result


def _load_dev_qrels(
    path: Path, candidates: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError("qrels_dev has empty/duplicate request identity")
        relevance = row.get("relevance") or {}
        if not isinstance(relevance, dict):
            raise ValueError("qrels_dev relevance must be an object")
        gains: dict[str, float] = {}
        for item_id, raw in relevance.items():
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError("qrels_dev contains invalid gain")
            if value > 0:
                gains[str(item_id)] = value
        result[request_id] = gains
    if set(result) != set(candidates):
        raise ValueError("qrels_dev request coverage differs from candidates")
    for request_id, gains in result.items():
        if set(gains) - set(candidates[request_id]):
            raise ValueError("qrels_dev contains an out-of-slate item")
    return result


def _dev_preference_labels(
    records: Sequence[ModelRecord],
    gains: Mapping[str, Mapping[str, float]],
    probe_metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    vocab = {
        task: set(probe_metadata["label_audit"][task]["vocabulary"])
        for task in ("brand", "category")
    }
    labels = {task: [] for task in ("brand", "category")}
    for record in records:
        values = [
            float(gains[record.request_id].get(str(row["item_id"]), 0.0))
            for row in record.candidates
        ]
        maximum = max(values, default=0.0)
        if maximum <= 0:
            labels["brand"].append("")
            labels["category"].append("")
            continue
        index = next(position for position, value in enumerate(values) if value == maximum)
        candidate = record.candidates[index]
        brand = _normalize_label(candidate.get("brand"))
        categories = candidate.get("cat") or []
        category = _normalize_label(categories[-1] if categories else "")
        labels["brand"].append(brand if brand in vocab["brand"] else "")
        labels["category"].append(category if category in vocab["category"] else "")
    return {task: np.asarray(values, dtype=np.str_) for task, values in labels.items()}


def _classification_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if target.size == 0:
        return {"requests": 0, "accuracy": None, "balanced_accuracy": None}
    accuracy = float(np.mean(target == prediction))
    classes = np.unique(target)
    balanced = float(
        np.mean([np.mean(prediction[target == value] == value) for value in classes])
    )
    return {
        "requests": int(target.size),
        "classes_observed": int(classes.size),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
    }


def _assert_shard_alignment(
    left: Any, right: Any, left_name: str, right_name: str
) -> None:
    for key in ("request_ids", "normalized_queries", "candidate_offsets", "candidate_ids"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"M2 shard alignment differs for {left_name}/{right_name}: {key}")


def _normalized_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return np.linalg.norm(difference, axis=-1) / math.sqrt(difference.shape[-1])


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    result = np.ones_like(numerator)
    valid = denominator > 1.0e-12
    result[valid] = 1.0 - np.clip(numerator[valid] / denominator[valid], -1.0, 1.0)
    both_zero = (np.linalg.norm(a, axis=-1) <= 1.0e-12) & (
        np.linalg.norm(b, axis=-1) <= 1.0e-12
    )
    result[both_zero] = 0.0
    return result


def _request_mean(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    rows = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        if stop <= start:
            raise ValueError("dev candidate activation request has no candidates")
        rows.append(values[int(start) : int(stop)].mean(axis=0))
    return np.stack(rows)


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _sd_or_none(values: np.ndarray) -> float | None:
    return float(np.std(values, ddof=1)) if values.size > 1 else None


def _normalize_label(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip().casefold()


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
