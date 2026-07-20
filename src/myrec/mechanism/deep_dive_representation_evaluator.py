"""Dev-qrels-gated D1 evaluator for all-layer full/null representations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.eval.target_aware_surfaces import build_target_aware_surface_memberships
from myrec.mechanism.deep_dive_representation_analysis import (
    ALL_POSITIONS,
    AllStateBundle,
    audit_all_state_bundle,
    load_all_request_states,
    load_selected_candidate_states,
)
from myrec.mechanism.deep_dive_representation_runtime import (
    ALL_HIDDEN_STATE_INDICES,
    REQUEST_POSITIONS,
)
from myrec.mechanism.representation_evaluator import (
    STRICT_TRANSFER_SURFACE,
    _audit_candidate_and_request_manifests,
    _dev_preference_labels,
    _load_dev_qrels,
)
from myrec.mechanism.representation_probe import (
    LinearReadout,
    load_m2_probe_manifest,
    normalize_query,
    normalized_query_fold,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


BLOCK_REGIONS = {
    "blocks_00_06": tuple(range(1, 8)),
    "blocks_07_13": tuple(range(8, 15)),
    "blocks_14_20": tuple(range(15, 22)),
    "blocks_21_27": tuple(range(22, 29)),
}


def evaluate_deep_dive_representations(
    standardized_dir: str | Path,
    full_bundle_dir: str | Path,
    null_bundle_dir: str | Path,
    probe_model_dir: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit two 29-state bundles, then open frozen internal-dev qrels."""

    if not analysis_run_id or any(value in analysis_run_id for value in ("/", "\\")):
        raise ValueError("invalid D1 analysis_run_id")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D1 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_dev.jsonl"
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    dataset_manifest_path = standardized_dir / "manifest.json"
    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("D1 evaluator requires all 8000 internal-dev requests")
    candidates = _audit_candidate_and_request_manifests(
        candidate_manifest_path,
        request_manifest_path,
        records,
        raw_records,
    )
    bundles = {
        "full": audit_all_state_bundle(
            full_bundle_dir,
            expected_records=records,
            expected_role="dev_representation",
            expected_condition="full",
        ),
        "null": audit_all_state_bundle(
            null_bundle_dir,
            expected_records=records,
            expected_role="dev_representation",
            expected_condition="null",
        ),
    }
    reference = bundles["full"].metadata
    invariants = (
        "method_id",
        "checkpoint_id",
        "config_sha256",
        "records_sha256",
        "candidate_manifest_sha256",
        "request_manifest_sha256",
        "dataset_manifest_sha256",
        "deep_dive_manifest_sha256",
        "request_positions",
        "candidate_positions",
        "hidden_state_indices",
    )
    for condition, bundle in bundles.items():
        for key in invariants:
            if bundle.metadata.get(key) != reference.get(key):
                raise ValueError(f"D1 bundle invariant differs for {condition}: {key}")
        if bundle.metadata.get("qrels_read") is not False:
            raise ValueError("D1 activation bundle crossed qrels boundary")

    probe_model_dir = Path(probe_model_dir)
    probe_metadata = _read_json(probe_model_dir / "metadata.json")
    weights_path = probe_model_dir / "probe_weights.npz"
    if probe_metadata.get("weights_sha256") != sha256_file(weights_path):
        raise ValueError("D1 probe weights changed after fitting")
    if probe_metadata.get("method_id") != reference.get("method_id"):
        raise ValueError("D1 probe/model method mismatch")
    if probe_metadata.get("dev_qrels_read") is not False:
        raise ValueError("D1 probe crossed dev qrels boundary")
    old_manifest = load_m2_probe_manifest()
    import yaml

    deep_manifest = yaml.safe_load(
        Path(reference["deep_dive_manifest_path"]).read_text(encoding="utf-8")
    )
    if deep_manifest["scientific_plan"]["prior_probe_manifest_sha256"] != (
        old_manifest["sha256"]
    ):
        raise ValueError("D1 deep-dive manifest is not bound to frozen qrels identity")
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d1_all_layer_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "full_and_null_present": True,
            "all_8000_requests_complete": True,
            "all_160753_candidates_complete_finite": True,
            "all_29_states_present": True,
            "candidate_and_request_manifests_reconstructed": True,
            "train_only_probe_hash_checked": True,
        },
        "invariants": {key: reference.get(key) for key in invariants},
        "bundles": {
            condition: {
                "path": str(bundle.root),
                "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
                "index_sha256": sha256_file(bundle.root / "index.json"),
                "request_count": len(bundle.request_ids),
                "candidate_count": bundle.candidate_count,
            }
            for condition, bundle in bundles.items()
        },
        "probe": {
            "path": str(probe_model_dir),
            "metadata_sha256": sha256_file(probe_model_dir / "metadata.json"),
            "weights_sha256": sha256_file(weights_path),
        },
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    # First qrels byte read occurs only below this durable boundary.
    qrels_sha256 = sha256_file(qrels_path)
    if qrels_sha256 != old_manifest["frozen_inputs"]["qrels_dev_sha256"]:
        raise ValueError("D1 qrels_dev differs from frozen first-stage identity")
    gains = _load_dev_qrels(qrels_path, candidates)
    memberships = build_target_aware_surface_memberships(records_path, candidates, gains)
    strict_mask = np.asarray(
        [row.request_id in memberships[STRICT_TRANSFER_SURFACE] for row in records],
        dtype=bool,
    )
    folds = np.asarray([normalized_query_fold(row.query) for row in records], dtype=np.int8)
    labels = _dev_preference_labels(records, gains, probe_metadata)
    target_ordinals = _target_ordinals(records, gains)

    correctness: dict[str, np.ndarray] = {}
    accuracy_rows: list[dict[str, Any]] = []
    with np.load(weights_path, allow_pickle=False) as weights:
        for condition, bundle in bundles.items():
            position_features = {
                position: load_all_request_states(bundle, position)
                for position in REQUEST_POSITIONS
            }
            position_features["candidate_readout"] = load_selected_candidate_states(
                bundle, target_ordinals
            )
            for position in ALL_POSITIONS:
                matrix = position_features[position]
                for task in ("brand", "category"):
                    target = labels[task]
                    label_present = target != ""
                    for control in ("real_labels", "random_labels"):
                        for state in ALL_HIDDEN_STATE_INDICES:
                            readout = _readout(
                                weights,
                                position=position,
                                task=task,
                                state=state,
                                control=control,
                            )
                            prediction = readout.predict(matrix[:, state])
                            is_correct = prediction == target
                            key = _correctness_key(
                                condition, position, task, control, state
                            )
                            correctness[key] = is_correct.astype(np.bool_)
                            for surface, surface_mask in (
                                ("all", np.ones(len(records), dtype=bool)),
                                (STRICT_TRANSFER_SURFACE, strict_mask),
                            ):
                                for fold_name, fold_mask in (
                                    ("all", np.ones(len(records), dtype=bool)),
                                    ("0", folds == 0),
                                    ("1", folds == 1),
                                ):
                                    mask = label_present & surface_mask & fold_mask
                                    accuracy_rows.append(
                                        {
                                            "condition": condition,
                                            "position": position,
                                            "task": task,
                                            "label_control": control,
                                            "hidden_state_index": state,
                                            "surface": surface,
                                            "normalized_query_fold": fold_name,
                                            **_classification_summary(
                                                target[mask], prediction[mask]
                                            ),
                                        }
                                    )
    correctness_path = output_dir / "correctness.npz"
    temporary = output_dir / ".correctness.writing.npz"
    np.savez(
        temporary,
        **correctness,
        brand_labels=labels["brand"],
        category_labels=labels["category"],
        strict_mask=strict_mask,
        folds=folds,
        normalized_queries=np.asarray(
            [normalize_query(row.query) for row in records], dtype=np.str_
        ),
    )
    temporary.replace(correctness_path)
    geometry_rows = _full_null_geometry(bundles, folds, strict_mask)
    region_rows = _region_point_estimates(
        accuracy_rows,
        surface=STRICT_TRANSFER_SURFACE,
        fold="all",
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d1_all_layer_representation",
        "analysis_run_id": analysis_run_id,
        "method_id": reference["method_id"],
        "checkpoint_id": reference["checkpoint_id"],
        "request_count": len(records),
        "candidate_count": bundles["full"].candidate_count,
        "strict_transfer_requests": int(strict_mask.sum()),
        "qrels_read": True,
        "qrels_opened_only_after_bundle_integrity": True,
        "qrels_dev_sha256": qrels_sha256,
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "correctness_path": str(correctness_path),
        "correctness_sha256": sha256_file(correctness_path),
        "hidden_state_indices": list(ALL_HIDDEN_STATE_INDICES),
        "block_regions": {key: list(value) for key, value in BLOCK_REGIONS.items()},
        "accuracy_rows": accuracy_rows,
        "region_point_estimates": region_rows,
        "full_null_geometry": geometry_rows,
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
            "strict_transfer_requests": int(strict_mask.sum()),
        },
    )
    return metrics


def _full_null_geometry(
    bundles: Mapping[str, AllStateBundle],
    folds: np.ndarray,
    strict_mask: np.ndarray,
) -> list[dict[str, Any]]:
    request_l2: list[np.ndarray] = []
    request_cosine: list[np.ndarray] = []
    request_rms_ratio: list[np.ndarray] = []
    candidate_l2: list[np.ndarray] = []
    candidate_cosine: list[np.ndarray] = []
    candidate_rms_ratio: list[np.ndarray] = []
    full_shards = bundles["full"].index["shards"]
    null_shards = bundles["null"].index["shards"]
    if len(full_shards) != len(null_shards):
        raise ValueError("D1 full/null shard partitions differ")
    for full_row, null_row in zip(full_shards, null_shards):
        with np.load(
            bundles["full"].root / "shards" / full_row["path"],
            allow_pickle=False,
        ) as full, np.load(
            bundles["null"].root / "shards" / null_row["path"],
            allow_pickle=False,
        ) as null:
            for key in (
                "request_ids",
                "normalized_queries",
                "candidate_offsets",
                "candidate_ids",
            ):
                if not np.array_equal(full[key], null[key]):
                    raise ValueError(f"D1 full/null shard alignment differs: {key}")
            full_request = np.asarray(full["request_activations"], dtype=np.float32)
            null_request = np.asarray(null["request_activations"], dtype=np.float32)
            l2, cosine, ratio = _geometry(full_request, null_request)
            request_l2.append(l2)
            request_cosine.append(cosine)
            request_rms_ratio.append(ratio)
            full_candidate = np.asarray(full["candidate_activations"], dtype=np.float32)
            null_candidate = np.asarray(null["candidate_activations"], dtype=np.float32)
            l2, cosine, ratio = _geometry(full_candidate, null_candidate)
            offsets = np.asarray(full["candidate_offsets"], dtype=np.int64)
            candidate_l2.append(_request_mean(l2, offsets))
            candidate_cosine.append(_request_mean(cosine, offsets))
            candidate_rms_ratio.append(_request_mean(ratio, offsets))
    request_values = {
        "l2": np.concatenate(request_l2),
        "cosine": np.concatenate(request_cosine),
        "rms_ratio": np.concatenate(request_rms_ratio),
    }
    candidate_values = {
        "l2": np.concatenate(candidate_l2),
        "cosine": np.concatenate(candidate_cosine),
        "rms_ratio": np.concatenate(candidate_rms_ratio),
    }
    rows: list[dict[str, Any]] = []
    for state in ALL_HIDDEN_STATE_INDICES:
        for position_ordinal, position in enumerate(REQUEST_POSITIONS):
            rows.extend(
                _geometry_summaries(
                    {key: value[:, position_ordinal, state] for key, value in request_values.items()},
                    folds,
                    strict_mask,
                    position,
                    state,
                )
            )
        rows.extend(
            _geometry_summaries(
                {key: value[:, state] for key, value in candidate_values.items()},
                folds,
                strict_mask,
                "candidate_readout",
                state,
            )
        )
    return rows


def _geometry(full: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(full, dtype=np.float64)
    b = np.asarray(null, dtype=np.float64)
    difference = np.linalg.norm(a - b, axis=-1) / math.sqrt(a.shape[-1])
    a_norm = np.linalg.norm(a, axis=-1)
    b_norm = np.linalg.norm(b, axis=-1)
    denominator = a_norm * b_norm
    cosine = np.ones_like(denominator)
    valid = denominator > 1.0e-12
    cosine[valid] = 1.0 - np.clip(
        np.sum(a[valid] * b[valid], axis=-1) / denominator[valid], -1.0, 1.0
    )
    cosine[(a_norm <= 1.0e-12) & (b_norm <= 1.0e-12)] = 0.0
    rms_ratio = (a_norm / math.sqrt(a.shape[-1])) / (
        b_norm / math.sqrt(a.shape[-1]) + 1.0e-12
    )
    return difference, cosine, rms_ratio


def _geometry_summaries(
    values: Mapping[str, np.ndarray],
    folds: np.ndarray,
    strict_mask: np.ndarray,
    position: str,
    state: int,
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
                    "position": position,
                    "hidden_state_index": state,
                    "surface": surface,
                    "normalized_query_fold": fold_name,
                    "requests": int(mask.sum()),
                    "mean_l2_per_sqrt_hidden": float(np.mean(values["l2"][mask])),
                    "mean_cosine_distance": float(np.mean(values["cosine"][mask])),
                    "mean_full_over_null_rms": float(np.mean(values["rms_ratio"][mask])),
                }
            )
    return rows


def _region_point_estimates(
    rows: Sequence[Mapping[str, Any]], *, surface: str, fold: str
) -> list[dict[str, Any]]:
    lookup = {
        (
            row["condition"],
            row["position"],
            row["task"],
            row["label_control"],
            row["hidden_state_index"],
        ): row["balanced_accuracy"]
        for row in rows
        if row["surface"] == surface and row["normalized_query_fold"] == fold
    }
    result: list[dict[str, Any]] = []
    for position in ALL_POSITIONS:
        for task in ("brand", "category"):
            for region, states in BLOCK_REGIONS.items():
                full_real = np.mean(
                    [lookup[("full", position, task, "real_labels", state)] for state in states]
                )
                full_random = np.mean(
                    [lookup[("full", position, task, "random_labels", state)] for state in states]
                )
                null_real = np.mean(
                    [lookup[("null", position, task, "real_labels", state)] for state in states]
                )
                result.extend(
                    [
                        {
                            "position": position,
                            "task": task,
                            "region": region,
                            "contrast": "real_minus_random",
                            "estimate": float(full_real - full_random),
                        },
                        {
                            "position": position,
                            "task": task,
                            "region": region,
                            "contrast": "full_minus_null_excess",
                            "estimate": float(full_real - null_real),
                        },
                    ]
                )
    return result


def _readout(
    payload: Any, *, position: str, task: str, state: int, control: str
) -> LinearReadout:
    key = f"{position}__{task}__state_{state}__{control}"
    return LinearReadout(
        classes=tuple(str(value) for value in payload[f"{key}__classes"].tolist()),
        mean=np.asarray(payload[f"{key}__mean"], dtype=np.float64),
        scale=np.asarray(payload[f"{key}__scale"], dtype=np.float64),
        coefficient=np.asarray(payload[f"{key}__coefficient"], dtype=np.float64),
        intercept=np.asarray(payload[f"{key}__intercept"], dtype=np.float64),
    )


def _target_ordinals(records: Sequence[Any], gains: Mapping[str, Mapping[str, float]]) -> list[int]:
    result = []
    for record in records:
        values = [
            float(gains[record.request_id].get(str(candidate["item_id"]), 0.0))
            for candidate in record.candidates
        ]
        maximum = max(values, default=0.0)
        result.append(
            0
            if maximum <= 0
            else next(index for index, value in enumerate(values) if value == maximum)
        )
    return result


def _classification_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if target.size == 0:
        return {"requests": 0, "accuracy": None, "balanced_accuracy": None}
    classes = np.unique(target)
    return {
        "requests": int(target.size),
        "classes_observed": int(classes.size),
        "accuracy": float(np.mean(target == prediction)),
        "balanced_accuracy": float(
            np.mean([np.mean(prediction[target == value] == value) for value in classes])
        ),
    }


def _request_mean(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.stack(
        [values[int(start) : int(stop)].mean(axis=0) for start, stop in zip(offsets[:-1], offsets[1:])]
    )


def _correctness_key(
    condition: str, position: str, task: str, control: str, state: int
) -> str:
    return f"{condition}__{position}__{task}__{control}__state_{state}"


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
