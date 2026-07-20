"""Auditing and train-only readouts for D1 all-layer representations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.deep_dive_representation_runtime import (
    ALL_HIDDEN_STATE_INDICES,
    REQUEST_POSITIONS,
)
from myrec.mechanism.representation_probe import (
    M2_MAX_CLASSES,
    M2_MIN_CLASS_FREQUENCY,
    LinearReadout,
    _load_positive_qrels,
    _permuted_labels,
    build_preference_labels,
    fit_linear_readout,
    normalize_query,
    representation_holdout,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


ALL_POSITIONS = (*REQUEST_POSITIONS, "candidate_readout")


@dataclass(frozen=True)
class AllStateBundle:
    root: Path
    metadata: dict[str, Any]
    index: dict[str, Any]
    request_ids: tuple[str, ...]
    candidate_count: int
    hidden_size: int


def audit_all_state_bundle(
    bundle_dir: str | Path,
    *,
    expected_records: Sequence[ModelRecord],
    expected_role: str,
    expected_condition: str,
    require_result_eligible: bool = True,
    allowed_method_ids: Sequence[str] = (
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    ),
) -> AllStateBundle:
    """Audit every all-state shard without reading a qrels file."""

    root = Path(bundle_dir)
    metadata = _read_json(root / "metadata.json")
    index = _read_json(root / "index.json")
    expected_meta = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d1_all_layer_representation",
        "bundle_role": expected_role,
        "condition_id": expected_condition,
        "hidden_state_indices": list(ALL_HIDDEN_STATE_INDICES),
        "request_positions": list(REQUEST_POSITIONS),
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_activation_coverage": True,
        "status": "completed",
    }
    for key, expected in expected_meta.items():
        if metadata.get(key) != expected:
            raise ValueError(f"all-state bundle metadata mismatch: {key}")
    if require_result_eligible and metadata.get("result_eligible") is not True:
        raise ValueError("all-state bundle is a smoke non-result")
    if metadata.get("method_id") not in set(map(str, allowed_method_ids)):
        raise ValueError("all-state bundle method is outside the caller boundary")
    expected_ids = [row.request_id for row in expected_records]
    observed_ids: list[str] = []
    observed_candidate_ids: list[str] = []
    hidden_size: int | None = None
    total_candidates = 0
    for shard in index.get("shards", []):
        relative = str(shard.get("path") or "")
        if Path(relative).name != relative or not relative.endswith(".npz"):
            raise ValueError("all-state shard path is invalid")
        path = root / "shards" / relative
        if sha256_file(path) != shard.get("sha256"):
            raise ValueError("all-state shard hash mismatch")
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "request_ids",
                "normalized_queries",
                "request_activations",
                "candidate_offsets",
                "candidate_ids",
                "candidate_activations",
                "hidden_state_indices",
                "request_positions",
            }
            if set(payload.files) != required:
                raise ValueError("all-state shard field set differs")
            request_ids = [str(value) for value in payload["request_ids"].tolist()]
            queries = [str(value) for value in payload["normalized_queries"].tolist()]
            request_values = np.asarray(payload["request_activations"])
            candidate_values = np.asarray(payload["candidate_activations"])
            candidate_ids = [str(value) for value in payload["candidate_ids"].tolist()]
            offsets = np.asarray(payload["candidate_offsets"], dtype=np.int64)
            if payload["hidden_state_indices"].tolist() != list(
                ALL_HIDDEN_STATE_INDICES
            ):
                raise ValueError("all-state hidden-state order drift")
            if payload["request_positions"].tolist() != list(REQUEST_POSITIONS):
                raise ValueError("all-state request-position order drift")
            if request_values.ndim != 4 or request_values.shape[1:3] != (
                len(REQUEST_POSITIONS),
                len(ALL_HIDDEN_STATE_INDICES),
            ):
                raise ValueError("all-state request tensor shape mismatch")
            if candidate_values.ndim != 3 or candidate_values.shape[1] != len(
                ALL_HIDDEN_STATE_INDICES
            ):
                raise ValueError("all-state candidate tensor shape mismatch")
            if request_values.shape[0] != len(request_ids) or len(queries) != len(
                request_ids
            ):
                raise ValueError("all-state request arrays are misaligned")
            if offsets.shape != (len(request_ids) + 1,) or int(offsets[0]) != 0:
                raise ValueError("all-state offsets have invalid shape")
            if np.any(np.diff(offsets) < 0) or int(offsets[-1]) != len(candidate_ids):
                raise ValueError("all-state offsets are inconsistent")
            if len(candidate_values) != len(candidate_ids):
                raise ValueError("all-state candidate arrays are misaligned")
            if not np.isfinite(request_values).all() or not np.isfinite(
                candidate_values
            ).all():
                raise ValueError("all-state bundle contains non-finite values")
            expected_slice = expected_records[
                len(observed_ids) : len(observed_ids) + len(request_ids)
            ]
            if queries != [normalize_query(row.query) for row in expected_slice]:
                raise ValueError("all-state normalized-query order mismatch")
            current_hidden = int(request_values.shape[-1])
            if candidate_values.shape[-1] != current_hidden:
                raise ValueError("all-state request/candidate hidden sizes differ")
            if hidden_size is None:
                hidden_size = current_hidden
            elif hidden_size != current_hidden:
                raise ValueError("all-state hidden size differs between shards")
        if int(shard.get("request_count", -1)) != len(request_ids):
            raise ValueError("all-state shard request count mismatch")
        if int(shard.get("candidate_count", -1)) != len(candidate_ids):
            raise ValueError("all-state shard candidate count mismatch")
        observed_ids.extend(request_ids)
        observed_candidate_ids.extend(candidate_ids)
        total_candidates += len(candidate_ids)
    if observed_ids != expected_ids:
        raise ValueError("all-state request identity/order coverage mismatch")
    expected_candidate_ids = [
        str(candidate["item_id"])
        for record in expected_records
        for candidate in record.candidates
    ]
    if metadata.get("candidate_positions") == ["candidate_readout"]:
        if observed_candidate_ids != expected_candidate_ids:
            raise ValueError("all-state candidate identity/order coverage mismatch")
    elif observed_candidate_ids:
        raise ValueError("all-state metadata omits stored candidate readouts")
    if int(index.get("request_count", -1)) != len(expected_ids):
        raise ValueError("all-state index request count mismatch")
    if int(index.get("candidate_count", -1)) != total_candidates:
        raise ValueError("all-state index candidate count mismatch")
    if hidden_size is None or hidden_size <= 0:
        raise ValueError("all-state hidden size is invalid")
    return AllStateBundle(
        root=root,
        metadata=metadata,
        index=index,
        request_ids=tuple(observed_ids),
        candidate_count=total_candidates,
        hidden_size=hidden_size,
    )


def load_all_request_states(bundle: AllStateBundle, position: str) -> np.ndarray:
    """Load [request, state, hidden] for one causally registered position."""

    if position not in REQUEST_POSITIONS:
        raise ValueError(f"unsupported request position={position}")
    position_ordinal = REQUEST_POSITIONS.index(position)
    values: list[np.ndarray] = []
    for shard in bundle.index["shards"]:
        with np.load(
            bundle.root / "shards" / shard["path"], allow_pickle=False
        ) as payload:
            values.append(
                np.asarray(
                    payload["request_activations"][:, position_ordinal],
                    dtype=np.float32,
                )
            )
    result = np.concatenate(values, axis=0)
    expected = (
        len(bundle.request_ids),
        len(ALL_HIDDEN_STATE_INDICES),
        bundle.hidden_size,
    )
    if result.shape != expected:
        raise ValueError("loaded all-state request matrix has invalid shape")
    return result


def load_selected_candidate_states(
    bundle: AllStateBundle, candidate_ordinals: Sequence[int]
) -> np.ndarray:
    """Load one qrels-selected candidate per request after the qrels boundary."""

    ordinals = [int(value) for value in candidate_ordinals]
    if len(ordinals) != len(bundle.request_ids):
        raise ValueError("candidate ordinal/request count mismatch")
    result: list[np.ndarray] = []
    request_ordinal = 0
    for shard in bundle.index["shards"]:
        with np.load(
            bundle.root / "shards" / shard["path"], allow_pickle=False
        ) as payload:
            offsets = np.asarray(payload["candidate_offsets"], dtype=np.int64)
            candidate_values = np.asarray(
                payload["candidate_activations"], dtype=np.float32
            )
            for local in range(len(offsets) - 1):
                selected = ordinals[request_ordinal]
                width = int(offsets[local + 1] - offsets[local])
                if not 0 <= selected < width:
                    raise ValueError("selected candidate ordinal is outside slate")
                result.append(candidate_values[int(offsets[local]) + selected])
                request_ordinal += 1
    matrix = np.stack(result)
    expected = (
        len(bundle.request_ids),
        len(ALL_HIDDEN_STATE_INDICES),
        bundle.hidden_size,
    )
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise ValueError("selected candidate activation matrix is invalid")
    return matrix


def fit_deep_dive_representation_probes(
    standardized_dir: str | Path,
    activation_bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_method_id: str,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit all 3 positions x 29 states after a durable train-bundle audit."""

    standardized_dir = Path(standardized_dir)
    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    selected, selection = select_deep_dive_train_records(records_path)
    bundle = audit_all_state_bundle(
        activation_bundle_dir,
        expected_records=selected,
        expected_role="train_probe",
        expected_condition="full",
        require_result_eligible=True,
    )
    if bundle.metadata.get("method_id") != expected_method_id:
        raise ValueError("deep-dive train bundle method mismatch")
    if bundle.metadata.get("candidate_positions") != ["candidate_readout"]:
        raise ValueError("deep-dive train bundle lacks candidate readouts")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"deep-dive probe output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d1_train_probe_pre_qrels_integrity",
        "status": "passed",
        "qrels_read": False,
        "method_id": expected_method_id,
        "request_count": len(selected),
        "candidate_count": bundle.candidate_count,
        "bundle_metadata_sha256": sha256_file(bundle.root / "metadata.json"),
        "bundle_index_sha256": sha256_file(bundle.root / "index.json"),
        "records_train_sha256": sha256_file(records_path),
        "selection_audit": selection,
    }
    _write_json(output_dir / "pre_qrels_audit.json", pre_qrels)

    qrels_sha256 = sha256_file(qrels_path)
    qrels = _load_positive_qrels(qrels_path)
    labels, label_audit = build_preference_labels(
        selected,
        qrels,
        max_classes=M2_MAX_CLASSES,
        min_frequency=M2_MIN_CLASS_FREQUENCY,
    )
    candidate_ordinals = _target_candidate_ordinals(selected, qrels)
    features = {
        position: load_all_request_states(bundle, position)
        for position in REQUEST_POSITIONS
    }
    features["candidate_readout"] = load_selected_candidate_states(
        bundle, candidate_ordinals
    )
    holdout = np.asarray([representation_holdout(row.query) for row in selected])
    clusters = [normalize_query(row.query) for row in selected]
    if not holdout.any() or holdout.all():
        raise ValueError("deep-dive normalized-query split has an empty side")
    if {
        cluster for cluster, flag in zip(clusters, holdout) if flag
    } & {cluster for cluster, flag in zip(clusters, holdout) if not flag}:
        raise ValueError("deep-dive normalized-query clusters crossed split")

    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for position in ALL_POSITIONS:
        matrix = features[position]
        for task in ("brand", "category"):
            task_labels = labels[task]
            eligible = np.asarray(
                [value is not None for value in task_labels], dtype=bool
            )
            real = np.asarray([value or "" for value in task_labels], dtype=np.str_)
            random_values = _permuted_labels(real, eligible, task=task)
            for control, target in (
                ("real_labels", real),
                ("random_labels", random_values),
            ):
                train_mask = eligible & ~holdout
                test_mask = eligible & holdout
                for state in ALL_HIDDEN_STATE_INDICES:
                    readout = fit_linear_readout(
                        matrix[train_mask, state], target[train_mask]
                    )
                    prediction = readout.predict(matrix[test_mask, state])
                    key = f"{position}__{task}__state_{state}__{control}"
                    _store_readout(arrays, key, readout)
                    rows.append(
                        {
                            "position": position,
                            "task": task,
                            "hidden_state_index": state,
                            "label_control": control,
                            "embedding_state_negative_control": state == 0,
                            "train_requests": int(train_mask.sum()),
                            "holdout_requests": int(test_mask.sum()),
                            "holdout_accuracy": float(
                                np.mean(prediction == target[test_mask])
                            ),
                            "holdout_balanced_accuracy": _balanced_accuracy(
                                target[test_mask], prediction
                            ),
                        }
                    )
    weights_path = output_dir / "probe_weights.npz"
    temporary = output_dir / ".probe_weights.writing.npz"
    np.savez(temporary, **arrays)
    temporary.replace(weights_path)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d1_all_layer_representation",
        "artifact_role": "train_only_all_position_linear_probes",
        "method_id": expected_method_id,
        "activation_bundle_path": str(bundle.root),
        "activation_bundle_metadata_sha256": sha256_file(bundle.root / "metadata.json"),
        "activation_bundle_index_sha256": sha256_file(bundle.root / "index.json"),
        "pre_qrels_audit_sha256": sha256_file(output_dir / "pre_qrels_audit.json"),
        "records_train_sha256": sha256_file(records_path),
        "qrels_train_sha256": qrels_sha256,
        "qrels_read": "train_only_after_activation_bundle_integrity",
        "dev_qrels_read": False,
        "selection_audit": selection,
        "label_audit": label_audit,
        "positions": list(ALL_POSITIONS),
        "hidden_state_indices": list(ALL_HIDDEN_STATE_INDICES),
        "classifier": {
            "type": "standardized_multiclass_ridge_linear_readout",
            "alpha": 1.0,
            "solver": "lsqr",
            "tuning": False,
        },
        "negative_controls": ["random_labels", "embedding_state_index_0"],
        "probe_rows": rows,
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "command": list(command or []),
        "result_eligible": True,
        "status": "completed",
    }
    metadata["probe_checkpoint_id"] = (
        f"d1_{expected_method_id}@{metadata['weights_sha256'][:20]}"
    )
    _write_json(output_dir / "metadata.json", metadata)
    return metadata


def load_deep_dive_readout(
    model_dir: str | Path,
    *,
    position: str,
    task: str,
    hidden_state_index: int,
    label_control: str,
) -> LinearReadout:
    model_dir = Path(model_dir)
    metadata = _read_json(model_dir / "metadata.json")
    weights_path = model_dir / "probe_weights.npz"
    if metadata.get("weights_sha256") != sha256_file(weights_path):
        raise ValueError("deep-dive probe weights changed after fitting")
    if metadata.get("dev_qrels_read") is not False:
        raise ValueError("deep-dive fitted probe crossed dev qrels boundary")
    key = f"{position}__{task}__state_{int(hidden_state_index)}__{label_control}"
    with np.load(weights_path, allow_pickle=False) as payload:
        names = {
            field: f"{key}__{field}"
            for field in ("classes", "mean", "scale", "coefficient", "intercept")
        }
        if any(name not in payload.files for name in names.values()):
            raise ValueError(f"deep-dive probe weights omit {key}")
        return LinearReadout(
            classes=tuple(str(value) for value in payload[names["classes"]].tolist()),
            mean=np.asarray(payload[names["mean"]], dtype=np.float64),
            scale=np.asarray(payload[names["scale"]], dtype=np.float64),
            coefficient=np.asarray(payload[names["coefficient"]], dtype=np.float64),
            intercept=np.asarray(payload[names["intercept"]], dtype=np.float64),
        )


def select_deep_dive_train_records(
    records_path: str | Path,
) -> tuple[list[ModelRecord], dict[str, Any]]:
    from myrec.mechanism.representation_probe import select_train_probe_records

    return select_train_probe_records(iter_jsonl(records_path))


def _target_candidate_ordinals(
    records: Sequence[ModelRecord], qrels: Mapping[str, Mapping[str, float]]
) -> list[int]:
    result: list[int] = []
    for record in records:
        gains = qrels.get(record.request_id)
        if gains is None:
            raise ValueError("qrels lacks selected train request")
        values = [float(gains.get(str(row["item_id"]), 0.0)) for row in record.candidates]
        maximum = max(values, default=0.0)
        result.append(
            0
            if maximum <= 0
            else next(index for index, value in enumerate(values) if value == maximum)
        )
    return result


def _store_readout(arrays: dict[str, np.ndarray], key: str, readout: LinearReadout) -> None:
    arrays[f"{key}__classes"] = np.asarray(readout.classes, dtype=np.str_)
    arrays[f"{key}__mean"] = readout.mean.astype(np.float32)
    arrays[f"{key}__scale"] = readout.scale.astype(np.float32)
    arrays[f"{key}__coefficient"] = readout.coefficient.astype(np.float32)
    arrays[f"{key}__intercept"] = readout.intercept.astype(np.float32)


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    classes = np.unique(target)
    return float(
        np.mean([np.mean(prediction[target == value] == value) for value in classes])
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def analysis_identity(values: Sequence[str]) -> str:
    return sha256_text(json.dumps(list(values), separators=(",", ":")))
