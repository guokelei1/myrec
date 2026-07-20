"""Train-only, ID-free recoverability probe for Motivation mechanism analysis.

The probe is deliberately small and interpretable.  It learns a pairwise linear
ranker over frozen, label-free BGE similarities and visible brand/category
matches.  Training labels come only from ``qrels_train.jsonl``.  Scoring accepts
only the label-free internal-dev records; dev qrels remain behind the shared
mechanism evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.frozen_text_features import (
    FrozenTextFeatureStore,
    serialize_item_semantic_content,
)
from myrec.baselines.motivation_v12_contracts import (
    SERIALIZED_INPUT_FIELDS,
    ModelRecord,
    load_training_groups,
    pairwise_index_pairs,
    sanitize_record_for_model,
)
from myrec.baselines.representative_sequence_adapter import serialize_item_content
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json


METHOD_ID = "m0_bge_pairwise_transfer_probe"
SEED = 20260717
HISTORY_BUDGET = 6
MAX_PAIRS_PER_REQUEST = 16
ROUTING_TEMPERATURE = 5.0
PROBE_MANIFEST_PATH = Path("experiments/motivation/probe_manifest.yaml")
RUN_ID_PATTERN = re.compile(r"^\d{8}_[a-z0-9][a-z0-9_]*$")
FEATURE_NAMES = (
    "query_candidate_cosine",
    "routed_history_candidate_cosine",
    "max_history_candidate_cosine",
    "mean_history_candidate_cosine",
    "event_routed_history_candidate_cosine",
    "brand_match_any",
    "brand_match_routed",
    "deepest_category_match_any",
    "category_prefix_overlap_max",
    "category_prefix_overlap_routed",
)


@dataclass(frozen=True)
class FittedRecoverabilityProbe:
    coefficient: np.ndarray
    checkpoint_id: str
    model_sha256: str
    metadata: dict[str, Any]


class RecoverabilityFeatureExtractor:
    """Extract candidate features without serializing raw item identity."""

    def __init__(
        self,
        feature_store: FrozenTextFeatureStore,
        *,
        history_budget: int = HISTORY_BUDGET,
        routing_temperature: float = ROUTING_TEMPERATURE,
    ) -> None:
        if history_budget <= 0 or routing_temperature <= 0:
            raise ValueError("history budget and routing temperature must be positive")
        self.feature_store = feature_store
        self.history_budget = int(history_budget)
        self.routing_temperature = float(routing_temperature)

    def candidate_features(
        self,
        record: ModelRecord,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        routing_query: str | None = None,
    ) -> dict[str, np.ndarray]:
        selected = list(record.history if history is None else history)[
            -self.history_budget :
        ]
        query = record.query if routing_query is None else str(routing_query).strip()
        if not query:
            raise ValueError("routing query must be non-empty")
        actual_query_vector = _unit(self.feature_store(f"query: {record.query}"))
        routing_query_vector = _unit(self.feature_store(f"query: {query}"))
        candidate_vectors = np.stack(
            [
                _unit(self.feature_store(serialize_item_semantic_content(candidate)))
                for candidate in record.candidates
            ]
        )
        query_candidate = candidate_vectors @ actual_query_vector

        if selected:
            canonical_history = np.stack(
                [
                    _unit(self.feature_store(serialize_item_semantic_content(event)))
                    for event in selected
                ]
            )
            contextual_history = np.stack(
                [
                    _unit(self.feature_store(serialize_item_content(event)))
                    for event in selected
                ]
            )
            routing_logits = (
                contextual_history @ routing_query_vector * self.routing_temperature
            )
            routing_weights = _softmax(routing_logits)
            event_strength = np.asarray(
                [_event_strength(event.get("event")) for event in selected],
                dtype=np.float32,
            )
            event_weights = routing_weights * event_strength
            event_weights /= max(float(event_weights.sum()), 1.0e-12)
            routed_profile = _unit(routing_weights @ canonical_history)
            candidate_history = candidate_vectors @ canonical_history.T
            routed_similarity = candidate_vectors @ routed_profile
            max_similarity = candidate_history.max(axis=1)
            mean_similarity = candidate_history.mean(axis=1)
            event_similarity = candidate_history @ event_weights
        else:
            routing_weights = np.zeros(0, dtype=np.float32)
            routed_similarity = np.zeros(len(record.candidates), dtype=np.float32)
            max_similarity = np.zeros(len(record.candidates), dtype=np.float32)
            mean_similarity = np.zeros(len(record.candidates), dtype=np.float32)
            event_similarity = np.zeros(len(record.candidates), dtype=np.float32)

        rows: dict[str, np.ndarray] = {}
        for index, candidate in enumerate(record.candidates):
            if selected:
                brand_matches = np.asarray(
                    [_brand_match(candidate, event) for event in selected],
                    dtype=np.float32,
                )
                deepest_matches = np.asarray(
                    [_deepest_category_match(candidate, event) for event in selected],
                    dtype=np.float32,
                )
                category_overlaps = np.asarray(
                    [_category_prefix_overlap(candidate, event) for event in selected],
                    dtype=np.float32,
                )
                brand_any = float(brand_matches.max(initial=0.0))
                brand_routed = float(brand_matches @ routing_weights)
                deepest_any = float(deepest_matches.max(initial=0.0))
                category_max = float(category_overlaps.max(initial=0.0))
                category_routed = float(category_overlaps @ routing_weights)
            else:
                brand_any = brand_routed = deepest_any = 0.0
                category_max = category_routed = 0.0
            value = np.asarray(
                [
                    query_candidate[index],
                    routed_similarity[index],
                    max_similarity[index],
                    mean_similarity[index],
                    event_similarity[index],
                    brand_any,
                    brand_routed,
                    deepest_any,
                    category_max,
                    category_routed,
                ],
                dtype=np.float32,
            )
            if value.shape != (len(FEATURE_NAMES),) or not np.isfinite(value).all():
                raise FloatingPointError(
                    f"invalid recoverability features for {record.request_id}"
                )
            rows[str(candidate["item_id"])] = value
        return rows


def fit_recoverability_probe(
    standardized_dir: str | Path,
    feature_store_path: str | Path,
    output_model_dir: str | Path,
    run_id: str,
    *,
    runs_dir: str | Path = "runs",
    label_shuffle: bool = False,
    seed: int = SEED,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit the fixed transfer-only pairwise probe on train labels."""

    started = time.perf_counter()
    _validate_run_id(run_id)
    probe_manifest = _probe_manifest_identity()
    standardized_dir = Path(standardized_dir)
    _reject_non_train_dev_paths(standardized_dir)
    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    manifest_path = standardized_dir / "manifest.json"
    for path in (records_path, qrels_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    _verify_train_identity(
        standardized_dir,
        records_path=records_path,
        qrels_path=qrels_path,
        manifest_path=manifest_path,
    )
    output_model_dir = Path(output_model_dir)
    if output_model_dir.exists() and any(output_model_dir.iterdir()):
        raise FileExistsError(f"output model directory is not empty: {output_model_dir}")
    output_model_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"fit run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    store = FrozenTextFeatureStore(feature_store_path, require_fingerprints=True)
    if store.metadata.get("qrels_read") is not False:
        raise ValueError("frozen text store crossed the qrels boundary")
    extractor = RecoverabilityFeatureExtractor(store)
    groups, group_stats = load_training_groups(
        records_path,
        qrels_path,
        seed=seed,
        negatives_per_positive=2,
        max_group_size=8,
    )
    pair_features, pair_labels, pair_stats = _build_pairwise_dataset(
        groups,
        extractor,
        label_shuffle=label_shuffle,
        seed=seed,
        max_pairs_per_request=MAX_PAIRS_PER_REQUEST,
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=False)
    scaled = scaler.fit_transform(pair_features)
    model = LogisticRegression(
        C=1.0,
        fit_intercept=False,
        max_iter=500,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(scaled, pair_labels)
    coefficient = np.asarray(model.coef_[0] / scaler.scale_, dtype=np.float64)
    if coefficient.shape != (len(FEATURE_NAMES),) or not np.isfinite(coefficient).all():
        raise FloatingPointError("recoverability coefficient is invalid")
    model_path = output_model_dir / "model.npz"
    np.savez(
        model_path,
        coefficient=coefficient,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    model_sha256 = sha256_file(model_path)
    checkpoint_id = f"{METHOD_ID}@{model_sha256[:20]}"
    train_prediction = ((pair_features @ coefficient) > 0).astype(np.int64)
    train_accuracy = float((train_prediction == pair_labels).mean())
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": METHOD_ID,
        "checkpoint_id": checkpoint_id,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "probe_manifest": probe_manifest,
        "evidence_mode": "motivation_mechanism_train_only_recoverability_control",
        "feature_names": list(FEATURE_NAMES),
        "feature_store_path": str(feature_store_path),
        "feature_store_metadata_sha256": sha256_file(Path(feature_store_path) / "metadata.json"),
        "feature_store_fingerprint_sha256": store.store_fingerprint_sha256,
        "group_stats": group_stats,
        "pair_stats": pair_stats,
        "label_shuffle": bool(label_shuffle),
        "label_shuffle_scope": "within_request_candidate_gains" if label_shuffle else None,
        "history_budget": HISTORY_BUDGET,
        "max_pairs_per_request": MAX_PAIRS_PER_REQUEST,
        "routing_temperature": ROUTING_TEMPERATURE,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "records_train_path": str(records_path),
        "records_train_sha256": sha256_file(records_path),
        "qrels_train_path": str(qrels_path),
        "qrels_train_sha256": sha256_file(qrels_path),
        "qrels_train_read": True,
        "dev_or_confirmation_qrels_read": False,
        "seed": int(seed),
        "train_pair_accuracy": train_accuracy,
        "package_versions": _package_versions(),
        "job_boundary": {
            "max_continuous_seconds": 13500,
            "resumption": "deterministic_restart_from_frozen_inputs",
        },
    }
    write_json(output_model_dir / "training_metadata.json", metadata)
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def score_recoverability_probe(
    standardized_dir: str | Path,
    feature_store_path: str | Path,
    model_dir: str | Path,
    condition_id: str,
    run_id: str,
    *,
    runs_dir: str | Path = "runs",
    wrong_history_assignments: str | Path | None = None,
    seed: int = SEED,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score a label-free internal-dev condition without resolving dev qrels."""

    _validate_run_id(run_id)
    probe_manifest = _probe_manifest_identity()
    if condition_id not in {"full", "null", "history_shuffle", "routing_query_shuffle"}:
        raise ValueError(f"unsupported recoverability condition: {condition_id}")
    standardized_dir = Path(standardized_dir)
    _reject_non_train_dev_paths(standardized_dir)
    records_path = standardized_dir / "records_dev.jsonl"
    dataset_manifest_path = standardized_dir / "manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    _verify_internal_dev_identity(
        standardized_dir,
        records_path=records_path,
        dataset_manifest_path=dataset_manifest_path,
        candidate_manifest_path=candidate_manifest_path,
        request_manifest_path=request_manifest_path,
    )
    model = load_fitted_probe(model_dir)
    store = FrozenTextFeatureStore(feature_store_path, require_fingerprints=True)
    if store.metadata.get("qrels_read") is not False:
        raise ValueError("frozen text store crossed the qrels boundary")
    if (
        model.metadata.get("feature_store_fingerprint_sha256")
        != store.store_fingerprint_sha256
    ):
        raise ValueError("feature store differs from the fitted probe")
    extractor = RecoverabilityFeatureExtractor(store)

    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    wrong_histories: dict[str, list[dict[str, Any]]] | None = None
    assignment_sha: str
    if condition_id == "history_shuffle":
        if wrong_history_assignments is None:
            raise ValueError("history_shuffle requires wrong history assignments")
        wrong_path = Path(wrong_history_assignments)
        wrong_histories = _load_wrong_histories(wrong_path, raw_records, records)
        assignment_sha = sha256_file(wrong_path)
    elif condition_id == "null":
        assignment_sha = sha256_text("motivation-mechanism-null-history-v1")
    elif condition_id == "routing_query_shuffle":
        assignment_sha = sha256_text(
            json.dumps(
                _routing_query_map(records, seed=seed),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        assignment_sha = sha256_file(records_path)
    routing_queries = (
        _routing_query_map(records, seed=seed)
        if condition_id == "routing_query_shuffle"
        else {}
    )

    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"score run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    scores_path = run_dir / "scores.jsonl"
    request_ranges: list[float] = []
    score_rows = 0
    with scores_path.open("w", encoding="utf-8") as handle:
        for raw, record in zip(raw_records, records):
            if condition_id == "null":
                history: Sequence[Mapping[str, Any]] = ()
            elif wrong_histories is not None:
                history = wrong_histories[record.request_id]
            else:
                history = record.history
            features = extractor.candidate_features(
                record,
                history=history,
                routing_query=routing_queries.get(record.request_id),
            )
            values: list[float] = []
            for candidate in record.candidates:
                item_id = str(candidate["item_id"])
                score = float(features[item_id] @ model.coefficient)
                if not math.isfinite(score):
                    raise FloatingPointError(
                        f"non-finite probe score request={record.request_id} item={item_id}"
                    )
                values.append(score)
                handle.write(
                    json.dumps(
                        {
                            "candidate_item_id": item_id,
                            "method_id": METHOD_ID,
                            "request_id": record.request_id,
                            "score": score,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                score_rows += 1
            request_ranges.append(max(values) - min(values))

    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    base_signature = {
        "schema_version": 1,
        "probe": "pairwise_linear_visible_field_recoverability_v1",
        "checkpoint_id": model.checkpoint_id,
        "feature_names": list(FEATURE_NAMES),
        "feature_store_fingerprint_sha256": store.store_fingerprint_sha256,
        "history_budget": HISTORY_BUDGET,
        "routing_temperature": ROUTING_TEMPERATURE,
        "input_fields": list(SERIALIZED_INPUT_FIELDS),
        "probe_manifest_sha256": probe_manifest["sha256"],
    }
    metadata = {
        "schema_version": 1,
        "analysis_stage": "motivation_mechanism",
        "run_id": run_id,
        "method_id": METHOD_ID,
        "checkpoint_id": model.checkpoint_id,
        "condition_id": condition_id,
        "history_condition": condition_id,
        "history_assignment_sha256": assignment_sha,
        "base_scoring_signature": base_signature,
        "scoring_signature": base_signature,
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "request_manifest_path": str(request_manifest_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "dataset_id": str(dataset_manifest["dataset_id"]),
        "dataset_version": str(dataset_manifest["dataset_version"]),
        "split": "dev",
        "request_count": len(records),
        "score_rows": score_rows,
        "scores_sha256": sha256_file(scores_path),
        "score_non_degeneracy": {
            "nonconstant_requests_at_1e_8": sum(value > 1.0e-8 for value in request_ranges),
            "mean_request_range": float(np.mean(request_ranges)),
            "max_request_range": max(request_ranges),
        },
        "qrels_read": False,
        "qrels_train_used_by_fitted_checkpoint": True,
        "input_fields_used": list(SERIALIZED_INPUT_FIELDS),
        "raw_item_id_serialized_as_feature": False,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probe_manifest": probe_manifest,
        "evidence_mode": "motivation_mechanism_internal_dev",
        "model_metadata_path": str(Path(model_dir) / "training_metadata.json"),
        "model_sha256": model.model_sha256,
        "feature_store_path": str(feature_store_path),
        "feature_store_metadata_sha256": sha256_file(Path(feature_store_path) / "metadata.json"),
        "package_versions": _package_versions(),
    }
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def load_fitted_probe(model_dir: str | Path) -> FittedRecoverabilityProbe:
    model_dir = Path(model_dir)
    model_path = model_dir / "model.npz"
    metadata_path = model_dir / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observed_sha = sha256_file(model_path)
    if observed_sha != metadata.get("model_sha256"):
        raise ValueError("recoverability model changed after training")
    payload = np.load(model_path, allow_pickle=False)
    names = tuple(str(value) for value in payload["feature_names"].tolist())
    if names != FEATURE_NAMES:
        raise ValueError("recoverability feature order drifted")
    coefficient = np.asarray(payload["coefficient"], dtype=np.float64)
    checkpoint_id = f"{METHOD_ID}@{observed_sha[:20]}"
    if checkpoint_id != metadata.get("checkpoint_id"):
        raise ValueError("recoverability checkpoint identity drifted")
    return FittedRecoverabilityProbe(
        coefficient=coefficient,
        checkpoint_id=checkpoint_id,
        model_sha256=observed_sha,
        metadata=metadata,
    )


def _build_pairwise_dataset(
    groups: Sequence[Any],
    extractor: RecoverabilityFeatureExtractor,
    *,
    label_shuffle: bool,
    seed: int,
    max_pairs_per_request: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    eligible_groups = 0
    skipped_nontransfer = 0
    selected_pairs = 0
    for group in groups:
        history_ids = {str(row["item_id"]) for row in group.record.history}
        candidate_ids = {str(row["item_id"]) for row in group.record.candidates}
        if not history_ids or history_ids & candidate_ids:
            skipped_nontransfer += 1
            continue
        eligible_groups += 1
        by_item = extractor.candidate_features(group.record)
        gains = list(group.gains)
        if label_shuffle:
            random.Random(_stable_seed(seed, "label_shuffle", group.record.request_id)).shuffle(gains)
        pairs = pairwise_index_pairs(gains)
        random.Random(_stable_seed(seed, "pair_cap", group.record.request_id)).shuffle(pairs)
        pairs = pairs[:max_pairs_per_request]
        for high, low in pairs:
            high_id = str(group.candidates[high]["item_id"])
            low_id = str(group.candidates[low]["item_id"])
            difference = by_item[high_id] - by_item[low_id]
            features.extend((difference, -difference))
            labels.extend((1, 0))
            selected_pairs += 1
    if not features or len(set(labels)) != 2:
        raise ValueError("recoverability training constructed no usable pairs")
    matrix = np.stack(features).astype(np.float64, copy=False)
    target = np.asarray(labels, dtype=np.int64)
    return matrix, target, {
        "eligible_strict_transfer_groups": eligible_groups,
        "skipped_nontransfer_groups": skipped_nontransfer,
        "selected_unmirrored_pairs": selected_pairs,
        "fit_rows_after_mirroring": len(target),
        "label_shuffle": bool(label_shuffle),
    }


def _load_wrong_histories(
    path: Path,
    raw_records: Sequence[dict[str, Any]],
    records: Sequence[ModelRecord],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(path):
        request_id = str(row.get("request_id") or "")
        history = row.get("history")
        if not request_id or not isinstance(history, list) or request_id in rows:
            raise ValueError("invalid wrong-history assignment")
        rows[request_id] = history
    expected = {record.request_id for record in records}
    if set(rows) != expected:
        raise ValueError("wrong-history assignment request coverage differs")
    for raw, record in zip(raw_records, records):
        candidate_ids = {str(row["item_id"]) for row in record.candidates}
        history = rows[record.request_id]
        if any(str(event.get("item_id")) in candidate_ids for event in history):
            raise ValueError("wrong history leaks a current candidate")
        if any(int(event.get("ts", raw["ts"])) >= int(raw["ts"]) for event in history):
            raise ValueError("wrong history is not causal")
        sanitize_record_for_model({**raw, "history": history})
    return rows


def _routing_query_map(records: Sequence[ModelRecord], *, seed: int) -> dict[str, str]:
    ordered = sorted(
        records,
        key=lambda record: (_stable_seed(seed, "routing_query", record.request_id), record.request_id),
    )
    if len(ordered) < 2:
        raise ValueError("routing-query shuffle requires at least two requests")
    queries = [record.query for record in ordered]
    result: dict[str, str] = {}
    for index, record in enumerate(ordered):
        offset = 1
        donor = queries[(index + offset) % len(queries)]
        while donor == record.query and offset < len(queries):
            offset += 1
            donor = queries[(index + offset) % len(queries)]
        result[record.request_id] = donor
    return result


def _verify_internal_dev_identity(
    standardized_dir: Path,
    *,
    records_path: Path,
    dataset_manifest_path: Path,
    candidate_manifest_path: Path,
    request_manifest_path: Path,
) -> None:
    import yaml

    protocol_path = Path("experiments/motivation/protocol.yaml")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    development = protocol["data"]["development_population"]
    expected_dir = Path(str(development["standardized_dir"])).resolve()
    if standardized_dir.resolve() != expected_dir:
        raise ValueError("recoverability scoring is restricted to frozen v11 development data")
    expected = {
        records_path: development["records_dev_sha256"],
        dataset_manifest_path: development["manifest_sha256"],
        candidate_manifest_path: development["candidate_manifest_sha256"],
        request_manifest_path: development["request_manifest_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != str(digest):
            raise ValueError(f"frozen internal-dev identity mismatch: {path}")


def _verify_train_identity(
    standardized_dir: Path,
    *,
    records_path: Path,
    qrels_path: Path,
    manifest_path: Path,
) -> None:
    import yaml

    protocol_path = Path("experiments/motivation/protocol.yaml")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    development = protocol["data"]["development_population"]
    if standardized_dir.resolve() != Path(development["standardized_dir"]).resolve():
        raise ValueError("recoverability fitting is restricted to frozen v11 train")
    expected = {
        records_path: development["records_train_sha256"],
        qrels_path: development["qrels_train_sha256"],
        manifest_path: development["manifest_sha256"],
    }
    # Training qrels are opened only after their frozen identity is checked.
    for path, digest in expected.items():
        if sha256_file(path) != str(digest):
            raise ValueError(f"frozen train identity mismatch: {path}")


def _reject_non_train_dev_paths(path: Path) -> None:
    lowered = str(path).lower()
    if "test" in lowered or "newholdout" in lowered:
        raise ValueError("source test/new holdout is closed for mechanism development")


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value.astype(np.float64) - float(np.max(value))
    result = np.exp(shifted)
    result /= result.sum()
    return result.astype(np.float32)


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("frozen text feature has zero/non-finite norm")
    return value / norm


def _event_strength(value: Any) -> float:
    text = str(value or "").lower()
    return 2.0 if "purchase" in text or "buy" in text else 1.0


def _brand_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = str(left.get("brand") or "").strip().casefold()
    b = str(right.get("brand") or "").strip().casefold()
    return float(bool(a and b and a == b))


def _categories(value: Mapping[str, Any]) -> tuple[str, ...]:
    raw = value.get("cat") or []
    return tuple(str(row).strip().casefold() for row in raw if str(row).strip())


def _deepest_category_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a, b = _categories(left), _categories(right)
    return float(bool(a and b and a[-1] == b[-1]))


def _category_prefix_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a, b = _categories(left), _categories(right)
    if not a or not b:
        return 0.0
    matched = 0
    for x, y in zip(a, b):
        if x != y:
            break
        matched += 1
    return matched / max(len(a), len(b))


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run id must use YYYYMMDD_<dataset>_<method>_<purpose>")


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    import sklearn

    return {
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }


def _probe_manifest_identity() -> dict[str, str]:
    import yaml

    payload = yaml.safe_load(PROBE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("probe_manifest_id") != "motivation_mechanism_first_diagnosis_v1":
        raise ValueError("unexpected mechanism probe manifest")
    if payload.get("status") != "frozen_before_mechanism_outcomes":
        raise ValueError("mechanism probe manifest is not frozen")
    return {"path": str(PROBE_MANIFEST_PATH), "sha256": sha256_file(PROBE_MANIFEST_PATH)}


def implementation_identity() -> dict[str, str]:
    path = Path(__file__)
    return {"path": str(path), "sha256": sha256_file(path)}
