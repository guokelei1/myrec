"""Train-only semantic recoverability witness for strict-nonrepeat history."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from myrec.baselines.core import document_text, tokenize_text
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


FEATURE_VERSION = "bge_small_zh_histgb_v1"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
BASE_FEATURE_NAMES = (
    "dense_query_candidate",
    "char_ngram_jaccard",
    "query_ngram_coverage",
    "query_exact_substring",
    "candidate_log_text_length",
)
HISTORY_FEATURE_NAMES = (
    "history_present",
    "history_log_length",
    "history_candidate_max",
    "history_candidate_mean",
    "history_candidate_recency",
    "history_candidate_latest",
    "history_candidate_query_conditioned",
    "history_query_candidate_bridge_max",
    "history_candidate_top_query_event",
    "history_candidate_click_max",
    "history_candidate_purchase_max",
    "history_profile_mean",
    "history_profile_recency",
    "history_cat_l1_max",
    "history_cat_l2_max",
    "history_cat_l3_max",
    "history_cat_recency",
    "history_brand_any",
    "history_brand_recency",
    "query_history_max",
)
FULL_FEATURE_NAMES = BASE_FEATURE_NAMES + HISTORY_FEATURE_NAMES


class EmbeddingStore:
    def __init__(self, index: dict[str, int], values: np.ndarray):
        self.index = index
        self.values = values

    def query(self, text: str) -> np.ndarray:
        return self.values[self.index[_query_key(text)]]

    def document(self, row: dict[str, Any]) -> np.ndarray:
        return self.values[self.index[_document_key(document_text(row))]]


def ensure_embedding_cache(
    records_paths: Iterable[str | Path],
    cache_dir: str | Path,
    *,
    model_name: str = "BAAI/bge-small-zh-v1.5",
    device: str = "cuda:2",
    batch_size: int = 512,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Encode unique train/dev information objects without reading qrels."""

    records_paths = [Path(path) for path in records_paths]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    values_path = cache_dir / "embeddings.npy"
    metadata_path = cache_dir / "metadata.json"
    source_hashes = {str(path): sha256_file(path) for path in records_paths}
    if index_path.exists() and values_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            metadata.get("feature_version") == FEATURE_VERSION
            and metadata.get("model_name") == model_name
            and metadata.get("source_hashes") == source_hashes
        ):
            return metadata

    keyed_text: dict[str, str] = {}
    for records_path in records_paths:
        for record in iter_jsonl(records_path):
            query = str(record.get("query") or "")
            keyed_text[_query_key(query)] = QUERY_INSTRUCTION + query
            for candidate in record.get("candidates", []):
                text = document_text(candidate)
                keyed_text[_document_key(text)] = text
            for event in record.get("history", []):
                text = document_text(event)
                keyed_text[_document_key(text)] = text
    keys = sorted(keyed_text)
    texts = [keyed_text[key] for key in keys]

    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(
        model_name,
        device=device,
        local_files_only=local_files_only,
    )
    values = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float16)
    np.save(values_path, values)
    index = {key: index for index, key in enumerate(keys)}
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, sort_keys=True)
    metadata = {
        "device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "embedding_dimension": int(values.shape[1]),
        "feature_version": FEATURE_VERSION,
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "local_files_only": local_files_only,
        "model_name": model_name,
        "rows": int(values.shape[0]),
        "source_hashes": source_hashes,
        "values_path": str(values_path),
        "values_sha256": sha256_file(values_path),
    }
    write_json(metadata_path, metadata)
    return metadata


def load_embedding_store(cache_dir: str | Path) -> EmbeddingStore:
    cache_dir = Path(cache_dir)
    with (cache_dir / "index.json").open("r", encoding="utf-8") as handle:
        index = {str(key): int(value) for key, value in json.load(handle).items()}
    values = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
    return EmbeddingStore(index, values)


def candidate_features(
    record: dict[str, Any],
    candidate: dict[str, Any],
    history: list[dict[str, Any]],
    store: EmbeddingStore,
    *,
    history_budget: int = 6,
) -> np.ndarray:
    query_text = str(record.get("query") or "")
    candidate_text = document_text(candidate)
    query_embedding = np.asarray(store.query(query_text), dtype=np.float32)
    candidate_embedding = np.asarray(store.document(candidate), dtype=np.float32)
    query_terms = set(tokenize_text(query_text, mode="cjk_2_3gram"))
    candidate_terms = set(tokenize_text(candidate_text, mode="cjk_2_3gram"))
    union = query_terms | candidate_terms
    compact_query = "".join(query_text.lower().split())
    compact_candidate = "".join(candidate_text.lower().split())
    base = [
        float(query_embedding @ candidate_embedding),
        len(query_terms & candidate_terms) / len(union) if union else 0.0,
        len(query_terms & candidate_terms) / len(query_terms) if query_terms else 0.0,
        float(bool(compact_query and compact_query in compact_candidate)),
        math.log1p(len(candidate_text)),
    ]

    selected = list(history[-history_budget:]) if history_budget else []
    if not selected:
        return np.asarray(base + [0.0] * len(HISTORY_FEATURE_NAMES), dtype=np.float32)
    history_embeddings = np.stack(
        [np.asarray(store.document(event), dtype=np.float32) for event in selected]
    )
    history_candidate = history_embeddings @ candidate_embedding
    query_history = history_embeddings @ query_embedding
    recency = np.arange(1, len(selected) + 1, dtype=np.float32)
    recency /= recency.sum()
    conditioned = np.exp(5.0 * (query_history - float(query_history.max())))
    conditioned /= conditioned.sum()
    click_values = [
        float(history_candidate[index])
        for index, event in enumerate(selected)
        if str(event.get("event")) == "click"
    ]
    purchase_values = [
        float(history_candidate[index])
        for index, event in enumerate(selected)
        if str(event.get("event")) == "purchase"
    ]
    cat_matches = np.asarray(
        [_category_matches(candidate, event) for event in selected],
        dtype=np.float32,
    )
    brand_matches = np.asarray(
        [_brand_match(candidate, event) for event in selected],
        dtype=np.float32,
    )
    history_mean = history_embeddings.mean(axis=0)
    history_recency = recency @ history_embeddings
    top_query_index = int(np.argmax(query_history))
    history_features = [
        1.0,
        math.log1p(len(selected)),
        float(history_candidate.max()),
        float(history_candidate.mean()),
        float(recency @ history_candidate),
        float(history_candidate[-1]),
        float(conditioned @ history_candidate),
        float(np.max(query_history * history_candidate)),
        float(history_candidate[top_query_index]),
        max(click_values) if click_values else 0.0,
        max(purchase_values) if purchase_values else 0.0,
        float(history_mean @ candidate_embedding),
        float(history_recency @ candidate_embedding),
        float(cat_matches[:, 0].max()),
        float(cat_matches[:, 1].max()),
        float(cat_matches[:, 2].max()),
        float(recency @ cat_matches[:, 2]),
        float(brand_matches.max()),
        float(recency @ brand_matches),
        float(query_history.max()),
    ]
    return np.asarray(base + history_features, dtype=np.float32)


def train_recoverability_witness(
    standardized_dir: str | Path,
    output_model_dir: str | Path,
    run_id: str,
    *,
    runs_dir: str | Path = "runs",
    embedding_cache_dir: str | Path,
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
    embedding_device: str = "cuda:2",
    history_budget: int = 6,
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    seed: int = 20260714,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit nested base/full probes using strict-nonrepeat train clicks only."""

    from joblib import dump
    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import HistGradientBoostingClassifier

    standardized_dir = Path(standardized_dir)
    output_model_dir = Path(output_model_dir)
    run_dir = Path(runs_dir) / run_id
    if output_model_dir.exists() and any(output_model_dir.iterdir()):
        raise FileExistsError(f"model directory is not empty: {output_model_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    output_model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    records_train = standardized_dir / "records_train.jsonl"
    records_dev = standardized_dir / "records_dev.jsonl"
    qrels_train = standardized_dir / "qrels_train.jsonl"
    embedding_metadata = ensure_embedding_cache(
        [records_train, records_dev],
        embedding_cache_dir,
        model_name=embedding_model_name,
        device=embedding_device,
    )
    store = load_embedding_store(embedding_cache_dir)
    labels = {
        str(row["request_id"]): {str(item) for item in row.get("clicked", [])}
        for row in iter_jsonl(qrels_train)
    }
    full_features = []
    targets = []
    sample_weights = []
    selected_requests = 0
    skipped_no_click = 0
    skipped_not_strict = 0
    started = time.perf_counter()
    for record in iter_jsonl(records_train):
        request_id = str(record["request_id"])
        if not bool(record.get("masks", {}).get("strict_nonrepeat")):
            skipped_not_strict += 1
            continue
        positives = labels[request_id]
        candidates = list(record["candidates"])
        positive_count = sum(str(row["item_id"]) in positives for row in candidates)
        negative_count = len(candidates) - positive_count
        if positive_count == 0 or negative_count == 0:
            skipped_no_click += 1
            continue
        selected_requests += 1
        for candidate in candidates:
            target = int(str(candidate["item_id"]) in positives)
            full_features.append(
                candidate_features(
                    record,
                    candidate,
                    list(record.get("history", [])),
                    store,
                    history_budget=history_budget,
                )
            )
            targets.append(target)
            sample_weights.append(
                0.5 / positive_count if target else 0.5 / negative_count
            )
    x_full = np.stack(full_features)
    x_base = x_full[:, : len(BASE_FEATURE_NAMES)]
    y = np.asarray(targets, dtype=np.int8)
    weights = np.asarray(sample_weights, dtype=np.float64)
    common = {
        "early_stopping": False,
        "l2_regularization": l2_regularization,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "random_state": seed,
    }
    base_model = HistGradientBoostingClassifier(**common).fit(
        x_base, y, sample_weight=weights
    )
    full_model = HistGradientBoostingClassifier(**common).fit(
        x_full, y, sample_weight=weights
    )
    bundle_path = output_model_dir / "witness.joblib"
    dump(
        {
            "base_feature_names": BASE_FEATURE_NAMES,
            "base_model": base_model,
            "embedding_cache_dir": str(embedding_cache_dir),
            "embedding_model_name": embedding_model_name,
            "feature_version": FEATURE_VERSION,
            "full_feature_names": FULL_FEATURE_NAMES,
            "full_model": full_model,
            "history_budget": history_budget,
            "seed": seed,
        },
        bundle_path,
    )
    checkpoint_id = f"recoverability-witness@{sha256_file(bundle_path)[:20]}"
    metadata = {
        "checkpoint_id": checkpoint_id,
        "config_path": str(config_path) if config_path else None,
        "dev_labels_read": False,
        "elapsed_seconds": time.perf_counter() - started,
        "embedding_cache": embedding_metadata,
        "feature_version": FEATURE_VERSION,
        "input_fields_used": [
            "query",
            "history.title",
            "history.brand",
            "history.cat",
            "history.event",
            "candidates.title",
            "candidates.brand",
            "candidates.cat",
        ],
        "learner": "sklearn.HistGradientBoostingClassifier",
        "objective": "request-balanced candidate click classification",
        "output_model_dir": str(output_model_dir),
        "qrels_train_sha256": sha256_file(qrels_train),
        "records_train_sha256": sha256_file(records_train),
        "run_id": run_id,
        "seed": seed,
        "sklearn_version": sklearn_version,
        "strict_nonrepeat_only": True,
        "training": {
            **common,
            "candidate_rows": len(targets),
            "positive_rows": int(y.sum()),
            "selected_requests": selected_requests,
            "skipped_no_click": skipped_no_click,
            "skipped_not_strict": skipped_not_strict,
        },
        "training_labels_path": str(qrels_train),
        "training_labels_read": True,
        "witness_role": "recoverability_diagnostic_not_a_baseline_or_proposed_system",
        "witness_sha256": sha256_file(bundle_path),
    }
    write_json(output_model_dir / "metadata.json", metadata)
    write_json(run_dir / "metadata.json", metadata)
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")
    return metadata


def write_recoverability_witness_scores(
    standardized_dir: str | Path,
    model_dir: str | Path,
    run_id: str,
    *,
    mode: str,
    history_condition: str,
    history_assignments_path: str | Path | None = None,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score dev without reading dev qrels."""

    from joblib import load

    if mode not in {"base", "full"}:
        raise ValueError(f"unsupported mode={mode}")
    if history_condition not in {"base", "true", "null", "wrong"}:
        raise ValueError(f"unsupported history_condition={history_condition}")
    if mode == "full" and history_condition == "base":
        raise ValueError("full mode requires true/null/wrong history")
    standardized_dir = Path(standardized_dir)
    model_dir = Path(model_dir)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / "witness.joblib"
    bundle = load(bundle_path)
    store = load_embedding_store(bundle["embedding_cache_dir"])
    assignments = None
    if mode == "full":
        if history_assignments_path is None:
            raise ValueError("full mode requires history assignments")
        assignments = {
            str(row["request_id"]): list(row.get("history", []))
            for row in iter_jsonl(history_assignments_path)
        }
    records_path = standardized_dir / "records_dev.jsonl"
    rows = 0
    requests = 0
    method_id = f"recoverability_witness_{mode}"
    scores_path = run_dir / "scores.jsonl"
    feature_rows = []
    output_keys = []
    for record in iter_jsonl(records_path):
        request_id = str(record["request_id"])
        requests += 1
        history = assignments[request_id] if assignments is not None else []
        for candidate in record["candidates"]:
            feature_rows.append(
                candidate_features(
                    record,
                    candidate,
                    history,
                    store,
                    history_budget=int(bundle["history_budget"]),
                )
            )
            output_keys.append((request_id, str(candidate["item_id"])))
    x_full = np.stack(feature_rows)
    if mode == "base":
        x = x_full[:, : len(BASE_FEATURE_NAMES)]
        model = bundle["base_model"]
    else:
        x = x_full
        model = bundle["full_model"]
    scores = model.predict_proba(x)[:, 1]
    with scores_path.open("w", encoding="utf-8") as handle:
        for (request_id, candidate_item_id), score in zip(output_keys, scores):
            handle.write(
                json.dumps(
                    {
                        "candidate_item_id": candidate_item_id,
                        "method_id": method_id,
                        "request_id": request_id,
                        "score": float(score),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            rows += 1
    with (model_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        training_metadata = json.load(handle)
    candidate_manifest = standardized_dir / "candidate_manifest.json"
    request_manifest = standardized_dir / "request_manifest.json"
    scoring_signature = {
        "embedding_model_name": bundle["embedding_model_name"],
        "feature_version": bundle["feature_version"],
        "history_budget": int(bundle["history_budget"]),
        "learner": "sklearn.HistGradientBoostingClassifier",
        "mode": mode,
    }
    metadata = {
        "candidate_manifest_path": str(candidate_manifest),
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "checkpoint_id": training_metadata["checkpoint_id"],
        "config_path": str(config_path) if config_path else None,
        "dataset_id": "kuaisearch",
        "dataset_version": "lite_scout10k_v1",
        "history_assignment_sha256": (
            sha256_file(history_assignments_path)
            if history_assignments_path is not None
            else None
        ),
        "history_assignments_path": (
            str(history_assignments_path)
            if history_assignments_path is not None
            else None
        ),
        "history_condition": history_condition,
        "method_id": method_id,
        "qrels_read": False,
        "request_count": requests,
        "request_manifest_sha256": sha256_file(request_manifest),
        "run_id": run_id,
        "score_rows": rows,
        "scoring_signature": scoring_signature,
        "split": "dev",
        "standardized_dir": str(standardized_dir),
        "witness_role": "recoverability_diagnostic_not_a_baseline_or_proposed_system",
    }
    write_json(run_dir / "metadata.json", metadata)
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")
    return metadata


def _query_key(text: str) -> str:
    return "q:" + text


def _document_key(text: str) -> str:
    return "d:" + text


def _category_matches(
    candidate: dict[str, Any], event: dict[str, Any]
) -> tuple[float, float, float]:
    left = [str(value) for value in candidate.get("cat", [])]
    right = [str(value) for value in event.get("cat", [])]
    return tuple(
        float(
            index < len(left)
            and index < len(right)
            and bool(left[index])
            and left[index].upper() != "UNKNOWN"
            and left[index] == right[index]
        )
        for index in range(3)
    )


def _brand_match(candidate: dict[str, Any], event: dict[str, Any]) -> float:
    invalid = {"", "无品牌", "UNKNOWN"}
    left = str(candidate.get("brand") or "")
    right = str(event.get("brand") or "")
    return float(left not in invalid and left == right)
