"""M4 request-level feature construction for PPS motivation audits."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from myrec.baselines.core import document_text, tokenize_text
from myrec.utils.jsonl import iter_jsonl


UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TrainStats:
    query_freq: dict[str, int]
    query_click_entropy: dict[str, float]
    query_term_df: dict[str, int]
    train_request_count: int


@dataclass(frozen=True)
class TrainSubset:
    request_ids: list[str]
    eligible_count: int
    sample_size: int
    seed: int
    filters: dict[str, int]


def build_train_stats(records_train_path: str | Path) -> TrainStats:
    query_freq: Counter[str] = Counter()
    query_term_df: Counter[str] = Counter()
    query_item_clicks: dict[str, Counter[str]] = defaultdict(Counter)
    train_request_count = 0
    for record in iter_jsonl(records_train_path):
        train_request_count += 1
        query = str(record.get("query") or "")
        query_freq[query] += 1
        terms = set(tokenize_text(query, mode="cjk_2_3gram"))
        query_term_df.update(terms)
        for candidate in record.get("candidates", []):
            if int(candidate.get("clicked", 0) or 0) > 0:
                query_item_clicks[query][str(candidate["item_id"])] += 1
    query_click_entropy = {
        query: _entropy(counts.values()) for query, counts in query_item_clicks.items()
    }
    return TrainStats(
        query_freq=dict(query_freq),
        query_click_entropy=query_click_entropy,
        query_term_df=dict(query_term_df),
        train_request_count=train_request_count,
    )


def sample_train_subset(
    records_train_path: str | Path,
    sample_size: int,
    seed: int,
) -> TrainSubset:
    rng = random.Random(seed)
    reservoir: list[str] = []
    eligible_count = 0
    filters = Counter()
    for record in iter_jsonl(records_train_path):
        request_id = str(record["request_id"])
        candidates = record.get("candidates", [])
        if len(candidates) < 5:
            filters["candidate_count_lt_5"] += 1
            continue
        positives = sum(int(c.get("clicked", 0) or 0) > 0 for c in candidates)
        if positives < 1:
            filters["no_clicked_positive"] += 1
            continue
        eligible_count += 1
        if len(reservoir) < sample_size:
            reservoir.append(request_id)
        else:
            index = rng.randrange(eligible_count)
            if index < sample_size:
                reservoir[index] = request_id
    reservoir.sort()
    return TrainSubset(
        request_ids=reservoir,
        eligible_count=eligible_count,
        sample_size=len(reservoir),
        seed=seed,
        filters=dict(filters),
    )


def build_feature_frame(
    records_path: str | Path,
    stats: TrainStats,
    split: str,
    request_ids: set[str] | None = None,
    semantic_model_name: str = "BAAI/bge-small-zh-v1.5",
    cache_folder: str | Path = "models/huggingface/sentence_transformers",
    device: str = "cuda:0",
    batch_size: int = 512,
    max_seq_length: int = 256,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = _load_feature_records(records_path, stats, request_ids=request_ids)
    semantic_meta = _attach_history_query_similarity(
        records=records,
        model_name=semantic_model_name,
        cache_folder=cache_folder,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    frame = pd.DataFrame([record["features"] for record in records])
    frame.insert(0, "request_id", [record["request_id"] for record in records])
    frame.insert(1, "split", split)
    return frame, {
        "feature_rows": len(frame),
        "semantic_similarity": semantic_meta,
    }


def _load_feature_records(
    records_path: str | Path,
    stats: TrainStats,
    request_ids: set[str] | None,
) -> list[dict[str, Any]]:
    rows = []
    for record in iter_jsonl(records_path):
        request_id = str(record["request_id"])
        if request_ids is not None and request_id not in request_ids:
            continue
        query = str(record.get("query") or "")
        candidates = record.get("candidates", [])
        history = record.get("history", [])
        query_terms = tokenize_text(query, mode="cjk_2_3gram")
        query_known = query in stats.query_freq
        query_entropy_known = query in stats.query_click_entropy
        features = {
            "query_len_chars": len(query),
            "query_len_terms": len(query_terms),
            "query_avg_idf": _avg_query_idf(query_terms, stats),
            "query_avg_idf_missing": int(not query_terms),
            "query_train_freq": stats.query_freq.get(query, 0),
            "query_train_missing": int(not query_known),
            "query_click_entropy": stats.query_click_entropy.get(query, 0.0),
            "query_click_entropy_missing": int(not query_entropy_known),
            "candidate_count": len(candidates),
            "candidate_cat_entropy": _entropy(_counter_values(_deepest_cats(candidates))),
            "candidate_brand_entropy": _entropy(_counter_values(_brands(candidates))),
            "history_length": len(history),
            "history_query_semantic_sim": np.nan,
            "history_query_semantic_sim_missing": int(len(history) == 0 or not query.strip()),
            "history_candidate_cat_overlap": _history_candidate_cat_overlap(history, candidates),
        }
        rows.append(
            {
                "features": features,
                "history_texts": [document_text(item) for item in history],
                "query": query,
                "request_id": request_id,
            }
        )
    return rows


def _attach_history_query_similarity(
    records: list[dict[str, Any]],
    model_name: str,
    cache_folder: str | Path,
    device: str,
    batch_size: int,
    max_seq_length: int,
) -> dict[str, Any]:
    pending = [
        record
        for record in records
        if record["history_texts"] and str(record["query"]).strip()
    ]
    if not pending:
        return {
            "computed": False,
            "model_name": model_name,
            "records_with_similarity": 0,
            "reason": "no records with both query and history",
        }

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=str(cache_folder), device=device)
    model.max_seq_length = max_seq_length
    queries = [record["query"] for record in pending]
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    unique_texts = sorted({text for record in pending for text in record["history_texts"] if text})
    item_embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    text_index = {text: index for index, text in enumerate(unique_texts)}
    for row_index, record in enumerate(pending):
        indices = [text_index[text] for text in record["history_texts"] if text in text_index]
        if not indices:
            continue
        mean_embedding = item_embeddings[np.asarray(indices)].mean(axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
            sim = float(mean_embedding @ query_embeddings[row_index])
            record["features"]["history_query_semantic_sim"] = sim
            record["features"]["history_query_semantic_sim_missing"] = 0
    return {
        "computed": True,
        "model_name": model_name,
        "records_with_similarity": len(pending),
        "unique_history_texts_encoded": len(unique_texts),
        "device": device,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
    }


def _avg_query_idf(terms: list[str], stats: TrainStats) -> float:
    if not terms:
        return 0.0
    values = []
    n = stats.train_request_count
    for term in terms:
        df = stats.query_term_df.get(term, 0)
        values.append(math.log((n + 1.0) / (df + 1.0)) + 1.0)
    return sum(values) / len(values)


def _deepest_cats(items: list[dict[str, Any]]) -> list[str]:
    values = []
    for item in items:
        cats = [str(value) for value in item.get("cat", []) if _valid_value(value)]
        values.append(cats[-1] if cats else UNKNOWN)
    return values


def _brands(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("brand") or UNKNOWN) for item in items]


def _history_candidate_cat_overlap(
    history: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> float:
    history_cats = {cat for cat in _deepest_cats(history) if cat != UNKNOWN}
    candidate_cats = {cat for cat in _deepest_cats(candidates) if cat != UNKNOWN}
    if not history_cats or not candidate_cats:
        return 0.0
    return len(history_cats & candidate_cats) / len(history_cats | candidate_cats)


def _counter_values(values: list[str]) -> list[int]:
    return list(Counter(values).values())


def _entropy(counts: Any) -> float:
    counts = [float(value) for value in counts if float(value) > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in counts)


def _valid_value(value: Any) -> bool:
    text = str(value) if value is not None else ""
    return bool(text and text.upper() != UNKNOWN)
