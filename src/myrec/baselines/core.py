"""Self-implemented PPS Batch 1 baseline scorers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


ScoreRecordFn = Callable[[dict[str, Any]], dict[str, float]]

_ASCII_RE = re.compile(r"[a-z0-9]+")


def write_source_order_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score candidates by their preserved source-slate position."""

    standardized_dir = Path(standardized_dir)

    def score_record(record: dict[str, Any]) -> dict[str, float]:
        size = len(record["candidates"])
        return {
            str(candidate["item_id"]): float(size - index)
            for index, candidate in enumerate(record["candidates"])
        }

    return _write_scores_from_records(
        standardized_dir=standardized_dir,
        split=split,
        run_id=run_id,
        method_id="source_order",
        runs_dir=runs_dir,
        config_path=config_path,
        score_record=score_record,
        metadata_extra={
            "input_fields_used": ["candidate source position"],
            "score_definition": "candidate_count - zero_based_source_position",
        },
    )


def write_popularity_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    runs_dir: str | Path = "runs",
    artifacts_dir: str | Path = "artifacts/baselines",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    standardized_dir = Path(standardized_dir)
    artifacts_dir = Path(artifacts_dir)
    _, dataset_version = _dataset_identity(standardized_dir)
    stats_dir = artifacts_dir / _safe_path_component(dataset_version)
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "popularity_stats.jsonl"
    if stats_path.exists():
        click_counts = _load_popularity_stats(stats_path)
    else:
        click_counts = _build_popularity_stats(standardized_dir / "records_train.jsonl", stats_path)

    def score_record(record: dict[str, Any]) -> dict[str, float]:
        return {
            str(candidate["item_id"]): math.log1p(click_counts.get(str(candidate["item_id"]), 0))
            for candidate in record["candidates"]
        }

    return _write_scores_from_records(
        standardized_dir=standardized_dir,
        split=split,
        run_id=run_id,
        method_id="b0a_popularity",
        runs_dir=runs_dir,
        config_path=config_path,
        score_record=score_record,
        metadata_extra={
            "input_fields_used": ["records_train.candidates.clicked", "records_<split>.candidates.item_id"],
            "popularity_stats_path": str(stats_path),
            "popularity_stats_sha256": sha256_file(stats_path),
            "score_definition": "log1p(train_clicked_candidate_count)",
        },
    )


def write_recent_behavior_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    standardized_dir = Path(standardized_dir)

    def score_record(record: dict[str, Any]) -> dict[str, float]:
        return recent_behavior_scores(record)

    return _write_scores_from_records(
        standardized_dir=standardized_dir,
        split=split,
        run_id=run_id,
        method_id="b0b_recent_behavior",
        runs_dir=runs_dir,
        config_path=config_path,
        score_record=score_record,
        metadata_extra={
            "input_fields_used": ["history.item_id", "history.cat", "history.event", "candidates.item_id", "candidates.cat"],
            "score_definition": (
                "sum over history: recency_decay * event_weight * "
                "(3.0 item_match + 1.0 cat_l3 + 0.5 cat_l2 + 0.2 cat_l1)"
            ),
            "weights": {
                "cat_l1": 0.2,
                "cat_l2": 0.5,
                "cat_l3": 1.0,
                "click_event": 1.0,
                "item_match": 3.0,
                "purchase_event": 1.5,
                "recency_decay": "1/sqrt(reverse_age), most recent reverse_age=1",
            },
        },
    )


def recent_behavior_scores(record: dict[str, Any]) -> dict[str, float]:
    """Score one record with the frozen B0b recent-behavior definition."""
    return _recent_behavior_scores(record)


def write_bm25_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    runs_dir: str | Path = "runs",
    artifacts_dir: str | Path = "artifacts/baselines",
    config_path: str | Path | None = None,
    k1: float = 1.2,
    b: float = 0.75,
    tokenizer_mode: str = "cjk_2_3gram",
    exact_match_boost: float = 2.0,
    idf_scope: str = "global",
    char_coverage_boost: float = 0.0,
) -> dict[str, Any]:
    standardized_dir = Path(standardized_dir)
    bm25_stats = None
    stats_metadata = {}
    if idf_scope == "global":
        _, dataset_version = _dataset_identity(standardized_dir)
        stats_path = (
            Path(artifacts_dir)
            / _safe_path_component(dataset_version)
            / f"bm25_stats_{tokenizer_mode}.json"
        )
        bm25_stats = _load_or_build_bm25_stats(
            item_catalog_path=standardized_dir / "item_catalog.jsonl",
            stats_path=stats_path,
            tokenizer_mode=tokenizer_mode,
        )
        stats_metadata = {
            "bm25_stats_path": str(stats_path),
            "bm25_stats_sha256": sha256_file(stats_path),
        }
    elif idf_scope != "request":
        raise ValueError(f"unknown idf_scope: {idf_scope}")

    def score_record(record: dict[str, Any]) -> dict[str, float]:
        return _bm25_scores(
            record,
            k1=k1,
            b=b,
            tokenizer_mode=tokenizer_mode,
            exact_match_boost=exact_match_boost,
            bm25_stats=bm25_stats,
            char_coverage_boost=char_coverage_boost,
        )

    return _write_scores_from_records(
        standardized_dir=standardized_dir,
        split=split,
        run_id=run_id,
        method_id="b1_bm25",
        runs_dir=runs_dir,
        config_path=config_path,
        score_record=score_record,
        metadata_extra={
            "document_template": "title + brand + seller + cat_l1 + cat_l2 + cat_l3",
            "idf_scope": idf_scope,
            "input_fields_used": ["query", "candidates.title", "candidates.brand", "candidates.seller", "candidates.cat"],
            "k1": k1,
            "b": b,
            "exact_match_boost": exact_match_boost,
            "char_coverage_boost": char_coverage_boost,
            "tokenizer": tokenizer_mode,
            **stats_metadata,
        },
    )


def write_static_mixture_scores(
    query_scores_path: str | Path,
    history_scores_path: str | Path,
    query_run_id: str,
    history_run_id: str,
    run_id: str,
    method_id: str,
    alpha: float,
    candidate_manifest_path: str | Path,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    query_scores = _load_score_map(query_scores_path)
    history_scores = _load_score_map(history_scores_path)
    if set(query_scores) != set(history_scores):
        raise ValueError("query/history score request_id sets differ")

    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scores_path = run_dir / "scores.jsonl"
    rows = 0
    with scores_path.open("w", encoding="utf-8") as handle:
        for request_id in sorted(query_scores):
            query_items = query_scores[request_id]
            history_items = history_scores[request_id]
            if set(query_items) != set(history_items):
                raise ValueError(f"query/history score candidate sets differ for {request_id}")
            query_z = _zscore_map(query_items)
            history_z = _zscore_map(history_items)
            for item_id in sorted(query_items):
                score = alpha * query_z[item_id] + (1.0 - alpha) * history_z[item_id]
                handle.write(
                    json.dumps(
                        {
                            "candidate_item_id": item_id,
                            "method_id": method_id,
                            "request_id": request_id,
                            "score": score,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                rows += 1

    metadata = _base_metadata(
        standardized_dir=None,
        split="dev",
        run_id=run_id,
        method_id=method_id,
        candidate_manifest_path=Path(candidate_manifest_path),
        config_path=config_path,
    )
    metadata.update(
        {
            "alpha": alpha,
            "history_run_id": history_run_id,
            "input_fields_used": ["upstream query scores", "upstream history scores"],
            "query_run_id": query_run_id,
            "score_definition": "alpha * z(query_score) + (1 - alpha) * z(history_score), z-scored within request",
            "score_rows": rows,
            "upstream_scores": {
                "history_scores_path": str(history_scores_path),
                "history_scores_sha256": sha256_file(history_scores_path),
                "query_scores_path": str(query_scores_path),
                "query_scores_sha256": sha256_file(query_scores_path),
            },
        }
    )
    _copy_config(config_path, run_dir)
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def document_text(item: dict[str, Any]) -> str:
    return " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("brand") or ""),
            str(item.get("seller") or ""),
            " ".join(str(part) for part in item.get("cat", [])),
        ]
    )


def tokenize_text(text: str, mode: str = "cjk_unigram_bigram") -> list[str]:
    text = text.lower()
    tokens = _ASCII_RE.findall(text)
    cjk_chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    if mode == "cjk_unigram_bigram":
        tokens.extend(cjk_chars)
        tokens.extend(a + b for a, b in zip(cjk_chars, cjk_chars[1:]))
    elif mode == "cjk_2_3gram":
        tokens.extend(_cjk_ngrams(cjk_chars, 2))
        tokens.extend(_cjk_ngrams(cjk_chars, 3))
        if not tokens:
            tokens.extend(cjk_chars)
    elif mode == "jieba":
        import jieba

        tokens.extend(
            term.strip().lower()
            for term in jieba.lcut(text, cut_all=False)
            if _valid_token(term)
        )
        if not tokens and cjk_chars:
            tokens.extend(cjk_chars)
    else:
        raise ValueError(f"unknown tokenizer mode: {mode}")
    return tokens


def _build_popularity_stats(train_records_path: Path, stats_path: Path) -> dict[str, int]:
    exposure = Counter()
    clicked = Counter()
    purchased = Counter()
    for record in iter_jsonl(train_records_path):
        for candidate in record["candidates"]:
            item_id = str(candidate["item_id"])
            exposure[item_id] += 1
            clicked[item_id] += int(candidate.get("clicked", 0) or 0)
            purchased[item_id] += int(candidate.get("purchased", 0) or 0)
    with stats_path.open("w", encoding="utf-8") as handle:
        for item_id in sorted(exposure, key=lambda value: (0, int(value)) if value.isdigit() else (1, value)):
            handle.write(
                json.dumps(
                    {
                        "clicked": clicked[item_id],
                        "exposed": exposure[item_id],
                        "item_id": item_id,
                        "purchased": purchased[item_id],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return dict(clicked)


def _load_popularity_stats(stats_path: Path) -> dict[str, int]:
    return {str(row["item_id"]): int(row["clicked"]) for row in iter_jsonl(stats_path)}


def _recent_behavior_scores(record: dict[str, Any]) -> dict[str, float]:
    history = record.get("history", [])
    if not history:
        return {str(candidate["item_id"]): 0.0 for candidate in record["candidates"]}

    history_features = []
    size = len(history)
    for index, event in enumerate(history):
        reverse_age = size - index
        recency = 1.0 / math.sqrt(reverse_age)
        event_weight = 1.5 if event.get("event") == "purchase" else 1.0
        history_features.append(
            {
                "cat": [str(part) for part in event.get("cat", [])],
                "event_weight": event_weight,
                "item_id": str(event["item_id"]),
                "weight": recency * event_weight,
            }
        )

    result = {}
    for candidate in record["candidates"]:
        item_id = str(candidate["item_id"])
        cats = [str(part) for part in candidate.get("cat", [])]
        score = 0.0
        for event in history_features:
            weight = float(event["weight"])
            if item_id == event["item_id"]:
                score += 3.0 * weight
            score += _category_overlap_score(cats, event["cat"]) * weight
        result[item_id] = score
    return result


def _category_overlap_score(left: list[str], right: list[str]) -> float:
    left = (left + ["", "", ""])[:3]
    right = (right + ["", "", ""])[:3]
    if _valid_category(left[2]) and left[2] == right[2]:
        return 1.0
    if _valid_category(left[1]) and left[1] == right[1]:
        return 0.5
    if _valid_category(left[0]) and left[0] == right[0]:
        return 0.2
    return 0.0


def _valid_category(value: str) -> bool:
    return bool(value and value.upper() != "UNKNOWN")


def _bm25_scores(
    record: dict[str, Any],
    k1: float,
    b: float,
    tokenizer_mode: str,
    exact_match_boost: float,
    bm25_stats: dict[str, Any] | None,
    char_coverage_boost: float,
) -> dict[str, float]:
    query = str(record.get("query") or "")
    query_terms = Counter(tokenize_text(query, mode=tokenizer_mode))
    if not query_terms:
        return {str(candidate["item_id"]): 0.0 for candidate in record["candidates"]}

    doc_tfs = []
    doc_lens = []
    df = Counter()
    candidates = record["candidates"]
    documents = [document_text(candidate) for candidate in candidates]
    for candidate, document in zip(candidates, documents):
        tf = Counter(tokenize_text(document, mode=tokenizer_mode))
        doc_tfs.append((str(candidate["item_id"]), tf))
        doc_lens.append(sum(tf.values()))
        for term in tf:
            df[term] += 1
    n_docs = len(doc_tfs)
    avgdl = float(bm25_stats["avgdl"]) if bm25_stats else sum(doc_lens) / n_docs if n_docs else 0.0
    corpus_docs = int(bm25_stats["doc_count"]) if bm25_stats else n_docs
    global_df = bm25_stats["df"] if bm25_stats else None
    if avgdl <= 0:
        return {item_id: 0.0 for item_id, _ in doc_tfs}

    scores = {}
    compact_query = _compact_text(query)
    query_chars = {char for char in compact_query if "\u4e00" <= char <= "\u9fff"}
    for (item_id, tf), doc_len, document in zip(doc_tfs, doc_lens, documents):
        score = 0.0
        for term, query_tf in query_terms.items():
            freq = tf.get(term, 0)
            if not freq:
                continue
            term_df = int(global_df.get(term, 0)) if global_df is not None else df[term]
            idf = math.log1p((corpus_docs - term_df + 0.5) / (term_df + 0.5))
            denom = freq + k1 * (1.0 - b + b * doc_len / avgdl)
            score += query_tf * idf * (freq * (k1 + 1.0) / denom)
        if exact_match_boost and len(compact_query) >= 2 and compact_query in _compact_text(document):
            score += exact_match_boost
        if char_coverage_boost and query_chars:
            document_chars = {char for char in _compact_text(document) if "\u4e00" <= char <= "\u9fff"}
            score += char_coverage_boost * (len(query_chars & document_chars) / len(query_chars))
        scores[item_id] = score
    return scores


def _cjk_ngrams(chars: list[str], n: int) -> list[str]:
    if len(chars) < n:
        return []
    return ["".join(chars[index : index + n]) for index in range(len(chars) - n + 1)]


def _load_or_build_bm25_stats(
    item_catalog_path: Path,
    stats_path: Path,
    tokenizer_mode: str,
) -> dict[str, Any]:
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    df = Counter()
    doc_count = 0
    total_len = 0
    for item in iter_jsonl(item_catalog_path):
        terms = tokenize_text(document_text(item), mode=tokenizer_mode)
        doc_count += 1
        total_len += len(terms)
        df.update(set(terms))
    stats = {
        "avgdl": total_len / doc_count if doc_count else 0.0,
        "df": dict(sorted(df.items())),
        "doc_count": doc_count,
        "item_catalog_path": str(item_catalog_path),
        "tokenizer": tokenizer_mode,
    }
    write_json(stats_path, stats)
    return stats


def _compact_text(text: str) -> str:
    return "".join(char.lower() for char in text if not char.isspace())


def _valid_token(term: str) -> bool:
    term = term.strip()
    if not term:
        return False
    if _ASCII_RE.fullmatch(term.lower()):
        return True
    return any("\u4e00" <= char <= "\u9fff" for char in term)


def _write_scores_from_records(
    standardized_dir: Path,
    split: str,
    run_id: str,
    method_id: str,
    runs_dir: str | Path,
    config_path: str | Path | None,
    score_record: ScoreRecordFn,
    metadata_extra: dict[str, Any],
) -> dict[str, Any]:
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / f"records_{split}.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    scores_path = run_dir / "scores.jsonl"

    rows = 0
    requests = 0
    with scores_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(records_path):
            requests += 1
            request_id = str(record["request_id"])
            scores = score_record(record)
            candidate_ids = {str(candidate["item_id"]) for candidate in record["candidates"]}
            if set(scores) != candidate_ids:
                raise ValueError(f"score/candidate mismatch for request_id={request_id}")
            for item_id in sorted(scores):
                score = float(scores[item_id])
                if not math.isfinite(score):
                    raise ValueError(f"non-finite score for {request_id} {item_id}: {score}")
                handle.write(
                    json.dumps(
                        {
                            "candidate_item_id": item_id,
                            "method_id": method_id,
                            "request_id": request_id,
                            "score": score,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                rows += 1

    metadata = _base_metadata(
        standardized_dir=standardized_dir,
        split=split,
        run_id=run_id,
        method_id=method_id,
        candidate_manifest_path=candidate_manifest_path,
        config_path=config_path,
    )
    metadata.update(metadata_extra)
    metadata.update({"request_count": requests, "score_rows": rows})
    _copy_config(config_path, run_dir)
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def _base_metadata(
    standardized_dir: Path | None,
    split: str,
    run_id: str,
    method_id: str,
    candidate_manifest_path: Path,
    config_path: str | Path | None,
) -> dict[str, Any]:
    dataset_id, dataset_version = _dataset_identity(standardized_dir)
    return {
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "config_path": str(config_path) if config_path else None,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method_id": method_id,
        "qrels_read": False,
        "run_id": run_id,
        "split": split,
        "standardized_dir": str(standardized_dir) if standardized_dir else None,
    }


def _dataset_identity(standardized_dir: Path | None) -> tuple[str, str]:
    if standardized_dir is None:
        return "unknown", "unknown"
    manifest_path = standardized_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"standardized manifest missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    dataset_id = str(manifest.get("dataset_id") or "").strip()
    dataset_version = str(manifest.get("dataset_version") or "").strip()
    if not dataset_id or not dataset_version:
        raise ValueError("standardized manifest must define dataset_id and dataset_version")
    return dataset_id, dataset_version


def _safe_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not safe:
        raise ValueError("dataset version cannot form a safe artifact path")
    return safe


def _copy_config(config_path: str | Path | None, run_dir: Path) -> None:
    if not config_path:
        return
    config_path = Path(config_path)
    if config_path.exists():
        shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")


def _load_score_map(path: str | Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        item_id = str(row["candidate_item_id"])
        if item_id in scores[request_id]:
            raise ValueError(f"duplicate score for {request_id} {item_id}")
        scores[request_id][item_id] = float(row["score"])
    return dict(scores)


def _zscore_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return {item_id: 0.0 for item_id in values}
    return {item_id: (value - mean) / std for item_id, value in values.items()}


def deterministic_hash_float(*parts: str) -> float:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)
