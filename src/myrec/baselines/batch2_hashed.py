"""Local trainable Batch 2 baseline adapters.

These scorers are intentionally small and auditable. They provide protocol-valid
fixed-candidate runs when upstream environments are unavailable, while the
baseline cards must still disclose the adapter gap versus official RecBole /
KuaiSearch / PPS-classic implementations.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from myrec.baselines.core import document_text, tokenize_text
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


METHOD_SPECS: dict[str, dict[str, Any]] = {
    "b4": {
        "method_id": "b4_sasrec_style_hashed",
        "role": "history-only sequence-style baseline",
        "input_fields_used": [
            "records_train.history.item_id",
            "records_train.candidates.item_id",
            "records_train.candidates.clicked",
            "records_dev.history.item_id",
            "records_dev.candidates.item_id",
        ],
        "score_definition": "online logistic ranker over candidate item bias and recent history-item to candidate-item transition features, plus train-only item-bias prior",
    },
    "b5": {
        "method_id": "b5_kuaisearch_dcn_din_style_hashed",
        "role": "full-feature industrial CTR-style baseline",
        "input_fields_used": [
            "records_train.user_id",
            "records_train.query",
            "records_train.history.item_id",
            "records_train.history.cat",
            "records_train.candidates.item_id",
            "records_train.candidates.title",
            "records_train.candidates.brand",
            "records_train.candidates.seller",
            "records_train.candidates.cat",
            "records_train.candidates.clicked",
            "records_dev.user_id",
            "records_dev.query",
            "records_dev.history.item_id",
            "records_dev.history.cat",
            "records_dev.candidates.item_id",
            "records_dev.candidates.title",
            "records_dev.candidates.brand",
            "records_dev.candidates.seller",
            "records_dev.candidates.cat",
        ],
        "score_definition": "online logistic CTR ranker over user, query, item text/category, and history overlap features",
    },
    "b6": {
        "method_id": "b6_pps_classic_style_hashed",
        "role": "PPS-classic shallow query-history fusion baseline",
        "input_fields_used": [
            "records_train.query",
            "records_train.history.item_id",
            "records_train.history.title",
            "records_train.history.cat",
            "records_train.candidates.item_id",
            "records_train.candidates.title",
            "records_train.candidates.brand",
            "records_train.candidates.seller",
            "records_train.candidates.cat",
            "records_train.candidates.clicked",
            "records_dev.query",
            "records_dev.history.item_id",
            "records_dev.history.title",
            "records_dev.history.cat",
            "records_dev.candidates.item_id",
            "records_dev.candidates.title",
            "records_dev.candidates.brand",
            "records_dev.candidates.seller",
            "records_dev.candidates.cat",
        ],
        "score_definition": "online logistic PPS-style ranker over query-document, history-document, and gated personalization features",
    },
}


def write_hashed_batch2_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    method: str,
    seed: int,
    runs_dir: str | Path = "runs",
    artifacts_dir: str | Path = "artifacts/baselines",
    config_path: str | Path | None = None,
    n_features: int = 1 << 20,
    chunk_size: int = 65536,
    epochs: int = 1,
    negatives_per_positive: int = 4,
    max_history_len: int = 20,
    max_query_tokens: int = 12,
    max_doc_tokens: int = 24,
    tokenizer_mode: str = "cjk_2_3gram",
    alpha: float = 1e-6,
) -> dict[str, Any]:
    """Train an online hashed-feature ranker on train and score a blind split."""

    if method not in METHOD_SPECS:
        raise ValueError(f"unknown Batch 2 hashed method: {method}")

    import sklearn
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import SGDClassifier

    standardized_dir = Path(standardized_dir)
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    spec = METHOD_SPECS[method]
    train_item_stats_path, train_item_stats = _load_or_build_train_item_stats(standardized_dir, artifacts_dir)
    hasher = FeatureHasher(n_features=n_features, input_type="dict", alternate_sign=False)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        random_state=seed,
        fit_intercept=True,
        average=True,
        max_iter=1,
        tol=None,
        shuffle=False,
    )

    train_started = time.perf_counter()
    train_stats = _train_online_ranker(
        model=model,
        hasher=hasher,
        standardized_dir=standardized_dir,
        method=method,
        seed=seed,
        chunk_size=chunk_size,
        epochs=epochs,
        negatives_per_positive=negatives_per_positive,
        max_history_len=max_history_len,
        max_query_tokens=max_query_tokens,
        max_doc_tokens=max_doc_tokens,
        tokenizer_mode=tokenizer_mode,
        train_item_stats=train_item_stats,
    )
    train_seconds = time.perf_counter() - train_started

    artifact_path = artifacts_dir / f"{run_id}_model.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump({"hasher": hasher, "model": model, "method": method}, handle)

    score_started = time.perf_counter()
    score_stats = _score_split(
        model=model,
        hasher=hasher,
        standardized_dir=standardized_dir,
        split=split,
        run_dir=run_dir,
        method=method,
        method_id=spec["method_id"],
        chunk_size=chunk_size,
        max_history_len=max_history_len,
        max_query_tokens=max_query_tokens,
        max_doc_tokens=max_doc_tokens,
        tokenizer_mode=tokenizer_mode,
        train_item_stats=train_item_stats,
    )
    score_seconds = time.perf_counter() - score_started

    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    metadata = {
        "alpha": alpha,
        "artifact_model_path": str(artifact_path),
        "artifact_model_sha256": sha256_file(artifact_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "chunk_size": chunk_size,
        "config_path": str(config_path) if config_path else None,
        "dataset_id": "kuaisearch",
        "dataset_version": "v0_lite",
        "epochs": epochs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "implementation_note": _implementation_note(method),
        "input_fields_used": spec["input_fields_used"],
        "max_doc_tokens": max_doc_tokens,
        "max_history_len": max_history_len,
        "max_query_tokens": max_query_tokens,
        "method_id": spec["method_id"],
        "method_key": method,
        "negatives_per_positive": negatives_per_positive,
        "n_features": n_features,
        "package_versions": {"sklearn": sklearn.__version__},
        "qrels_read": False,
        "role": spec["role"],
        "run_id": run_id,
        "score_definition": spec["score_definition"],
        "score_rows": score_stats["score_rows"],
        "seed": seed,
        "split": split,
        "standardized_dir": str(standardized_dir),
        "timing": {
            "score_seconds": score_seconds,
            "train_seconds": train_seconds,
        },
        "tokenizer": tokenizer_mode,
        "train_stats": train_stats,
        "train_item_stats_path": str(train_item_stats_path),
        "train_item_stats_sha256": sha256_file(train_item_stats_path),
        "tuning": {
            "class": "trainable",
            "dev_eval_budget": 16,
            "dev_evals_used_for_this_config": 1,
            "multi_seed_member": True,
        },
    }
    _copy_config(config_path, run_dir)
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def _train_online_ranker(
    model: Any,
    hasher: Any,
    standardized_dir: Path,
    method: str,
    seed: int,
    chunk_size: int,
    epochs: int,
    negatives_per_positive: int,
    max_history_len: int,
    max_query_tokens: int,
    max_doc_tokens: int,
    tokenizer_mode: str,
    train_item_stats: dict[str, dict[str, int]],
) -> dict[str, Any]:
    classes = [0, 1]
    first = True
    total_rows = 0
    positives = 0
    negatives = 0
    for epoch in range(epochs):
        rng = random.Random(seed + epoch * 1009)
        features: list[dict[str, float]] = []
        labels: list[int] = []
        for feature_row, label in _iter_train_examples(
            standardized_dir / "records_train.jsonl",
            method=method,
            rng=rng,
            negatives_per_positive=negatives_per_positive,
            max_history_len=max_history_len,
            max_query_tokens=max_query_tokens,
            max_doc_tokens=max_doc_tokens,
            tokenizer_mode=tokenizer_mode,
            train_item_stats=train_item_stats,
        ):
            features.append(feature_row)
            labels.append(label)
            if len(labels) >= chunk_size:
                first = _partial_fit(model, hasher, features, labels, classes, first)
                total_rows += len(labels)
                positives += sum(labels)
                negatives += len(labels) - sum(labels)
                features = []
                labels = []
        if labels:
            first = _partial_fit(model, hasher, features, labels, classes, first)
            total_rows += len(labels)
            positives += sum(labels)
            negatives += len(labels) - sum(labels)
    return {
        "epochs": epochs,
        "negative_examples": negatives,
        "positive_examples": positives,
        "rows": total_rows,
    }


def _partial_fit(
    model: Any,
    hasher: Any,
    features: list[dict[str, float]],
    labels: list[int],
    classes: list[int],
    first: bool,
) -> bool:
    matrix = hasher.transform(features)
    if first:
        model.partial_fit(matrix, labels, classes=classes)
        return False
    model.partial_fit(matrix, labels)
    return False


def _iter_train_examples(
    path: Path,
    method: str,
    rng: random.Random,
    negatives_per_positive: int,
    max_history_len: int,
    max_query_tokens: int,
    max_doc_tokens: int,
    tokenizer_mode: str,
    train_item_stats: dict[str, dict[str, int]],
) -> Iterable[tuple[dict[str, float], int]]:
    for record in iter_jsonl(path):
        positives = [candidate for candidate in record["candidates"] if int(candidate.get("clicked", 0) or 0) > 0]
        if not positives:
            continue
        negatives = [candidate for candidate in record["candidates"] if int(candidate.get("clicked", 0) or 0) <= 0]
        sample_size = min(len(negatives), negatives_per_positive * len(positives))
        if sample_size:
            negatives = rng.sample(negatives, sample_size)
        for candidate in positives:
            yield (
                _features_for_candidate(
                    record,
                    candidate,
                    method=method,
                    max_history_len=max_history_len,
                    max_query_tokens=max_query_tokens,
                    max_doc_tokens=max_doc_tokens,
                    tokenizer_mode=tokenizer_mode,
                    train_item_stats=train_item_stats,
                ),
                1,
            )
        for candidate in negatives:
            yield (
                _features_for_candidate(
                    record,
                    candidate,
                    method=method,
                    max_history_len=max_history_len,
                    max_query_tokens=max_query_tokens,
                    max_doc_tokens=max_doc_tokens,
                    tokenizer_mode=tokenizer_mode,
                    train_item_stats=train_item_stats,
                ),
                0,
            )


def _score_split(
    model: Any,
    hasher: Any,
    standardized_dir: Path,
    split: str,
    run_dir: Path,
    method: str,
    method_id: str,
    chunk_size: int,
    max_history_len: int,
    max_query_tokens: int,
    max_doc_tokens: int,
    tokenizer_mode: str,
    train_item_stats: dict[str, dict[str, int]],
) -> dict[str, int]:
    scores_path = run_dir / "scores.jsonl"
    score_rows = 0
    request_count = 0
    features: list[dict[str, float]] = []
    keys: list[tuple[str, str]] = []
    priors: list[float] = []

    def flush(handle: Any) -> None:
        nonlocal score_rows
        if not features:
            return
        matrix = hasher.transform(features)
        values = model.decision_function(matrix)
        for (request_id, item_id), value, prior in zip(keys, values, priors):
            score = float(value) + prior
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
            score_rows += 1
        features.clear()
        keys.clear()
        priors.clear()

    with scores_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(standardized_dir / f"records_{split}.jsonl"):
            request_count += 1
            request_id = str(record["request_id"])
            for candidate in record["candidates"]:
                features.append(
                    _features_for_candidate(
                        record,
                        candidate,
                        method=method,
                        max_history_len=max_history_len,
                        max_query_tokens=max_query_tokens,
                        max_doc_tokens=max_doc_tokens,
                        tokenizer_mode=tokenizer_mode,
                        train_item_stats=train_item_stats,
                    )
                )
                keys.append((request_id, str(candidate["item_id"])))
                priors.append(_score_prior(method, str(candidate["item_id"]), train_item_stats))
                if len(features) >= chunk_size:
                    flush(handle)
        flush(handle)
    return {"request_count": request_count, "score_rows": score_rows}


def _features_for_candidate(
    record: dict[str, Any],
    candidate: dict[str, Any],
    method: str,
    max_history_len: int,
    max_query_tokens: int,
    max_doc_tokens: int,
    tokenizer_mode: str,
    train_item_stats: dict[str, dict[str, int]],
) -> dict[str, float]:
    item_id = str(candidate["item_id"])
    history = list(record.get("history", []))[-max_history_len:]
    features: dict[str, float] = {"bias": 1.0}
    _add_train_item_bias_features(features, item_id, train_item_stats)
    if method == "b4":
        _add_b4_features(features, item_id, history)
    elif method == "b5":
        _add_b5_features(features, record, candidate, history, max_query_tokens, max_doc_tokens, tokenizer_mode)
    elif method == "b6":
        _add_b6_features(features, record, candidate, history, max_query_tokens, max_doc_tokens, tokenizer_mode)
    else:
        raise ValueError(f"unknown method: {method}")
    return features


def _add_b4_features(features: dict[str, float], item_id: str, history: list[dict[str, Any]]) -> None:
    features[f"cand_item={item_id}"] = 1.0
    features[f"hist_len_bucket={_bucket(len(history), [0, 1, 3, 5, 10, 20])}"] = 1.0
    seen = set()
    for age, event in enumerate(reversed(history), start=1):
        hist_item = str(event.get("item_id", ""))
        if not hist_item:
            continue
        weight = 1.0 / math.sqrt(age)
        if hist_item == item_id:
            features["same_item_in_history"] = 1.0
        if age <= 10:
            features[f"recent_transition={hist_item}>{item_id}"] = weight
        if hist_item not in seen and len(seen) < 20:
            features[f"hist_item_to_candidate={hist_item}>{item_id}"] = weight
            seen.add(hist_item)


def _add_train_item_bias_features(
    features: dict[str, float],
    item_id: str,
    train_item_stats: dict[str, dict[str, int]],
) -> None:
    stats = train_item_stats.get(item_id)
    if not stats:
        features["train_item_unseen"] = 1.0
        return
    clicked = int(stats.get("clicked", 0))
    exposed = int(stats.get("exposed", 0))
    purchased = int(stats.get("purchased", 0))
    features["train_item_log_clicked"] = math.log1p(clicked)
    features["train_item_log_exposed"] = math.log1p(exposed)
    features["train_item_ctr"] = clicked / exposed if exposed else 0.0
    features["train_item_purchase_ctr"] = purchased / exposed if exposed else 0.0
    features[f"train_item_clicked_bucket={_bucket(clicked, [0, 1, 2, 5, 10, 20, 50])}"] = 1.0
    features[f"train_item_exposed_bucket={_bucket(exposed, [0, 1, 2, 5, 10, 20, 50, 100])}"] = 1.0


def _score_prior(method: str, item_id: str, train_item_stats: dict[str, dict[str, int]]) -> float:
    if method != "b4":
        return 0.0
    stats = train_item_stats.get(item_id)
    if not stats:
        return 0.0
    return math.log1p(int(stats.get("clicked", 0) or 0))


def _add_b5_features(
    features: dict[str, float],
    record: dict[str, Any],
    candidate: dict[str, Any],
    history: list[dict[str, Any]],
    max_query_tokens: int,
    max_doc_tokens: int,
    tokenizer_mode: str,
) -> None:
    item_id = str(candidate["item_id"])
    features[f"user={record.get('user_id', '')}"] = 1.0
    features[f"cand_item={item_id}"] = 1.0
    _add_category_features(features, "cand", candidate.get("cat", []))
    query_tokens = _limited_counts(str(record.get("query") or ""), tokenizer_mode, max_query_tokens)
    doc_tokens = _limited_counts(document_text(candidate), tokenizer_mode, max_doc_tokens)
    _add_token_features(features, "q", query_tokens, max_terms=max_query_tokens)
    _add_token_features(features, "doc", doc_tokens, max_terms=max_doc_tokens)
    _add_query_doc_features(features, query_tokens, doc_tokens, record.get("query", ""), document_text(candidate))
    _add_history_overlap_features(features, history, candidate, include_text=False, tokenizer_mode=tokenizer_mode)


def _add_b6_features(
    features: dict[str, float],
    record: dict[str, Any],
    candidate: dict[str, Any],
    history: list[dict[str, Any]],
    max_query_tokens: int,
    max_doc_tokens: int,
    tokenizer_mode: str,
) -> None:
    item_id = str(candidate["item_id"])
    features[f"cand_item={item_id}"] = 1.0
    _add_category_features(features, "cand", candidate.get("cat", []))
    query_tokens = _limited_counts(str(record.get("query") or ""), tokenizer_mode, max_query_tokens)
    doc_text = document_text(candidate)
    doc_tokens = _limited_counts(doc_text, tokenizer_mode, max_doc_tokens)
    _add_query_doc_features(features, query_tokens, doc_tokens, record.get("query", ""), doc_text)
    _add_history_overlap_features(features, history, candidate, include_text=True, tokenizer_mode=tokenizer_mode)
    hist_text = " ".join(str(event.get("title") or "") for event in history[-10:])
    hist_tokens = _limited_counts(hist_text, tokenizer_mode, max_doc_tokens)
    overlap = set(hist_tokens) & set(doc_tokens)
    for token in list(overlap)[:16]:
        features[f"hist_doc_token={token}"] = min(hist_tokens[token], doc_tokens[token])
    if overlap and set(query_tokens) & overlap:
        features["query_history_doc_three_way_overlap"] = float(len(set(query_tokens) & overlap))


def _add_category_features(features: dict[str, float], prefix: str, cats: list[Any]) -> None:
    cats = [str(cat) for cat in (cats or [])]
    for level, cat in enumerate(cats[:3], start=1):
        if cat and cat.upper() != "UNKNOWN":
            features[f"{prefix}_cat{level}={cat}"] = 1.0
            if level >= 2:
                features[f"{prefix}_cat_path{level}={'/'.join(cats[:level])}"] = 1.0


def _add_token_features(
    features: dict[str, float],
    prefix: str,
    counts: Counter[str],
    max_terms: int,
) -> None:
    for token, count in counts.most_common(max_terms):
        features[f"{prefix}_tok={token}"] = float(min(count, 3))


def _add_query_doc_features(
    features: dict[str, float],
    query_tokens: Counter[str],
    doc_tokens: Counter[str],
    query: Any,
    doc_text: str,
) -> None:
    overlap = set(query_tokens) & set(doc_tokens)
    if overlap:
        features["query_doc_overlap_count"] = float(len(overlap))
        features["query_doc_overlap_rate"] = len(overlap) / max(1, len(query_tokens))
    for token in list(overlap)[:16]:
        features[f"query_doc_token={token}"] = min(query_tokens[token], doc_tokens[token])
    compact_query = _compact_text(str(query or ""))
    if len(compact_query) >= 2 and compact_query in _compact_text(doc_text):
        features["query_exact_substring_in_doc"] = 1.0


def _add_history_overlap_features(
    features: dict[str, float],
    history: list[dict[str, Any]],
    candidate: dict[str, Any],
    include_text: bool,
    tokenizer_mode: str,
) -> None:
    item_id = str(candidate["item_id"])
    candidate_cats = [str(cat) for cat in candidate.get("cat", [])]
    hist_items = [str(event.get("item_id", "")) for event in history]
    if item_id in hist_items:
        features["history_same_item"] = 1.0
    for level in range(3):
        cand_cat = candidate_cats[level] if level < len(candidate_cats) else ""
        if not cand_cat or cand_cat.upper() == "UNKNOWN":
            continue
        best = 0.0
        for age, event in enumerate(reversed(history), start=1):
            hist_cats = [str(cat) for cat in event.get("cat", [])]
            if level < len(hist_cats) and hist_cats[level] == cand_cat:
                best = max(best, 1.0 / math.sqrt(age))
        if best:
            features[f"history_cat{level + 1}_overlap"] = best
    if include_text and history:
        candidate_tokens = _limited_counts(document_text(candidate), tokenizer_mode, 24)
        recent_titles = " ".join(str(event.get("title") or "") for event in history[-10:])
        history_tokens = _limited_counts(recent_titles, tokenizer_mode, 24)
        overlap = set(candidate_tokens) & set(history_tokens)
        if overlap:
            features["history_doc_overlap_count"] = float(len(overlap))


def _limited_counts(text: str, tokenizer_mode: str, max_terms: int) -> Counter[str]:
    counts = Counter(tokenize_text(text, mode=tokenizer_mode))
    return Counter(dict(counts.most_common(max_terms)))


def _bucket(value: int, edges: list[int]) -> str:
    for edge in edges:
        if value <= edge:
            return f"<= {edge}"
    return f"> {edges[-1]}"


def _compact_text(text: str) -> str:
    return "".join(char.lower() for char in text if not char.isspace())


def _implementation_note(method: str) -> str:
    if method == "b4":
        return (
            "RecBole 1.2.1 could not be installed in the active Python 3.13 "
            "environment because its ray<=2.6.3 dependency has no cp313 wheel; "
            "this run is a transparent SASRec-style local adapter, not an "
            "official RecBole run."
        )
    if method == "b5":
        return (
            "The KuaiSearch official repository is available and contains DIN/DCN "
            "models, but its ranking pipeline expects precomputed query/title "
            "embeddings and raw user feature files outside the standardized "
            "blind-record interface. This run is a compact DCN/DIN-style local "
            "adapter over allowed standardized fields, not an official-number "
            "alignment run."
        )
    if method == "b6":
        return (
            "No official HEM/ZAM/TEM adapter is present in the repo. This run is "
            "a PPS-classic style shallow query-history fusion adapter over the "
            "allowed standardized fields, not an official reproduction."
        )
    raise ValueError(method)


def _load_or_build_train_item_stats(
    standardized_dir: Path,
    artifacts_dir: Path,
) -> tuple[Path, dict[str, dict[str, int]]]:
    stats_path = artifacts_dir / "kuaisearch_train_item_label_stats.jsonl"
    if not stats_path.exists():
        clicked = Counter()
        exposed = Counter()
        purchased = Counter()
        for record in iter_jsonl(standardized_dir / "records_train.jsonl"):
            for candidate in record["candidates"]:
                item_id = str(candidate["item_id"])
                exposed[item_id] += 1
                clicked[item_id] += int(candidate.get("clicked", 0) or 0)
                purchased[item_id] += int(candidate.get("purchased", 0) or 0)
        with stats_path.open("w", encoding="utf-8") as handle:
            for item_id in sorted(exposed, key=_item_sort_key):
                handle.write(
                    json.dumps(
                        {
                            "clicked": clicked[item_id],
                            "exposed": exposed[item_id],
                            "item_id": item_id,
                            "purchased": purchased[item_id],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    stats = {}
    for row in iter_jsonl(stats_path):
        stats[str(row["item_id"])] = {
            "clicked": int(row.get("clicked", 0) or 0),
            "exposed": int(row.get("exposed", 0) or 0),
            "purchased": int(row.get("purchased", 0) or 0),
        }
    return stats_path, stats


def _item_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _copy_config(config_path: str | Path | None, run_dir: Path) -> None:
    if not config_path:
        return
    config_path = Path(config_path)
    if config_path.exists():
        shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")
