"""Diagnostics for the KuaiSearch B1 BM25 C2 failure."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.baselines.core import _bm25_scores, _load_or_build_bm25_stats
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


DEFAULT_B1_RUN_ID = "20260708_kuaisearch_b1_bm25_globalidf_exact10_dev"
DEFAULT_B0A_RUN_ID = "20260708_kuaisearch_b0a_popularity_dev"


def run_b1_diagnostics(
    standardized_dir: str | Path,
    raw_dir: str | Path,
    output_json_path: str | Path,
    output_markdown_path: str | Path | None = None,
    runs_dir: str | Path = "runs",
    artifacts_dir: str | Path = "artifacts/baselines",
    split: str = "dev",
    seed: int = 20260708,
    max_requests: int | None = None,
    random_catalog_size: int = 100_000,
    bootstrap_samples: int = 5000,
    tokenizer_mode: str = "cjk_2_3gram",
    k1: float = 1.2,
    b: float = 0.75,
    exact_match_boost: float = 10.0,
    char_coverage_boost: float = 0.0,
    b1_run_id: str = DEFAULT_B1_RUN_ID,
    b0a_run_id: str = DEFAULT_B0A_RUN_ID,
) -> dict[str, Any]:
    """Run non-label B1 diagnostics and write a report."""

    standardized_dir = Path(standardized_dir)
    raw_dir = Path(raw_dir)
    runs_dir = Path(runs_dir)
    artifacts_dir = Path(artifacts_dir)
    output_json_path = Path(output_json_path)
    output_markdown_path = Path(output_markdown_path) if output_markdown_path else None

    bm25_stats_path = artifacts_dir / f"kuaisearch_b1_bm25_stats_{tokenizer_mode}.json"
    bm25_stats = _load_or_build_bm25_stats(
        item_catalog_path=standardized_dir / "item_catalog.jsonl",
        stats_path=bm25_stats_path,
        tokenizer_mode=tokenizer_mode,
    )
    bm25_config = {
        "b": b,
        "bm25_stats_path": str(bm25_stats_path),
        "bm25_stats_sha256": sha256_file(bm25_stats_path),
        "char_coverage_boost": char_coverage_boost,
        "exact_match_boost": exact_match_boost,
        "idf_scope": "global_item_catalog",
        "k1": k1,
        "tokenizer": tokenizer_mode,
    }
    queries = _load_queries(standardized_dir / f"records_{split}.jsonl", max_requests=max_requests)
    shuffled_queries = _shuffled_queries(queries, seed=seed)
    random_catalog = _reservoir_sample_items(
        standardized_dir / "item_catalog.jsonl",
        sample_size=random_catalog_size,
        seed=seed + 17,
    )
    query_pool_diagnostics = _run_query_pool_diagnostics(
        records_path=standardized_dir / f"records_{split}.jsonl",
        shuffled_queries=shuffled_queries,
        random_catalog=random_catalog,
        bm25_stats=bm25_stats,
        bm25_kwargs={
            "k1": k1,
            "b": b,
            "tokenizer_mode": tokenizer_mode,
            "exact_match_boost": exact_match_boost,
            "char_coverage_boost": char_coverage_boost,
        },
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        max_requests=max_requests,
    )
    relevance_diagnostics = _run_relevance_pairwise_diagnostic(
        relevance_path=raw_dir / "relevance" / "train.jsonl",
        bm25_stats=bm25_stats,
        bm25_kwargs={
            "k1": k1,
            "b": b,
            "tokenizer_mode": tokenizer_mode,
            "exact_match_boost": exact_match_boost,
            "char_coverage_boost": char_coverage_boost,
        },
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    b0a_diagnostics = _run_b0a_audit(
        standardized_dir=standardized_dir,
        runs_dir=runs_dir,
        b0a_run_id=b0a_run_id,
    )
    report = {
        "b0a_audit": b0a_diagnostics,
        "b1_run_id": b1_run_id,
        "bm25_config": bm25_config,
        "candidate_pool_vs_random_catalog": query_pool_diagnostics["candidate_pool_vs_random_catalog"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": _interpret_diagnostics(
            query_pool_diagnostics["shuffled_query_canary"],
            query_pool_diagnostics["candidate_pool_vs_random_catalog"],
            relevance_diagnostics,
            b0a_diagnostics,
        ),
        "label_policy": {
            "click_qrels_read": False,
            "dev_eval_log_written": False,
            "notes": (
                "Diagnostics do not produce dev metrics and do not read qrels_dev/test. "
                "The relevance-table check uses KuaiSearch relevance/train.jsonl, not click labels."
            ),
        },
        "random_seed": seed,
        "relevance_pairwise": relevance_diagnostics,
        "shuffled_query_canary": query_pool_diagnostics["shuffled_query_canary"],
        "split": split,
    }
    write_json(output_json_path, report)
    if output_markdown_path:
        _write_markdown(output_markdown_path, report)
    return report


def _load_queries(records_path: Path, max_requests: int | None) -> list[str]:
    queries = []
    for index, record in enumerate(iter_jsonl(records_path)):
        if max_requests is not None and index >= max_requests:
            break
        queries.append(str(record.get("query") or ""))
    if len(queries) < 2:
        raise ValueError(f"need at least two records for query shuffle: {records_path}")
    return queries


def _shuffled_queries(queries: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    shuffled = list(queries)
    rng.shuffle(shuffled)
    if len(shuffled) > 1:
        for index, query in enumerate(queries):
            if shuffled[index] == query:
                swap_index = (index + 1) % len(shuffled)
                shuffled[index], shuffled[swap_index] = shuffled[swap_index], shuffled[index]
    return shuffled


def _reservoir_sample_items(path: Path, sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("random_catalog_size must be positive")
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for item in iter_jsonl(path):
        normalized = {
            "brand": item.get("brand") or "",
            "cat": item.get("cat") or [],
            "item_id": str(item["item_id"]),
            "seller": item.get("seller") or "",
            "title": item.get("title") or "",
        }
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(normalized)
            continue
        replacement = rng.randrange(seen)
        if replacement < sample_size:
            reservoir[replacement] = normalized
    if not reservoir:
        raise ValueError(f"empty item catalog: {path}")
    return reservoir


def _run_query_pool_diagnostics(
    records_path: Path,
    shuffled_queries: list[str],
    random_catalog: list[dict[str, Any]],
    bm25_stats: dict[str, Any],
    bm25_kwargs: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
    max_requests: int | None,
) -> dict[str, Any]:
    rng = random.Random(seed + 31)
    shuffled_deltas: list[float] = []
    shuffled_original_means: list[float] = []
    shuffled_random_means: list[float] = []
    candidate_deltas: list[float] = []
    actual_candidate_means: list[float] = []
    random_candidate_means: list[float] = []
    same_query_after_shuffle = 0
    nondegenerate = Counter()
    examples = {
        "candidate_pool_advantage": [],
        "shuffled_query_failure": [],
    }
    requests = 0
    random_candidate_rows = 0
    actual_candidate_rows = 0
    for index, record in enumerate(iter_jsonl(records_path)):
        if max_requests is not None and index >= max_requests:
            break
        query = str(record.get("query") or "")
        shuffled_query = shuffled_queries[index]
        if shuffled_query == query:
            same_query_after_shuffle += 1
        original_scores = _score_record(record, bm25_stats=bm25_stats, **bm25_kwargs)
        shuffled_record = dict(record)
        shuffled_record["query"] = shuffled_query
        shuffled_scores = _score_record(shuffled_record, bm25_stats=bm25_stats, **bm25_kwargs)
        random_candidates = _sample_random_candidates(
            random_catalog,
            actual_item_ids={str(candidate["item_id"]) for candidate in record["candidates"]},
            count=len(record["candidates"]),
            rng=rng,
        )
        random_record = {
            "candidates": random_candidates,
            "query": query,
            "request_id": record["request_id"],
        }
        random_scores = _score_record(random_record, bm25_stats=bm25_stats, **bm25_kwargs)

        original_values = list(original_scores.values())
        shuffled_values = list(shuffled_scores.values())
        random_values = list(random_scores.values())
        original_mean = _mean(original_values)
        shuffled_mean = _mean(shuffled_values)
        random_mean = _mean(random_values)

        shuffled_original_means.append(original_mean)
        shuffled_random_means.append(shuffled_mean)
        shuffled_deltas.append(original_mean - shuffled_mean)
        actual_candidate_means.append(original_mean)
        random_candidate_means.append(random_mean)
        candidate_deltas.append(original_mean - random_mean)
        actual_candidate_rows += len(original_values)
        random_candidate_rows += len(random_values)
        requests += 1

        if _all_zero(original_values):
            nondegenerate["all_zero_requests"] += 1
        if _all_tie(original_values):
            nondegenerate["all_tie_requests"] += 1
        if original_mean <= shuffled_mean and len(examples["shuffled_query_failure"]) < 10:
            examples["shuffled_query_failure"].append(
                {
                    "original_mean": original_mean,
                    "query": query,
                    "request_id": record["request_id"],
                    "shuffled_mean": shuffled_mean,
                    "shuffled_query": shuffled_query,
                }
            )
        if original_mean <= random_mean and len(examples["candidate_pool_advantage"]) < 10:
            examples["candidate_pool_advantage"].append(
                {
                    "actual_candidate_mean": original_mean,
                    "query": query,
                    "random_catalog_mean": random_mean,
                    "request_id": record["request_id"],
                }
            )

    if requests == 0:
        raise ValueError(f"empty records file: {records_path}")
    shuffled = _paired_diagnostic_summary(
        left_name="original_query_candidate_mean",
        right_name="shuffled_query_candidate_mean",
        left_values=shuffled_original_means,
        right_values=shuffled_random_means,
        deltas=shuffled_deltas,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    shuffled.update(
        {
            "examples_original_not_above_shuffled": examples["shuffled_query_failure"],
            "request_count": requests,
            "same_query_after_shuffle": same_query_after_shuffle,
            "status": (
                "passed"
                if shuffled["paired_delta_ci95"][0] > 0.0
                and shuffled["left_greater_rate"] >= 0.70
                else "failed"
            ),
        }
    )
    candidate = _paired_diagnostic_summary(
        left_name="actual_candidate_mean",
        right_name="random_catalog_mean",
        left_values=actual_candidate_means,
        right_values=random_candidate_means,
        deltas=candidate_deltas,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 1,
    )
    candidate.update(
        {
            "actual_candidate_rows": actual_candidate_rows,
            "examples_actual_not_above_random": examples["candidate_pool_advantage"],
            "random_candidate_rows": random_candidate_rows,
            "random_catalog_sample_size": len(random_catalog),
            "request_count": requests,
            "status": "passed" if candidate["paired_delta_ci95"][0] > 0.0 else "failed",
        }
    )
    nondegenerate_report = {
        "all_tie_request_rate": nondegenerate["all_tie_requests"] / requests,
        "all_tie_requests": nondegenerate["all_tie_requests"],
        "all_zero_request_rate": nondegenerate["all_zero_requests"] / requests,
        "all_zero_requests": nondegenerate["all_zero_requests"],
        "request_count": requests,
    }
    shuffled["original_score_nondegeneracy"] = nondegenerate_report
    return {
        "candidate_pool_vs_random_catalog": candidate,
        "shuffled_query_canary": shuffled,
    }


def _score_record(
    record: dict[str, Any],
    k1: float,
    b: float,
    tokenizer_mode: str,
    exact_match_boost: float,
    bm25_stats: dict[str, Any],
    char_coverage_boost: float,
) -> dict[str, float]:
    return _bm25_scores(
        record,
        k1=k1,
        b=b,
        tokenizer_mode=tokenizer_mode,
        exact_match_boost=exact_match_boost,
        bm25_stats=bm25_stats,
        char_coverage_boost=char_coverage_boost,
    )


def _sample_random_candidates(
    catalog: list[dict[str, Any]],
    actual_item_ids: set[str],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    attempts = 0
    max_attempts = max(100, count * 50)
    while len(selected) < count and attempts < max_attempts:
        attempts += 1
        item = catalog[rng.randrange(len(catalog))]
        item_id = str(item["item_id"])
        if item_id in actual_item_ids or item_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item_id)
    if len(selected) < count:
        for item in catalog:
            item_id = str(item["item_id"])
            if item_id in actual_item_ids or item_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item_id)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f"could not sample {count} random catalog items")
    return selected


def _run_relevance_pairwise_diagnostic(
    relevance_path: Path,
    bm25_stats: dict[str, Any],
    bm25_kwargs: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    relevance_rows = []
    for row_index, row in enumerate(iter_jsonl(relevance_path)):
        rel = int(row.get("score", -1))
        if rel < 0:
            continue
        relevance_rows.append(
            {
                "document": _relevance_document_text(row),
                "query": str(row.get("query") or ""),
                "rel": rel,
                "row_index": row_index,
            }
        )
    shuffled_queries = _shuffled_queries(
        [str(row["query"]) for row in relevance_rows],
        seed=seed + 41,
    )
    by_query: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    advantages_by_score: dict[int, list[float]] = defaultdict(list)
    true_scores_by_score: dict[int, list[float]] = defaultdict(list)
    score_counts = Counter()
    for index, row in enumerate(relevance_rows):
        query = str(row["query"])
        rel = int(row["rel"])
        document = str(row["document"])
        item_id = f"rel_{row['row_index']}"
        true_score = _score_record(
            {
                "candidates": [
                    {
                        "brand": "",
                        "cat": [],
                        "item_id": item_id,
                        "seller": "",
                        "title": document,
                    }
                ],
                "query": query,
            },
            bm25_stats=bm25_stats,
            **bm25_kwargs,
        )[item_id]
        shuffled_score = _score_record(
            {
                "candidates": [
                    {
                        "brand": "",
                        "cat": [],
                        "item_id": item_id,
                        "seller": "",
                        "title": document,
                    }
                ],
                "query": shuffled_queries[index],
            },
            bm25_stats=bm25_stats,
            **bm25_kwargs,
        )[item_id]
        advantage = true_score - shuffled_score
        by_query[query][rel].append(true_score)
        true_scores_by_score[rel].append(true_score)
        advantages_by_score[rel].append(advantage)
        score_counts[rel] += 1
    same_query_accuracies = []
    total_pairs = 0
    total_wins = 0
    total_ties = 0
    example_failures = []
    for query, groups in by_query.items():
        high = groups.get(3, [])
        low = groups.get(0, [])
        if not high or not low:
            continue
        wins = 0
        ties = 0
        pairs = 0
        for high_score in high:
            for low_score in low:
                pairs += 1
                if high_score > low_score:
                    wins += 1
                elif high_score == low_score:
                    ties += 1
        if pairs:
            accuracy = (wins + 0.5 * ties) / pairs
            same_query_accuracies.append(accuracy)
            total_pairs += pairs
            total_wins += wins
            total_ties += ties
            if accuracy < 0.5 and len(example_failures) < 10:
                example_failures.append(
                    {
                        "pairwise_accuracy": accuracy,
                        "query": query,
                        "rel0_count": len(low),
                        "rel3_count": len(high),
                    }
                )
    same_query_accuracy = (total_wins + 0.5 * total_ties) / total_pairs if total_pairs else 0.0
    same_query_ci = (
        _bootstrap_ci(same_query_accuracies, samples=bootstrap_samples, seed=seed + 2)
        if same_query_accuracies
        else [0.0, 0.0]
    )
    same_query_status = (
        "passed"
        if len(same_query_accuracies) >= 30 and same_query_accuracy >= 0.60 and same_query_ci[0] > 0.50
        else "low_support"
        if len(same_query_accuracies) < 30
        else "failed"
    )
    high_advantages = advantages_by_score.get(3, [])
    low_advantages = advantages_by_score.get(0, [])
    if high_advantages and low_advantages:
        advantage_auc = _auc(positive=high_advantages, negative=low_advantages)
        advantage_ci = _bootstrap_auc_ci(
            positive=high_advantages,
            negative=low_advantages,
            samples=bootstrap_samples,
            seed=seed + 3,
        )
        label_advantage_status = (
            "passed" if advantage_auc >= 0.60 and advantage_ci[0] > 0.55 else "failed"
        )
    else:
        advantage_auc = 0.0
        advantage_ci = [0.0, 0.0]
        label_advantage_status = "failed"
    return {
        "advantage_by_score": {
            str(score): _summarize_values(values)
            for score, values in sorted(advantages_by_score.items())
        },
        "example_queries_below_random": example_failures,
        "label_advantage_auc_rel3_over_rel0": advantage_auc,
        "label_advantage_auc_rel3_over_rel0_ci95": advantage_ci,
        "label_advantage_status": label_advantage_status,
        "pairwise_accuracy": same_query_accuracy,
        "pairwise_accuracy_macro_ci95": same_query_ci,
        "query_count": len(by_query),
        "queries_with_rel3_and_rel0": len(same_query_accuracies),
        "relevance_path": str(relevance_path),
        "relevance_rows": len(relevance_rows),
        "rows_by_score": {str(score): count for score, count in sorted(score_counts.items())},
        "same_query_pairwise_status": same_query_status,
        "status": label_advantage_status,
        "total_pairs_rel3_vs_rel0": total_pairs,
        "true_score_by_score": {
            str(score): _summarize_values(values)
            for score, values in sorted(true_scores_by_score.items())
        },
    }


def _relevance_document_text(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("item_title") or ""),
            str(row.get("brand") or ""),
            str(row.get("seller_name") or ""),
            str(row.get("attr_value") or ""),
        ]
    )


def _run_b0a_audit(
    standardized_dir: Path,
    runs_dir: Path,
    b0a_run_id: str,
) -> dict[str, Any]:
    metadata_path = runs_dir / b0a_run_id / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    stats_path = Path(metadata["popularity_stats_path"])
    train_counts = _count_train_popularity(standardized_dir / "records_train.jsonl")
    stats_compare = _compare_popularity_stats(stats_path, train_counts)
    split_ranges = {
        split: _scan_split_time_range(standardized_dir / f"records_{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    position = _audit_popularity_position_correlation(
        records_path=standardized_dir / "records_dev.jsonl",
        click_counts=train_counts["clicked"],
    )
    train_before_dev = (
        split_ranges["train"]["max_ts"] is not None
        and split_ranges["dev"]["min_ts"] is not None
        and split_ranges["train"]["max_ts"] < split_ranges["dev"]["min_ts"]
    )
    dev_before_test = (
        split_ranges["dev"]["max_ts"] is not None
        and split_ranges["test"]["min_ts"] is not None
        and split_ranges["dev"]["max_ts"] < split_ranges["test"]["min_ts"]
    )
    return {
        "b0a_metadata": {
            "input_fields_used": metadata.get("input_fields_used"),
            "popularity_stats_path": str(stats_path),
            "popularity_stats_sha256": metadata.get("popularity_stats_sha256"),
            "qrels_read": metadata.get("qrels_read"),
            "score_definition": metadata.get("score_definition"),
        },
        "dev_candidate_order_vs_train_popularity": position,
        "split_time_ranges": split_ranges,
        "stats_match_train_records": stats_compare,
        "status": "passed" if stats_compare["exact_match"] and train_before_dev else "failed",
        "train_before_dev": train_before_dev,
        "dev_before_test": dev_before_test,
    }


def _count_train_popularity(path: Path) -> dict[str, Any]:
    clicked = Counter()
    exposed = Counter()
    purchased = Counter()
    for record in iter_jsonl(path):
        for candidate in record["candidates"]:
            item_id = str(candidate["item_id"])
            exposed[item_id] += 1
            clicked[item_id] += int(candidate.get("clicked", 0) or 0)
            purchased[item_id] += int(candidate.get("purchased", 0) or 0)
    return {
        "clicked": dict(clicked),
        "exposed": dict(exposed),
        "purchased": dict(purchased),
    }


def _compare_popularity_stats(stats_path: Path, train_counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    mismatches = []
    stats_items = set()
    totals = Counter()
    for row in iter_jsonl(stats_path):
        item_id = str(row["item_id"])
        stats_items.add(item_id)
        expected = {
            "clicked": int(train_counts["clicked"].get(item_id, 0)),
            "exposed": int(train_counts["exposed"].get(item_id, 0)),
            "purchased": int(train_counts["purchased"].get(item_id, 0)),
        }
        observed = {
            "clicked": int(row.get("clicked", 0) or 0),
            "exposed": int(row.get("exposed", 0) or 0),
            "purchased": int(row.get("purchased", 0) or 0),
        }
        for key, value in observed.items():
            totals[f"stats_{key}"] += value
        if observed != expected and len(mismatches) < 10:
            mismatches.append({"expected": expected, "item_id": item_id, "observed": observed})
    train_items = set(train_counts["exposed"])
    for name, values in train_counts.items():
        totals[f"train_{name}"] = sum(int(value) for value in values.values())
    missing_in_stats = sorted(train_items - stats_items)[:10]
    extra_in_stats = sorted(stats_items - train_items)[:10]
    exact_match = not mismatches and not missing_in_stats and not extra_in_stats
    return {
        "exact_match": exact_match,
        "extra_items_in_stats_examples": extra_in_stats,
        "missing_train_items_in_stats_examples": missing_in_stats,
        "mismatch_examples": mismatches,
        "stats_item_count": len(stats_items),
        "stats_path": str(stats_path),
        "stats_sha256": sha256_file(stats_path),
        "totals": dict(totals),
        "train_item_count": len(train_items),
    }


def _scan_split_time_range(path: Path) -> dict[str, Any]:
    min_ts = None
    max_ts = None
    rows = 0
    for record in iter_jsonl(path):
        ts = int(record["ts"])
        min_ts = ts if min_ts is None else min(min_ts, ts)
        max_ts = ts if max_ts is None else max(max_ts, ts)
        rows += 1
    return {"max_ts": max_ts, "min_ts": min_ts, "request_count": rows}


def _audit_popularity_position_correlation(
    records_path: Path,
    click_counts: dict[str, int],
) -> dict[str, Any]:
    positions = []
    log_popularities = []
    bucket_sums = Counter()
    bucket_counts = Counter()
    for record in iter_jsonl(records_path):
        candidates = record["candidates"]
        denom = max(1, len(candidates) - 1)
        for index, candidate in enumerate(candidates):
            position = index / denom
            log_popularity = math.log1p(int(click_counts.get(str(candidate["item_id"]), 0)))
            positions.append(position)
            log_popularities.append(log_popularity)
            bucket = min(4, int(position * 5))
            bucket_sums[bucket] += log_popularity
            bucket_counts[bucket] += 1
    bucket_means = {
        str(bucket): {
            "candidate_rows": bucket_counts[bucket],
            "mean_log1p_train_clicks": bucket_sums[bucket] / bucket_counts[bucket],
        }
        for bucket in range(5)
        if bucket_counts[bucket]
    }
    return {
        "candidate_rows": len(positions),
        "mean_log1p_train_clicks_by_normalized_position_quintile": bucket_means,
        "pearson_normalized_position_vs_log1p_train_clicks": _pearson(positions, log_popularities),
        "spearman_normalized_position_vs_log1p_train_clicks": _spearman(positions, log_popularities),
    }


def _paired_diagnostic_summary(
    left_name: str,
    right_name: str,
    left_values: list[float],
    right_values: list[float],
    deltas: list[float],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        f"{left_name}_summary": _summarize_values(left_values),
        f"{right_name}_summary": _summarize_values(right_values),
        "left_greater_rate": sum(1 for value in deltas if value > 0.0) / len(deltas),
        "paired_delta_ci95": _bootstrap_ci(deltas, samples=bootstrap_samples, seed=seed),
        "paired_delta_summary": _summarize_values(deltas),
    }


def _interpret_diagnostics(
    shuffled_query: dict[str, Any],
    candidate_pool: dict[str, Any],
    relevance: dict[str, Any],
    b0a: dict[str, Any],
) -> dict[str, Any]:
    bm25_instrument_ok = (
        shuffled_query["status"] == "passed" and relevance["status"] == "passed"
    )
    candidate_query_conditioned = candidate_pool["status"] == "passed"
    b0a_no_train_window_leak = b0a["status"] == "passed"
    if bm25_instrument_ok and candidate_query_conditioned and b0a_no_train_window_leak:
        conclusion = (
            "Diagnostics support the data-property explanation: BM25 responds to the true query, "
            "the fixed candidate pools are already much more query-relevant than random catalog items, "
            "and B0a statistics match train-only records."
        )
    elif not bm25_instrument_ok:
        conclusion = (
            "Diagnostics do not yet clear the BM25 instrument; inspect query fields, tokenizer, "
            "and document template before amending C2."
        )
    elif not b0a_no_train_window_leak:
        conclusion = (
            "Diagnostics found a B0a audit problem; resolve popularity leakage before amending C2."
        )
    else:
        conclusion = (
            "Diagnostics are mixed; keep C2 failed unless a human explicitly amends the gate."
        )
    return {
        "b0a_no_train_window_leak": b0a_no_train_window_leak,
        "bm25_instrument_ok": bm25_instrument_ok,
        "candidate_pool_query_conditioned": candidate_query_conditioned,
        "conclusion": conclusion,
        "c2_gate_effect": (
            "No automatic C2 pass is granted. These diagnostics can support an explicit "
            "post-hoc C2 amendment, with the original B1-vs-B0a failure retained."
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    shuffled = report["shuffled_query_canary"]
    candidate = report["candidate_pool_vs_random_catalog"]
    relevance = report["relevance_pairwise"]
    b0a = report["b0a_audit"]
    lines = [
        "# C2 B1 BM25 Diagnostics",
        "",
        f"Date: {report['generated_at']}",
        "",
        "## Summary",
        "",
        report["interpretation"]["conclusion"],
        "",
        "This report does not change C2 status by itself. It is evidence for an explicit gate amendment only.",
        "",
        "## Results",
        "",
        f"- Shuffled-query canary: `{shuffled['status']}`; mean delta "
        f"{shuffled['paired_delta_summary']['mean']:.6f}; CI95 "
        f"[{shuffled['paired_delta_ci95'][0]:.6f}, {shuffled['paired_delta_ci95'][1]:.6f}]; "
        f"left-greater rate {shuffled['left_greater_rate']:.3f}.",
        f"- Candidate pool vs random catalog: `{candidate['status']}`; mean delta "
        f"{candidate['paired_delta_summary']['mean']:.6f}; CI95 "
        f"[{candidate['paired_delta_ci95'][0]:.6f}, {candidate['paired_delta_ci95'][1]:.6f}]; "
        f"left-greater rate {candidate['left_greater_rate']:.3f}.",
        f"- Relevance rel=3 vs rel=0 pairwise: `{relevance['status']}`; accuracy "
        f"{relevance['label_advantage_auc_rel3_over_rel0']:.4f}; AUC CI95 "
        f"[{relevance['label_advantage_auc_rel3_over_rel0_ci95'][0]:.4f}, "
        f"{relevance['label_advantage_auc_rel3_over_rel0_ci95'][1]:.4f}]; "
        f"same-query pairwise `{relevance['same_query_pairwise_status']}`.",
        f"- B0a train-only audit: `{b0a['status']}`; stats exact match "
        f"{b0a['stats_match_train_records']['exact_match']}; train_before_dev {b0a['train_before_dev']}.",
        "",
        "## Caveats",
        "",
        "- The random catalog pool is `data/standardized/kuaisearch/v0_lite/item_catalog.jsonl`, "
        "not the full raw item table.",
        "- The relevance-table check uses independent relevance labels, not dev/test click qrels.",
        "- Candidate-order popularity correlation can indicate online-ranking or position bias, "
        "but it is not by itself a train-window leakage finding.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if samples <= 0:
        return [min(values), max(values)]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def _bootstrap_auc_ci(
    positive: list[float],
    negative: list[float],
    samples: int,
    seed: int,
) -> list[float]:
    if not positive or not negative:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n_pos = len(positive)
    n_neg = len(negative)
    values = []
    capped_samples = min(samples, 500)
    for _ in range(capped_samples):
        pos_sample = [positive[rng.randrange(n_pos)] for _ in range(n_pos)]
        neg_sample = [negative[rng.randrange(n_neg)] for _ in range(n_neg)]
        values.append(_auc(pos_sample, neg_sample))
    values.sort()
    return [
        values[int(0.025 * capped_samples)],
        values[min(capped_samples - 1, int(0.975 * capped_samples))],
    ]


def _auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return 0.0
    combined = [(value, 1) for value in positive] + [(value, 0) for value in negative]
    combined.sort(key=lambda pair: pair[0])
    rank_sum_positive = 0.0
    position = 0
    while position < len(combined):
        end = position + 1
        while end < len(combined) and combined[end][0] == combined[position][0]:
            end += 1
        average_rank = (position + end - 1) / 2.0
        positives_in_tie = sum(label for _, label in combined[position:end])
        rank_sum_positive += positives_in_tie * average_rank
        position = end
    u_stat = rank_sum_positive - (len(positive) * (len(positive) - 1) / 2.0)
    return u_stat / (len(positive) * len(negative))


def _summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "max": None, "mean": None, "min": None, "p50": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "max": ordered[-1],
        "mean": sum(values) / len(values),
        "min": ordered[0],
        "p05": ordered[int(0.05 * (len(ordered) - 1))],
        "p50": ordered[int(0.50 * (len(ordered) - 1))],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _all_zero(values: list[float]) -> bool:
    return all(value == 0.0 for value in values)


def _all_tie(values: list[float]) -> bool:
    if not values:
        return True
    first = values[0]
    return all(value == first for value in values)


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denom = math.sqrt(left_var * right_var)
    return numerator / denom if denom else 0.0


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks
