"""Evaluator-side data/signal/power audit for the mechanism stage.

Unlike materializers and scorers, this module intentionally crosses the dev
label boundary.  It does so only after verifying the frozen development hashes,
and emits aggregate architecture-facing evidence rather than model inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.frozen_text_features import (
    FrozenTextFeatureStore,
    serialize_item_semantic_content,
)
from myrec.baselines.representative_sequence_adapter import serialize_item_content
from myrec.eval.motivation_v12_evidence import normalize_query_cluster
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


RUN_ID_PATTERN = re.compile(r"^\d{8}_[a-z0-9][a-z0-9_]*$")
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260715
Z_975 = 1.959963984540054
Z_80 = 0.8416212335729143


def run_data_power_audit(
    standardized_dir: str | Path,
    feature_store_path: str | Path,
    run_id: str,
    *,
    frozen_analysis_dirs: Mapping[str, str | Path],
    runs_dir: str | Path = "runs",
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
) -> dict[str, Any]:
    """Verify the boundary, open only dev qrels, and write aggregate evidence."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("invalid mechanism run id")
    standardized_dir = Path(standardized_dir)
    paths, protocol = _verify_frozen_internal_dev(standardized_dir)
    output_dir = Path(runs_dir) / run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"audit output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # This is the sole label-opening point in the audit.
    gains = _load_graded_qrels(paths["qrels"])
    records = list(iter_jsonl(paths["records"]))
    if {str(row["request_id"]) for row in records} != set(gains):
        raise ValueError("dev record/qrels request coverage differs")
    store = FrozenTextFeatureStore(feature_store_path, require_fingerprints=True)
    if store.metadata.get("qrels_read") is not False:
        raise ValueError("label-free feature store declares qrels access")

    surface_ids: dict[str, list[str]] = {
        "recurrence": [],
        "strict_transfer": [],
        "other_overlap": [],
        "no_history": [],
        "no_observed_positive": [],
    }
    strict_rows: list[dict[str, Any]] = []
    for record in records:
        request_id = str(record["request_id"])
        candidates = list(record["candidates"])
        candidate_ids = [str(row["item_id"]) for row in candidates]
        history = list(record.get("history") or [])
        history_ids = {str(row["item_id"]) for row in history}
        positive = {
            item_id: float(value)
            for item_id, value in gains[request_id].items()
            if float(value) > 0
        }
        unknown = set(positive) - set(candidate_ids)
        if unknown:
            raise ValueError(f"qrels contain non-candidate items: {request_id}")
        if not positive:
            surface_ids["no_observed_positive"].append(request_id)
            continue
        if history_ids & set(positive):
            surface_ids["recurrence"].append(request_id)
        elif not history:
            surface_ids["no_history"].append(request_id)
        elif history_ids & set(candidate_ids):
            surface_ids["other_overlap"].append(request_id)
        else:
            surface_ids["strict_transfer"].append(request_id)
            strict_rows.append(_strict_request_audit(record, positive, store))
    _assert_partition(surface_ids, len(records))

    data_audit = _aggregate_strict_rows(strict_rows, records, surface_ids)
    power = {
        method_id: _power_from_frozen_analysis(
            Path(analysis_dir),
            records,
            strict_ids=set(surface_ids["strict_transfer"]),
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        for method_id, analysis_dir in sorted(frozen_analysis_dirs.items())
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": 1,
        "analysis_type": "motivation_mechanism_data_signal_power_audit",
        "run_id": run_id,
        "generated_at": generated_at,
        "dataset_id": "kuaisearch",
        "dataset_version": protocol["data"]["development_population"]["dataset_version"],
        "split": "dev",
        "qrels_read": True,
        "source_test_opened": False,
        "qrels_path": str(paths["qrels"]),
        "qrels_sha256": sha256_file(paths["qrels"]),
        "records_path": str(paths["records"]),
        "records_sha256": sha256_file(paths["records"]),
        "candidate_manifest_sha256": sha256_file(paths["candidate_manifest"]),
        "request_manifest_sha256": sha256_file(paths["request_manifest"]),
        "feature_store_path": str(feature_store_path),
        "feature_store_fingerprint_sha256": store.store_fingerprint_sha256,
        "surface_counts": {key: len(value) for key, value in surface_ids.items()},
        "data_signal_audit": data_audit,
        "power_audit": power,
        "bootstrap": {
            "cluster": "normalized_query",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "architecture_links": {
            "query_history_relevance": "bounds the need for explicit query-conditioned history routing; diffuse relevance predicts benefit from sparse selection",
            "brand_category_alignment": "bounds what a factorized cross-item preference bottleneck can recover from current visible fields",
            "semantic_target_competitor_margin": "tests whether candidate-conditioned matching can separate the positive using history semantics without raw ID",
            "minimum_detectable_effect": "bounds claims about null transfer response; it does not by itself diagnose an architecture",
        },
    }
    write_json(output_dir / "data_power_audit.json", result)
    write_json(
        output_dir / "metadata.json",
        {
            "analysis_type": result["analysis_type"],
            "run_id": run_id,
            "generated_at": generated_at,
            "qrels_read": True,
            "source_test_opened": False,
            "output_path": str(output_dir / "data_power_audit.json"),
            "output_sha256": sha256_file(output_dir / "data_power_audit.json"),
        },
    )
    _append_dev_log(
        Path(dev_eval_log_path),
        {
            "timestamp": generated_at,
            "run_id": run_id,
            "analysis_type": result["analysis_type"],
            "method_id": "shared_mechanism_data_evaluator",
            "split": "dev",
            "qrels_sha256": result["qrels_sha256"],
            "strict_transfer_requests": len(surface_ids["strict_transfer"]),
        },
    )
    return result


def _strict_request_audit(
    record: Mapping[str, Any],
    positive: Mapping[str, float],
    store: FrozenTextFeatureStore,
) -> dict[str, Any]:
    candidates = list(record["candidates"])
    history = list(record.get("history") or [])
    candidate_order = {str(row["item_id"]): index for index, row in enumerate(candidates)}
    target_id = min(
        positive,
        key=lambda item_id: (-float(positive[item_id]), candidate_order[item_id]),
    )
    target = next(row for row in candidates if str(row["item_id"]) == target_id)
    competitors = [row for row in candidates if str(row["item_id"]) != target_id]
    query = _unit(store(f"query: {str(record['query']).strip()}"))
    history_semantic = np.stack(
        [_unit(store(serialize_item_semantic_content(row))) for row in history]
    )
    history_contextual = np.stack(
        [_unit(store(serialize_item_content(row))) for row in history]
    )
    target_vector = _unit(store(serialize_item_semantic_content(target)))
    competitor_vectors = np.stack(
        [_unit(store(serialize_item_semantic_content(row))) for row in competitors]
    )
    query_history = history_contextual @ query
    target_history = history_semantic @ target_vector
    competitor_history = competitor_vectors @ history_semantic.T
    target_query = float(target_vector @ query)
    competitor_query = competitor_vectors @ query
    target_brand = _brand(target)
    target_cat = _categories(target)
    history_brands = {_brand(row) for row in history} - {""}
    history_categories = [_categories(row) for row in history]
    category_overlaps = [_category_prefix_overlap(target_cat, value) for value in history_categories]
    better_query_competitors = int((competitor_query >= target_query).sum())
    return {
        "request_id": str(record["request_id"]),
        "normalized_query": normalize_query_cluster(str(record["query"])),
        "history_length": len(history),
        "candidate_count": len(candidates),
        "positive_grade": float(positive[target_id]),
        "query_history_max": float(query_history.max()),
        "query_history_mean": float(query_history.mean()),
        "target_history_max": float(target_history.max()),
        "best_competitor_history_max": float(competitor_history.max()),
        "history_semantic_target_minus_competitor": float(
            target_history.max() - competitor_history.max()
        ),
        "query_target_minus_best_competitor": float(
            target_query - competitor_query.max()
        ),
        "query_better_or_tied_competitors": better_query_competitors,
        "brand_aligned": bool(target_brand and target_brand in history_brands),
        "deepest_category_aligned": bool(
            target_cat
            and any(value and value[-1] == target_cat[-1] for value in history_categories)
        ),
        "category_prefix_aligned": bool(max(category_overlaps, default=0.0) > 0),
        "category_prefix_overlap_max": max(category_overlaps, default=0.0),
        "target_brand_present": bool(target_brand),
        "target_category_present": bool(target_cat),
    }


def _aggregate_strict_rows(
    rows: Sequence[Mapping[str, Any]],
    all_records: Sequence[Mapping[str, Any]],
    surface_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("strict-transfer population is empty")
    clusters = Counter(str(row["normalized_query"]) for row in rows)
    cluster_sizes = np.asarray(list(clusters.values()), dtype=np.float64)
    kish = float(cluster_sizes.sum() ** 2 / np.square(cluster_sizes).sum())
    numeric = (
        "history_length",
        "candidate_count",
        "positive_grade",
        "query_history_max",
        "query_history_mean",
        "target_history_max",
        "best_competitor_history_max",
        "history_semantic_target_minus_competitor",
        "query_target_minus_best_competitor",
        "query_better_or_tied_competitors",
        "category_prefix_overlap_max",
    )
    summary = {key: _distribution([float(row[key]) for row in rows]) for key in numeric}
    for key in (
        "brand_aligned",
        "deepest_category_aligned",
        "category_prefix_aligned",
        "target_brand_present",
        "target_category_present",
    ):
        summary[key] = {
            "count": sum(bool(row[key]) for row in rows),
            "fraction": sum(bool(row[key]) for row in rows) / len(rows),
        }
    summary["effective_query_clusters"] = {
        "unique": len(clusters),
        "kish_effective": kish,
        "max_requests_per_cluster": int(cluster_sizes.max()),
    }
    summary["population_reconstruction"] = {
        "all_requests": len(all_records),
        "partition_total": sum(len(value) for value in surface_ids.values()),
    }
    return summary


def _power_from_frozen_analysis(
    analysis_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    strict_ids: set[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    per_request_path = analysis_dir / "per_request_history_response.jsonl"
    metadata_path = analysis_dir / "metadata.json"
    if not per_request_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"incomplete frozen analysis: {analysis_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("qrels_read") is not True or metadata.get("split") != "dev":
        raise ValueError("power source is not a qrels-reading dev evaluator output")
    query_by_request = {
        str(record["request_id"]): normalize_query_cluster(str(record["query"]))
        for record in records
    }
    by_cluster: dict[str, list[float]] = {}
    observed_ids: set[str] = set()
    for row in iter_jsonl(per_request_path):
        request_id = str(row["request_id"])
        if request_id not in strict_ids:
            continue
        observed_ids.add(request_id)
        by_cluster.setdefault(query_by_request[request_id], []).append(
            float(row["true_minus_null_ndcg@10"])
        )
    if observed_ids != strict_ids:
        raise ValueError("frozen evaluator strict request coverage differs")
    observed = np.asarray([value for rows in by_cluster.values() for value in rows])
    boot = _cluster_bootstrap_means(by_cluster, samples=samples, seed=seed)
    standard_error = float(boot.std(ddof=1))
    mde_80 = (Z_975 + Z_80) * standard_error
    return {
        "analysis_dir": str(analysis_dir),
        "analysis_metadata_sha256": sha256_file(metadata_path),
        "per_request_sha256": sha256_file(per_request_path),
        "requests": len(observed),
        "query_clusters": len(by_cluster),
        "mean": float(observed.mean()),
        "bootstrap_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "bootstrap_standard_error": standard_error,
        "two_sided_alpha_0_05_power_0_80_mde": mde_80,
        "normal_approx_power": {
            str(effect): _normal_power(effect, standard_error)
            for effect in (0.005, 0.01, 0.02)
        },
    }


def _cluster_bootstrap_means(
    by_cluster: Mapping[str, Sequence[float]], *, samples: int, seed: int
) -> np.ndarray:
    keys = sorted(by_cluster)
    if not keys or samples <= 0:
        raise ValueError("cluster bootstrap requires clusters and samples")
    arrays = [np.asarray(by_cluster[key], dtype=np.float64) for key in keys]
    rng = np.random.default_rng(seed)
    result = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        total = sum(float(arrays[row].sum()) for row in selected)
        count = sum(int(arrays[row].size) for row in selected)
        result[index] = total / count
    return result


def _normal_power(effect: float, standard_error: float) -> float:
    if standard_error <= 0:
        return 1.0 if effect else 0.05
    from scipy.stats import norm

    signal = abs(float(effect)) / standard_error
    return float(norm.cdf(-Z_975 - signal) + 1.0 - norm.cdf(Z_975 - signal))


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _verify_frozen_internal_dev(standardized_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    import yaml

    protocol_path = Path("experiments/motivation/protocol.yaml")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    development = protocol["data"]["development_population"]
    if standardized_dir.resolve() != Path(development["standardized_dir"]).resolve():
        raise ValueError("data audit is restricted to frozen v11 internal-dev")
    paths = {
        "records": standardized_dir / "records_dev.jsonl",
        "qrels": standardized_dir / "qrels_dev.jsonl",
        "manifest": standardized_dir / "manifest.json",
        "candidate_manifest": standardized_dir / "candidate_manifest.json",
        "request_manifest": standardized_dir / "request_manifest.json",
    }
    expected = {
        "records": development["records_dev_sha256"],
        "qrels": development["qrels_dev_sha256"],
        "manifest": development["manifest_sha256"],
        "candidate_manifest": development["candidate_manifest_sha256"],
        "request_manifest": development["request_manifest_sha256"],
    }
    # Hash all inputs before the first qrels parse.
    for key, path in paths.items():
        if not path.is_file() or sha256_file(path) != str(expected[key]):
            raise ValueError(f"frozen development input mismatch: {key}")
    return paths, protocol


def _load_graded_qrels(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError("duplicate dev qrels request")
        relevance = row.get("relevance") or {}
        if not isinstance(relevance, dict):
            raise ValueError("graded relevance must be an object")
        gains = {str(key): float(value) for key, value in relevance.items() if float(value) > 0}
        if not gains:
            gains = {
                **{str(value): 1.0 for value in row.get("clicked", [])},
                **{str(value): 2.0 for value in row.get("purchased", [])},
            }
        result[request_id] = gains
    return result


def _assert_partition(surface_ids: Mapping[str, Sequence[str]], expected: int) -> None:
    flattened = [request_id for rows in surface_ids.values() for request_id in rows]
    if len(flattened) != expected or len(set(flattened)) != expected:
        raise AssertionError("mechanism surface partition is not exhaustive/disjoint")


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid frozen feature vector")
    return value / norm


def _brand(row: Mapping[str, Any]) -> str:
    return str(row.get("brand") or "").strip().casefold()


def _categories(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(value).strip().casefold() for value in (row.get("cat") or []) if str(value).strip())


def _category_prefix_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count / max(len(left), len(right))


def _append_dev_log(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
