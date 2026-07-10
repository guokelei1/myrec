"""Matched wrong-user history controls for the repaired PPS motivation."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from myrec.baselines.core import recent_behavior_scores, write_static_mixture_scores
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


UNKNOWN = "UNKNOWN"
EXPECTED_DONOR_PRIORITY = [
    "normalized_query_and_history_length_bin",
    "normalized_query",
    "candidate_major_category_and_history_length_bin",
    "candidate_major_category",
    "history_length_bin",
    "global",
]


@dataclass(frozen=True)
class Donor:
    request_id: str
    user_id: str
    query_key: str
    request_ts: int
    history: tuple[dict[str, Any], ...]
    major_category: str
    length_bin: int


@dataclass(frozen=True)
class Target:
    request_id: str
    user_id: str
    query_key: str
    request_ts: int
    history_length: int
    major_category: str
    length_bin: int
    candidates: tuple[dict[str, Any], ...]


class BoundedDonorPools:
    """Keep stable hash reservoirs for each matching tier and key."""

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._pools: dict[str, dict[Any, list[tuple[int, Donor]]]] = defaultdict(dict)

    def offer(self, tier: str, key: Any, donor: Donor) -> None:
        tier_pools = self._pools[tier]
        bucket = tier_pools.setdefault(key, [])
        rank = _stable_int("pool", tier, _stable_key(key), donor.request_id)
        entry = (rank, donor)
        if len(bucket) < self.max_size:
            bucket.append(entry)
            return
        worst_index = max(range(len(bucket)), key=lambda index: bucket[index][0])
        if rank < bucket[worst_index][0]:
            bucket[worst_index] = entry

    def get(self, tier: str, key: Any) -> list[Donor]:
        entries = self._pools.get(tier, {}).get(key, [])
        return [donor for _, donor in sorted(entries, key=lambda value: value[0])]

    def pool_sizes(self) -> dict[str, dict[str, int]]:
        return {
            tier: {
                "keys": len(pools),
                "retained_references": sum(len(values) for values in pools.values()),
            }
            for tier, pools in sorted(self._pools.items())
        }


def normalize_query(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return "".join(text.split())


def history_length_bin(length: int, upper_bounds: Iterable[int]) -> int:
    if length <= 0:
        return 0
    bounds = [int(value) for value in upper_bounds]
    for upper in bounds:
        if length <= upper:
            return upper
    raise ValueError(f"history length {length} exceeds configured bounds {bounds}")


def majority_top_category(candidates: Iterable[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        categories = candidate.get("cat") or []
        value = str(categories[0]) if categories else UNKNOWN
        if value and value.upper() != UNKNOWN:
            counts[value] += 1
    if not counts:
        return UNKNOWN
    return min(counts, key=lambda value: (-counts[value], value))


def load_targets(
    records_path: str | Path,
    upper_bounds: Iterable[int],
) -> list[Target]:
    targets = []
    for record in iter_jsonl(records_path):
        history = record.get("history") or []
        candidates = tuple(
            {
                "cat": [str(value) for value in candidate.get("cat", [])],
                "item_id": str(candidate["item_id"]),
            }
            for candidate in record["candidates"]
        )
        targets.append(
            Target(
                request_id=str(record["request_id"]),
                user_id=str(record["user_id"]),
                query_key=normalize_query(record.get("query")),
                request_ts=int(record["ts"]),
                history_length=len(history),
                major_category=majority_top_category(candidates),
                length_bin=history_length_bin(len(history), upper_bounds),
                candidates=candidates,
            )
        )
    if not targets:
        raise ValueError(f"no target records in {records_path}")
    return targets


def build_donor_pools(
    records_path: str | Path,
    targets: list[Target],
    upper_bounds: Iterable[int],
    max_pool_size: int,
) -> tuple[BoundedDonorPools, dict[str, int]]:
    history_targets = [target for target in targets if target.history_length > 0]
    if not history_targets:
        raise ValueError("no history-present targets")
    min_target_ts = min(target.request_ts for target in history_targets)
    query_length_keys = {(target.query_key, target.length_bin) for target in history_targets}
    query_keys = {target.query_key for target in history_targets}
    category_length_keys = {
        (target.major_category, target.length_bin) for target in history_targets
    }
    category_keys = {target.major_category for target in history_targets}
    length_keys = {target.length_bin for target in history_targets}

    pools = BoundedDonorPools(max_pool_size)
    stats = Counter()
    for record in iter_jsonl(records_path):
        stats["records_scanned"] += 1
        history = record.get("history") or []
        if not history:
            continue
        request_ts = int(record["ts"])
        if request_ts >= min_target_ts:
            stats["not_strictly_before_dev"] += 1
            continue
        compact_history = tuple(
            {
                "cat": [str(value) for value in event.get("cat", [])],
                "event": str(event.get("event") or "click"),
                "item_id": str(event["item_id"]),
                "ts": int(event["ts"]),
            }
            for event in history
        )
        if any(event["ts"] >= request_ts for event in compact_history):
            stats["donor_history_not_before_donor_request"] += 1
            continue
        candidates = record.get("candidates") or []
        query_key = normalize_query(record.get("query"))
        length_key = history_length_bin(len(compact_history), upper_bounds)
        category_key = majority_top_category(candidates)
        donor = Donor(
            request_id=str(record["request_id"]),
            user_id=str(record["user_id"]),
            query_key=query_key,
            request_ts=request_ts,
            history=compact_history,
            major_category=category_key,
            length_bin=length_key,
        )

        if (query_key, length_key) in query_length_keys:
            pools.offer("query_length", (query_key, length_key), donor)
        if query_key in query_keys:
            pools.offer("query", query_key, donor)
        if (category_key, length_key) in category_length_keys:
            pools.offer("category_length", (category_key, length_key), donor)
        if category_key in category_keys:
            pools.offer("category", category_key, donor)
        if length_key in length_keys:
            pools.offer("length", length_key, donor)
        pools.offer("global", "all", donor)
        stats["eligible_donors"] += 1
    return pools, dict(sorted(stats.items()))


def select_donor(
    target: Target,
    pools: BoundedDonorPools,
    seed: int,
) -> tuple[str, Donor]:
    keys = [
        ("query_length", (target.query_key, target.length_bin)),
        ("query", target.query_key),
        ("category_length", (target.major_category, target.length_bin)),
        ("category", target.major_category),
        ("length", target.length_bin),
        ("global", "all"),
    ]
    for tier, key in keys:
        candidates = [
            donor
            for donor in pools.get(tier, key)
            if donor.user_id != target.user_id and donor.request_ts < target.request_ts
        ]
        if not candidates:
            continue
        donor = min(
            candidates,
            key=lambda value: _stable_int(
                "select", str(seed), target.request_id, tier, value.request_id
            ),
        )
        if any(int(event["ts"]) >= target.request_ts for event in donor.history):
            raise AssertionError(f"future donor event for target {target.request_id}")
        return tier, donor
    raise ValueError(f"no eligible wrong-history donor for {target.request_id}")


def materialize_controls(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    validate_control_config(config)
    standardized_dir = Path(config["standardized_dir"])
    runs_dir = Path(config.get("runs_dir", "runs"))
    artifacts_dir = Path(
        config.get("artifacts_dir", "artifacts/analysis/c3_history_identity_controls")
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    config_sha256 = sha256_file(config_path)
    bounds = config["wrong_history"]["history_length_bins"]
    seeds = [int(value) for value in config["seeds"]]
    max_pool_size = int(config["wrong_history"]["max_donors_per_pool"])

    targets = load_targets(standardized_dir / "records_dev.jsonl", bounds)
    pools, donor_scan = build_donor_pools(
        standardized_dir / "records_train.jsonl",
        targets,
        bounds,
        max_pool_size,
    )

    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    candidate_manifest_sha256 = sha256_file(candidate_manifest_path)
    history_run_ids = {
        seed: f"20260710_kuaisearch_c3r_b0b_wrong_history_dev_s{seed}" for seed in seeds
    }
    mixture_run_ids = {
        seed: f"20260710_kuaisearch_c3r_b7_wrong_history_dev_s{seed}" for seed in seeds
    }
    handles = {}
    assignment_handles = {}
    tier_counts = {seed: Counter() for seed in seeds}
    same_query_ids = {seed: set() for seed in seeds}
    assignment_paths = {}
    score_paths = {}
    try:
        for seed in seeds:
            run_dir = runs_dir / history_run_ids[seed]
            run_dir.mkdir(parents=True, exist_ok=True)
            score_paths[seed] = run_dir / "scores.jsonl"
            assignment_paths[seed] = artifacts_dir / f"donor_assignments_s{seed}.jsonl"
            handles[seed] = score_paths[seed].open("w", encoding="utf-8")
            assignment_handles[seed] = assignment_paths[seed].open("w", encoding="utf-8")

        score_rows = Counter()
        for target in targets:
            for seed in seeds:
                if target.history_length == 0:
                    history: tuple[dict[str, Any], ...] = ()
                    tier = "target_history_absent"
                    donor = None
                else:
                    tier, donor = select_donor(target, pools, seed)
                    history = donor.history
                    if tier in {"query_length", "query"}:
                        same_query_ids[seed].add(target.request_id)
                tier_counts[seed][tier] += 1
                record = {"candidates": list(target.candidates), "history": list(history)}
                scores = recent_behavior_scores(record)
                expected = {str(candidate["item_id"]) for candidate in target.candidates}
                if set(scores) != expected:
                    raise AssertionError(f"candidate mismatch for {target.request_id}")
                for item_id in sorted(scores):
                    handles[seed].write(
                        json.dumps(
                            {
                                "candidate_item_id": item_id,
                                "method_id": "c3r_b0b_wrong_history",
                                "request_id": target.request_id,
                                "score": float(scores[item_id]),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    score_rows[seed] += 1
                assignment_handles[seed].write(
                    json.dumps(
                        {
                            "donor_request_id": donor.request_id if donor else None,
                            "donor_user_id": donor.user_id if donor else None,
                            "match_tier": tier,
                            "request_id": target.request_id,
                            "target_history_length": target.history_length,
                            "target_user_id": target.user_id,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    finally:
        for handle in handles.values():
            handle.close()
        for handle in assignment_handles.values():
            handle.close()

    history_present_ids = {target.request_id for target in targets if target.history_length > 0}
    history_absent_ids = {target.request_id for target in targets if target.history_length == 0}
    same_query_all_seeds = set.intersection(*(same_query_ids[seed] for seed in seeds))
    subset_paths = {
        "history_present": artifacts_dir / "history_present_request_ids.txt",
        "history_absent": artifacts_dir / "history_absent_request_ids.txt",
        "same_query_all_seeds": artifacts_dir / "same_query_all_seeds_request_ids.txt",
    }
    _write_request_ids(subset_paths["history_present"], history_present_ids)
    _write_request_ids(subset_paths["history_absent"], history_absent_ids)
    _write_request_ids(subset_paths["same_query_all_seeds"], same_query_all_seeds)

    for seed in seeds:
        metadata = {
            "analysis_id": config["analysis_id"],
            "assignment_path": str(assignment_paths[seed]),
            "assignment_sha256": sha256_file(assignment_paths[seed]),
            "candidate_manifest_path": str(candidate_manifest_path),
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "dataset_id": config["dataset_id"],
            "dataset_version": config["dataset_version"],
            "donor_split": "train",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_fields_used": [
                "records_train.history",
                "records_train.query",
                "records_train.user_id",
                "records_dev.candidates",
                "records_dev.history_length",
                "records_dev.query",
                "records_dev.user_id",
            ],
            "method_id": "c3r_b0b_wrong_history",
            "qrels_read": False,
            "request_count": len(targets),
            "run_id": history_run_ids[seed],
            "score_definition": "frozen B0b score with matched wrong-user train history",
            "score_rows": score_rows[seed],
            "seed": seed,
            "split": "dev",
            "tier_counts": dict(sorted(tier_counts[seed].items())),
        }
        write_json(runs_dir / history_run_ids[seed] / "metadata.json", metadata)

    query_run_id = config["frozen_runs"]["query"]
    alpha = float(config["static_mixture"]["alpha"])
    for seed in seeds:
        write_static_mixture_scores(
            query_scores_path=runs_dir / query_run_id / "scores.jsonl",
            history_scores_path=score_paths[seed],
            query_run_id=query_run_id,
            history_run_id=history_run_ids[seed],
            run_id=mixture_run_ids[seed],
            method_id="c3r_b7_wrong_history",
            alpha=alpha,
            candidate_manifest_path=candidate_manifest_path,
            runs_dir=runs_dir,
            config_path=config_path,
        )
        metadata_path = runs_dir / mixture_run_ids[seed] / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.update(
            {
                "analysis_id": config["analysis_id"],
                "assignment_path": str(assignment_paths[seed]),
                "assignment_sha256": sha256_file(assignment_paths[seed]),
                "config_sha256": config_sha256,
                "control": "matched_wrong_user_history",
                "qrels_read": False,
                "seed": seed,
                "tier_counts": dict(sorted(tier_counts[seed].items())),
            }
        )
        write_json(metadata_path, metadata)

    manifest = {
        "analysis_id": config["analysis_id"],
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "donor_pool_sizes": pools.pool_sizes(),
        "donor_scan": donor_scan,
        "history_run_ids": {str(seed): history_run_ids[seed] for seed in seeds},
        "mixture_run_ids": {str(seed): mixture_run_ids[seed] for seed in seeds},
        "qrels_read": False,
        "seeds": seeds,
        "subset_counts": {
            "history_absent": len(history_absent_ids),
            "history_present": len(history_present_ids),
            "same_query_all_seeds": len(same_query_all_seeds),
        },
        "subset_paths": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in subset_paths.items()
        },
        "tier_counts": {
            str(seed): dict(sorted(tier_counts[seed].items())) for seed in seeds
        },
    }
    manifest_path = artifacts_dir / "materialization_manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def validate_control_config(config: dict[str, Any]) -> None:
    wrong = config["wrong_history"]
    required = {
        "source_split": "train",
        "target_split": "dev",
    }
    for field, expected in required.items():
        if config.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r}")
    if wrong.get("donor_split") != "train":
        raise ValueError("wrong-history donors must come from train")
    for field in [
        "require_different_user",
        "require_donor_request_before_target_request",
        "keep_empty_target_history_empty",
    ]:
        if wrong.get(field) is not True:
            raise ValueError(f"wrong_history.{field} must be true")
    if wrong.get("donor_priority") != EXPECTED_DONOR_PRIORITY:
        raise ValueError("wrong_history.donor_priority differs from the implemented rule")
    if config["static_mixture"].get("zscore_scope") != "within_request":
        raise ValueError("static mixture must use within-request z-scoring")


def _write_request_ids(path: Path, request_ids: set[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for request_id in sorted(request_ids):
            handle.write(request_id + "\n")


def _stable_int(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
