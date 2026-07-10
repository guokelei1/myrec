"""Freshness-matched, strictly-prior wrong-user history controls."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from myrec.analysis.history_identity import (
    history_length_bin,
    majority_top_category,
    normalize_query,
)
from myrec.baselines.core import recent_behavior_scores, write_static_mixture_scores
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


MATCH_TIERS = (
    "query_length",
    "query",
    "category_length",
    "category",
    "length",
    "global",
)

EXPECTED_DONOR_PRIORITY = [
    "normalized_query_and_history_length_bin",
    "normalized_query",
    "candidate_major_category_and_history_length_bin",
    "candidate_major_category",
    "history_length_bin",
    "global",
]


@dataclass(frozen=True)
class TemporalSnapshot:
    request_id: str
    user_id: str
    query_key: str
    request_ts: int
    latest_event_ts: int
    history: tuple[dict[str, Any], ...]
    major_category: str
    length_bin: int
    source_split: str


@dataclass(frozen=True)
class TemporalTarget:
    request_id: str
    user_id: str
    query_key: str
    request_ts: int
    history: tuple[dict[str, Any], ...]
    major_category: str
    length_bin: int
    candidates: tuple[dict[str, Any], ...]

    @property
    def latest_event_ts(self) -> int | None:
        if not self.history:
            return None
        return max(int(event["ts"]) for event in self.history)


@dataclass(frozen=True)
class TemporalAssignment:
    tier: str
    donor: TemporalSnapshot
    log2_age_gap: float
    balanced: bool


class RecentSnapshotPools:
    """Retain the most recent snapshots for every matching key."""

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._pools: dict[str, dict[Any, list[TemporalSnapshot]]] = defaultdict(dict)

    def offer(self, tier: str, key: Any, snapshot: TemporalSnapshot) -> None:
        values = self._pools[tier].setdefault(key, [])
        values.append(snapshot)
        values.sort(
            key=lambda row: (row.latest_event_ts, row.request_ts, row.request_id),
            reverse=True,
        )
        del values[self.max_size :]

    def get(self, tier: str, key: Any) -> tuple[TemporalSnapshot, ...]:
        return tuple(self._pools.get(tier, {}).get(key, ()))

    def pool_sizes(self) -> dict[str, dict[str, int]]:
        return {
            tier: {
                "keys": len(values),
                "retained_references": sum(len(rows) for rows in values.values()),
            }
            for tier, values in sorted(self._pools.items())
        }


def history_age(target_ts: int, latest_event_ts: int) -> int:
    age = int(target_ts) - int(latest_event_ts)
    if age <= 0:
        raise ValueError(
            f"history must be strictly prior: target_ts={target_ts} "
            f"latest_event_ts={latest_event_ts}"
        )
    return age


def log2_age_gap(target: TemporalTarget, donor: TemporalSnapshot) -> float:
    if target.latest_event_ts is None:
        raise ValueError("cannot compare freshness for an empty target history")
    target_age = history_age(target.request_ts, target.latest_event_ts)
    donor_age = history_age(target.request_ts, donor.latest_event_ts)
    return abs(math.log2((donor_age + 1.0) / (target_age + 1.0)))


def select_temporal_donor(
    target: TemporalTarget,
    pools: RecentSnapshotPools,
    seed: int,
    max_log2_age_gap: float,
    top_k: int,
) -> TemporalAssignment:
    """Select the first context-matched donor that also passes freshness balance."""

    if not target.history:
        raise ValueError("donor selection requires a non-empty target history")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    tier_keys = (
        ("query_length", (target.query_key, target.length_bin)),
        ("query", target.query_key),
        ("category_length", (target.major_category, target.length_bin)),
        ("category", target.major_category),
        ("length", target.length_bin),
        ("global", "all"),
    )
    fallback: tuple[str, list[TemporalSnapshot]] | None = None
    for tier, key in tier_keys:
        eligible = [
            donor
            for donor in pools.get(tier, key)
            if donor.user_id != target.user_id
            and donor.request_ts < target.request_ts
            and donor.latest_event_ts < donor.request_ts
        ]
        if not eligible:
            continue
        ranked = sorted(
            eligible,
            key=lambda donor: (
                log2_age_gap(target, donor),
                abs(len(donor.history) - len(target.history)),
                -donor.latest_event_ts,
                donor.request_id,
            ),
        )
        if fallback is None:
            fallback = (tier, ranked)
        balanced = [
            donor
            for donor in ranked
            if log2_age_gap(target, donor) <= max_log2_age_gap
        ][:top_k]
        if balanced:
            donor = min(
                balanced,
                key=lambda row: _stable_int(
                    "temporal-select", str(seed), target.request_id, tier, row.request_id
                ),
            )
            return TemporalAssignment(
                tier=tier,
                donor=donor,
                log2_age_gap=log2_age_gap(target, donor),
                balanced=True,
            )

    if fallback is None:
        raise ValueError(f"no eligible temporal donor for {target.request_id}")
    tier, ranked = fallback
    donor = min(
        ranked[:top_k],
        key=lambda row: _stable_int(
            "temporal-fallback", str(seed), target.request_id, tier, row.request_id
        ),
    )
    return TemporalAssignment(
        tier=tier,
        donor=donor,
        log2_age_gap=log2_age_gap(target, donor),
        balanced=False,
    )


def materialize_temporal_controls(
    config: dict[str, Any],
    config_path: str | Path,
) -> dict[str, Any]:
    """Create temporally matched wrong B0b and D2s scores without qrels."""

    validate_temporal_config(config)
    config_path = Path(config_path)
    config_sha256 = sha256_file(config_path)
    inputs = config["inputs"]
    train_path = Path(inputs["records_train"])
    dev_path = Path(inputs["records_dev"])
    candidate_manifest_path = Path(inputs["candidate_manifest"])
    window_path = Path(inputs["window_requests"])
    expected_window_sha256 = str(inputs["expected_window_requests_sha256"])
    if sha256_file(window_path) != expected_window_sha256:
        raise ValueError("window request hash differs from the frozen config")

    seeds = [int(seed) for seed in config["seeds"]]
    matching = config["matching"]
    upper_bounds = [int(value) for value in matching["history_length_bins"]]
    max_pool_size = int(matching["max_donors_per_pool"])
    max_gap = float(matching["max_log2_age_gap"])
    top_k = int(matching["freshness_top_k"])
    artifacts_dir = Path(config["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    targets = _load_targets(dev_path, upper_bounds)
    target_keys = _target_keys(targets)
    pools = RecentSnapshotPools(max_pool_size)
    donor_scan = Counter()
    for record in iter_jsonl(train_path):
        donor_scan["train_records_scanned"] += 1
        snapshot = _snapshot_from_record(record, upper_bounds, "train")
        if snapshot is None:
            donor_scan["train_empty_history"] += 1
            continue
        _offer_relevant(pools, snapshot, target_keys)
        donor_scan["train_snapshots_offered"] += 1

    candidate_manifest_sha256 = sha256_file(candidate_manifest_path)
    b0b_run_ids = {
        seed: config["runs"]["wrong_b0b_pattern"].format(seed=seed) for seed in seeds
    }
    d2s_run_ids = {
        seed: config["runs"]["wrong_d2s_pattern"].format(seed=seed) for seed in seeds
    }
    b0b_handles: dict[int, Any] = {}
    assignment_handles: dict[int, Any] = {}
    b0b_score_paths: dict[int, Path] = {}
    assignment_paths: dict[int, Path] = {}
    score_rows = Counter()
    tier_counts = {seed: Counter() for seed in seeds}
    balanced_ids = {seed: set() for seed in seeds}
    same_query_balanced_ids = {seed: set() for seed in seeds}
    age_gaps: dict[int, list[float]] = {seed: [] for seed in seeds}
    source_counts = {seed: Counter() for seed in seeds}
    history_present_ids = {target.request_id for target in targets if target.history}
    history_absent_ids = {target.request_id for target in targets if not target.history}

    try:
        for seed in seeds:
            run_dir = Path("runs") / b0b_run_ids[seed]
            run_dir.mkdir(parents=True, exist_ok=True)
            b0b_score_paths[seed] = run_dir / "scores.jsonl"
            assignment_paths[seed] = artifacts_dir / f"donor_assignments_s{seed}.jsonl"
            b0b_handles[seed] = b0b_score_paths[seed].open("w", encoding="utf-8")
            assignment_handles[seed] = assignment_paths[seed].open(
                "w", encoding="utf-8"
            )

        index = 0
        while index < len(targets):
            end = index + 1
            while end < len(targets) and targets[end].request_ts == targets[index].request_ts:
                end += 1

            for target in targets[index:end]:
                for seed in seeds:
                    if target.history:
                        assignment = select_temporal_donor(
                            target=target,
                            pools=pools,
                            seed=seed,
                            max_log2_age_gap=max_gap,
                            top_k=top_k,
                        )
                        donor = assignment.donor
                        history = donor.history
                        tier = assignment.tier
                        balanced = assignment.balanced
                        gap = assignment.log2_age_gap
                        if donor.user_id == target.user_id:
                            raise AssertionError("same-user temporal donor")
                        if donor.request_ts >= target.request_ts:
                            raise AssertionError("future or same-time donor request")
                        if any(int(event["ts"]) >= donor.request_ts for event in history):
                            raise AssertionError("donor history is not prior to donor request")
                        if balanced and gap > max_gap:
                            raise AssertionError("balanced donor exceeds freshness threshold")
                        age_gaps[seed].append(gap)
                        source_counts[seed][donor.source_split] += 1
                        if balanced:
                            balanced_ids[seed].add(target.request_id)
                        if balanced and tier in {"query", "query_length"}:
                            same_query_balanced_ids[seed].add(target.request_id)
                    else:
                        donor = None
                        history = ()
                        tier = "target_history_absent"
                        balanced = True
                        gap = None

                    tier_counts[seed][tier] += 1
                    scores = recent_behavior_scores(
                        {
                            "candidates": list(target.candidates),
                            "history": list(history),
                        }
                    )
                    expected_items = {
                        str(candidate["item_id"]) for candidate in target.candidates
                    }
                    if set(scores) != expected_items:
                        raise AssertionError(f"candidate mismatch for {target.request_id}")
                    for item_id in sorted(scores):
                        b0b_handles[seed].write(
                            json.dumps(
                                {
                                    "candidate_item_id": item_id,
                                    "method_id": "c5r2_b0b_temporal_wrong_history",
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
                            _assignment_row(target, donor, tier, balanced, gap),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )

            # Insert current dev snapshots only after the complete same-time group.
            for target in targets[index:end]:
                snapshot = _snapshot_from_target(target, "earlier_dev")
                if snapshot is not None:
                    _offer_relevant(pools, snapshot, target_keys)
                    donor_scan["dev_snapshots_offered_after_time_group"] += 1
            index = end
    finally:
        for handle in b0b_handles.values():
            handle.close()
        for handle in assignment_handles.values():
            handle.close()

    balanced_all = set.intersection(*(balanced_ids[seed] for seed in seeds))
    same_query_balanced_all = set.intersection(
        *(same_query_balanced_ids[seed] for seed in seeds)
    )
    subset_paths = {
        "history_present": artifacts_dir / "history_present_request_ids.txt",
        "history_absent": artifacts_dir / "history_absent_request_ids.txt",
        "freshness_balanced_all_seeds": (
            artifacts_dir / "freshness_balanced_all_seeds_request_ids.txt"
        ),
        "same_query_freshness_balanced_all_seeds": (
            artifacts_dir / "same_query_freshness_balanced_all_seeds_request_ids.txt"
        ),
    }
    _write_request_ids(subset_paths["history_present"], history_present_ids)
    _write_request_ids(subset_paths["history_absent"], history_absent_ids)
    _write_request_ids(subset_paths["freshness_balanced_all_seeds"], balanced_all)
    _write_request_ids(
        subset_paths["same_query_freshness_balanced_all_seeds"],
        same_query_balanced_all,
    )

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
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_fields_used": [
                "records_train.history/query/user_id/candidates",
                "records_dev.history/query/user_id/candidates",
            ],
            "method_id": "c5r2_b0b_temporal_wrong_history",
            "qrels_read": False,
            "run_id": b0b_run_ids[seed],
            "score_rows": score_rows[seed],
            "seed": seed,
            "split": "dev",
            "test_read": False,
        }
        write_json(Path("runs") / b0b_run_ids[seed] / "metadata.json", metadata)
        shutil.copyfile(
            config_path,
            Path("runs") / b0b_run_ids[seed] / "config_snapshot.yaml",
        )

    beta = float(config["static_mixture"]["beta"])
    d2s_metadata = {}
    for seed in seeds:
        d2p_run_id = config["runs"]["d2p_pattern"].format(seed=seed)
        metadata = write_static_mixture_scores(
            query_scores_path=Path("runs") / d2p_run_id / "scores.jsonl",
            history_scores_path=b0b_score_paths[seed],
            query_run_id=d2p_run_id,
            history_run_id=b0b_run_ids[seed],
            run_id=d2s_run_ids[seed],
            method_id="c5r2_d2s_temporal_wrong_history",
            alpha=beta,
            candidate_manifest_path=candidate_manifest_path,
            config_path=config_path,
        )
        metadata.update(
            {
                "analysis_id": config["analysis_id"],
                "assignment_path": str(assignment_paths[seed]),
                "assignment_sha256": sha256_file(assignment_paths[seed]),
                "config_sha256": config_sha256,
                "history_condition": "temporally_matched_wrong_user",
                "prequential": True,
                "qrels_read": False,
                "seed": seed,
                "test_read": False,
            }
        )
        write_json(Path("runs") / d2s_run_ids[seed] / "metadata.json", metadata)
        d2s_metadata[str(seed)] = metadata

    freshness_stats = {}
    for seed in seeds:
        values = sorted(age_gaps[seed])
        freshness_stats[str(seed)] = {
            "balanced": len(balanced_ids[seed]),
            "history_present": len(history_present_ids),
            "log2_age_gap_max": values[-1],
            "log2_age_gap_median": statistics.median(values),
            "log2_age_gap_p90": _quantile(values, 0.9),
            "same_query_balanced": len(same_query_balanced_ids[seed]),
            "source_counts": dict(sorted(source_counts[seed].items())),
        }

    manifest = {
        "analysis_id": config["analysis_id"],
        "b0b_run_ids": {str(seed): b0b_run_ids[seed] for seed in seeds},
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "d2s_run_ids": {str(seed): d2s_run_ids[seed] for seed in seeds},
        "donor_scan": dict(sorted(donor_scan.items())),
        "freshness": freshness_stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {
            "records_dev": sha256_file(dev_path),
            "records_train": sha256_file(train_path),
            "window_requests": expected_window_sha256,
        },
        "pool_sizes": pools.pool_sizes(),
        "qrels_read": False,
        "seeds": seeds,
        "subset_counts": {
            "freshness_balanced_all_seeds": len(balanced_all),
            "history_absent": len(history_absent_ids),
            "history_present": len(history_present_ids),
            "same_query_freshness_balanced_all_seeds": len(
                same_query_balanced_all
            ),
        },
        "subset_paths": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in subset_paths.items()
        },
        "test_read": False,
        "tier_counts": {
            str(seed): dict(sorted(tier_counts[seed].items())) for seed in seeds
        },
    }
    manifest_path = artifacts_dir / "materialization_manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path), "runs": d2s_metadata}


def validate_temporal_config(config: dict[str, Any]) -> None:
    if config.get("target_split") != "dev":
        raise ValueError("target_split must be dev")
    matching = config["matching"]
    if matching.get("donor_sources") != ["train", "earlier_dev"]:
        raise ValueError("donor_sources must be train then earlier_dev")
    for field in (
        "require_different_user",
        "require_donor_request_strictly_before_target",
        "insert_dev_donors_after_same_timestamp_group",
        "keep_empty_target_history_empty",
    ):
        if matching.get(field) is not True:
            raise ValueError(f"matching.{field} must be true")
    if matching.get("donor_priority") != EXPECTED_DONOR_PRIORITY:
        raise ValueError("matching.donor_priority differs from the frozen rule")
    if float(matching.get("max_log2_age_gap", -1.0)) <= 0.0:
        raise ValueError("matching.max_log2_age_gap must be positive")
    if int(matching.get("freshness_top_k", 0)) < 1:
        raise ValueError("matching.freshness_top_k must be positive")
    if config["static_mixture"].get("zscore_scope") != "within_request":
        raise ValueError("static mixture must use within-request z-scoring")


def adjudicate_temporal_gate(
    gate: dict[str, Any],
    subset_counts: dict[str, int],
    freshness_comparisons: dict[str, dict[str, Any]],
    same_query_comparisons: dict[str, dict[str, Any]],
    integrity_passed: bool,
) -> dict[str, Any]:
    """Apply only the decision rule frozen in the C5-R2 config."""

    freshness_count_ok = (
        int(subset_counts["freshness_balanced_all_seeds"])
        >= int(gate["min_freshness_balanced_requests"])
    )
    same_query_count_ok = (
        int(subset_counts["same_query_freshness_balanced_all_seeds"])
        >= int(gate["min_same_query_freshness_balanced_requests"])
    )
    freshness_all_significant = all(
        float(row["ci95"][0]) > 0.0 for row in freshness_comparisons.values()
    )
    same_query_significant_seeds = sum(
        float(row["ci95"][0]) > 0.0 for row in same_query_comparisons.values()
    )
    same_query_mean_delta = statistics.mean(
        float(row["delta"]) for row in same_query_comparisons.values()
    )
    checks = {
        "freshness_balanced_count_at_least_minimum": freshness_count_ok,
        "all_freshness_balanced_ci_lower_gt_zero": freshness_all_significant,
        "same_query_freshness_balanced_count_at_least_minimum": (
            same_query_count_ok
        ),
        "same_query_mean_delta_gt_zero": same_query_mean_delta > 0.0,
        "same_query_significant_seed_count_at_least_minimum": (
            same_query_significant_seeds
            >= int(gate["same_query_min_significant_seeds"])
        ),
        "assignment_and_no_history_integrity": bool(integrity_passed),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "same_query_mean_delta": same_query_mean_delta,
        "same_query_significant_seed_count": same_query_significant_seeds,
    }


def _load_targets(path: Path, upper_bounds: Iterable[int]) -> list[TemporalTarget]:
    targets = []
    for record in iter_jsonl(path):
        history = _compact_history(record.get("history") or [], int(record["ts"]))
        candidates = tuple(
            {
                "cat": [str(value) for value in candidate.get("cat", [])],
                "item_id": str(candidate["item_id"]),
            }
            for candidate in record["candidates"]
        )
        targets.append(
            TemporalTarget(
                request_id=str(record["request_id"]),
                user_id=str(record["user_id"]),
                query_key=normalize_query(record.get("query")),
                request_ts=int(record["ts"]),
                history=history,
                major_category=majority_top_category(candidates),
                length_bin=history_length_bin(len(history), upper_bounds),
                candidates=candidates,
            )
        )
    targets.sort(key=lambda row: (row.request_ts, row.request_id))
    if not targets:
        raise ValueError(f"no dev targets in {path}")
    return targets


def _snapshot_from_record(
    record: dict[str, Any],
    upper_bounds: Iterable[int],
    source_split: str,
) -> TemporalSnapshot | None:
    request_ts = int(record["ts"])
    history = _compact_history(record.get("history") or [], request_ts)
    if not history:
        return None
    candidates = tuple(
        {
            "cat": [str(value) for value in candidate.get("cat", [])],
            "item_id": str(candidate["item_id"]),
        }
        for candidate in record["candidates"]
    )
    return TemporalSnapshot(
        request_id=str(record["request_id"]),
        user_id=str(record["user_id"]),
        query_key=normalize_query(record.get("query")),
        request_ts=request_ts,
        latest_event_ts=max(int(event["ts"]) for event in history),
        history=history,
        major_category=majority_top_category(candidates),
        length_bin=history_length_bin(len(history), upper_bounds),
        source_split=source_split,
    )


def _snapshot_from_target(
    target: TemporalTarget, source_split: str
) -> TemporalSnapshot | None:
    if not target.history or target.latest_event_ts is None:
        return None
    return TemporalSnapshot(
        request_id=target.request_id,
        user_id=target.user_id,
        query_key=target.query_key,
        request_ts=target.request_ts,
        latest_event_ts=target.latest_event_ts,
        history=target.history,
        major_category=target.major_category,
        length_bin=target.length_bin,
        source_split=source_split,
    )


def _compact_history(
    history: Iterable[dict[str, Any]], request_ts: int
) -> tuple[dict[str, Any], ...]:
    result = tuple(
        {
            "cat": [str(value) for value in event.get("cat", [])],
            "event": str(event.get("event") or "click"),
            "item_id": str(event["item_id"]),
            "ts": int(event["ts"]),
        }
        for event in history
    )
    if any(int(event["ts"]) >= request_ts for event in result):
        raise ValueError("history event is not strictly prior to its request")
    return result


def _target_keys(targets: Iterable[TemporalTarget]) -> dict[str, set[Any]]:
    history_targets = [target for target in targets if target.history]
    return {
        "query_length": {
            (target.query_key, target.length_bin) for target in history_targets
        },
        "query": {target.query_key for target in history_targets},
        "category_length": {
            (target.major_category, target.length_bin) for target in history_targets
        },
        "category": {target.major_category for target in history_targets},
        "length": {target.length_bin for target in history_targets},
    }


def _offer_relevant(
    pools: RecentSnapshotPools,
    snapshot: TemporalSnapshot,
    target_keys: dict[str, set[Any]],
) -> None:
    keys = {
        "query_length": (snapshot.query_key, snapshot.length_bin),
        "query": snapshot.query_key,
        "category_length": (snapshot.major_category, snapshot.length_bin),
        "category": snapshot.major_category,
        "length": snapshot.length_bin,
    }
    for tier, key in keys.items():
        if key in target_keys[tier]:
            pools.offer(tier, key, snapshot)
    pools.offer("global", "all", snapshot)


def _assignment_row(
    target: TemporalTarget,
    donor: TemporalSnapshot | None,
    tier: str,
    balanced: bool,
    gap: float | None,
) -> dict[str, Any]:
    return {
        "balanced": balanced,
        "donor_history_age_at_target": (
            history_age(target.request_ts, donor.latest_event_ts) if donor else None
        ),
        "donor_history_length": len(donor.history) if donor else 0,
        "donor_latest_event_ts": donor.latest_event_ts if donor else None,
        "donor_request_id": donor.request_id if donor else None,
        "donor_request_ts": donor.request_ts if donor else None,
        "donor_source_split": donor.source_split if donor else None,
        "donor_user_id": donor.user_id if donor else None,
        "log2_age_gap": gap,
        "match_tier": tier,
        "request_id": target.request_id,
        "target_history_age": (
            history_age(target.request_ts, target.latest_event_ts)
            if target.latest_event_ts is not None
            else None
        ),
        "target_history_length": len(target.history),
        "target_latest_event_ts": target.latest_event_ts,
        "target_request_ts": target.request_ts,
        "target_user_id": target.user_id,
    }


def _write_request_ids(path: Path, request_ids: set[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for request_id in sorted(request_ids):
            handle.write(request_id + "\n")


def _quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of no values")
    index = min(len(sorted_values) - 1, int(quantile * len(sorted_values)))
    return float(sorted_values[index])


def _stable_int(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
