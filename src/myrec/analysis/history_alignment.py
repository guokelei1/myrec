"""Label-free C5-R3 candidate-history component controls and gate logic."""

from __future__ import annotations

import json
import math
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from myrec.baselines.core import recent_behavior_scores, write_static_mixture_scores
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


EXPECTED_SEEDS = [20260708, 20260709, 20260710]
DECOMPOSITION_TOLERANCE = 1e-12


def candidate_history_components(
    record: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return exact-item and deepest-exclusive category B0b components.

    This intentionally mirrors the frozen executable B0b implementation.  It
    is kept in the analysis module so the component claim can be tested against
    the public full scorer rather than changing shared baseline behavior.
    """

    history = record.get("history") or []
    candidates = record["candidates"]
    if not history:
        zeros = {str(candidate["item_id"]): 0.0 for candidate in candidates}
        return dict(zeros), dict(zeros)

    features = []
    size = len(history)
    for index, event in enumerate(history):
        reverse_position = size - index
        recency = 1.0 / math.sqrt(reverse_position)
        event_weight = 1.5 if event.get("event") == "purchase" else 1.0
        features.append(
            {
                "cat": [str(value) for value in event.get("cat", [])],
                "item_id": str(event["item_id"]),
                "weight": recency * event_weight,
            }
        )

    item_scores: dict[str, float] = {}
    category_scores: dict[str, float] = {}
    for candidate in candidates:
        item_id = str(candidate["item_id"])
        candidate_cat = [str(value) for value in candidate.get("cat", [])]
        item_score = 0.0
        category_score = 0.0
        for event in features:
            weight = float(event["weight"])
            if item_id == event["item_id"]:
                item_score += 3.0 * weight
            category_score += _deepest_category_match(
                candidate_cat, event["cat"]
            ) * weight
        item_scores[item_id] = item_score
        category_scores[item_id] = category_score
    return item_scores, category_scores


def materialize_alignment_controls(
    config: dict[str, Any], config_path: str | Path
) -> dict[str, Any]:
    """Materialize the two locked history ablations without reading qrels."""

    validate_alignment_config(config)
    config_path = Path(config_path)
    config_sha256 = sha256_file(config_path)
    records_path = Path(config["inputs"]["records_dev"])
    candidate_manifest_path = Path(config["inputs"]["candidate_manifest"])
    candidate_manifest_sha256 = sha256_file(candidate_manifest_path)
    present_path = Path(config["inputs"]["history_present_ids"])
    absent_path = Path(config["inputs"]["history_absent_ids"])
    present_ids = _load_ids(present_path)
    absent_ids = _load_ids(absent_path)
    if present_ids & absent_ids:
        raise ValueError("history-present and history-absent subsets overlap")

    runs = config["runs"]
    component_runs = {
        "item": str(runs["item_history"]),
        "category": str(runs["category_history"]),
    }
    component_paths = {
        name: Path("runs") / run_id / "scores.jsonl"
        for name, run_id in component_runs.items()
    }
    for path in component_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    full_history_run = str(runs["full_history"])
    full_history_dir = Path("runs") / full_history_run
    full_history_path = full_history_dir / "scores.jsonl"
    full_metadata = _load_json(full_history_dir / "metadata.json")
    _assert_upstream_metadata(
        full_metadata,
        candidate_manifest_sha256,
        expected_run_id=full_history_run,
    )

    counts = Counter()
    max_public_error = 0.0
    max_upstream_error = 0.0
    public_violations = 0
    upstream_violations = 0
    full_rows: Iterable[dict[str, Any]] = iter_jsonl(full_history_path)
    with component_paths["item"].open("w", encoding="utf-8") as item_handle, \
            component_paths["category"].open("w", encoding="utf-8") as category_handle:
        full_iter = iter(full_rows)
        seen_requests: set[str] = set()
        for record in iter_jsonl(records_path):
            request_id = str(record["request_id"])
            if request_id in seen_requests:
                raise ValueError(f"duplicate dev request_id: {request_id}")
            seen_requests.add(request_id)
            history_present = bool(record.get("history"))
            if history_present != (request_id in present_ids):
                raise ValueError(f"history subset mismatch for {request_id}")
            if (not history_present) != (request_id in absent_ids):
                raise ValueError(f"history subset mismatch for {request_id}")

            item_scores, category_scores = candidate_history_components(record)
            public_scores = recent_behavior_scores(record)
            candidate_ids = sorted(str(row["item_id"]) for row in record["candidates"])
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"duplicate candidate for {request_id}")
            if set(item_scores) != set(candidate_ids) or set(category_scores) != set(candidate_ids):
                raise ValueError(f"component/candidate mismatch for {request_id}")

            counts["requests"] += 1
            counts["history_present_requests" if history_present else "history_absent_requests"] += 1
            request_item_nonzero = False
            request_category_nonzero = False
            for item_id in candidate_ids:
                upstream = next(full_iter, None)
                if upstream is None:
                    raise ValueError("full B0b score file ended before dev records")
                if (
                    str(upstream["request_id"]) != request_id
                    or str(upstream["candidate_item_id"]) != item_id
                ):
                    raise ValueError(
                        "full B0b score order/key mismatch: "
                        f"expected {request_id}/{item_id}, got "
                        f"{upstream.get('request_id')}/{upstream.get('candidate_item_id')}"
                    )
                item_value = float(item_scores[item_id])
                category_value = float(category_scores[item_id])
                decomposed = item_value + category_value
                public_error = abs(decomposed - float(public_scores[item_id]))
                upstream_error = abs(decomposed - float(upstream["score"]))
                max_public_error = max(max_public_error, public_error)
                max_upstream_error = max(max_upstream_error, upstream_error)
                public_violations += int(public_error > DECOMPOSITION_TOLERANCE)
                upstream_violations += int(upstream_error > DECOMPOSITION_TOLERANCE)
                request_item_nonzero = request_item_nonzero or item_value != 0.0
                request_category_nonzero = request_category_nonzero or category_value != 0.0
                counts["item_nonzero_candidates"] += int(item_value != 0.0)
                counts["category_nonzero_candidates"] += int(category_value != 0.0)
                _write_score_row(
                    item_handle,
                    request_id,
                    item_id,
                    item_value,
                    "c5r3_b0b_item_component",
                )
                _write_score_row(
                    category_handle,
                    request_id,
                    item_id,
                    category_value,
                    "c5r3_b0b_category_component",
                )
                counts["score_rows"] += 1
            counts["item_nonzero_requests"] += int(request_item_nonzero)
            counts["category_nonzero_requests"] += int(request_category_nonzero)
        if next(full_iter, None) is not None:
            raise ValueError("full B0b score file has rows beyond dev records")

    if seen_requests != present_ids | absent_ids:
        raise ValueError("frozen history subsets do not cover dev exactly")
    if public_violations or upstream_violations:
        raise AssertionError(
            "B0b component decomposition differs from the public or upstream score"
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    component_metadata: dict[str, dict[str, Any]] = {}
    for name, run_id in component_runs.items():
        metadata = {
            "analysis_id": config["analysis_id"],
            "candidate_manifest_path": str(candidate_manifest_path),
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "component": name,
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "dataset_id": config["dataset_id"],
            "dataset_version": config["dataset_version"],
            "full_history_run_id": full_history_run,
            "full_history_scores_sha256": sha256_file(full_history_path),
            "generated_at": generated_at,
            "input_fields_used": [
                "records_dev.history.item_id/cat/event",
                "records_dev.candidates.item_id/cat",
            ],
            "method_id": f"c5r3_b0b_{name}_component",
            "qrels_read": False,
            "request_count": counts["requests"],
            "run_id": run_id,
            "score_rows": counts["score_rows"],
            "split": "dev",
            "test_read": False,
        }
        write_json(Path("runs") / run_id / "metadata.json", metadata)
        shutil.copyfile(config_path, Path("runs") / run_id / "config_snapshot.yaml")
        component_metadata[name] = metadata

    mixture_metadata: dict[str, dict[str, Any]] = {}
    beta = float(config["static_mixture"]["beta"])
    for seed in EXPECTED_SEEDS:
        d2p_run_id = str(runs["d2p_pattern"]).format(seed=seed)
        d2p_dir = Path("runs") / d2p_run_id
        d2p_metadata = _load_json(d2p_dir / "metadata.json")
        _assert_upstream_metadata(
            d2p_metadata,
            candidate_manifest_sha256,
            expected_run_id=d2p_run_id,
        )
        for name in ("item", "category"):
            run_id = str(runs[f"{name}_d2s_pattern"]).format(seed=seed)
            metadata = write_static_mixture_scores(
                query_scores_path=d2p_dir / "scores.jsonl",
                history_scores_path=component_paths[name],
                query_run_id=d2p_run_id,
                history_run_id=component_runs[name],
                run_id=run_id,
                method_id=f"c5r3_d2s_{name}_only",
                alpha=beta,
                candidate_manifest_path=candidate_manifest_path,
                config_path=config_path,
            )
            metadata.update(
                {
                    "analysis_id": config["analysis_id"],
                    "beta": beta,
                    "component": name,
                    "config_sha256": config_sha256,
                    "qrels_read": False,
                    "seed": seed,
                    "test_read": False,
                }
            )
            write_json(Path("runs") / run_id / "metadata.json", metadata)
            mixture_metadata[f"{name}_s{seed}"] = metadata

    artifacts_dir = Path(config["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "analysis_id": config["analysis_id"],
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "component_run_ids": component_runs,
        "component_scores": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in component_paths.items()
        },
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "counts": dict(sorted(counts.items())),
        "decomposition_audit": {
            "candidate_rows_checked": counts["score_rows"],
            "max_abs_error_vs_public_scorer": max_public_error,
            "max_abs_error_vs_upstream_full_b0b": max_upstream_error,
            "public_tolerance_violations": public_violations,
            "tolerance": DECOMPOSITION_TOLERANCE,
            "upstream_tolerance_violations": upstream_violations,
        },
        "generated_at": generated_at,
        "input_sha256": {
            "history_absent_ids": sha256_file(absent_path),
            "history_present_ids": sha256_file(present_path),
            "records_dev": sha256_file(records_path),
            "upstream_full_b0b_scores": sha256_file(full_history_path),
        },
        "mixture_run_ids": {
            key: value["run_id"] for key, value in sorted(mixture_metadata.items())
        },
        "qrels_read": False,
        "test_read": False,
    }
    manifest_path = artifacts_dir / "materialization_manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def adjudicate_alignment_gate(
    gate: dict[str, Any],
    comparisons: dict[str, dict[str, dict[str, Any]]],
    subset_means: dict[str, dict[str, float]],
    integrity_passed: bool,
) -> dict[str, Any]:
    """Apply the finite primary/fallback ladder exactly as frozen."""

    expected = {
        "item_vs_d2p",
        "category_vs_d2p",
        "full_vs_item",
        "full_vs_category",
    }
    if set(comparisons) != expected:
        raise ValueError("comparison families differ from the frozen C5-R3 gate")
    seed_keys = {str(seed) for seed in EXPECTED_SEEDS}
    if any(set(rows) != seed_keys for rows in comparisons.values()):
        raise ValueError("comparison seeds differ from the frozen C5-R3 gate")

    minimum = int(gate["primary_min_significant_seeds"])
    significant_counts = {
        name: sum(
            float(row["delta"]) > 0.0 and float(row["ci95"][0]) > 0.0
            for row in rows.values()
        )
        for name, rows in comparisons.items()
    }
    primary_checks = {
        "item_only_vs_d2p_significant_in_at_least_two_seeds": (
            significant_counts["item_vs_d2p"] >= minimum
        ),
        "category_only_vs_d2p_significant_in_at_least_two_seeds": (
            significant_counts["category_vs_d2p"] >= minimum
        ),
        "full_vs_item_only_significant_in_at_least_two_seeds": (
            significant_counts["full_vs_item"] >= minimum
        ),
        "full_vs_category_only_significant_in_at_least_two_seeds": (
            significant_counts["full_vs_category"] >= minimum
        ),
        "integrity_and_no_history_fallback": bool(integrity_passed),
    }
    primary_passed = all(primary_checks.values())

    relative_gains = []
    for seed in EXPECTED_SEEDS:
        values = subset_means[str(seed)]
        base = float(values["d2p"])
        if base <= 0.0:
            raise ValueError("D2p history-present NDCG mean must be positive")
        relative_gains.append((float(values["category"]) - base) / base)
    category_relative_gain = statistics.mean(relative_gains)
    full_vs_category_deltas = [
        float(row["delta"])
        for row in comparisons["full_vs_category"].values()
    ]
    fallback_checks = {
        "category_only_vs_d2p_significant_in_all_seeds": (
            significant_counts["category_vs_d2p"] == len(EXPECTED_SEEDS)
        ),
        "category_only_mean_relative_gain_at_least_two_percent": (
            category_relative_gain
            >= float(gate["fallback_category_min_relative_gain"])
        ),
        "full_not_significantly_worse_than_category_any_seed": all(
            float(row["ci95"][1]) >= 0.0
            for row in comparisons["full_vs_category"].values()
        ),
        "full_vs_category_three_seed_mean_delta_nonnegative": (
            statistics.mean(full_vs_category_deltas) >= 0.0
        ),
        "integrity_and_no_history_fallback": bool(integrity_passed),
    }
    fallback_passed = (not primary_passed) and all(fallback_checks.values())
    if primary_passed:
        outcome = "PRIMARY_PASS"
        authorized_primitive = "multi_granular_candidate_history_evidence_matching"
    elif fallback_passed:
        outcome = "FALLBACK_PASS"
        authorized_primitive = "coarse_candidate_history_semantic_matching"
    else:
        outcome = "TERMINAL_FAIL"
        authorized_primitive = None
    return {
        "architecture_ready": outcome in {"PRIMARY_PASS", "FALLBACK_PASS"},
        "authorized_primitive": authorized_primitive,
        "category_relative_gains_by_seed": relative_gains,
        "category_three_seed_mean_relative_gain": category_relative_gain,
        "fallback_checks": fallback_checks,
        "fallback_passed": fallback_passed,
        "outcome": outcome,
        "primary_checks": primary_checks,
        "primary_passed": primary_passed,
        "significant_seed_counts": significant_counts,
    }


def validate_alignment_config(config: dict[str, Any]) -> None:
    """Reject policy drift before materialization or adjudication."""

    if config.get("analysis_id") != "c5r3_candidate_history_alignment":
        raise ValueError("unexpected analysis_id")
    if config.get("status") != "locked_before_outcome_evaluation":
        raise ValueError("C5-R3 config must be outcome-locked")
    if config.get("target_split") != "dev":
        raise ValueError("C5-R3 may run only on dev")
    if [int(seed) for seed in config.get("seeds", [])] != EXPECTED_SEEDS:
        raise ValueError("C5-R3 seed set differs from the frozen set")
    inputs = config.get("inputs", {})
    if set(inputs) != {
        "records_dev",
        "candidate_manifest",
        "history_present_ids",
        "history_absent_ids",
    }:
        raise ValueError("C5-R3 input set differs from the label-free frozen set")
    forbidden = ("qrels", "records_test", "test.json", "test_")
    for value in inputs.values():
        lowered = str(value).lower()
        if any(token in lowered for token in forbidden):
            raise ValueError(f"forbidden C5-R3 input: {value}")
    components = config.get("components", {})
    expected_components = {
        "recency": "1/sqrt(reverse_position)",
        "click_weight": 1.0,
        "purchase_weight": 1.5,
        "item_match_weight": 3.0,
        "category_match_semantics": "deepest_exclusive",
        "category_l3_weight": 1.0,
        "category_l2_weight": 0.5,
        "category_l1_weight": 0.2,
    }
    if components != expected_components:
        raise ValueError("component semantics differ from frozen executable B0b")
    mixture = config.get("static_mixture", {})
    if float(mixture.get("beta", -1.0)) != 0.3:
        raise ValueError("C5-R3 beta must remain 0.3")
    if mixture.get("zscore_scope") != "within_request":
        raise ValueError("C5-R3 requires within-request z-scoring")
    gate = config.get("gate", {})
    expected_gate = {
        "bootstrap_samples": 10000,
        "bootstrap_seed": 20260708,
        "primary_min_significant_seeds": 2,
        "fallback_category_require_all_significant_seeds": True,
        "fallback_category_min_relative_gain": 0.02,
        "fallback_relative_gain_definition": "mean_of_per_seed_subset_mean_ratios",
    }
    if gate != expected_gate:
        raise ValueError("gate differs from the frozen C5-R3 ladder")


def _deepest_category_match(left: list[str], right: list[str]) -> float:
    left_values = (left + ["", "", ""])[:3]
    right_values = (right + ["", "", ""])[:3]
    if _valid_category(left_values[2]) and left_values[2] == right_values[2]:
        return 1.0
    if _valid_category(left_values[1]) and left_values[1] == right_values[1]:
        return 0.5
    if _valid_category(left_values[0]) and left_values[0] == right_values[0]:
        return 0.2
    return 0.0


def _valid_category(value: str) -> bool:
    return bool(value and value.upper() != "UNKNOWN")


def _write_score_row(
    handle: Any,
    request_id: str,
    item_id: str,
    score: float,
    method_id: str,
) -> None:
    if not math.isfinite(score):
        raise ValueError(f"non-finite component score: {request_id}/{item_id}")
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


def _assert_upstream_metadata(
    metadata: dict[str, Any],
    candidate_manifest_sha256: str,
    expected_run_id: str,
) -> None:
    if metadata.get("run_id") != expected_run_id:
        raise ValueError(f"upstream run_id mismatch: {expected_run_id}")
    if metadata.get("candidate_manifest_sha256") != candidate_manifest_sha256:
        raise ValueError(f"upstream candidate hash mismatch: {expected_run_id}")
    if metadata.get("qrels_read") is not False:
        raise ValueError(f"upstream scoring read qrels: {expected_run_id}")
    if metadata.get("test_read", False) is not False:
        raise ValueError(f"upstream scoring read test: {expected_run_id}")


def _load_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
