"""Post-evaluator completion audit for the single C04 dev screening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import (
    assert_candidate_manifest,
    finite_float,
    iter_jsonl,
    sha256_file,
    write_json,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _request_ids(path: str | Path) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _scores_for(path: str | Path, request_ids: set[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id not in request_ids:
            continue
        result.setdefault(request_id, {})[str(row["candidate_item_id"])] = finite_float(
            row["score"]
        )
    if set(result) != request_ids:
        raise ValueError(
            f"score subset coverage mismatch: expected={len(request_ids)} actual={len(result)}"
        )
    return result


def _rank_mismatches(
    left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]
) -> tuple[int, int]:
    from myrec.eval.metrics import ScoredCandidate, sort_candidates

    mismatches = 0
    score_order_requests = 0
    for request_id in sorted(left):
        if set(left[request_id]) != set(right[request_id]):
            raise ValueError(f"candidate mismatch while auditing {request_id}")
        left_rank = [
            row.item_id
            for row in sort_candidates(
                request_id,
                [
                    ScoredCandidate(item_id, score)
                    for item_id, score in left[request_id].items()
                ],
            )
        ]
        right_rank = [
            row.item_id
            for row in sort_candidates(
                request_id,
                [
                    ScoredCandidate(item_id, score)
                    for item_id, score in right[request_id].items()
                ],
            )
        ]
        mismatches += int(left_rank != right_rank)
        score_order_requests += 1
    return mismatches, score_order_requests


def _dev_log_count(run_id: str) -> int:
    path = Path("reports/dev_eval_log.jsonl")
    if not path.exists():
        return 0
    return sum(1 for row in iter_jsonl(path) if str(row.get("run_id")) == run_id)


def _gpu_hours(config: dict[str, Any], screening_metadata: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    total = float(screening_metadata.get("gpu_hours", 0.0))
    for mode, run_id in config["run_ids"].items():
        if mode == "screening":
            continue
        summary_path = Path("runs") / run_id / "train_summary.json"
        if not summary_path.exists():
            continue
        value = float(_read_json(summary_path).get("gpu_hours", 0.0))
        rows[mode] = value
        total += value
    materialization_path = Path(config["paths"]["probe_data_dir"]) / "manifest.json"
    if materialization_path.exists():
        materialization = float(_read_json(materialization_path).get("elapsed_seconds", 0.0)) / 3600.0
        rows["materialization"] = materialization
        total += materialization
    rows["screening"] = float(screening_metadata.get("gpu_hours", 0.0))
    return {"by_stage": rows, "total": total}


def audit_screening(
    config: dict[str, Any],
    output_path: str | Path,
    nonrepeat_comparison_path: str | Path,
    repeat_comparison_path: str | Path,
    deterministic_a: str | Path,
    deterministic_b: str | Path,
) -> dict[str, Any]:
    expected_hash = assert_candidate_manifest(
        config["paths"]["candidate_manifest"], config["candidate_manifest_sha256"]
    )
    run_id = config["run_ids"]["screening"]
    run_dir = Path("runs") / run_id
    metadata = _read_json(run_dir / "metadata.json")
    metrics = _read_json(run_dir / "metrics.json")
    diagnostics = _read_json(run_dir / "delta_diagnostics.json")
    nonrepeat = _read_json(nonrepeat_comparison_path)
    repeat = _read_json(repeat_comparison_path)
    no_history_ids = _request_ids(config["paths"]["no_history_ids"])
    candidate_scores = _scores_for(run_dir / "scores.jsonl", no_history_ids)
    d2p_scores = _scores_for(config["paths"]["d2p_dev_scores"], no_history_ids)
    rank_mismatches, rank_requests = _rank_mismatches(candidate_scores, d2p_scores)
    thresholds = config["screening_stop_loss"]
    corruption_ratios = diagnostics["means"]["corruption_ratios_to_factual"]
    corruption_checks = {
        name: float(value) <= float(thresholds["corruption_delta_ratio_max"])
        for name, value in corruption_ratios.items()
    }
    main_train = _read_json(
        Path("runs")
        / config["run_ids"]["paired_delta"]
        / "train_summary.json"
    )
    internal = main_train["epochs"][-1]["internal"]
    checks = {
        "candidate_hash": (
            metadata.get("candidate_manifest_sha256") == expected_hash
            and metrics.get("candidate_manifest_sha256") == expected_hash
        ),
        "corruptions_attenuated": all(corruption_checks.values()),
        "deterministic_1000_byte_identity": sha256_file(deterministic_a)
        == sha256_file(deterministic_b),
        "dev_evaluator_called_exactly_once": _dev_log_count(run_id) == 1,
        "internal_anchor_concordance": float(internal["anchor_pair_concordance"])
        >= float(thresholds["internal_anchor_pair_concordance_min"]),
        "latency_budget": float(
            metadata["mean_latency_ms_per_request_including_tokenization"]
        )
        <= float(thresholds["latency_ms_per_request_max"]),
        "no_history_rank_equivalence": rank_mismatches
        <= int(thresholds["no_history_rank_mismatches_max"]),
        "nonrepeat_positive_screen": (
            float(nonrepeat["delta"])
            >= float(thresholds["nonrepeat_vs_d2p_delta_min"])
            and float(nonrepeat["ci95"][0])
            >= float(thresholds["nonrepeat_vs_d2p_ci_low_min"])
        ),
        "overall_not_obviously_degraded": (
            float(metrics["ndcg@10"]) >= float(thresholds["overall_ndcg_min"])
            and float(metrics["ndcg@10"])
            >= float(thresholds["item_only_seed_ndcg"])
            - float(thresholds["max_absolute_gap_below_item_only"])
        ),
        "repeat_noninferiority_screen": float(repeat["delta"])
        >= float(thresholds["repeat_vs_item_only_delta_min"]),
        "memory_budget": float(metadata["peak_allocated_gpu_gib"])
        <= float(thresholds["peak_gpu_gib_max"]),
        "score_contract": int(metadata.get("request_count", 0)) == 12229
        and int(metadata.get("score_rows", 0)) == 575609,
    }
    integrity_names = {
        "candidate_hash",
        "deterministic_1000_byte_identity",
        "dev_evaluator_called_exactly_once",
        "score_contract",
    }
    integrity_passed = all(checks[name] for name in integrity_names)
    screening_passed = all(checks.values())
    if not integrity_passed or not checks["no_history_rank_equivalence"]:
        recommendation = "stop"
    elif screening_passed:
        recommendation = "advance-to-full-gate"
    else:
        recommendation = "pivot-before-more-dev"
    gpu = _gpu_hours(config, metadata)
    report = {
        "candidate_id": config["candidate_id"],
        "checks": checks,
        "corruption_checks": corruption_checks,
        "corruption_ratios": corruption_ratios,
        "dev_evaluator_calls": _dev_log_count(run_id),
        "full_gate_executed": False,
        "gpu_hours": gpu,
        "integrity_passed": integrity_passed,
        "metrics": metrics,
        "no_history": {
            "rank_mismatches": rank_mismatches,
            "requests": rank_requests,
        },
        "nonrepeat_vs_d2p": nonrepeat,
        "primary_screening_passed": screening_passed,
        "qrels_read_by_candidate_code": False,
        "recommendation": recommendation,
        "repeat_vs_item_only": repeat,
        "run_id": run_id,
        "test_read": False,
        "thresholds": thresholds,
    }
    write_json(output_path, report)
    return report
