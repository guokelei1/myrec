#!/usr/bin/env python
"""Build the pre-registered KuaiSearch Full motivation confirmation decision."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.controlled_composition import (  # noqa: E402
    build_endpoint_aligned_accounting_rows,
    cluster_bootstrap_group_mean_contrast,
    cluster_bootstrap_mean_ci,
    summarize_endpoint_aligned_accounting,
    summarize_partition_contributions,
)
from myrec.eval.motivation_confirmation import (  # noqa: E402
    resolve_confirmation_decision,
)
from myrec.eval.target_aware_surfaces import (  # noqa: E402
    ALL_REQUEST_PARTITION,
    OBSERVED_POSITIVE_PARTITION,
)
from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import iter_jsonl, write_json  # noqa: E402


CONFIG = Path("configs/baselines/kuaisearch_confirmation_qwen3_pointwise.yaml")
PROTOCOL = Path("experiments/history_response_gap/kuaisearch_confirmation_protocol.md")
ADMISSION = Path("reports/pps_history_response_e0_confirmation_data_admission.json")
ASSIGNMENTS = Path("reports/pps_history_response_confirmation_assignments.json")
PREPROCESS_TRAIN = Path(
    "reports/pps_history_response_confirmation_qwen3_preprocess_train.json"
)
PREPROCESS_CONFIRMATION = Path(
    "reports/pps_history_response_confirmation_qwen3_preprocess_confirmation.json"
)
SURFACE_REPORT = Path(
    "reports/pps_history_response_confirmation_target_aware_surfaces.json"
)
OUTPUT = Path("reports/pps_motivation_confirmation_decision.json")


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    runs = config["runs"]
    evaluation = config["evaluation"]
    standardized_dir = Path(config["dataset"]["standardized_dir"])
    records_path = standardized_dir / "records_confirmation.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"

    records = {
        str(row["request_id"]): row for row in iter_jsonl(records_path)
    }
    clusters = {
        request_id: "".join(str(row.get("query", "")).casefold().split())
        for request_id, row in records.items()
    }
    if not records:
        raise ValueError("empty confirmation records")

    bm25_eval_dir = Path("runs") / runs["bm25_evaluation"]
    qc_eval_dir = Path("runs") / runs["qc_evaluation"]
    response_dir = Path("runs") / runs["response"]
    bm25_rows = _rows_by_request(bm25_eval_dir / "per_request_metrics.jsonl")
    qc_rows = _rows_by_request(qc_eval_dir / "per_request_metrics.jsonl")
    response_rows = _rows_by_request(
        response_dir / "per_request_history_response.jsonl"
    )
    if set(records) != set(bm25_rows) or set(records) != set(qc_rows):
        raise ValueError("BM25/QC confirmation request coverage mismatch")
    if set(records) != set(response_rows):
        raise ValueError("history-response confirmation request coverage mismatch")

    surface_report = _read_json(SURFACE_REPORT)
    surfaces = surface_report["surfaces"]
    surface_dir = response_dir / "target_aware_surfaces"
    surface_ids = {
        name: _load_surface(surface_dir / f"{name}.txt")
        for name in (
            "all",
            "observed_positive",
            *ALL_REQUEST_PARTITION,
        )
    }
    _assert_surface_counts(surface_ids, surfaces)

    bootstrap_samples = int(evaluation["query_cluster_bootstrap_samples"])
    bootstrap_seed = int(evaluation["bootstrap_seed"])
    accounting_rows = build_endpoint_aligned_accounting_rows(
        qc_rows, response_rows, records
    )
    accounting_by_id = {
        str(row["request_id"]): row for row in accounting_rows
    }
    accounting = {}
    for offset, name in enumerate(("all", "observed_positive", *ALL_REQUEST_PARTITION)):
        ids = surface_ids[name]
        accounting[name] = (
            summarize_endpoint_aligned_accounting(
                [accounting_by_id[request_id] for request_id in sorted(ids)],
                clusters,
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed + 100 + offset,
            )
            if ids
            else None
        )

    qc_minus_bm25_rows = [
        {
            "request_id": request_id,
            "qc_minus_bm25_ndcg@10": (
                float(qc_rows[request_id]["ndcg@10"])
                - float(bm25_rows[request_id]["ndcg@10"])
            ),
        }
        for request_id in sorted(records)
    ]
    qc_minus_bm25_ci = cluster_bootstrap_mean_ci(
        qc_minus_bm25_rows,
        clusters,
        ("qc_minus_bm25_ndcg@10",),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )["qc_minus_bm25_ndcg@10"]
    qc_minus_bm25_mean = sum(
        row["qc_minus_bm25_ndcg@10"] for row in qc_minus_bm25_rows
    ) / len(qc_minus_bm25_rows)

    contrast_rows = []
    contrast_groups = (
        "target_repeat",
        "target_nonrepeat_no_candidate_overlap",
    )
    for name in contrast_groups:
        contrast_rows.extend(
            {
                "request_id": request_id,
                "surface": name,
                "same_checkpoint_recovery": float(
                    response_rows[request_id]["true_minus_null_ndcg@10"]
                ),
            }
            for request_id in sorted(surface_ids[name])
        )
    recurrence_transfer_contrast = cluster_bootstrap_group_mean_contrast(
        contrast_rows,
        clusters,
        group_field="surface",
        value_field="same_checkpoint_recovery",
        left_group="target_repeat",
        right_group="target_nonrepeat_no_candidate_overlap",
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )

    training = _training_integrity(runs)
    score_integrity = _score_integrity(
        runs,
        candidate_manifest_path=candidate_manifest_path,
        request_manifest_path=request_manifest_path,
        expected_requests=len(records),
    )
    finite_metrics = _all_finite_request_metrics(
        bm25_rows, qc_rows, response_rows
    )

    repeat_surface = surfaces["target_repeat"]
    no_overlap_surface = surfaces[
        "target_nonrepeat_no_candidate_overlap"
    ]
    repeat_ci = _query_ci(repeat_surface, "mean_true_minus_null_ndcg@10")
    no_overlap_ci = _query_ci(
        no_overlap_surface, "mean_true_minus_null_ndcg@10"
    )
    power_passed = (
        len(surface_ids["target_repeat"])
        >= int(evaluation["power_min_target_repeat"])
        and len(surface_ids["target_nonrepeat_no_candidate_overlap"])
        >= int(evaluation["power_min_target_nonrepeat_no_candidate_overlap"])
    )
    task_adequacy_passed = bool(
        qc_minus_bm25_ci is not None
        and qc_minus_bm25_ci[1] >= 0.0
        and training["passed"]
        and score_integrity["passed"]
        and finite_metrics
    )
    gate_passes = {
        "population_power": power_passed,
        "task_adequacy": task_adequacy_passed,
        "target_repeat_positive_control": repeat_ci[0] > 0.0,
        "recurrence_transfer_separation": (
            recurrence_transfer_contrast["bootstrap_ci95"][0] > 0.0
        ),
        "practical_nonrepeat_bound": (
            no_overlap_ci[1]
            < float(evaluation["practical_nonrepeat_upper_bound"])
        ),
    }
    hierarchical_decision = resolve_confirmation_decision(gate_passes)

    bm25_metrics = _read_json(bm25_eval_dir / "metrics.json")
    qc_metrics = _read_json(qc_eval_dir / "metrics.json")
    response_metrics = _read_json(response_dir / "metrics.json")
    all_partition = {
        name: [accounting_by_id[request_id] for request_id in sorted(surface_ids[name])]
        for name in ALL_REQUEST_PARTITION
    }
    positive_partition = {
        name: [accounting_by_id[request_id] for request_id in sorted(surface_ids[name])]
        for name in OBSERVED_POSITIVE_PARTITION
    }

    report = {
        "schema_version": 1,
        "analysis_id": "kuaisearch_full_disjoint_window_motivation_confirmation_v1",
        "evidence_mode": "frozen_confirmation",
        "protocol": {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)},
        "config": {"path": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "population": {
            "requests": len(records),
            "target_repeat_requests": len(surface_ids["target_repeat"]),
            "target_nonrepeat_no_candidate_overlap_requests": len(
                surface_ids["target_nonrepeat_no_candidate_overlap"]
            ),
            "power_min_target_repeat": int(evaluation["power_min_target_repeat"]),
            "power_min_target_nonrepeat_no_candidate_overlap": int(
                evaluation["power_min_target_nonrepeat_no_candidate_overlap"]
            ),
        },
        "integrity": {
            "training": training,
            "scores": score_integrity,
            "all_request_metrics_finite": finite_metrics,
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
            "request_manifest_sha256": sha256_file(request_manifest_path),
        },
        "task_adequacy": {
            "bm25_ndcg@10_all_requests": bm25_metrics["ndcg@10_all_requests"],
            "qc_ndcg@10_all_requests": qc_metrics["ndcg@10_all_requests"],
            "qc_minus_bm25_ndcg@10": qc_minus_bm25_mean,
            "qc_minus_bm25_query_cluster_ci95": qc_minus_bm25_ci,
            "not_significantly_below_bm25": qc_minus_bm25_ci[1] >= 0.0,
        },
        "primary_scientific_endpoints": {
            "target_repeat": _surface_extract(repeat_surface),
            "target_nonrepeat_no_candidate_overlap": _surface_extract(
                no_overlap_surface
            ),
            "target_repeat_minus_no_overlap_recovery": recurrence_transfer_contrast,
            "practical_nonrepeat_upper_bound": float(
                evaluation["practical_nonrepeat_upper_bound"]
            ),
        },
        "hierarchical_decision": hierarchical_decision,
        "all_request_and_conditional_positive": {
            "bm25": _endpoint_extract(bm25_metrics),
            "qc": _endpoint_extract(qc_metrics),
            "full": {
                "true_ndcg@10_all_requests": response_metrics[
                    "mean_true_ndcg@10"
                ],
                "true_ndcg@10_positive_requests": response_metrics[
                    "mean_true_ndcg@10_positive"
                ],
                "true_minus_null_ndcg@10_all_requests": response_metrics[
                    "mean_true_minus_null_ndcg@10"
                ],
                "true_minus_null_ndcg@10_positive_requests": response_metrics[
                    "mean_true_minus_null_ndcg@10_positive"
                ],
            },
            "accounting": accounting,
        },
        "secondary_diagnostics": {
            "surface_report": str(SURFACE_REPORT),
            "surfaces": {
                name: _surface_extract(surfaces[name])
                for name in ALL_REQUEST_PARTITION
            },
            "all_request_recovery_contributions": summarize_partition_contributions(
                all_rows=accounting_rows,
                partition_rows=all_partition,
                metric="same_checkpoint_recovery",
            ),
            "observed_positive_recovery_contributions": summarize_partition_contributions(
                all_rows=[
                    accounting_by_id[request_id]
                    for request_id in sorted(surface_ids["observed_positive"])
                ],
                partition_rows=positive_partition,
                metric="same_checkpoint_recovery",
            ),
        },
        "claim_boundary": {
            "supports_if_confirmed": (
                "A task-adequate ordinary full-token decoder ranker can learn "
                "large target-recurrence utility while failing to establish a "
                "practically material recovery on target-nonrepeat requests whose "
                "history is disjoint from the full candidate slate."
            ),
            "does_not_support": [
                "That Transformers cannot read history.",
                "That every dataset or model family exhibits the same state.",
                "A causal architecture mechanism or a proposed-system necessity.",
                "Test-set performance.",
            ],
        },
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                ADMISSION,
                ASSIGNMENTS,
                PREPROCESS_TRAIN,
                PREPROCESS_CONFIRMATION,
                bm25_eval_dir / "metrics.json",
                bm25_eval_dir / "per_request_metrics.jsonl",
                qc_eval_dir / "metrics.json",
                qc_eval_dir / "per_request_metrics.jsonl",
                response_dir / "metrics.json",
                response_dir / "per_request_history_response.jsonl",
                SURFACE_REPORT,
            )
        },
    }
    write_json(OUTPUT, report)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "decision": hierarchical_decision["decision"],
                "gates": hierarchical_decision["gates"],
            },
            indent=2,
        )
    )
    return 0


def _training_integrity(runs: dict[str, str]) -> dict[str, Any]:
    qc_path = Path("runs") / runs["qc_train"] / "metadata.json"
    full_path = Path("runs") / runs["full_train"] / "metadata.json"
    qc = _read_json(qc_path)
    full = _read_json(full_path)
    fields = ("examples", "labeled_requests", "negatives_per_positive")
    examples_match = all(
        qc["example_stats"][field] == full["example_stats"][field]
        for field in fields
    )
    updates_match = all(
        qc["training"][field] == full["training"][field]
        for field in ("candidate_presentations", "micro_steps", "optimizer_steps")
    )
    no_confirmation_labels = (
        not qc["dev_labels_read"]
        and not full["dev_labels_read"]
        and qc["training_labels_path"].endswith("qrels_train.jsonl")
        and full["training_labels_path"].endswith("qrels_train.jsonl")
    )
    finite_training = all(
        math.isfinite(float(row["training"]["mean_microbatch_loss"]))
        and bool(row.get("weights_sha256"))
        and bool(row.get("checkpoint_id"))
        for row in (qc, full)
    )
    passed = (
        examples_match
        and updates_match
        and no_confirmation_labels
        and finite_training
    )
    return {
        "passed": passed,
        "example_counts_match": examples_match,
        "optimizer_counts_match": updates_match,
        "confirmation_labels_not_read": no_confirmation_labels,
        "finite_losses_and_checkpoints": finite_training,
        "qc": {
            "checkpoint_id": qc["checkpoint_id"],
            "example_stats": qc["example_stats"],
            "training": qc["training"],
            "metadata_sha256": sha256_file(qc_path),
        },
        "full": {
            "checkpoint_id": full["checkpoint_id"],
            "example_stats": full["example_stats"],
            "training": full["training"],
            "metadata_sha256": sha256_file(full_path),
        },
    }


def _score_integrity(
    runs: dict[str, str],
    *,
    candidate_manifest_path: Path,
    request_manifest_path: Path,
    expected_requests: int,
) -> dict[str, Any]:
    run_keys = ("bm25_score", "qc_score", "full_true", "full_null", "full_wrong")
    metadata_paths = {
        key: Path("runs") / runs[key] / "metadata.json" for key in run_keys
    }
    metadata = {key: _read_json(path) for key, path in metadata_paths.items()}
    candidate_sha = sha256_file(candidate_manifest_path)
    request_sha = sha256_file(request_manifest_path)
    common = all(
        row["candidate_manifest_sha256"] == candidate_sha
        and row["dataset_version"] == "full_confirm_preceding10k_v1"
        and row["split"] == "confirmation"
        and row["request_count"] == expected_requests
        and not row["qrels_read"]
        for row in metadata.values()
    )
    neural = {key: metadata[key] for key in run_keys if key != "bm25_score"}
    neural_request_hashes = all(
        row["request_manifest_sha256"] == request_sha for row in neural.values()
    )
    full = {key: metadata[key] for key in ("full_true", "full_null", "full_wrong")}
    reference = full["full_true"]
    full_invariant_fields = (
        "checkpoint_id",
        "candidate_manifest_sha256",
        "request_manifest_sha256",
        "dataset_id",
        "dataset_version",
        "split",
        "scoring_signature",
    )
    full_counterfactual_exact = all(
        row[field] == reference[field]
        for row in full.values()
        for field in full_invariant_fields
    )
    score_rows_match = len({row["score_rows"] for row in metadata.values()}) == 1
    passed = (
        common
        and neural_request_hashes
        and full_counterfactual_exact
        and score_rows_match
    )
    return {
        "passed": passed,
        "common_population_and_label_isolation": common,
        "neural_request_manifest_hashes_match": neural_request_hashes,
        "full_counterfactual_bundle_exact": full_counterfactual_exact,
        "score_row_counts_match": score_rows_match,
        "checkpoint_id": reference["checkpoint_id"],
        "metadata": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in metadata_paths.items()
        },
    }


def _all_finite_request_metrics(
    bm25_rows: dict[str, dict[str, Any]],
    qc_rows: dict[str, dict[str, Any]],
    response_rows: dict[str, dict[str, Any]],
) -> bool:
    values = [
        *(float(row["ndcg@10"]) for row in bm25_rows.values()),
        *(float(row["ndcg@10"]) for row in qc_rows.values()),
        *(
            float(row[field])
            for row in response_rows.values()
            for field in ("true_ndcg@10", "null_ndcg@10", "wrong_ndcg@10")
        ),
    ]
    return all(math.isfinite(value) for value in values)


def _assert_surface_counts(
    surface_ids: dict[str, set[str]], surfaces: dict[str, dict[str, Any]]
) -> None:
    for name, ids in surface_ids.items():
        if len(ids) != int(surfaces[name]["num_requests"]):
            raise ValueError(f"surface count mismatch for {name}")


def _surface_extract(surface: dict[str, Any]) -> dict[str, Any]:
    return {
        "requests": surface["num_requests"],
        "active_response_rate": surface["active_response_rate"],
        "pairwise_directional_accuracy": surface[
            "mean_pairwise_directional_accuracy"
        ],
        "pairwise_directional_accuracy_query_cluster_ci95": _query_ci(
            surface, "mean_pairwise_directional_accuracy"
        ),
        "true_minus_null_ndcg@10": surface["mean_true_minus_null_ndcg@10"],
        "true_minus_null_ndcg@10_query_cluster_ci95": _query_ci(
            surface, "mean_true_minus_null_ndcg@10"
        ),
        "true_minus_wrong_ndcg@10": surface["mean_true_minus_wrong_ndcg@10"],
        "true_minus_wrong_ndcg@10_query_cluster_ci95": _query_ci(
            surface, "mean_true_minus_wrong_ndcg@10"
        ),
    }


def _endpoint_extract(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "ndcg@10_all_requests": metrics["ndcg@10_all_requests"],
        "ndcg@10_positive_requests": metrics["ndcg@10_positive_requests"],
        "num_requests": metrics["num_requests"],
        "num_positive_eligible_requests": metrics["num_positive_eligible_requests"],
    }


def _query_ci(surface: dict[str, Any], metric: str) -> list[float]:
    return surface["bootstrap_ci95"]["query_cluster"][metric]


def _rows_by_request(path: Path) -> dict[str, dict[str, Any]]:
    rows = {str(row["request_id"]): row for row in iter_jsonl(path)}
    if not rows:
        raise ValueError(f"empty request artifact: {path}")
    return rows


def _load_surface(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
