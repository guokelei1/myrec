#!/usr/bin/env python
"""Audit the frozen confirmation score bundle without reading confirmation qrels."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.history_response_evaluator import (  # noqa: E402
    COUNTERFACTUAL_IDENTITY_KEYS,
    _assert_score_coverage,
    _load_candidates,
    _load_scores,
)
from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import write_json  # noqa: E402


CONFIG = Path("configs/baselines/kuaisearch_confirmation_qwen3_pointwise.yaml")
ASSIGNMENTS = Path("reports/pps_history_response_confirmation_assignments.json")
OUTPUT = Path("reports/pps_history_response_confirmation_score_integrity.json")


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    runs = config["runs"]
    standardized_dir = Path(config["dataset"]["standardized_dir"])
    candidate_manifest = standardized_dir / "candidate_manifest.json"
    request_manifest = standardized_dir / "request_manifest.json"
    candidates = _load_candidates(candidate_manifest, "confirmation")
    expected_rows = sum(len(items) for items in candidates.values())
    candidate_sha = sha256_file(candidate_manifest)
    request_sha = sha256_file(request_manifest)

    run_keys = ("bm25_score", "qc_score", "full_true", "full_null", "full_wrong")
    run_dirs = {key: Path("runs") / runs[key] for key in run_keys}
    metadata = {
        key: _read_json(run_dir / "metadata.json")
        for key, run_dir in run_dirs.items()
    }
    score_files = {key: run_dir / "scores.jsonl" for key, run_dir in run_dirs.items()}
    score_coverage = {}
    for key, path in score_files.items():
        scores = _load_scores(path)
        _assert_score_coverage(candidates, scores)
        score_coverage[key] = {
            "requests": len(scores),
            "rows": sum(len(items) for items in scores.values()),
            "scores_sha256": sha256_file(path),
        }

    common_population = all(
        row["candidate_manifest_sha256"] == candidate_sha
        and row["dataset_version"] == config["dataset"]["dataset_version"]
        and row["split"] == "confirmation"
        and row["request_count"] == len(candidates)
        and row["score_rows"] == expected_rows
        and not row["qrels_read"]
        for row in metadata.values()
    )
    neural_keys = ("qc_score", "full_true", "full_null", "full_wrong")
    neural_request_hashes = all(
        metadata[key]["request_manifest_sha256"] == request_sha
        for key in neural_keys
    )
    full = {key: metadata[key] for key in ("full_true", "full_null", "full_wrong")}
    reference = full["full_true"]
    full_counterfactual_exact = all(
        row[field] == reference[field]
        for row in full.values()
        for field in COUNTERFACTUAL_IDENTITY_KEYS
    )

    assignments = _read_json(ASSIGNMENTS)["files"]
    expected_assignment_sha = {
        "qc_score": assignments["null"]["sha256"],
        "full_true": assignments["true"]["sha256"],
        "full_null": assignments["null"]["sha256"],
        "full_wrong": assignments["wrong"]["sha256"],
    }
    assignments_exact = all(
        metadata[key]["history_assignment_sha256"] == expected_assignment_sha[key]
        for key in neural_keys
    )

    train_metadata = {
        key: _read_json(Path("runs") / runs[key] / "metadata.json")
        for key in ("qc_train", "full_train")
    }
    qc_train = train_metadata["qc_train"]
    full_train = train_metadata["full_train"]
    training_examples_match = all(
        qc_train["example_stats"][field] == full_train["example_stats"][field]
        for field in ("examples", "labeled_requests", "negatives_per_positive")
    )
    optimizer_counts_match = all(
        qc_train["training"][field] == full_train["training"][field]
        for field in ("candidate_presentations", "micro_steps", "optimizer_steps")
    )
    checkpoints_exact = (
        metadata["qc_score"]["checkpoint_id"] == qc_train["checkpoint_id"]
        and reference["checkpoint_id"] == full_train["checkpoint_id"]
    )
    train_label_boundary = (
        not qc_train["dev_labels_read"]
        and not full_train["dev_labels_read"]
        and qc_train["training_labels_path"].endswith("qrels_train.jsonl")
        and full_train["training_labels_path"].endswith("qrels_train.jsonl")
    )

    checks = {
        "all_score_files_exact_candidate_coverage": all(
            row["requests"] == len(candidates) and row["rows"] == expected_rows
            for row in score_coverage.values()
        ),
        "assignments_exact": assignments_exact,
        "checkpoints_match_training": checkpoints_exact,
        "common_population_and_qrels_unread": common_population,
        "full_counterfactual_identity_exact": full_counterfactual_exact,
        "neural_request_manifest_hashes_match": neural_request_hashes,
        "optimizer_counts_match": optimizer_counts_match,
        "training_examples_match": training_examples_match,
        "training_reads_train_labels_only": train_label_boundary,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "analysis_type": "frozen_confirmation_pre_label_score_integrity",
        "evidence_mode": "confirmation_labels_unopened",
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_manifest": {
            "path": str(candidate_manifest),
            "sha256": candidate_sha,
            "requests": len(candidates),
            "candidate_rows": expected_rows,
        },
        "request_manifest": {
            "path": str(request_manifest),
            "sha256": request_sha,
        },
        "score_coverage": score_coverage,
        "score_metadata": {
            key: {
                "path": str(run_dirs[key] / "metadata.json"),
                "sha256": sha256_file(run_dirs[key] / "metadata.json"),
                "checkpoint_id": metadata[key].get("checkpoint_id"),
                "qrels_read": metadata[key]["qrels_read"],
            }
            for key in run_keys
        },
        "training_metadata": {
            key: {
                "checkpoint_id": row["checkpoint_id"],
                "examples": row["example_stats"]["examples"],
                "optimizer_steps": row["training"]["optimizer_steps"],
                "metadata_sha256": sha256_file(
                    Path("runs") / runs[key] / "metadata.json"
                ),
            }
            for key, row in train_metadata.items()
        },
        "qrels_confirmation_read": False,
    }
    write_json(OUTPUT, report)
    print(json.dumps({"output": str(OUTPUT), "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 2


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
