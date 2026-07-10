#!/usr/bin/env python
"""Audit supervised diagnostic dev scores without reading any qrels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.supervised_diagnostics import PackedRequestData
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


RUN_NAMES = {
    "d1q": "d1q_supervised_query_dev",
    "d1m": "d1m_mean_history_residual_dev",
    "d1a": "d1a_query_attn_residual_dev",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/analysis/supervised_motivation_diagnostics.yaml",
    )
    parser.add_argument(
        "--final-config",
        default="configs/analysis/supervised_motivation_diagnostics_final.yaml",
    )
    parser.add_argument(
        "--output",
        default="reports/pps_supervised_diagnostics_score_audit.json",
    )
    return parser.parse_args()


def run_id(variant: str, seed: int) -> str:
    return f"20260710_kuaisearch_{RUN_NAMES[variant]}_s{seed}"


def score_path(variant: str, seed: int) -> Path:
    return Path("runs") / run_id(variant, seed) / "scores.jsonl"


def iter_score_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def audit_coverage(path: Path, expected_rows: list[tuple[str, str]]) -> dict:
    count = 0
    methods = set()
    for count, (row, expected) in enumerate(
        zip(iter_score_rows(path), expected_rows, strict=True), start=1
    ):
        expected_request, expected_item = expected
        actual = (str(row["request_id"]), str(row["candidate_item_id"]))
        if actual != expected:
            raise AssertionError(
                f"score order mismatch at row {count}: {actual} != {expected}"
            )
        score = float(row["score"])
        if not math.isfinite(score):
            raise AssertionError(f"non-finite score at {path}:{count}")
        methods.add(str(row["method_id"]))
    if count != len(expected_rows):
        raise AssertionError(f"score row count mismatch: {count} != {len(expected_rows)}")
    if len(methods) != 1:
        raise AssertionError(f"mixed method IDs in {path}: {sorted(methods)}")
    return {
        "method_id": methods.pop(),
        "row_count": count,
        "scores_sha256": sha256_file(path),
        "status": "passed",
    }


def compare_residual_to_base(
    base_path: Path,
    residual_path: Path,
    empty_history_requests: set[str],
) -> dict:
    empty_rows = 0
    empty_mismatches = 0
    nonempty_rows = 0
    nonempty_changed = 0
    max_nonempty_abs_delta = 0.0
    for base, residual in zip(
        iter_score_rows(base_path), iter_score_rows(residual_path), strict=True
    ):
        base_key = (str(base["request_id"]), str(base["candidate_item_id"]))
        residual_key = (
            str(residual["request_id"]),
            str(residual["candidate_item_id"]),
        )
        if base_key != residual_key:
            raise AssertionError(f"paired score mismatch: {base_key} != {residual_key}")
        base_score = float(base["score"])
        residual_score = float(residual["score"])
        if base_key[0] in empty_history_requests:
            empty_rows += 1
            if base_score != residual_score:
                empty_mismatches += 1
        else:
            nonempty_rows += 1
            delta = abs(residual_score - base_score)
            max_nonempty_abs_delta = max(max_nonempty_abs_delta, delta)
            nonempty_changed += int(delta > 0.0)
    return {
        "empty_history_candidate_rows": empty_rows,
        "empty_history_exact_mismatches": empty_mismatches,
        "empty_history_exact_equal": empty_mismatches == 0,
        "nonempty_history_candidate_rows": nonempty_rows,
        "nonempty_history_changed_rows": nonempty_changed,
        "max_nonempty_abs_score_delta": max_nonempty_abs_delta,
        "status": (
            "passed" if empty_mismatches == 0 and nonempty_changed > 0 else "failed"
        ),
    }


def main() -> int:
    args = parse_args()
    with Path(args.base_config).open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    with Path(args.final_config).open("r", encoding="utf-8") as handle:
        final_config = yaml.safe_load(handle)

    manifest_path = Path(base_config["standardized_dir"]) / "candidate_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_rows = [
        (str(entry["request_id"]), str(item_id))
        for entry in manifest["entries"]
        if entry["split"] == "dev"
        for item_id in entry["candidate_item_ids"]
    ]

    packed = PackedRequestData.load(
        Path(base_config["materialized_data"]["output_dir"]), "dev"
    )
    empty_history_requests = {
        packed.request_ids[index]
        for index in range(len(packed))
        if packed.history_offsets[index] == packed.history_offsets[index + 1]
    }
    seeds = [int(seed) for seed in final_config["final_training"]["seeds"]]

    coverage = {}
    residual_checks = {}
    wrong_history_checks = {}
    for seed in seeds:
        for variant in RUN_NAMES:
            path = score_path(variant, seed)
            metadata_path = path.parent / "metadata.json"
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("qrels_read") is not False or metadata.get("test_read") is not False:
                raise AssertionError(f"label-isolation metadata failed: {metadata_path}")
            if metadata.get("candidate_manifest_sha256") != sha256_file(manifest_path):
                raise AssertionError(f"candidate manifest hash mismatch: {metadata_path}")
            coverage[run_id(variant, seed)] = audit_coverage(path, expected_rows)
        for residual in ("d1m", "d1a"):
            residual_checks[f"{residual}_vs_d1q_s{seed}"] = compare_residual_to_base(
                score_path("d1q", seed),
                score_path(residual, seed),
                empty_history_requests,
            )
        wrong_run_id = f"20260710_kuaisearch_d1a_wrong_history_dev_s{seed}"
        wrong_path = Path("runs") / wrong_run_id / "scores.jsonl"
        if wrong_path.exists():
            metadata_path = wrong_path.parent / "metadata.json"
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("qrels_read") is not False or metadata.get("test_read") is not False:
                raise AssertionError(f"label-isolation metadata failed: {metadata_path}")
            if metadata.get("candidate_manifest_sha256") != sha256_file(manifest_path):
                raise AssertionError(f"candidate manifest hash mismatch: {metadata_path}")
            coverage[wrong_run_id] = audit_coverage(wrong_path, expected_rows)
            wrong_history_checks[f"d1a_true_vs_wrong_s{seed}"] = (
                compare_residual_to_base(
                    score_path("d1a", seed),
                    wrong_path,
                    empty_history_requests,
                )
            )

    passed = all(row["status"] == "passed" for row in coverage.values()) and all(
        row["status"] == "passed" for row in residual_checks.values()
    ) and all(row["status"] == "passed" for row in wrong_history_checks.values())
    report = {
        "analysis_id": base_config["analysis_id"],
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "dev_candidate_rows": len(expected_rows),
        "dev_requests": len(packed),
        "empty_history_requests": len(empty_history_requests),
        "final_config_path": args.final_config,
        "final_config_sha256": sha256_file(args.final_config),
        "qrels_read": False,
        "score_coverage": coverage,
        "residual_base_exactness": residual_checks,
        "true_wrong_history_exactness": wrong_history_checks,
        "status": "passed" if passed else "failed",
        "test_read": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
