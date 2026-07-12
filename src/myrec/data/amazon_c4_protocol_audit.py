"""C1-style protocol audit specialized to the Amazon-C4 secondary track."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.data.protocol_audit import (
    _build_file_checks,
    _build_split_summary,
    _check_candidate_manifest,
    _check_history_causality,
    _check_label_isolation,
    _check_qrels_consistency,
    _check_split_manifest,
    _run_candidate_hash_check,
    _run_metric_unit_tests,
    _scan_candidate_manifest,
    _scan_qrels,
    _scan_records,
    _scan_split_manifest,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


def audit_amazon_c4_protocol(
    *,
    standardized_dir: str | Path,
    c0_report_path: str | Path,
    report_path: str | Path,
    tmp_dir: str | Path = "tmp/amazon_c4_c1_protocol_audit",
    run_unit_tests: bool = True,
) -> dict[str, Any]:
    standardized_dir = Path(standardized_dir)
    c0_report_path = Path(c0_report_path)
    manifest_path = standardized_dir / "manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    split_manifest_path = standardized_dir / "split_manifest.json"
    manifest = _read_json(manifest_path)
    c0 = _read_json(c0_report_path)
    record_scans = {
        split: _scan_records(standardized_dir / f"records_{split}.jsonl", split)
        for split in ("train", "dev", "test")
    }
    qrel_scans = {
        split: _scan_qrels(standardized_dir / f"qrels_{split}.jsonl")
        for split in ("dev", "test")
    }
    candidate_scan = _scan_candidate_manifest(candidate_manifest_path)
    split_scan = _scan_split_manifest(split_manifest_path)
    checks = {
        "amazon_c4_c0": {
            "status": "passed" if c0.get("overall_status") == "passed" else "failed",
            "evidence": {
                "path": str(c0_report_path),
                "sha256": sha256_file(c0_report_path),
                "overall_status": c0.get("overall_status"),
            },
        },
        "split_manifest": _check_split_manifest(record_scans, split_scan),
        "candidate_manifest": _check_candidate_manifest(record_scans, candidate_scan),
        "label_isolation": _check_label_isolation(record_scans),
        "train_blind_equivalence": check_train_blind_equivalence(
            standardized_dir / "records_train.jsonl",
            standardized_dir / "records_train_blind.jsonl",
        ),
        "qrels_consistency": _check_qrels_consistency(qrel_scans, candidate_scan),
        "history_causality": _check_history_causality(record_scans),
        "metric_unit_tests": _run_metric_unit_tests(run_unit_tests),
        "candidate_hash_assertion": _run_candidate_hash_check(
            standardized_dir=standardized_dir,
            candidate_manifest_path=candidate_manifest_path,
            tmp_dir=tmp_dir,
        ),
    }
    report = {
        "report_id": "pps_c1_amazon_c4_protocol",
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "standardized_dir": str(standardized_dir),
        "standardized_manifest_path": str(manifest_path),
        "standardized_manifest_sha256": sha256_file(manifest_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": candidate_scan["sha256"],
        "split_summary": _build_split_summary(record_scans, qrel_scans),
        "files": _build_file_checks(
            manifest,
            manifest_path,
            record_scans,
            qrel_scans,
        ),
        "checks": checks,
        "canary_scope": {
            "status": "not_run",
            "reason": (
                "C38 is train-internal only and cannot call the Amazon dev evaluator; "
                "shared metric tests and the candidate-hash negative control remain binding."
            ),
        },
    }
    report["overall_status"] = (
        "passed" if all(row["status"] == "passed" for row in checks.values()) else "failed"
    )
    write_json(report_path, report)
    return report


def check_train_blind_equivalence(
    labeled_path: str | Path,
    blind_path: str | Path,
) -> dict[str, Any]:
    labeled_path = Path(labeled_path)
    blind_path = Path(blind_path)
    rows = 0
    mismatches = []
    with labeled_path.open("r", encoding="utf-8") as labeled, blind_path.open(
        "r", encoding="utf-8"
    ) as blind:
        while True:
            labeled_line = labeled.readline()
            blind_line = blind.readline()
            if not labeled_line and not blind_line:
                break
            rows += 1
            if not labeled_line or not blind_line:
                mismatches.append({"line": rows, "reason": "row_count"})
                break
            labeled_row = json.loads(labeled_line)
            blind_row = json.loads(blind_line)
            for candidate in labeled_row["candidates"]:
                candidate.pop("clicked", None)
                candidate.pop("purchased", None)
            if labeled_row != blind_row and len(mismatches) < 10:
                mismatches.append(
                    {
                        "line": rows,
                        "reason": "content",
                        "labeled_request_id": labeled_row.get("request_id"),
                        "blind_request_id": blind_row.get("request_id"),
                    }
                )
    return {
        "status": "passed" if not mismatches else "failed",
        "evidence": {
            "rows": rows,
            "mismatches": mismatches,
            "labeled_path": str(labeled_path),
            "labeled_sha256": sha256_file(labeled_path),
            "blind_path": str(blind_path),
            "blind_sha256": sha256_file(blind_path),
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
